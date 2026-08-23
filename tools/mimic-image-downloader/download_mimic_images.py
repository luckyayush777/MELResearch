#!/usr/bin/env python3
"""
Checkpointable downloader for the MIMIC image archives used by FAST-MEL.

The three datasets (WikiMEL, RichpediaMEL, WikiDiverse) are published by the
MIMIC authors as password-protected OneDrive/SharePoint links, as uncompressed
.tar files totalling roughly 7.4 GiB. Every stage here is resumable: interrupt
with Ctrl-C at any point and re-run the same command to pick up where it left
off.

    WikiMEL.tar        2.96 GiB
    RichpediaMEL.tar   2.63 GiB
    WikiDiverse.tar    1.80 GiB

Pipeline per dataset, each stage checkpointed independently:

    download -> clear the SharePoint password gate, then stream to
                <name>.tar.part using HTTP Range to resume (the server
                returns 206, so resume genuinely works)
    verify   -> confirm the archive is a readable tar, record its sha256
    extract  -> unpack member-by-member, skipping members already on disk
    arrange  -> normalise the top-level directory name and report the
                kb_image / mention_image counts the configs depend on

Two ways to get the archives:

  automated   python download_mimic_images.py --dataset all
  manual      download the .tar files yourself (password: kdd2023), drop them
              into mimic-dataset-archives/, then run with --manual. An archive
              already present is used as-is and the run skips to extraction.

Useful flags:
  --status            show checkpoint progress for all datasets
  --reset <dataset>   throw away one dataset's progress and start it over

Stdlib only, so this runs under any Python 3.8+ without installing anything.
"""

import argparse
import hashlib
import html
import http.cookiejar
import json
import os
import re
import sys
import tarfile
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile

# --------------------------------------------------------------------------
# Layout
# --------------------------------------------------------------------------

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(HERE)
ARCHIVE_DIR = os.path.join(PROJECT_ROOT, "mimic-dataset-archives")
IMAGE_DIR = os.path.join(PROJECT_ROOT, "mimic-dataset-images")
CHECKPOINT_PATH = os.path.join(ARCHIVE_DIR, "download-checkpoint.json")

# Share links and password are taken verbatim from the MIMIC readme:
# https://github.com/pengfei-luo/MIMIC  ("Step 2: Download the data")
SHARE_PASSWORD = "kdd2023"

DATASETS = {
    "WikiMEL": {
        "share_url": "https://mailustceducn-my.sharepoint.com/:u:/g/personal/pfluo_mail_ustc_edu_cn/ETtT1zwqdDdAmE-uxHMX5EAB7bCGb1Eh2AuafB0tijDdyg?e=IT9E8a",
        "archive_name": "WikiMEL.tar",
    },
    "RichpediaMEL": {
        "share_url": "https://mailustceducn-my.sharepoint.com/:u:/g/personal/pfluo_mail_ustc_edu_cn/ERikbOQuoWFHrA_AizcuCbgB8PBOiRqCV4U0lZfxUN-6kg?e=speIdh",
        "archive_name": "RichpediaMEL.tar",
    },
    "WikiDiverse": {
        "share_url": "https://mailustceducn-my.sharepoint.com/:u:/g/personal/pfluo_mail_ustc_edu_cn/EQgQKn4VeghChY_lhUoyBIMBKz6aTS00DFKOL1dqxP_bEg?e=yRpKkU",
        "archive_name": "WikiDiverse.tar",
    },
}

STAGES = ["pending", "downloaded", "verified", "extracted", "unpacked", "arranged", "done"]

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
)

CHUNK = 1 << 20          # 1 MiB read size
FLUSH_EVERY = 16 << 20   # persist checkpoint every 16 MiB of download
EXTRACT_FLUSH_EVERY = 2000   # persist checkpoint every N tar members
MAX_ATTEMPTS = 12        # re-auth + resume attempts before giving up
RETRY_BACKOFF = 10       # seconds, multiplied by the attempt number


# --------------------------------------------------------------------------
# Checkpoint
# --------------------------------------------------------------------------

def _blank_entry(name):
    return {
        "stage": "pending",
        "archive_name": DATASETS[name]["archive_name"],
        "bytes_downloaded": 0,
        "total_bytes": None,
        "sha256": None,
        "members_extracted": 0,
        "members_total": None,
        "top_level_dir": None,
        "image_folders": None,
        "updated_at": None,
    }


def load_checkpoint():
    state = {}
    if os.path.exists(CHECKPOINT_PATH):
        try:
            with open(CHECKPOINT_PATH, "r", encoding="utf-8") as f:
                state = json.load(f)
        except (ValueError, OSError) as exc:
            # A checkpoint torn by a hard kill must not wedge the tool forever.
            print("  ! checkpoint unreadable (%s), starting a fresh one" % exc)
            state = {}

    state.setdefault("version", 1)
    state.setdefault("datasets", {})
    for name in DATASETS:
        entry = state["datasets"].setdefault(name, _blank_entry(name))
        for key, default in _blank_entry(name).items():
            entry.setdefault(key, default)
    return state


def save_checkpoint(state):
    """Atomic write: a Ctrl-C mid-save must never corrupt the checkpoint."""
    os.makedirs(ARCHIVE_DIR, exist_ok=True)
    tmp = CHECKPOINT_PATH + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, CHECKPOINT_PATH)


def touch(entry):
    entry["updated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")


def stage_before(entry, stage):
    return STAGES.index(entry["stage"]) < STAGES.index(stage)


def human(n):
    if n is None:
        return "?"
    n = float(n)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if abs(n) < 1024:
            return "%.1f %s" % (n, unit)
        n /= 1024.0
    return "%.1f PiB" % n


# --------------------------------------------------------------------------
# SharePoint: clear the anonymous-password gate, then build a direct file URL
# --------------------------------------------------------------------------

HIDDEN_INPUT_RE = re.compile(
    r'<input[^>]*type="hidden"[^>]*name="([^"]+)"[^>]*value="([^"]*)"[^>]*>',
    re.IGNORECASE,
)
# Attribute order inside <form> is not guaranteed, so grab the whole tag first
# and pull `action` out of it separately.
FORM_TAG_RE = re.compile(r'<form[^>]*\bid="inputForm"[^>]*>', re.IGNORECASE)
ACTION_ATTR_RE = re.compile(r'\baction="([^"]+)"', re.IGNORECASE)


def _build_opener():
    jar = http.cookiejar.CookieJar()
    opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(jar))
    opener.addheaders = [("User-Agent", USER_AGENT)]
    return opener, jar


def resolve_direct_url(share_url, password, opener):
    """Authenticate against the share link and return a raw-bytes URL.

    The share page is an ASP.NET WebForms password gate. Replaying its postback
    with the password puts a FedAuth cookie on the session and redirects to the
    OneDrive viewer, whose `id` query parameter is the file's server-relative
    path. Feeding that to download.aspx yields the file itself, with Range
    support intact.
    """
    with opener.open(share_url, timeout=60) as resp:
        page = resp.read().decode("utf-8", "replace")
        landed_on = resp.geturl()

    if "txtPassword" in page:
        fields = {name: html.unescape(val) for name, val in HIDDEN_INPUT_RE.findall(page)}
        fields["txtPassword"] = password
        fields["__EVENTTARGET"] = "btnSubmitPassword"
        fields["__EVENTARGUMENT"] = ""

        form = FORM_TAG_RE.search(page)
        if not form:
            raise RuntimeError("could not locate the password form on the share page")
        action_attr = ACTION_ATTR_RE.search(form.group(0))
        if not action_attr:
            raise RuntimeError("password form has no action attribute")
        action = urllib.parse.urljoin(landed_on, html.unescape(action_attr.group(1)))

        req = urllib.request.Request(
            action,
            data=urllib.parse.urlencode(fields).encode("utf-8"),
            headers={
                "User-Agent": USER_AGENT,
                "Content-Type": "application/x-www-form-urlencoded",
                "Referer": landed_on,
            },
        )
        with opener.open(req, timeout=60) as resp:
            after = resp.read().decode("utf-8", "replace")
            landed_on = resp.geturl()

        if "txtPassword" in after:
            raise RuntimeError("SharePoint rejected the share password")

    parts = urllib.parse.urlsplit(landed_on)
    source_url = dict(urllib.parse.parse_qsl(parts.query)).get("id")
    if not source_url:
        raise RuntimeError("authenticated, but could not determine the file path "
                           "from %s" % landed_on)

    m = re.search(r"(/personal/[^/]+)/", source_url)
    if not m:
        raise RuntimeError("unexpected file path layout: %s" % source_url)

    return urllib.parse.urlunsplit((
        parts.scheme,
        parts.netloc,
        m.group(1) + "/_layouts/15/download.aspx",
        "SourceUrl=" + urllib.parse.quote(source_url, safe=""),
        "",
    ))


# --------------------------------------------------------------------------
# Stage: download  (HTTP Range resume)
# --------------------------------------------------------------------------

def download_with_resume(url, dest, entry, state, opener):
    part = dest + ".part"
    have = os.path.getsize(part) if os.path.exists(part) else 0

    headers = {"User-Agent": USER_AGENT}
    if have:
        headers["Range"] = "bytes=%d-" % have

    try:
        resp = opener.open(urllib.request.Request(url, headers=headers), timeout=120)
    except urllib.error.HTTPError as exc:
        if exc.code == 416 and entry.get("total_bytes") and have >= entry["total_bytes"]:
            os.replace(part, dest)   # already complete
            return
        raise

    with resp:
        if have and resp.getcode() != 206:
            print("  ! server ignored Range, restarting from 0")
            have = 0
            mode = "wb"
        else:
            mode = "ab" if have else "wb"

        length = resp.headers.get("Content-Length")
        total = (have + int(length)) if length is not None else entry.get("total_bytes")
        entry["total_bytes"] = total
        entry["bytes_downloaded"] = have
        touch(entry)
        save_checkpoint(state)

        if have:
            print("  resuming at %s / %s" % (human(have), human(total)))

        since_flush = 0
        started = time.time()
        at_start = have
        with open(part, mode) as f:
            while True:
                buf = resp.read(CHUNK)
                if not buf:
                    break
                f.write(buf)
                have += len(buf)
                since_flush += len(buf)

                if since_flush >= FLUSH_EVERY:
                    f.flush()
                    os.fsync(f.fileno())
                    entry["bytes_downloaded"] = have
                    touch(entry)
                    save_checkpoint(state)
                    since_flush = 0

                elapsed = max(time.time() - started, 1e-6)
                rate = (have - at_start) / elapsed / (1 << 20)
                eta = ((total - have) / max(have - at_start, 1) * elapsed) if total else 0
                sys.stdout.write(
                    "\r  %s / %s (%.1f%%)  %.1f MiB/s  ETA %s   "
                    % (human(have), human(total),
                       (100.0 * have / total) if total else 0.0,
                       rate, time.strftime("%H:%M:%S", time.gmtime(eta)))
                )
                sys.stdout.flush()

            f.flush()
            os.fsync(f.fileno())

    sys.stdout.write("\n")
    entry["bytes_downloaded"] = have
    touch(entry)
    save_checkpoint(state)

    if total and have < total:
        raise RuntimeError("connection closed early: got %s of %s — re-run to resume"
                           % (human(have), human(total)))

    os.replace(part, dest)


def download_with_retries(cfg, dest, entry, state, password):
    """Download, re-authenticating and resuming across dropped connections.

    A multi-GB transfer routinely outlives the FedAuth cookie, and the server
    also just drops connections. Each attempt re-clears the password gate for a
    fresh cookie and resumes from whatever is already in the .part file, so no
    downloaded bytes are ever thrown away.
    """
    part = dest + ".part"
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            opener, _ = _build_opener()
            print("  authenticating (attempt %d/%d) ..." % (attempt, MAX_ATTEMPTS))
            url = resolve_direct_url(cfg["share_url"], password, opener)
            print("  downloading %s ..." % cfg["archive_name"])
            download_with_resume(url, dest, entry, state, opener)
            return
        except KeyboardInterrupt:
            raise
        except Exception as exc:
            have = os.path.getsize(part) if os.path.exists(part) else 0
            save_checkpoint(state)
            if attempt == MAX_ATTEMPTS:
                raise
            wait = RETRY_BACKOFF * attempt
            print("\n  ! %s" % exc)
            print("  have %s so far; retrying in %ds (progress is kept)"
                  % (human(have), wait))
            time.sleep(wait)


# --------------------------------------------------------------------------
# Stage: verify
# --------------------------------------------------------------------------

def sha256_of(path):
    h = hashlib.sha256()
    size = os.path.getsize(path)
    read = 0
    with open(path, "rb") as f:
        while True:
            buf = f.read(CHUNK)
            if not buf:
                break
            h.update(buf)
            read += len(buf)
            sys.stdout.write("\r  hashing %.1f%%" % (100.0 * read / size if size else 100.0))
            sys.stdout.flush()
    sys.stdout.write("\n")
    return h.hexdigest()


def verify_archive(path, entry):
    if not tarfile.is_tarfile(path):
        raise RuntimeError(
            "%s is not a valid tar. It is most likely a SharePoint error page "
            "saved under the wrong name — delete it and re-run." % os.path.basename(path)
        )
    entry["sha256"] = sha256_of(path)


# --------------------------------------------------------------------------
# Stage: extract  (resumable, member by member)
# --------------------------------------------------------------------------

def _is_within(base, target):
    """Guard against tar members that escape the destination via ../ or absolute paths."""
    base = os.path.abspath(base)
    target = os.path.abspath(target)
    return target == base or target.startswith(base + os.sep)


def extract_with_resume(archive, out_dir, entry, state):
    """Unpack into out_dir, skipping members already present at the right size.

    Resume is driven by what is actually on disk rather than purely by the
    member counter, so an extraction killed midway through a file is redone
    correctly rather than left truncated.
    """
    os.makedirs(out_dir, exist_ok=True)
    done = 0
    skipped = 0
    top_levels = set()

    with tarfile.open(archive, "r:") as tf:
        for member in tf:
            done += 1
            top_levels.add(member.name.split("/")[0])

            target = os.path.join(out_dir, member.name)
            if not _is_within(out_dir, target):
                raise RuntimeError("refusing unsafe tar member path: %r" % member.name)

            if member.isreg() and os.path.exists(target) \
                    and os.path.getsize(target) == member.size:
                skipped += 1
            else:
                tf.extract(member, out_dir)

            if done % EXTRACT_FLUSH_EVERY == 0:
                entry["members_extracted"] = done
                touch(entry)
                save_checkpoint(state)
                sys.stdout.write("\r  %d members (%d already on disk)" % (done, skipped))
                sys.stdout.flush()

    sys.stdout.write("\r  %d members (%d already on disk)\n" % (done, skipped))
    entry["members_extracted"] = done
    entry["members_total"] = done
    entry["top_level_dir"] = sorted(top_levels)[0] if len(top_levels) == 1 else None
    touch(entry)
    save_checkpoint(state)


# --------------------------------------------------------------------------
# Stage: unpack  (the tars hold nested zips, not loose images)
# --------------------------------------------------------------------------

def unpack_nested_zips(dataset_dir, entry, state):
    """Unzip kb_image.zip / mention_image.zip in place.

    Each .tar turns out to contain only ~9 members: the JSON metadata plus the
    images bundled as *separate zip files*. Those zips already carry the right
    top-level folder (`kb_image/`, `mention_image/`), so extracting them into
    the dataset directory produces exactly the layout the configs expect.

    Resumable on the same principle as the tar stage: a member already on disk
    at the correct size is skipped, so an interrupted unzip is repaired rather
    than restarted.
    """
    zips = sorted(f for f in os.listdir(dataset_dir) if f.lower().endswith(".zip"))
    if not zips:
        print("  no nested zips found (images may already be loose)")
        return

    done_zips = entry.get("unpacked_zips") or []

    for zname in zips:
        zpath = os.path.join(dataset_dir, zname)
        print("  unpacking %s ..." % zname)
        extracted = skipped = 0

        with zipfile.ZipFile(zpath) as zf:
            members = zf.infolist()
            for i, member in enumerate(members, 1):
                target = os.path.join(dataset_dir, member.filename)
                if not _is_within(dataset_dir, target):
                    raise RuntimeError("refusing unsafe zip member path: %r" % member.filename)

                if member.is_dir():
                    os.makedirs(target, exist_ok=True)
                elif os.path.exists(target) and os.path.getsize(target) == member.file_size:
                    skipped += 1
                else:
                    zf.extract(member, dataset_dir)
                    extracted += 1

                if i % EXTRACT_FLUSH_EVERY == 0:
                    entry["unpack_progress"] = "%s %d/%d" % (zname, i, len(members))
                    touch(entry)
                    save_checkpoint(state)
                    sys.stdout.write("\r    %d / %d  (%d new, %d already on disk)"
                                     % (i, len(members), extracted, skipped))
                    sys.stdout.flush()

            sys.stdout.write("\r    %d / %d  (%d new, %d already on disk)\n"
                             % (len(members), len(members), extracted, skipped))

        if zname not in done_zips:
            done_zips.append(zname)
        entry["unpacked_zips"] = done_zips
        entry["unpack_progress"] = None
        touch(entry)
        save_checkpoint(state)


# --------------------------------------------------------------------------
# Stage: arrange
# --------------------------------------------------------------------------

# What the config YAMLs + codes/utils/dataset.py expect to find, per dataset.
# NOTE: RichpediaMEL ships mention images in `mention_images` (plural) and its
# shipped config already points there; both spellings are accepted.
EXPECTED_FOLDERS = {
    "WikiMEL": ["kb_image", "mention_image"],
    "RichpediaMEL": ["kb_image", "mention_images", "mention_image"],
    "WikiDiverse": ["kb_image", "mention_image"],
}


def arrange(name, entry):
    """Normalise <IMAGE_DIR>/<top-level dir> to <IMAGE_DIR>/<dataset name>."""
    dest = os.path.join(IMAGE_DIR, name)
    top = entry.get("top_level_dir")

    if top and top != name:
        src = os.path.join(IMAGE_DIR, top)
        if os.path.isdir(src) and not os.path.isdir(dest):
            print("  renaming %r -> %r" % (top, name))
            os.rename(src, dest)

    if not os.path.isdir(dest):
        raise RuntimeError("expected %s after extraction, but it does not exist" % dest)

    found = {}
    for folder in EXPECTED_FOLDERS[name]:
        path = os.path.join(dest, folder)
        found[folder] = sum(1 for _ in os.scandir(path)) if os.path.isdir(path) else None

    for folder, count in found.items():
        if count is not None:
            print("  %-16s %d files" % (folder + ":", count))
    missing = [f for f, c in found.items() if c is None]
    if missing and len(missing) == len(found):
        print("  ! none of the expected image folders were found under %s" % dest)
        print("    contents: %s" % sorted(os.listdir(dest))[:20])

    entry["image_folders"] = found
    entry["image_root"] = dest


# --------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------

def process(name, state, password, manual_only):
    entry = state["datasets"][name]
    cfg = DATASETS[name]
    archive = os.path.join(ARCHIVE_DIR, cfg["archive_name"])

    print("\n=== %s ===" % name)

    if entry["stage"] == "done":
        print("  already complete — skipping (use --reset %s to redo)" % name)
        return

    os.makedirs(ARCHIVE_DIR, exist_ok=True)
    os.makedirs(IMAGE_DIR, exist_ok=True)

    # -- download -----------------------------------------------------------
    if stage_before(entry, "downloaded"):
        if os.path.exists(archive):
            print("  archive already present, skipping download")
        elif manual_only:
            print("  ! %s not found, and --manual was given." % cfg["archive_name"])
            print("    Download it here (password: %s):" % password)
            print("      %s" % cfg["share_url"])
            print("    Save it as: %s" % archive)
            print("    Then re-run this command.")
            return
        else:
            # Cookies live on the opener, not in the checkpoint, so every run
            # (including a resumed one) authenticates first.
            download_with_retries(cfg, archive, entry, state, password)

        entry["stage"] = "downloaded"
        touch(entry)
        save_checkpoint(state)

    # -- verify -------------------------------------------------------------
    if stage_before(entry, "verified"):
        print("  verifying archive ...")
        verify_archive(archive, entry)
        entry["stage"] = "verified"
        touch(entry)
        save_checkpoint(state)

    # -- extract ------------------------------------------------------------
    if stage_before(entry, "extracted"):
        print("  extracting into %s ..." % IMAGE_DIR)
        extract_with_resume(archive, IMAGE_DIR, entry, state)
        entry["stage"] = "extracted"
        touch(entry)
        save_checkpoint(state)

    # -- unpack nested zips -------------------------------------------------
    if stage_before(entry, "unpacked"):
        dataset_dir = os.path.join(IMAGE_DIR, entry.get("top_level_dir") or name)
        print("  unpacking nested image zips ...")
        unpack_nested_zips(dataset_dir, entry, state)
        entry["stage"] = "unpacked"
        touch(entry)
        save_checkpoint(state)

    # -- arrange ------------------------------------------------------------
    if stage_before(entry, "arranged"):
        print("  arranging ...")
        arrange(name, entry)
        entry["stage"] = "arranged"
        touch(entry)
        save_checkpoint(state)

    entry["stage"] = "done"
    touch(entry)
    save_checkpoint(state)
    print("  done")


def cmd_status(state):
    print("checkpoint: %s" % CHECKPOINT_PATH)
    print()
    print("%-14s %-11s %-26s %-14s %s" % ("DATASET", "STAGE", "DOWNLOADED", "MEMBERS", "UPDATED"))
    for name in DATASETS:
        e = state["datasets"][name]
        pct = ""
        if e["total_bytes"]:
            pct = " (%.1f%%)" % (100.0 * e["bytes_downloaded"] / e["total_bytes"])
        dl = "%s / %s%s" % (human(e["bytes_downloaded"]), human(e["total_bytes"]), pct)
        print("%-14s %-11s %-26s %-14s %s"
              % (name, e["stage"], dl, e["members_extracted"], e["updated_at"] or "-"))


def main():
    p = argparse.ArgumentParser(
        description="Checkpointable downloader for the MIMIC image archives.")
    p.add_argument("--dataset", default="all", choices=["all"] + list(DATASETS),
                   help="which dataset to process (default: all)")
    p.add_argument("--password", default=SHARE_PASSWORD,
                   help="SharePoint share password (default: the one in the MIMIC readme)")
    p.add_argument("--manual", action="store_true",
                   help="never hit the network; only process archives already "
                        "present in mimic-dataset-archives/")
    p.add_argument("--status", action="store_true",
                   help="print checkpoint progress and exit")
    p.add_argument("--reset", metavar="DATASET",
                   help="clear the checkpoint for one dataset (or 'all') and start over")
    p.add_argument("--reunpack", action="store_true",
                   help="rewind finished datasets to the 'extracted' stage so the "
                        "nested-zip unpack and arrange steps run again (keeps the "
                        "downloaded archives; the unpack itself is idempotent)")
    args = p.parse_args()

    state = load_checkpoint()

    if args.status:
        cmd_status(state)
        return 0

    if args.reunpack:
        names = list(DATASETS) if args.dataset == "all" else [args.dataset]
        for name in names:
            entry = state["datasets"][name]
            if STAGES.index(entry["stage"]) > STAGES.index("extracted"):
                entry["stage"] = "extracted"
                entry["unpacked_zips"] = []
                print("rewound %s to 'extracted'" % name)
        save_checkpoint(state)

    if args.reset:
        targets = list(DATASETS) if args.reset == "all" else [args.reset]
        for name in targets:
            if name not in DATASETS:
                print("unknown dataset: %s" % name)
                return 2
            state["datasets"][name] = _blank_entry(name)
            print("reset checkpoint for %s" % name)
        save_checkpoint(state)
        return 0

    names = list(DATASETS) if args.dataset == "all" else [args.dataset]

    try:
        for name in names:
            process(name, state, args.password, args.manual)
    except KeyboardInterrupt:
        save_checkpoint(state)
        print("\n\ninterrupted — progress saved. Re-run the same command to resume.")
        return 130
    except Exception as exc:
        save_checkpoint(state)
        print("\n\nFAILED: %s" % exc)
        print("Progress is saved; re-run the same command to resume from here.")
        return 1

    print("\nAll requested datasets complete.")
    cmd_status(state)
    return 0


if __name__ == "__main__":
    sys.exit(main())
