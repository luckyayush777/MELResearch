import os
import json
import time
import requests

# -- CONFIG --------------------------------------------------------------------
VALID_META_PATH = os.path.join("data", "embeddings", "valid_metadata.json")
OUT_PATH        = os.path.join("data", "entity_kb.json")
CHECKPOINT_PATH = os.path.join("data", "entity_kb_checkpoint.json")
TARGET          = 30000
BATCH_SIZE      = 50      # Wikipedia API max titles per extracts call
SLEEP           = 1.1     # seconds between requests
MIN_DESC_LEN    = 20
HEADERS = {
    "User-Agent": "MEL-Quantization-Research/1.0 (ayushrocks456@gmail.com)"
}
API = "https://en.wikipedia.org/w/api.php"
# ------------------------------------------------------------------------------

os.makedirs("data", exist_ok=True)

# Resume if interrupted
if os.path.exists(OUT_PATH):
    with open(OUT_PATH, "r", encoding="utf-8") as f:
        kb = json.load(f)
    print(f"Resuming: {len(kb)} entities already collected.")
else:
    kb = []

seen_titles = set(e["title"] for e in kb)

# Load checkpoint (Phase 2 cursor) if it exists
checkpoint = {}
if os.path.exists(CHECKPOINT_PATH):
    with open(CHECKPOINT_PATH, "r", encoding="utf-8") as f:
        checkpoint = json.load(f)
    print(f"Checkpoint loaded: phase1_done={checkpoint.get('phase1_done')}, gapcontinue={checkpoint.get('gapcontinue')!r}")

def save():
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(kb, f, indent=2, ensure_ascii=False)

def save_checkpoint(apcontinue, phase1_done=True):
    with open(CHECKPOINT_PATH, "w", encoding="utf-8") as f:
        json.dump({"phase1_done": phase1_done, "apcontinue": apcontinue}, f)

def is_junk(title, text):
    if title.startswith("List of"):
        return True
    if "(disambiguation)" in title:
        return True
    if not text or len(text.strip()) < MIN_DESC_LEN:
        return True
    return False

def fetch_descriptions(titles):
    """
    Fetch extracts for a list of titles (up to 50 at a time) via titles= param.
    Returns dict: {title -> description_text}
    """
    results = {}
    for i in range(0, len(titles), BATCH_SIZE):
        batch = titles[i : i + BATCH_SIZE]
        params = {
            "action":      "query",
            "titles":      "|".join(batch),
            "prop":        "extracts",
            "exintro":     1,
            "exsentences": 2,
            "explaintext": 1,
            "format":      "json",
        }
        try:
            r = requests.get(API, params=params, headers=HEADERS, timeout=15)
            if r.status_code in (429, 503):
                wait = int(r.headers.get("Retry-After", 60))
                print(f"\n  [RATE LIMIT] Waiting {wait}s ...")
                time.sleep(wait)
                # retry same batch
                r = requests.get(API, params=params, headers=HEADERS, timeout=15)
            r.raise_for_status()
            data = r.json()
            for page in data.get("query", {}).get("pages", {}).values():
                t = page.get("title", "")
                text = page.get("extract", "").strip()
                if t and text:
                    results[t] = text[:300]
        except Exception as e:
            print(f"\n  [WARN] fetch_descriptions failed: {e}")
        time.sleep(SLEEP)
    return results

# ── PHASE 1: seed KB with image entities (guaranteed correct matches) ──────────
if checkpoint.get("phase1_done"):
    print("Phase 1: already done (checkpoint). Skipping.")
else:
    print("Phase 1: seeding KB with image entities ...")
    if not os.path.exists(VALID_META_PATH):
        print(f"  [WARN] {VALID_META_PATH} not found. Run generate_embeddings.py first.")
    else:
        with open(VALID_META_PATH, "r", encoding="utf-8") as f:
            valid_meta = json.load(f)
        image_titles = [item["title"] for item in valid_meta if item["title"] not in seen_titles]
        print(f"  {len(image_titles)} image entities not yet in KB, fetching descriptions ...")

        desc_map = fetch_descriptions(image_titles)
        added = 0
        for item in valid_meta:
            t = item["title"]
            if t in seen_titles:
                continue
            text = item.get("lead_text", desc_map.get(t, ""))[:300]
            if not text:
                continue
            kb.append({"title": t, "description": text, "is_query_entity": True})
            seen_titles.add(t)
            added += 1

        save()
        save_checkpoint(apcontinue="A", phase1_done=True)
        print(f"  Phase 1 done: {added} image entities added. KB size: {len(kb)}")

# ── PHASE 2: fill remaining slots with alphabetical distractor entities ────────
# Two-step approach: list=allpages for reliable title pagination (apcontinue),
# then a separate titles= call for extracts. Decoupled so neither limits the other.
print(f"\nPhase 2: filling to {TARGET} with distractor entities ...")
apcontinue = checkpoint.get("apcontinue", "A")   # resume cursor; default start from "A"
if apcontinue != "A":
    print(f"  Resuming Phase 2 from cursor: {apcontinue!r}")

def api_request(params, label=""):
    """GET with rate-limit backoff. Returns parsed JSON or None on failure."""
    for attempt in range(3):
        try:
            r = requests.get(API, params=params, headers=HEADERS, timeout=15)
            if r.status_code in (429, 503):
                wait = int(r.headers.get("Retry-After", 60))
                print(f"\n  [RATE LIMIT] {label} HTTP {r.status_code}. Waiting {wait}s ...")
                time.sleep(wait)
                continue
            r.raise_for_status()
            data = r.json()
            if "error" in data:
                print(f"\n  [API ERROR] {label} {data['error'].get('code')}. Waiting 15s ...")
                time.sleep(15)
                continue
            return data
        except requests.exceptions.Timeout:
            print(f"\n  [WARN] {label} timeout. Waiting 10s ...")
            time.sleep(10)
        except Exception as e:
            print(f"\n  [WARN] {label} failed: {e}. Waiting 10s ...")
            time.sleep(10)
    return None

while len(kb) < TARGET:
    # Step 1: get a batch of titles via list=allpages
    data = api_request({
        "action":         "query",
        "list":           "allpages",
        "aplimit":        BATCH_SIZE,
        "apnamespace":    0,
        "apfilterredir":  "nonredirects",
        "apfrom":         apcontinue,
        "format":         "json",
    }, label="allpages")

    if data is None:
        continue

    titles_batch = [p["title"] for p in data.get("query", {}).get("allpages", [])]
    next_apcontinue = data.get("continue", {}).get("apcontinue")

    # Filter out already-seen titles before making the extracts call
    new_titles = [t for t in titles_batch if t not in seen_titles and not is_junk(t, "x")]

    time.sleep(SLEEP)

    if new_titles:
        # Step 2: fetch extracts for unseen titles only
        ext_data = api_request({
            "action":      "query",
            "titles":      "|".join(new_titles),
            "prop":        "extracts",
            "exintro":     1,
            "exsentences": 2,
            "explaintext": 1,
            "format":      "json",
        }, label="extracts")
        time.sleep(SLEEP)
    else:
        ext_data = None

    added = 0
    if ext_data:
        for page in ext_data.get("query", {}).get("pages", {}).values():
            title = page.get("title", "")
            text  = page.get("extract", "").strip()
            if title in seen_titles:
                continue
            if is_junk(title, text):
                continue
            kb.append({"title": title, "description": text[:300], "is_query_entity": False})
            seen_titles.add(title)
            added += 1
            if len(kb) >= TARGET:
                break

    # Advance cursor and checkpoint every batch
    apcontinue = next_apcontinue
    save_checkpoint(apcontinue, phase1_done=True)

    print(f"  Collected {len(kb):,}/{TARGET}  (+{added} this batch, cursor={apcontinue!r})", end="\r")

    if len(kb) % 1000 < BATCH_SIZE:
        save()
        print(f"\n  [SAVED] {len(kb):,} entities written to {OUT_PATH}")

    if not apcontinue:
        print("\n  [INFO] Reached end of allpages index.")
        break

save()
if os.path.exists(CHECKPOINT_PATH):
    os.remove(CHECKPOINT_PATH)

n_query  = sum(1 for e in kb if e.get("is_query_entity"))
n_distractor = len(kb) - n_query
print(f"\nDone. {len(kb):,} total entities in KB")
print(f"  Query entities (guaranteed correct matches): {n_query}")
print(f"  Distractor entities:                         {n_distractor}")
print(f"Saved to {OUT_PATH}")
