#!/usr/bin/env python3
"""
Check that the downloaded images actually line up with what the model will ask for.

This exists because of a trap in codes/utils/dataset.py::choose_image():

    try:
        img = Image.open(img_path)...
    except:
        pixel_values = torch.rand((3, 224, 224))

Every failure — wrong folder, missing file, corrupt jpeg — is swallowed and
replaced with *random noise*. Training therefore runs happily to completion on a
misconfigured image path and simply reports bad numbers. Nothing crashes, and
nothing warns you.

So before trusting a run, confirm the lookup actually hits. This script resolves
filenames using exactly the same rules as choose_image():

  * entity  (sample_type 0) -> kb_img_folder / <name from kb_entity image_list>
  * mention (sample_type 1) -> mention_img_folder / <basename>.jpg
    (dataset.py does img.split('/')[-1].split('.')[0] + '.jpg')

Usage:
    python verify_image_coverage.py --config ..\\local-configs\\wikimel_local.yaml
    python verify_image_coverage.py --all          # all local-configs/*_local.yaml
    python verify_image_coverage.py --all --full   # check every record, not a sample

Reads the YAML with a tiny purpose-built parser, so it needs no dependencies and
can run outside the venv.
"""

import argparse
import glob
import json
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(HERE)
CONFIG_DIR = os.path.join(PROJECT_ROOT, "local-configs")

SAMPLE_SIZE = 4000


def load_config(path):
    """Minimal YAML reader for these configs: two levels, scalar values only."""
    cfg, section = {}, None
    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.split("#", 1)[0].rstrip()
            if not line.strip():
                continue
            indented = line[0] in " \t"
            key, _, value = line.strip().partition(":")
            value = value.strip().strip("'\"")
            if not value:
                section = key.strip()
                cfg[section] = {}
            elif indented and section:
                cfg[section][key.strip()] = value
            else:
                cfg[key.strip()] = value
    return cfg


def check(paths, folder, label, full):
    """Report how many of `paths` exist inside `folder`."""
    if not os.path.isdir(folder):
        print("  %-9s FOLDER MISSING: %s" % (label, folder))
        return False

    total = len(paths)
    if not full and total > SAMPLE_SIZE:
        checked = random.sample(paths, SAMPLE_SIZE)
    else:
        checked = paths

    missing = [p for p in checked if not os.path.exists(os.path.join(folder, p))]
    hit = len(checked) - len(missing)
    pct = 100.0 * hit / len(checked) if checked else 100.0

    scope = "all %d" % total if checked is paths else "%d of %d sampled" % (len(checked), total)
    # Report the raw miss count too: a handful of misses out of tens of thousands
    # rounds to "100.00%", which would otherwise read as a clean sweep.
    print("  %-9s %6.2f%% found, %d missing  (%s)" % (label, pct, len(missing), scope))
    if missing:
        print("             e.g. missing: %s" % ", ".join(missing[:4]))
    return pct


def run(config_path, full):
    cfg = load_config(config_path)
    data = cfg["data"]
    name = cfg.get("run_name", os.path.basename(config_path))
    print("\n=== %s  (%s) ===" % (name, os.path.basename(config_path)))

    # --- entity images -----------------------------------------------------
    with open(data["entity"], "r", encoding="utf-8") as f:
        entities = json.load(f)
    ent_imgs = []
    for e in entities:
        for img in e.get("image_list") or []:
            ent_imgs.append(img)
            break                      # eval uses image_list[0]
    print("  entities: %d total, %d with >=1 image" % (len(entities), len(ent_imgs)))
    ent_pct = check(ent_imgs, data["kb_img_folder"], "kb:", full)

    # --- mention images ----------------------------------------------------
    ment_imgs = []
    for key in ("train_file", "dev_file", "test_file"):
        with open(data[key], "r", encoding="utf-8") as f:
            for s in json.load(f):
                p = s.get("imgPath") or ""
                if p:
                    # exactly what choose_image() does for sample_type == 1
                    ment_imgs.append(p.split("/")[-1].split(".")[0] + ".jpg")
    print("  mentions: %d with an imgPath" % len(ment_imgs))
    ment_pct = check(ment_imgs, data["mention_img_folder"], "mention:", full)

    ok = ent_pct is not False and ment_pct is not False \
        and ent_pct > 99.0 and ment_pct > 99.0
    print("  => %s" % ("OK" if ok else "PROBLEM — fix paths before training"))
    return ok


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", help="a single config YAML to check")
    p.add_argument("--all", action="store_true", help="check every local-configs/*_local.yaml")
    p.add_argument("--full", action="store_true",
                   help="check every record instead of a %d-item sample" % SAMPLE_SIZE)
    args = p.parse_args()

    random.seed(0)

    if args.all:
        configs = sorted(glob.glob(os.path.join(CONFIG_DIR, "*_local.yaml")))
    elif args.config:
        configs = [args.config]
    else:
        p.error("pass --config <file> or --all")

    if not configs:
        print("no configs found")
        return 2

    results = []
    for c in configs:
        try:
            results.append(run(c, args.full))
        except FileNotFoundError as exc:
            print("\n=== %s ===\n  ! %s" % (os.path.basename(c), exc))
            results.append(False)

    print("\n%d/%d config(s) OK" % (sum(1 for r in results if r), len(results)))
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())
