#!/usr/bin/env python3
"""
Generate the qid2id.json files that the FAST-MEL configs require.

Every config YAML points `data.qid2id` at data/<Dataset>/qid2id.json, but that
file is not in the repository — DataModuleForMIMIC would die on the very first
open(). It is fully derivable: kb_entity.json carries both `qid` (the Wikidata
id, e.g. "Q42", which is what the mention files reference in `answer`/`cands`)
and `id` (the dense row index into the entity embedding matrix).

dataset.py reads it with `json.loads(f.readline())`, so the whole mapping has to
sit on a single line — that is why this writes without indentation or newlines.

Usage:
    python generate_qid2id.py [--force]
"""

import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(HERE)
DATA_ROOT = os.path.join(PROJECT_ROOT, "fastmel-source-repo", "data")

DATASETS = ["WikiMEL", "RichpediaMEL", "WikiDiverse"]


def build(dataset, force):
    ds_dir = os.path.join(DATA_ROOT, dataset)
    kb_path = os.path.join(ds_dir, "kb_entity.json")
    out_path = os.path.join(ds_dir, "qid2id.json")

    if not os.path.exists(kb_path):
        print("  ! %s not found, skipping" % kb_path)
        return False

    if os.path.exists(out_path) and not force:
        print("  qid2id.json already exists, skipping (use --force to rebuild)")
        return True

    with open(kb_path, "r", encoding="utf-8") as f:
        entities = json.load(f)

    qid2id = {}
    duplicates = 0
    for ent in entities:
        qid = ent["qid"]
        if qid in qid2id:
            duplicates += 1
        qid2id[qid] = ent["id"]

    ids = sorted(qid2id.values())
    contiguous = ids == list(range(len(ids)))

    # Single line: dataset.py parses this with json.loads(f.readline()).
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(qid2id, f, separators=(",", ":"))

    print("  entities: %d   unique qids: %d   duplicate qids: %d"
          % (len(entities), len(qid2id), duplicates))
    print("  id range: %d..%d   contiguous: %s" % (ids[0], ids[-1], contiguous))
    if not contiguous:
        print("  ! ids are not a contiguous 0..N-1 range; make sure data.num_entity")
        print("    in the config is at least %d" % (ids[-1] + 1))
    print("  wrote %s (%s bytes)" % (out_path, os.path.getsize(out_path)))
    return True


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--force", action="store_true", help="rebuild even if the file exists")
    args = p.parse_args()

    ok = True
    for dataset in DATASETS:
        print("\n=== %s ===" % dataset)
        ok = build(dataset, args.force) and ok
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
