"""Build deterministic WikiMEL diagnostic variants for the clean experiment.

The script intentionally keeps image files in the immutable source directory and
writes JSON data plus a manifest.  This avoids duplicating several GB of images;
the manifest records the exact external image roots used by the experiment.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


SPLIT_FILES = {
    "train": "WIKIMEL_train.json",
    "dev": "WIKIMEL_dev.json",
    "test": "WIKIMEL_test.json",
}


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_json(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def dedupe_splits(splits: dict[str, list[dict[str, Any]]]) -> tuple[dict[str, list[dict[str, Any]]], dict[str, int]]:
    """Apply the historical clean-split rule in train/dev/test priority order.

    Train is retained as published.  Dev rows are removed when their raw context
    sentence or image path occurs in train.  Test is compared with the original
    train and dev inventories.  Comparing against the original higher-priority
    inventories makes the result independent of which higher-priority rows were
    themselves filtered and reproduces the established 18092/2083/4002 split.
    """

    clean: dict[str, list[dict[str, Any]]] = {"train": list(splits["train"])}
    removed = {"train": 0, "dev": 0, "test": 0}
    prior_sentences = {row.get("sentence") for row in splits["train"] if row.get("sentence")}
    prior_images = {row.get("imgPath") for row in splits["train"] if row.get("imgPath")}

    for split in ("dev", "test"):
        kept = [
            row
            for row in splits[split]
            if row.get("sentence") not in prior_sentences and row.get("imgPath") not in prior_images
        ]
        clean[split] = kept
        removed[split] = len(splits[split]) - len(kept)
        prior_sentences.update(row.get("sentence") for row in splits[split] if row.get("sentence"))
        prior_images.update(row.get("imgPath") for row in splits[split] if row.get("imgPath"))

    return clean, removed


def dedupe_normalized_splits(
    splits: dict[str, list[dict[str, Any]]]
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, int]]:
    from audit_wikimel_detailed import canonical_path, normalize_text

    clean: dict[str, list[dict[str, Any]]] = {"train": list(splits["train"])}
    removed = {"train": 0, "dev": 0, "test": 0}
    prior_contexts = {normalize_text(row.get("sentence")) for row in splits["train"]}
    prior_images = {canonical_path(row.get("imgPath")) for row in splits["train"]}
    for split in ("dev", "test"):
        kept = [
            row for row in splits[split]
            if normalize_text(row.get("sentence")) not in prior_contexts
            and canonical_path(row.get("imgPath")) not in prior_images
        ]
        clean[split] = kept
        removed[split] = len(splits[split]) - len(kept)
        prior_contexts.update(normalize_text(row.get("sentence")) for row in splits[split])
        prior_images.update(canonical_path(row.get("imgPath")) for row in splits[split])
    return clean, removed


def without_entity_images(kb: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result = []
    for entity in kb:
        item = dict(entity)
        item["image_list"] = []
        result.append(item)
    return result


def add_candidates(
    splits: dict[str, list[dict[str, Any]]], candidate_root: Path
) -> dict[str, list[dict[str, Any]]]:
    result: dict[str, list[dict[str, Any]]] = {}
    for split, rows in splits.items():
        candidates = load_json(candidate_root / f"WIKIMEL_{split}_100cands-BM25.json")
        by_key = {
            (row.get("sentence"), row.get("imgPath"), row.get("mentions"), row.get("answer")):
            row.get("cands", [])
            for row in candidates
        }
        enriched = []
        for row in rows:
            key = (row.get("sentence"), row.get("imgPath"), row.get("mentions"), row.get("answer"))
            if key not in by_key:
                raise ValueError(f"No BM25 candidate row for {split} key {key!r}")
            item = dict(row)
            item["cands"] = by_key[key]
            enriched.append(item)
        result[split] = enriched
    return result


def git_value(args: Iterable[str], cwd: Path) -> str | None:
    try:
        return subprocess.check_output(["git", *args], cwd=cwd, text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def build_variant(
    name: str,
    output_root: Path,
    source_root: Path,
    splits: dict[str, list[dict[str, Any]]],
    kb: list[dict[str, Any]],
    qid2id: dict[str, int],
    source_hashes: dict[str, str],
    removed: dict[str, int],
    repository_root: Path,
) -> None:
    variant_root = output_root / name
    variant_root.mkdir(parents=True, exist_ok=True)

    for split, filename in SPLIT_FILES.items():
        write_json(variant_root / filename, splits[split])
    write_json(variant_root / "kb_entity.json", kb)
    write_json(variant_root / "qid2id.json", qid2id)
    description_path = source_root / "kb_entity_desc.json"
    if description_path.exists():
        shutil.copy2(description_path, variant_root / description_path.name)

    ordered_qids = [entity.get("qid") for entity in kb]
    manifest = {
        "schema_version": 1,
        "dataset": "WikiMEL",
        "variant": name,
        "source": str(source_root.resolve()),
        "source_file_sha256": source_hashes,
        "preprocessing_script": str(Path(__file__).resolve()),
        "preprocessing_script_sha256": sha256_file(Path(__file__)),
        "preprocessing_git_commit": git_value(["rev-parse", "HEAD"], repository_root),
        "build_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "split_rows": {split: len(rows) for split, rows in splits.items()},
        "split_rows_removed": removed,
        "split_unique_mentions": {
            split: len({row.get("mentions") for row in rows}) for split, rows in splits.items()
        },
        "split_unique_images": {
            split: len({row.get("imgPath") for row in rows if row.get("imgPath")})
            for split, rows in splits.items()
        },
        "split_unique_gold_entities": {
            split: len({row.get("answer") for row in rows}) for split, rows in splits.items()
        },
        "kb_entity_count": len(kb),
        "entity_images_listed": sum(len(entity.get("image_list", [])) for entity in kb),
        "entity_id_ordering_sha256": sha256_json(ordered_qids),
        "qid2id_sha256": sha256_json(qid2id),
        "mention_image_root": str((source_root / "mention_image").resolve()),
        "entity_image_root": str((source_root / "kb_image").resolve()),
        "files": {split: str((variant_root / filename).resolve()) for split, filename in SPLIT_FILES.items()},
        "kb_file": str((variant_root / "kb_entity.json").resolve()),
        "qid2id_file": str((variant_root / "qid2id.json").resolve()),
        "notes": (
            "Image binaries remain in the immutable source roots recorded above. "
            "clean_sanitized_visual_draft is exact-path sanitized only; content-hash "
            "and perceptual sanitization must be applied after detailed audit review."
        ),
    }
    write_json(variant_root / "dataset_manifest.json", manifest)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--image-hash-cache", type=Path)
    parser.add_argument("--candidate-root", type=Path)
    parser.add_argument("--phash-threshold", type=int, default=4)
    args = parser.parse_args()

    source_root = args.source_root.resolve()
    output_root = args.output_root.resolve()
    repository_root = Path(__file__).resolve().parent
    splits = {split: load_json(source_root / filename) for split, filename in SPLIT_FILES.items()}
    if args.candidate_root:
        splits = add_candidates(splits, args.candidate_root.resolve())
    kb_source = source_root / "kb_entity_desc.json"
    if not kb_source.exists():
        kb_source = source_root / "kb_entity.json"
    kb = load_json(kb_source)
    qid2id = load_json(source_root / "qid2id.json")

    source_files = [*SPLIT_FILES.values(), "kb_entity.json", "qid2id.json"]
    if (source_root / "kb_entity_desc.json").exists():
        source_files.append("kb_entity_desc.json")
    source_hashes = {filename: sha256_file(source_root / filename) for filename in source_files}

    clean, removed = dedupe_splits(splits)
    build_variant(
        "official_original", output_root, source_root, splits, kb, qid2id, source_hashes,
        {key: 0 for key in SPLIT_FILES}, repository_root,
    )
    build_variant(
        "dedupe_only", output_root, source_root, clean, kb, qid2id, source_hashes,
        removed, repository_root,
    )
    build_variant(
        "clean_no_entity_images", output_root, source_root, clean, without_entity_images(kb),
        qid2id, source_hashes, removed, repository_root,
    )

    # Exact path sanitization is a useful reproducible starting point.  It is a
    # draft until exact-byte and reviewed perceptual duplicates are also removed.
    mention_paths = {row.get("imgPath") for rows in splits.values() for row in rows if row.get("imgPath")}
    sanitized_kb = []
    for entity in kb:
        item = dict(entity)
        item["image_list"] = [path for path in entity.get("image_list", []) if path not in mention_paths]
        sanitized_kb.append(item)
    build_variant(
        "clean_sanitized_visual_draft", output_root, source_root, clean, sanitized_kb,
        qid2id, source_hashes, removed, repository_root,
    )

    primary_summary = None
    if args.image_hash_cache and args.image_hash_cache.exists():
        from audit_wikimel_detailed import BKTree, canonical_path

        cache = load_json(args.image_hash_cache)
        tree = BKTree()
        for label, metadata in cache.items():
            if label.startswith("mention:") and metadata.get("decode_ok") and metadata.get("phash"):
                tree.add(int(metadata["phash"], 16), label)

        blacklisted_entity_labels: set[str] = set()
        for label, metadata in cache.items():
            if not label.startswith("entity:") or not metadata.get("decode_ok") or not metadata.get("phash"):
                continue
            if tree.query(int(metadata["phash"], 16), args.phash_threshold):
                blacklisted_entity_labels.add(label)

        primary_kb = []
        for entity in kb:
            item = dict(entity)
            item["image_list"] = [
                path for path in entity.get("image_list", [])
                if f"entity:{canonical_path(path)}" not in blacklisted_entity_labels
            ]
            primary_kb.append(item)
        primary_splits, primary_removed = dedupe_normalized_splits(splits)
        build_variant(
            "clean_sanitized_visual", output_root, source_root, primary_splits, primary_kb,
            qid2id, source_hashes, primary_removed, repository_root,
        )
        manifest_path = output_root / "clean_sanitized_visual" / "dataset_manifest.json"
        manifest = load_json(manifest_path)
        manifest["sanitization"] = {
            "normalized_context_and_path_split_filter": True,
            "perceptual_hash_algorithm": "64-bit DCT pHash",
            "perceptual_hash_hamming_threshold": args.phash_threshold,
            "entity_images_blacklisted": len(blacklisted_entity_labels),
            "global_blacklist_uses_mention_images_from": list(SPLIT_FILES),
            "manual_review": (
                "Representative gold pairs at distances 0, 2, and 4 were visually "
                "confirmed as the same images with recompression/cropping differences."
            ),
            "image_hash_cache": str(args.image_hash_cache.resolve()),
        }
        manifest["notes"] = (
            "Primary normalized-text and perceptual-image sanitized dataset. Missing "
            "mention/entity images must use an explicit missing-image mask in both models."
        )
        write_json(manifest_path, manifest)
        primary_summary = {
            "split_rows": {key: len(value) for key, value in primary_splits.items()},
            "removed": primary_removed,
            "entity_images_blacklisted": len(blacklisted_entity_labels),
            "entity_images_after": sum(len(entity.get("image_list", [])) for entity in primary_kb),
        }

    print(json.dumps({
        "output_root": str(output_root),
        "original_split_rows": {key: len(value) for key, value in splits.items()},
        "clean_split_rows": {key: len(value) for key, value in clean.items()},
        "removed": removed,
        "kb_entities": len(kb),
        "entity_images_before": sum(len(entity.get("image_list", [])) for entity in kb),
        "entity_images_after_exact_path_draft": sum(len(entity.get("image_list", [])) for entity in sanitized_kb),
        "primary": primary_summary,
    }, indent=2))


if __name__ == "__main__":
    main()
