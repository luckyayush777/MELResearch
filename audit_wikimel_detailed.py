"""Content-aware leakage audit for MIMIC-formatted WikiMEL data."""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import os
import re
import unicodedata
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable

import numpy as np
from PIL import Image, ImageOps


SPLITS = ("train", "dev", "test")
PAIRS = (("train", "dev"), ("train", "test"), ("dev", "test"))
PUNCTUATION = re.compile(r"[^\w\s]", re.UNICODE)
WHITESPACE = re.compile(r"\s+")


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def normalize_text(value: Any) -> str:
    text = html.unescape(str(value or ""))
    text = unicodedata.normalize("NFKC", text).lower()
    text = PUNCTUATION.sub(" ", text)
    return WHITESPACE.sub(" ", text).strip()


def canonical_path(value: Any) -> str:
    return os.path.normcase(os.path.normpath(str(value or "").strip())).replace("\\", "/")


def full_row(row: dict[str, Any]) -> tuple[str, str, str, str]:
    return (
        normalize_text(row.get("mentions")),
        normalize_text(row.get("sentence")),
        canonical_path(row.get("imgPath")),
        normalize_text(row.get("answer")),
    )


def mention_context(row: dict[str, Any]) -> tuple[str, str]:
    return normalize_text(row.get("mentions")), normalize_text(row.get("sentence"))


def context(row: dict[str, Any]) -> str:
    return normalize_text(row.get("sentence"))


def duplicate_count(rows: list[dict[str, Any]], key: Callable[[dict[str, Any]], Any]) -> int:
    values = [key(row) for row in rows]
    return len(values) - len(set(values))


def overlap(left: list[dict[str, Any]], right: list[dict[str, Any]], key: Callable[[dict[str, Any]], Any]) -> tuple[int, list[Any]]:
    values = {key(row) for row in left} & {key(row) for row in right}
    ordered = sorted(values, key=repr)
    return len(values), ordered[:20]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def dct_matrix(size: int) -> np.ndarray:
    positions = np.arange(size, dtype=np.float64)
    frequencies = positions[:, None]
    matrix = np.cos((np.pi / size) * (positions + 0.5) * frequencies)
    matrix[0] *= np.sqrt(1.0 / size)
    matrix[1:] *= np.sqrt(2.0 / size)
    return matrix


DCT = dct_matrix(32)


def phash(path: Path) -> str:
    with Image.open(path) as image:
        grayscale = ImageOps.grayscale(image).resize((32, 32), Image.Resampling.LANCZOS)
        pixels = np.asarray(grayscale, dtype=np.float64)
    low = (DCT @ pixels @ DCT.T)[:8, :8]
    flattened = low.flatten()
    median = np.median(flattened[1:])
    bits = flattened > median
    value = 0
    for bit in bits:
        value = (value << 1) | int(bit)
    return f"{value:016x}"


def inspect_image(item: tuple[str, Path]) -> tuple[str, dict[str, Any]]:
    label, path = item
    result: dict[str, Any] = {"path": str(path), "exists": path.is_file()}
    if not result["exists"]:
        return label, result
    try:
        result["size"] = path.stat().st_size
        result["sha256"] = sha256_file(path)
        result["phash"] = phash(path)
        result["decode_ok"] = True
    except Exception as exc:  # corrupt images are audit findings, not fatal errors
        result["decode_ok"] = False
        result["error"] = f"{type(exc).__name__}: {exc}"
    return label, result


class BKTree:
    def __init__(self) -> None:
        self.root: tuple[int, str, dict[int, Any]] | None = None

    @staticmethod
    def distance(left: int, right: int) -> int:
        return (left ^ right).bit_count()

    def add(self, value: int, label: str) -> None:
        if self.root is None:
            self.root = (value, label, {})
            return
        node = self.root
        while True:
            distance = self.distance(value, node[0])
            child = node[2].get(distance)
            if child is None:
                node[2][distance] = (value, label, {})
                return
            node = child

    def query(self, value: int, threshold: int) -> list[tuple[int, str]]:
        if self.root is None:
            return []
        matches: list[tuple[int, str]] = []
        pending = [self.root]
        while pending:
            node = pending.pop()
            distance = self.distance(value, node[0])
            if distance <= threshold:
                matches.append((distance, node[1]))
            low, high = distance - threshold, distance + threshold
            pending.extend(child for edge, child in node[2].items() if low <= edge <= high)
        return matches


def image_inventory(
    splits: dict[str, list[dict[str, Any]]],
    kb: list[dict[str, Any]],
    mention_root: Path,
    entity_root: Path,
) -> tuple[dict[str, Path], dict[str, set[str]], dict[str, str], dict[str, str]]:
    files: dict[str, Path] = {}
    split_labels: dict[str, set[str]] = {split: set() for split in SPLITS}
    mention_answers: dict[str, str] = {}
    entity_qids: dict[str, str] = {}
    for split, rows in splits.items():
        for row in rows:
            raw = row.get("imgPath")
            if not raw:
                continue
            label = f"mention:{canonical_path(raw)}"
            files[label] = mention_root / str(raw)
            split_labels[split].add(label)
            mention_answers[label] = str(row.get("answer") or "")
    for entity in kb:
        qid = str(entity.get("qid") or "")
        for raw in entity.get("image_list", []):
            label = f"entity:{canonical_path(raw)}"
            files[label] = entity_root / str(raw)
            entity_qids[label] = qid
    return files, split_labels, mention_answers, entity_qids


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--mention-image-root", type=Path)
    parser.add_argument("--entity-image-root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--phash-threshold", type=int, default=4)
    parser.add_argument("--workers", type=int, default=min(12, (os.cpu_count() or 4)))
    args = parser.parse_args()

    root = args.data_root.resolve()
    splits = {split: load_json(root / f"WIKIMEL_{split}.json") for split in SPLITS}
    kb = load_json(root / "kb_entity.json")
    qid2id = load_json(root / "qid2id.json")
    mention_root = (args.mention_image_root or root / "mention_image").resolve()
    entity_root = (args.entity_image_root or root / "kb_image").resolve()

    report: dict[str, Any] = {
        "schema_version": 1,
        "data_root": str(root),
        "phash_algorithm": "64-bit DCT pHash (32x32 grayscale, 8x8 low frequencies)",
        "phash_hamming_threshold": args.phash_threshold,
        "split_rows": {split: len(rows) for split, rows in splits.items()},
        "normalized_text": {"within_split_duplicates": {}, "cross_split": {}},
    }
    for split, rows in splits.items():
        report["normalized_text"]["within_split_duplicates"][split] = {
            "full_row": duplicate_count(rows, full_row),
            "mention_context": duplicate_count(rows, mention_context),
            "context_only": duplicate_count(rows, context),
        }
    for left, right in PAIRS:
        pair = f"{left}-{right}"
        report["normalized_text"]["cross_split"][pair] = {}
        for label, key in (("full_row", full_row), ("mention_context", mention_context), ("context_only", context)):
            count, sample = overlap(splits[left], splits[right], key)
            report["normalized_text"]["cross_split"][pair][label] = {"count": count, "sample": sample}

    files, split_labels, mention_answers, entity_qids = image_inventory(
        splits, kb, mention_root, entity_root
    )
    cached: dict[str, dict[str, Any]] = load_json(args.cache) if args.cache.exists() else {}
    pending = [(label, path) for label, path in files.items() if label not in cached]
    if pending:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            for index, (label, result) in enumerate(pool.map(inspect_image, pending), 1):
                cached[label] = result
                if index % 5000 == 0:
                    print(f"hashed {index}/{len(pending)} uncached images", flush=True)
        write_json(args.cache, cached)

    def valid(labels: set[str]) -> set[str]:
        return {label for label in labels if cached.get(label, {}).get("decode_ok")}

    image_report: dict[str, Any] = {
        "unique_files_referenced": len(files),
        "missing": sum(not item.get("exists") for item in cached.values()),
        "decode_failures": sum(item.get("exists") and not item.get("decode_ok") for item in cached.values()),
        "cross_split": {},
    }
    for left, right in PAIRS:
        pair = f"{left}-{right}"
        a, b = valid(split_labels[left]), valid(split_labels[right])
        byte_a: dict[str, list[str]] = {}
        byte_b: dict[str, list[str]] = {}
        for label in a:
            byte_a.setdefault(cached[label]["sha256"], []).append(label)
        for label in b:
            byte_b.setdefault(cached[label]["sha256"], []).append(label)
        exact_hashes = set(byte_a) & set(byte_b)
        path_overlap = split_labels[left] & split_labels[right]
        image_report["cross_split"][pair] = {
            "exact_path_count": len(path_overlap),
            "exact_path_sample": sorted(path_overlap)[:20],
            "exact_byte_hash_count": len(exact_hashes),
            "exact_byte_hash_sample": sorted(exact_hashes)[:20],
        }

    mention_labels = set().union(*split_labels.values())
    entity_labels = set(entity_qids)
    mention_by_sha: dict[str, list[str]] = {}
    for label in valid(mention_labels):
        mention_by_sha.setdefault(cached[label]["sha256"], []).append(label)
    entity_by_sha: dict[str, list[str]] = {}
    for label in valid(entity_labels):
        entity_by_sha.setdefault(cached[label]["sha256"], []).append(label)
    shared_sha = set(mention_by_sha) & set(entity_by_sha)
    exact_pairs = [
        (mention, entity)
        for digest in shared_sha
        for mention in mention_by_sha[digest]
        for entity in entity_by_sha[digest]
    ]
    gold_exact = [
        (mention, entity)
        for mention, entity in exact_pairs
        if mention_answers.get(mention) == entity_qids.get(entity)
    ]

    tree = BKTree()
    for label in valid(mention_labels):
        tree.add(int(cached[label]["phash"], 16), label)
    perceptual_pairs: list[tuple[int, str, str]] = []
    perceptual_gold: list[tuple[int, str, str]] = []
    for entity in valid(entity_labels):
        for distance, mention in tree.query(int(cached[entity]["phash"], 16), args.phash_threshold):
            if cached[mention]["sha256"] == cached[entity]["sha256"]:
                continue
            pair = (distance, mention, entity)
            perceptual_pairs.append(pair)
            if mention_answers.get(mention) == entity_qids.get(entity):
                perceptual_gold.append(pair)

    perceptual_by_distance: dict[int, list[tuple[int, str, str]]] = defaultdict(list)
    gold_perceptual_by_distance: dict[int, list[tuple[int, str, str]]] = defaultdict(list)
    for pair in perceptual_pairs:
        perceptual_by_distance[pair[0]].append(pair)
    for pair in perceptual_gold:
        gold_perceptual_by_distance[pair[0]].append(pair)

    image_report["mention_entity"] = {
        "exact_byte_pair_count": len(exact_pairs),
        "exact_byte_pair_sample": exact_pairs[:50],
        "gold_exact_byte_pair_count": len(gold_exact),
        "gold_exact_byte_pair_sample": gold_exact[:50],
        "perceptual_candidate_pair_count": len(perceptual_pairs),
        "perceptual_candidate_distance_counts": dict(sorted(Counter(pair[0] for pair in perceptual_pairs).items())),
        "perceptual_candidate_samples_by_distance": {
            str(distance): sorted(pairs)[:20] for distance, pairs in sorted(perceptual_by_distance.items())
        },
        "perceptual_candidate_pair_sample": sorted(perceptual_pairs)[:100],
        "gold_perceptual_candidate_pair_count": len(perceptual_gold),
        "gold_perceptual_candidate_distance_counts": dict(sorted(Counter(pair[0] for pair in perceptual_gold).items())),
        "gold_perceptual_candidate_samples_by_distance": {
            str(distance): sorted(pairs)[:20]
            for distance, pairs in sorted(gold_perceptual_by_distance.items())
        },
        "gold_perceptual_candidate_pair_sample": sorted(perceptual_gold)[:100],
        "perceptual_candidates_confirmed": False,
        "review_required": bool(perceptual_pairs),
    }
    report["images"] = image_report

    qids = [str(entity.get("qid")) for entity in kb]
    ordering_payload = json.dumps(qids, separators=(",", ":")).encode("utf-8")
    answer_filename_hits = sum(
        normalize_text(row.get("answer")) in normalize_text(row.get("imgPath"))
        for rows in splits.values() for row in rows if row.get("answer") and row.get("imgPath")
    )
    report["label_metadata"] = {
        "mention_filenames_containing_answer_id": answer_filename_hits,
        "candidate_lists_present": any("candidates" in row for rows in splits.values() for row in rows),
        "qid2id_matches_kb_order": all(qid2id.get(qid) == index for index, qid in enumerate(qids)),
        "entity_id_ordering_sha256": hashlib.sha256(ordering_payload).hexdigest(),
    }
    normalized_gate = all(
        report["normalized_text"]["cross_split"][pair]["mention_context"]["count"] == 0
        and report["normalized_text"]["cross_split"][pair]["context_only"]["count"] == 0
        for pair in report["normalized_text"]["cross_split"]
    )
    byte_gate = all(item["exact_byte_hash_count"] == 0 for item in image_report["cross_split"].values())
    report["acceptance_gate"] = {
        "normalized_cross_split_zero": normalized_gate,
        "exact_image_bytes_cross_split_zero": byte_gate,
        "gold_entity_exact_image_bytes_zero": len(gold_exact) == 0,
        "answer_id_filename_zero": answer_filename_hits == 0,
        "perceptual_review_complete": not perceptual_pairs,
        "accepted": normalized_gate and byte_gate and not gold_exact and not answer_filename_hits and not perceptual_pairs,
    }
    write_json(args.output, report)
    print(json.dumps({
        "output": str(args.output.resolve()),
        "cache": str(args.cache.resolve()),
        "files": len(files),
        "acceptance_gate": report["acceptance_gate"],
        "perceptual_candidates": len(perceptual_pairs),
    }, indent=2))


if __name__ == "__main__":
    main()
