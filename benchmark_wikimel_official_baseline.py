import argparse
import csv
import json
import os
import platform
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, UnidentifiedImageError

try:
    import psutil
except ImportError:
    psutil = None


def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def normalize_text(value) -> str:
    if value is None:
        return ""
    return str(value).strip()


def build_text(label: str, description: str, char_limit: int) -> str:
    label = normalize_text(label)
    description = normalize_text(description)[:char_limit]
    if label and description:
        return f"{label}. {description}"
    return label or description


def read_qid2label(path: str) -> Dict[str, Dict[str, str]]:
    qid_map: Dict[str, Dict[str, str]] = {}
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for raw_line in f:
            line = raw_line.rstrip("\n")
            if not line:
                continue
            parts = line.split("\t", 2)
            qid = normalize_text(parts[0]) if len(parts) > 0 else ""
            label = normalize_text(parts[1]) if len(parts) > 1 else ""
            desc = normalize_text(parts[2]) if len(parts) > 2 else ""
            if not qid:
                continue
            qid_map[qid] = {
                "entity_id": qid,
                "title": label or qid,
                "description": desc,
            }
    if not qid_map:
        raise RuntimeError(f"No QID rows loaded from {path}")
    return qid_map


def process_memory_snapshot() -> Dict[str, int]:
    if psutil is None:
        return {}
    info = psutil.Process(os.getpid()).memory_info()
    snapshot = {"rss_bytes": int(getattr(info, "rss", 0))}
    for attr in ("peak_wset", "wset", "vms", "peak_pagefile", "pagefile"):
        if hasattr(info, attr):
            snapshot[f"{attr}_bytes"] = int(getattr(info, attr))
    return snapshot


def bytes_to_mb(n_bytes: int) -> float:
    return float(n_bytes) / (1024.0 * 1024.0)


def collect_environment() -> Dict[str, object]:
    env = {
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "processor": platform.processor(),
    }
    try:
        import torch

        env["torch_version"] = torch.__version__
        env["cuda_available"] = bool(torch.cuda.is_available())
        if torch.cuda.is_available():
            env["gpu_name"] = torch.cuda.get_device_name(0)
    except Exception as exc:
        env["torch_info_error"] = f"{type(exc).__name__}: {exc}"
    if psutil is not None:
        env["psutil_version"] = psutil.__version__
    return env


def iter_chunks(values: Sequence, chunk_size: int) -> Iterable[Tuple[int, Sequence]]:
    for start in range(0, len(values), chunk_size):
        yield start, values[start : start + chunk_size]


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def encode_query_images(
    queries: List[Dict[str, object]],
    model,
    preprocess,
    device,
    batch_size: int,
    use_amp: bool,
) -> Tuple[np.ndarray, List[Dict[str, object]], Dict[str, int]]:
    import torch

    embeddings = []
    valid_queries: List[Dict[str, object]] = []
    skipped = {
        "missing_image": 0,
        "unreadable_image": 0,
    }

    batch_tensors = []
    batch_rows: List[Dict[str, object]] = []

    def flush_batch() -> None:
        nonlocal batch_tensors, batch_rows
        if not batch_tensors:
            return
        tensor = torch.stack(batch_tensors).to(device)
        if use_amp:
            with torch.no_grad(), torch.cuda.amp.autocast():
                feats = model.encode_image(tensor)
                feats = feats / feats.norm(dim=-1, keepdim=True)
        else:
            with torch.no_grad():
                feats = model.encode_image(tensor)
                feats = feats / feats.norm(dim=-1, keepdim=True)
        embeddings.append(feats.detach().cpu().float().numpy())
        valid_queries.extend(batch_rows)
        batch_tensors = []
        batch_rows = []

    for row in queries:
        image_path = normalize_text(row.get("image_path"))
        if not image_path or not os.path.exists(image_path):
            skipped["missing_image"] += 1
            continue
        try:
            tensor = preprocess(Image.open(image_path).convert("RGB"))
        except (FileNotFoundError, OSError, UnidentifiedImageError):
            skipped["unreadable_image"] += 1
            continue
        batch_tensors.append(tensor)
        batch_rows.append(row)
        if len(batch_tensors) >= batch_size:
            flush_batch()

    flush_batch()

    if not embeddings:
        raise RuntimeError("No valid query images could be encoded.")

    return np.vstack(embeddings).astype(np.float32), valid_queries, skipped


def encode_texts(
    texts: Sequence[str],
    model,
    tokenizer,
    device,
    batch_size: int,
    use_amp: bool,
) -> np.ndarray:
    import torch

    outputs = []
    for _, chunk in iter_chunks(list(texts), batch_size):
        tokens = tokenizer(list(chunk)).to(device)
        if use_amp:
            with torch.no_grad(), torch.cuda.amp.autocast():
                feats = model.encode_text(tokens)
                feats = feats / feats.norm(dim=-1, keepdim=True)
        else:
            with torch.no_grad():
                feats = model.encode_text(tokens)
                feats = feats / feats.norm(dim=-1, keepdim=True)
        outputs.append(feats.detach().cpu().float().numpy())

    if not outputs:
        raise RuntimeError("No text embeddings were produced.")
    return np.vstack(outputs).astype(np.float32)


def compute_exact_metrics(
    query_embeddings: np.ndarray,
    kb_embeddings: np.ndarray,
    gold_indices: np.ndarray,
    topk_values: Sequence[int],
    chunk_size: int,
) -> Tuple[Dict[str, float], Dict[str, object]]:
    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    kb_tensor = torch.from_numpy(kb_embeddings).to(device)
    query_tensor = torch.from_numpy(query_embeddings).to(device)
    gold_tensor = torch.as_tensor(gold_indices, dtype=torch.long, device=device)

    total = int(query_embeddings.shape[0])
    ranks = np.empty(total, dtype=np.int64)
    gold_scores = np.empty(total, dtype=np.float32)
    top1_scores = np.empty(total, dtype=np.float32)
    top1_indices = np.empty(total, dtype=np.int64)
    topk_hits = {int(k): 0 for k in topk_values}

    peak_rss_bytes = 0
    peak_wset_bytes = 0
    timings_ms = []

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    max_topk = max(int(k) for k in topk_values)

    for start in range(0, total, chunk_size):
        end = min(start + chunk_size, total)
        before = process_memory_snapshot()
        t0 = time.perf_counter()

        sim = query_tensor[start:end] @ kb_tensor.T
        local_gold = gold_tensor[start:end]
        row_ids = torch.arange(end - start, device=device)
        local_gold_scores = sim[row_ids, local_gold]
        local_ranks = 1 + (sim > local_gold_scores.unsqueeze(1)).sum(dim=1)
        _, local_topk = torch.topk(sim, k=max_topk, dim=1)

        t1 = time.perf_counter()
        after = process_memory_snapshot()

        ranks[start:end] = local_ranks.detach().cpu().numpy().astype(np.int64)
        gold_scores[start:end] = local_gold_scores.detach().cpu().numpy().astype(np.float32)
        top1_scores[start:end] = sim.max(dim=1).values.detach().cpu().numpy().astype(np.float32)
        top1_indices[start:end] = local_topk[:, 0].detach().cpu().numpy().astype(np.int64)

        local_topk_cpu = local_topk.detach().cpu().numpy()
        local_gold_cpu = gold_indices[start:end]
        for row_idx, gold_idx in enumerate(local_gold_cpu):
            for k in topk_values:
                if gold_idx in local_topk_cpu[row_idx, : int(k)]:
                    topk_hits[int(k)] += 1

        timings_ms.append((t1 - t0) * 1000.0)
        peak_rss_bytes = max(
            peak_rss_bytes,
            int(before.get("rss_bytes", 0)),
            int(after.get("rss_bytes", 0)),
        )
        peak_wset_bytes = max(
            peak_wset_bytes,
            int(before.get("peak_wset_bytes", 0)),
            int(after.get("peak_wset_bytes", 0)),
        )

    metrics = {
        "queries": total,
        "candidate_entities": int(kb_embeddings.shape[0]),
        "embedding_dim": int(kb_embeddings.shape[1]),
        "recall@1": float(np.mean(ranks <= 1)),
        "recall@5": float(np.mean(ranks <= 5)),
        "recall@10": float(np.mean(ranks <= 10)),
        "mrr": float(np.mean(1.0 / ranks)),
        "median_rank": float(np.median(ranks)),
        "mean_rank": float(np.mean(ranks)),
        "latency_ms": float(np.sum(timings_ms)),
        "latency_ms_mean_chunk": float(np.mean(timings_ms)),
        "latency_ms_std_chunk": float(np.std(timings_ms)),
        "per_query_latency_ms": float(np.sum(timings_ms) / max(total, 1)),
        "queries_per_second": float(total / max(np.sum(timings_ms) / 1000.0, 1e-12)),
        "peak_rss_mb": bytes_to_mb(peak_rss_bytes) if peak_rss_bytes else None,
        "peak_working_set_mb": bytes_to_mb(peak_wset_bytes) if peak_wset_bytes else None,
        "query_storage_mb": bytes_to_mb(int(query_embeddings.nbytes)),
        "candidate_storage_mb": bytes_to_mb(int(kb_embeddings.nbytes)),
        "similarity_chunk_storage_mb": bytes_to_mb(int(chunk_size * kb_embeddings.shape[0] * 4)),
    }
    for k in topk_values:
        metrics[f"recall@{int(k)}"] = float(topk_hits[int(k)] / max(total, 1))

    diagnostics = {
        "ranks": ranks.tolist(),
        "gold_scores": gold_scores.tolist(),
        "top1_scores": top1_scores.tolist(),
        "top1_indices": top1_indices.tolist(),
        "timings_ms_per_chunk": timings_ms,
        "chunk_size": int(chunk_size),
    }
    if torch.cuda.is_available():
        diagnostics["gpu_peak_memory_mb"] = bytes_to_mb(int(torch.cuda.max_memory_allocated()))

    return metrics, diagnostics


def save_entity_kb(path: Path, entity_rows: List[Dict[str, str]]) -> None:
    ensure_parent(path)
    with path.open("w", encoding="utf-8") as f:
        json.dump(entity_rows, f, indent=2, ensure_ascii=False)


def save_valid_metadata(path: Path, rows: List[Dict[str, object]]) -> None:
    ensure_parent(path)
    with path.open("w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)


def export_metrics_csv(path: Path, metrics: Dict[str, float]) -> str:
    ensure_parent(path)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        for key, value in metrics.items():
            writer.writerow([key, value])
    return str(path)


def create_plots(results: Dict[str, object], out_dir: Path) -> List[str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics = results["metrics"]
    ranks = np.array(results["diagnostics"]["ranks"], dtype=np.int64)
    gold_scores = np.array(results["diagnostics"]["gold_scores"], dtype=np.float32)
    top1_scores = np.array(results["diagnostics"]["top1_scores"], dtype=np.float32)
    margins = top1_scores - gold_scores

    created: List[str] = []

    recall_keys = [key for key in metrics.keys() if key.startswith("recall@")]
    recall_keys.sort(key=lambda name: int(name.split("@")[1]))
    recall_x = [int(name.split("@")[1]) for name in recall_keys]
    recall_y = [float(metrics[name]) for name in recall_keys]

    plt.figure(figsize=(8, 5))
    plt.plot(recall_x, recall_y, marker="o", linewidth=2)
    plt.xticks(recall_x)
    plt.ylim(0.0, 1.0)
    plt.xlabel("k")
    plt.ylabel("Recall@k")
    plt.title("Official WikiMEL Baseline Recall Curve")
    plt.grid(alpha=0.25)
    path1 = out_dir / "wikimel_recall_at_k.png"
    plt.tight_layout()
    plt.savefig(path1, dpi=160)
    plt.close()
    created.append(str(path1))

    sorted_ranks = np.sort(ranks)
    cdf = np.arange(1, len(sorted_ranks) + 1) / len(sorted_ranks)
    plt.figure(figsize=(8, 5))
    plt.plot(sorted_ranks, cdf, linewidth=2)
    plt.xscale("log")
    plt.xlabel("Rank (log scale)")
    plt.ylabel("Fraction of queries")
    plt.title("Official WikiMEL Rank CDF")
    plt.grid(alpha=0.25)
    path2 = out_dir / "wikimel_rank_cdf.png"
    plt.tight_layout()
    plt.savefig(path2, dpi=160)
    plt.close()
    created.append(str(path2))

    bins = np.logspace(0, np.log10(max(int(ranks.max()), 1)), num=30)
    plt.figure(figsize=(8, 5))
    plt.hist(ranks, bins=bins, color="tab:blue", alpha=0.85)
    plt.xscale("log")
    plt.xlabel("Rank (log scale)")
    plt.ylabel("Query count")
    plt.title("Official WikiMEL Rank Histogram")
    plt.grid(alpha=0.2)
    path3 = out_dir / "wikimel_rank_histogram.png"
    plt.tight_layout()
    plt.savefig(path3, dpi=160)
    plt.close()
    created.append(str(path3))

    plt.figure(figsize=(8, 5))
    plt.hist(margins, bins=40, color="tab:orange", alpha=0.85)
    plt.axvline(0.0, linestyle="--", linewidth=1, color="black")
    plt.xlabel("Top-1 score minus gold score")
    plt.ylabel("Query count")
    plt.title("Official WikiMEL Retrieval Margin Histogram")
    plt.grid(alpha=0.2)
    path4 = out_dir / "wikimel_margin_histogram.png"
    plt.tight_layout()
    plt.savefig(path4, dpi=160)
    plt.close()
    created.append(str(path4))

    timing_labels = ["image_encode_s", "kb_text_encode_s", "retrieval_s"]
    timing_values = [
        float(results["timing"]["image_encoding_seconds"]),
        float(results["timing"]["kb_text_encoding_seconds"]),
        float(metrics["latency_ms"]) / 1000.0,
    ]
    plt.figure(figsize=(8, 5))
    plt.bar(timing_labels, timing_values, color=["tab:green", "tab:purple", "tab:red"])
    plt.ylabel("Seconds")
    plt.title("Official WikiMEL Baseline Timing Breakdown")
    plt.grid(axis="y", alpha=0.2)
    path5 = out_dir / "wikimel_timing_breakdown.png"
    plt.tight_layout()
    plt.savefig(path5, dpi=160)
    plt.close()
    created.append(str(path5))

    return created


def main() -> None:
    parser = argparse.ArgumentParser(description="Build and benchmark an fp32 baseline on the official WikiMEL split.")
    parser.add_argument("--queries", default=os.path.join("data", "wikimel_source", "WikiMEL.json"))
    parser.add_argument("--qid2label", default=os.path.join("data", "wikimel_source", "QID2Label.tsv"))
    parser.add_argument(
        "--split-mapping",
        default=os.path.join("external", "KGMEL_source", "data", "dataset", "mapping", "ids_split_mappings.json"),
    )
    parser.add_argument(
        "--candidate-qids",
        default=os.path.join("external", "KGMEL_source", "data", "dataset", "mapping", "qids_candidate.json"),
    )
    parser.add_argument("--image-root", default=os.path.join("data", "wikimel_source", "image", "WikiMEL"))
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    parser.add_argument("--dataset-name", default="WikiMEL")
    parser.add_argument("--artifact-root", default=os.path.join("data", "wikimel_official_test_v1"))
    parser.add_argument("--out-dir", default=os.path.join("benchmarks", "wikimel_official_test_v1"))
    parser.add_argument("--model-name", default="ViT-B-32")
    parser.add_argument("--pretrained", default="openai")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--retrieval-chunk-size", type=int, default=64)
    parser.add_argument("--text-char-limit", type=int, default=300)
    parser.add_argument("--topk", nargs="+", type=int, default=[1, 5, 10, 20, 50, 100])
    parser.add_argument("--force-reencode", action="store_true")
    args = parser.parse_args()

    artifact_root = Path(args.artifact_root)
    emb_dir = artifact_root / "embeddings"
    out_dir = Path(args.out_dir)
    plot_dir = out_dir / "plots"

    image_emb_path = emb_dir / "image_embeddings.npy"
    kb_emb_path = emb_dir / "kb_text_embeddings.npy"
    entity_kb_path = artifact_root / "entity_kb.json"
    valid_meta_path = emb_dir / "valid_metadata.json"
    query_to_entity_path = emb_dir / "query_to_entity.json"
    spec_path = artifact_root / "benchmark_spec.json"

    split_map = load_json(args.split_mapping)
    split_ids = set(split_map[args.dataset_name][args.split])
    all_queries = load_json(args.queries)
    qid_map = read_qid2label(args.qid2label)
    candidate_qids_raw = load_json(args.candidate_qids)[args.dataset_name]

    candidate_qids = []
    missing_candidate_qids = 0
    for qid in candidate_qids_raw:
        if qid in qid_map:
            candidate_qids.append(qid)
        else:
            missing_candidate_qids += 1

    candidate_rows = [qid_map[qid] for qid in candidate_qids]
    candidate_index = {qid: idx for idx, qid in enumerate(candidate_qids)}

    filtered_queries: List[Dict[str, object]] = []
    dropped_missing_gold = 0
    for row in all_queries:
        query_id = int(row["id"])
        if query_id not in split_ids:
            continue
        gold_qid = normalize_text(row.get("answer", [""])[0] if row.get("answer") else "")
        if gold_qid not in candidate_index:
            dropped_missing_gold += 1
            continue
        filtered_queries.append(
            {
                "query_id": query_id,
                "sentence": normalize_text(row.get("sentence")),
                "mention": normalize_text((row.get("mention") or [""])[0]),
                "entity_label": normalize_text((row.get("entity") or [""])[0]),
                "entity_id": gold_qid,
                "image_path": str((Path(args.image_root) / normalize_text(row.get("imgPath"))).resolve()),
            }
        )

    import torch
    import open_clip

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = bool(torch.cuda.is_available())
    model, _, preprocess = open_clip.create_model_and_transforms(args.model_name, pretrained=args.pretrained)
    tokenizer = open_clip.get_tokenizer(args.model_name)
    model = model.to(device).eval()

    image_encoding_seconds = 0.0
    if image_emb_path.exists() and valid_meta_path.exists() and not args.force_reencode:
        query_embeddings = np.load(image_emb_path).astype(np.float32)
        valid_queries = load_json(str(valid_meta_path))
        for row in valid_queries:
            row["image_path"] = row["image_file"]
    else:
        t0 = time.perf_counter()
        query_embeddings, valid_queries, skipped = encode_query_images(
            queries=filtered_queries,
            model=model,
            preprocess=preprocess,
            device=device,
            batch_size=args.batch_size,
            use_amp=use_amp,
        )
        image_encoding_seconds = time.perf_counter() - t0

        emb_dir.mkdir(parents=True, exist_ok=True)
        np.save(image_emb_path, query_embeddings)
        valid_meta_rows = []
        mapping_rows = []
        for row in valid_queries:
            valid_meta_rows.append(
                {
                    "query_id": row["query_id"],
                    "title": row["entity_label"] or qid_map[row["entity_id"]]["title"],
                    "sentence": row["sentence"],
                    "mention": row["mention"],
                    "entity_id": row["entity_id"],
                    "image_file": row["image_path"],
                }
            )
            mapping_rows.append({"query_id": row["query_id"], "entity_id": row["entity_id"]})
        save_valid_metadata(valid_meta_path, valid_meta_rows)
        ensure_parent(query_to_entity_path)
        with query_to_entity_path.open("w", encoding="utf-8") as f:
            json.dump(mapping_rows, f, indent=2, ensure_ascii=False)
    if image_encoding_seconds == 0.0 and image_emb_path.exists():
        # Cached run: preserve a clear marker instead of pretending we timed it.
        image_encoding_seconds = 0.0

    valid_meta_rows = load_json(str(valid_meta_path))
    valid_queries = []
    for row in valid_meta_rows:
        valid_queries.append(
            {
                "query_id": row["query_id"],
                "entity_id": row["entity_id"],
                "entity_label": row["title"],
                "sentence": row.get("sentence", ""),
                "mention": row.get("mention", ""),
                "image_path": row["image_file"],
            }
        )

    kb_text_encoding_seconds = 0.0
    if kb_emb_path.exists() and entity_kb_path.exists() and not args.force_reencode:
        kb_embeddings = np.load(kb_emb_path).astype(np.float32)
        candidate_rows = load_json(str(entity_kb_path))
    else:
        kb_texts = [build_text(row["title"], row.get("description", ""), args.text_char_limit) for row in candidate_rows]
        t0 = time.perf_counter()
        kb_embeddings = encode_texts(
            texts=kb_texts,
            model=model,
            tokenizer=tokenizer,
            device=device,
            batch_size=args.batch_size,
            use_amp=use_amp,
        )
        kb_text_encoding_seconds = time.perf_counter() - t0
        emb_dir.mkdir(parents=True, exist_ok=True)
        np.save(kb_emb_path, kb_embeddings)
        save_entity_kb(entity_kb_path, candidate_rows)
    if kb_text_encoding_seconds == 0.0 and kb_emb_path.exists():
        kb_text_encoding_seconds = 0.0

    gold_indices = np.array([candidate_index[row["entity_id"]] for row in valid_queries], dtype=np.int64)

    metrics, diagnostics = compute_exact_metrics(
        query_embeddings=query_embeddings,
        kb_embeddings=kb_embeddings,
        gold_indices=gold_indices,
        topk_values=sorted(set(int(k) for k in args.topk)),
        chunk_size=args.retrieval_chunk_size,
    )

    spec = {
        "benchmark_name": f"Official {args.dataset_name} {args.split} fp32 baseline",
        "dataset": {
            "name": args.dataset_name,
            "split": args.split,
            "raw_queries": len(all_queries),
            "split_queries_before_image_filter": len(filtered_queries),
            "queries_evaluated": len(valid_queries),
            "official_split_sizes": {k: len(v) for k, v in split_map[args.dataset_name].items()},
            "candidate_entities": len(candidate_rows),
            "missing_candidate_qids_from_qid2label": missing_candidate_qids,
            "dropped_queries_missing_gold_in_candidates": dropped_missing_gold,
        },
        "model": {
            "name": args.model_name,
            "pretrained": args.pretrained,
            "device": str(device),
            "batch_size": args.batch_size,
            "mixed_precision": use_amp,
        },
        "artifacts": {
            "entity_kb": str(entity_kb_path),
            "valid_metadata": str(valid_meta_path),
            "image_embeddings": str(image_emb_path),
            "kb_text_embeddings": str(kb_emb_path),
            "query_to_entity": str(query_to_entity_path),
        },
    }
    ensure_parent(spec_path)
    with spec_path.open("w", encoding="utf-8") as f:
        json.dump(spec, f, indent=2, ensure_ascii=False)

    results = {
        "environment": collect_environment(),
        "config": {
            "queries": args.queries,
            "qid2label": args.qid2label,
            "split_mapping": args.split_mapping,
            "candidate_qids": args.candidate_qids,
            "image_root": args.image_root,
            "split": args.split,
            "artifact_root": str(artifact_root),
            "out_dir": str(out_dir),
            "topk": sorted(set(int(k) for k in args.topk)),
            "retrieval_chunk_size": args.retrieval_chunk_size,
        },
        "dataset": spec["dataset"],
        "timing": {
            "image_encoding_seconds": image_encoding_seconds,
            "kb_text_encoding_seconds": kb_text_encoding_seconds,
        },
        "metrics": metrics,
        "diagnostics": diagnostics,
        "artifacts": spec["artifacts"],
    }

    plot_paths = create_plots(results, plot_dir)
    results["plots"] = plot_paths

    out_dir.mkdir(parents=True, exist_ok=True)
    out_json = out_dir / "baseline_results.json"
    metrics_csv = out_dir / "baseline_metrics.csv"
    diagnostics_json = out_dir / "baseline_diagnostics.json"

    results["metrics_csv"] = export_metrics_csv(metrics_csv, metrics)
    with diagnostics_json.open("w", encoding="utf-8") as f:
        json.dump(diagnostics, f, indent=2)
    with out_json.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("\nOfficial WikiMEL fp32 baseline")
    print("--------------------------------")
    print(f"Split: {args.split}")
    print(f"Queries evaluated: {metrics['queries']}")
    print(f"Candidate entities: {metrics['candidate_entities']}")
    print(f"Recall@1: {metrics['recall@1']:.4f}")
    print(f"Recall@10: {metrics['recall@10']:.4f}")
    print(f"Recall@100: {metrics.get('recall@100', 0.0):.4f}")
    print(f"MRR: {metrics['mrr']:.4f}")
    print(f"Median rank: {metrics['median_rank']:.1f}")
    print(f"Mean rank: {metrics['mean_rank']:.2f}")
    print(f"Retrieval latency: {metrics['latency_ms']:.2f} ms")
    print(f"Per-query latency: {metrics['per_query_latency_ms']:.4f} ms")
    print(f"Queries/sec: {metrics['queries_per_second']:.2f}")
    print(f"Saved results to {out_json}")


if __name__ == "__main__":
    main()
