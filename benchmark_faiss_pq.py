import argparse
import csv
import json
import os
import platform
import sys
import time
from pathlib import Path
from typing import Dict, List

import faiss
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


DEFAULT_IMAGE_EMB = os.path.join("data", "embeddings", "image_embeddings.npy")
DEFAULT_VALID_META = os.path.join("data", "embeddings", "valid_metadata.json")
DEFAULT_KB_TEXT_EMB = os.path.join("data", "embeddings", "kb_text_embeddings.npy")
DEFAULT_ENTITY_KB = os.path.join("data", "entity_kb.json")


def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def bytes_to_mb(n_bytes: int) -> float:
    return float(n_bytes) / (1024.0 * 1024.0)


def collect_environment() -> Dict[str, object]:
    return {
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "processor": platform.processor(),
        "faiss_version": getattr(faiss, "__version__", "unknown"),
    }


def normalize_rows(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    return (x / norms).astype(np.float32)


def kb_gold_indices(valid_meta: List[Dict[str, str]], entity_kb: List[Dict[str, str]]):
    entity_to_index = {}
    title_to_index = {}
    for idx, row in enumerate(entity_kb):
        entity_id = row.get("entity_id", "")
        title = row.get("title", "")
        if entity_id:
            entity_to_index[entity_id] = idx
        if title:
            title_to_index[title] = idx
    keep_positions = []
    gold = []
    for idx, item in enumerate(valid_meta):
        entity_id = item.get("entity_id", "")
        title = item.get("title", "")
        if entity_id and entity_id in entity_to_index:
            keep_positions.append(idx)
            gold.append(entity_to_index[entity_id])
        elif title in title_to_index:
            keep_positions.append(idx)
            gold.append(title_to_index[title])
    return np.array(keep_positions, dtype=np.int64), np.array(gold, dtype=np.int64)


def create_index(index_type: str, dim: int):
    if index_type == "flat":
        return faiss.IndexFlatIP(dim)
    if index_type == "sq8":
        return faiss.IndexScalarQuantizer(dim, faiss.ScalarQuantizer.QT_8bit, faiss.METRIC_INNER_PRODUCT)
    if index_type == "pq_m32_nbits6":
        return faiss.IndexPQ(dim, 32, 6, faiss.METRIC_INNER_PRODUCT)
    if index_type == "pq_m32_nbits8":
        return faiss.IndexPQ(dim, 32, 8, faiss.METRIC_INNER_PRODUCT)
    if index_type == "pq_m64_nbits6":
        return faiss.IndexPQ(dim, 64, 6, faiss.METRIC_INNER_PRODUCT)
    if index_type == "pq_m64_nbits4":
        return faiss.IndexPQ(dim, 64, 4, faiss.METRIC_INNER_PRODUCT)
    if index_type == "opq_pq_m32_nbits6":
        opq = faiss.OPQMatrix(dim, 32)
        pq = faiss.IndexPQ(dim, 32, 6, faiss.METRIC_INNER_PRODUCT)
        return faiss.IndexPreTransform(opq, pq)
    if index_type == "opq_pq_m32_nbits8":
        opq = faiss.OPQMatrix(dim, 32)
        pq = faiss.IndexPQ(dim, 32, 8, faiss.METRIC_INNER_PRODUCT)
        return faiss.IndexPreTransform(opq, pq)
    if index_type == "opq_pq_m64_nbits4":
        opq = faiss.OPQMatrix(dim, 64)
        pq = faiss.IndexPQ(dim, 64, 4, faiss.METRIC_INNER_PRODUCT)
        return faiss.IndexPreTransform(opq, pq)
    if index_type == "opq_pq_m64_nbits6":
        opq = faiss.OPQMatrix(dim, 64)
        pq = faiss.IndexPQ(dim, 64, 6, faiss.METRIC_INNER_PRODUCT)
        return faiss.IndexPreTransform(opq, pq)
    raise ValueError(f"Unsupported index type: {index_type}")


def pareto_frontier(labels: List[str], x_values: List[float], y_values: List[float], minimize_x: bool = True) -> List[str]:
    frontier = []
    for i, label_i in enumerate(labels):
        xi = x_values[i]
        yi = y_values[i]
        dominated = False
        for j, label_j in enumerate(labels):
            if i == j:
                continue
            xj = x_values[j]
            yj = y_values[j]
            if minimize_x:
                no_worse = (xj <= xi) and (yj >= yi)
                strictly_better = (xj < xi) or (yj > yi)
            else:
                no_worse = (xj >= xi) and (yj >= yi)
                strictly_better = (xj > xi) or (yj > yi)
            if no_worse and strictly_better:
                dominated = True
                break
        if not dominated:
            frontier.append(label_i)
    return frontier


def evaluate_search(results_idx: np.ndarray, gold: np.ndarray) -> Dict[str, float]:
    def hit_at(k: int) -> float:
        return float(np.mean([gold[i] in results_idx[i, :k] for i in range(results_idx.shape[0])]))

    rr = []
    for i in range(results_idx.shape[0]):
        row = results_idx[i]
        hits = np.where(row == gold[i])[0]
        rr.append(1.0 / (hits[0] + 1) if hits.size else 0.0)

    metrics = {
        "queries": int(results_idx.shape[0]),
        "recall@1": hit_at(1),
        "recall@5": hit_at(min(5, results_idx.shape[1])),
        "recall@10": hit_at(min(10, results_idx.shape[1])),
        "mrr@k": float(np.mean(rr)),
    }
    for k in (20, 50, 100):
        if results_idx.shape[1] >= k:
            metrics[f"recall@{k}"] = hit_at(k)
    return metrics


def benchmark_index(
    index_type: str,
    kb_vectors: np.ndarray,
    query_vectors: np.ndarray,
    gold: np.ndarray,
    topk: int,
) -> Dict[str, object]:
    dim = int(kb_vectors.shape[1])
    index = create_index(index_type, dim)

    train_start = time.perf_counter()
    if not index.is_trained:
        index.train(kb_vectors)
    train_seconds = time.perf_counter() - train_start

    add_start = time.perf_counter()
    index.add(kb_vectors)
    add_seconds = time.perf_counter() - add_start

    search_start = time.perf_counter()
    _, I = index.search(query_vectors, topk)
    search_seconds = time.perf_counter() - search_start

    metrics = evaluate_search(I, gold)
    serialized = faiss.serialize_index(index)
    metrics.update(
        {
            "index_type": index_type,
            "topk": int(topk),
            "train_time_seconds": train_seconds,
            "add_time_seconds": add_seconds,
            "search_time_seconds": search_seconds,
            "latency_ms": search_seconds * 1000.0,
            "per_query_latency_ms": (search_seconds * 1000.0) / max(query_vectors.shape[0], 1),
            "queries_per_second": float(query_vectors.shape[0] / max(search_seconds, 1e-12)),
            "index_size_bytes": int(len(serialized)),
            "index_size_mb": bytes_to_mb(int(len(serialized))),
        }
    )
    return metrics


def create_plots(results: Dict[str, object], out_dir: str) -> List[str]:
    os.makedirs(out_dir, exist_ok=True)
    labels = list(results["indexes"].keys())
    r1 = [results["indexes"][k]["recall@1"] for k in labels]
    latency = [results["indexes"][k]["latency_ms"] for k in labels]
    size_mb = [results["indexes"][k]["index_size_mb"] for k in labels]
    r1_drop = [results["indexes"][k].get("recall@1_drop_vs_flat", 0.0) for k in labels]
    compression = [results["indexes"][k].get("compression_ratio_vs_flat", 1.0) for k in labels]
    speedup = [results["indexes"][k].get("speedup_vs_flat", 1.0) for k in labels]
    size_frontier = set(results.get("pareto_frontiers", {}).get("size_vs_recall@1", []))
    latency_frontier = set(results.get("pareto_frontiers", {}).get("latency_vs_recall@1", []))

    created = []

    plt.figure(figsize=(10, 5))
    plt.bar(labels, r1)
    plt.ylabel("Recall@1")
    plt.title("FAISS Index Quantization Accuracy")
    plt.xticks(rotation=20)
    plt.tight_layout()
    path1 = os.path.join(out_dir, "faiss_index_accuracy.png")
    plt.savefig(path1, dpi=160)
    plt.close()
    created.append(path1)

    plt.figure(figsize=(10, 5))
    for x, y, label in zip(size_mb, r1, labels):
        color = "tab:red" if label in size_frontier else "tab:blue"
        marker = "D" if label in size_frontier else "o"
        plt.scatter([x], [y], s=140, c=color, marker=marker)
        plt.annotate(label, (x, y), textcoords="offset points", xytext=(5, 5))
    plt.xlabel("Serialized index size (MB)")
    plt.ylabel("Recall@1")
    plt.title("FAISS Accuracy vs Index Size")
    plt.tight_layout()
    path2 = os.path.join(out_dir, "faiss_accuracy_vs_size.png")
    plt.savefig(path2, dpi=160)
    plt.close()
    created.append(path2)

    plt.figure(figsize=(10, 5))
    for x, y, label in zip(latency, r1, labels):
        color = "tab:red" if label in latency_frontier else "tab:blue"
        marker = "D" if label in latency_frontier else "o"
        plt.scatter([x], [y], s=140, c=color, marker=marker)
        plt.annotate(label, (x, y), textcoords="offset points", xytext=(5, 5))
    plt.xlabel("Search latency (ms)")
    plt.ylabel("Recall@1")
    plt.title("FAISS Accuracy vs Search Latency")
    plt.tight_layout()
    path3 = os.path.join(out_dir, "faiss_accuracy_vs_latency.png")
    plt.savefig(path3, dpi=160)
    plt.close()
    created.append(path3)

    plt.figure(figsize=(10, 5))
    plt.scatter(compression, r1_drop, s=120)
    for x, y, label in zip(compression, r1_drop, labels):
        plt.annotate(label, (x, y), textcoords="offset points", xytext=(5, 5))
    plt.axhline(0.0, linestyle="--", linewidth=1)
    plt.xlabel("Compression ratio vs flat")
    plt.ylabel("Recall@1 drop vs flat")
    plt.title("FAISS Quantization Tradeoff: Compression vs Recall Loss")
    plt.tight_layout()
    path4 = os.path.join(out_dir, "faiss_tradeoff_compression_vs_r1_drop.png")
    plt.savefig(path4, dpi=160)
    plt.close()
    created.append(path4)

    plt.figure(figsize=(10, 5))
    plt.scatter(speedup, r1_drop, s=120)
    for x, y, label in zip(speedup, r1_drop, labels):
        plt.annotate(label, (x, y), textcoords="offset points", xytext=(5, 5))
    plt.axhline(0.0, linestyle="--", linewidth=1)
    plt.axvline(1.0, linestyle="--", linewidth=1)
    plt.xlabel("Search speedup vs flat")
    plt.ylabel("Recall@1 drop vs flat")
    plt.title("FAISS Quantization Tradeoff: Speedup vs Recall Loss")
    plt.tight_layout()
    path5 = os.path.join(out_dir, "faiss_tradeoff_speedup_vs_r1_drop.png")
    plt.savefig(path5, dpi=160)
    plt.close()
    created.append(path5)

    return created


def export_tradeoff_csv(results: Dict[str, object], csv_path: str) -> str:
    rows = []
    size_frontier = set(results.get("pareto_frontiers", {}).get("size_vs_recall@1", []))
    latency_frontier = set(results.get("pareto_frontiers", {}).get("latency_vs_recall@1", []))

    for name, metrics in results["indexes"].items():
        rows.append(
            {
                "index": name,
                "recall@1": metrics["recall@1"],
                "recall@10": metrics["recall@10"],
                "r1_drop_vs_flat": metrics.get("recall@1_drop_vs_flat"),
                "size_mb": metrics["index_size_mb"],
                "size_saving_mb_vs_flat": metrics.get("size_saving_mb_vs_flat"),
                "compression_ratio_vs_flat": metrics.get("compression_ratio_vs_flat"),
                "latency_ms": metrics["latency_ms"],
                "latency_delta_ms_vs_flat": metrics.get("latency_delta_ms_vs_flat"),
                "speedup_vs_flat": metrics.get("speedup_vs_flat"),
                "queries_per_second": metrics["queries_per_second"],
                "pareto_size_recall": name in size_frontier,
                "pareto_latency_recall": name in latency_frontier,
            }
        )

    rows.sort(key=lambda row: row["recall@1"], reverse=True)
    fieldnames = list(rows[0].keys()) if rows else [
        "index",
        "recall@1",
        "recall@10",
        "r1_drop_vs_flat",
        "size_mb",
        "size_saving_mb_vs_flat",
        "compression_ratio_vs_flat",
        "latency_ms",
        "latency_delta_ms_vs_flat",
        "speedup_vs_flat",
        "queries_per_second",
        "pareto_size_recall",
        "pareto_latency_recall",
    ]

    out_path = Path(csv_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return str(out_path)


def main():
    parser = argparse.ArgumentParser(description="Benchmark FAISS/PQ index quantization for entity linking.")
    parser.add_argument("--image-embeddings", default=DEFAULT_IMAGE_EMB)
    parser.add_argument("--valid-metadata", default=DEFAULT_VALID_META)
    parser.add_argument("--kb-text-embeddings", default=DEFAULT_KB_TEXT_EMB)
    parser.add_argument("--entity-kb", default=DEFAULT_ENTITY_KB)
    parser.add_argument("--topk", type=int, default=10)
    parser.add_argument(
        "--indexes",
        nargs="+",
        default=[
            "flat",
            "sq8",
            "pq_m32_nbits6",
            "pq_m32_nbits8",
            "pq_m64_nbits6",
            "pq_m64_nbits4",
            "opq_pq_m32_nbits6",
            "opq_pq_m32_nbits8",
            "opq_pq_m64_nbits6",
            "opq_pq_m64_nbits4",
        ],
    )
    parser.add_argument("--out-json", default="")
    parser.add_argument("--plot-dir", default="")
    args = parser.parse_args()

    image_embeddings = normalize_rows(np.load(args.image_embeddings).astype(np.float32))
    valid_meta = load_json(args.valid_metadata)
    kb_vectors = normalize_rows(np.load(args.kb_text_embeddings).astype(np.float32))
    entity_kb = load_json(args.entity_kb)

    keep_positions, gold = kb_gold_indices(valid_meta, entity_kb)
    query_vectors = image_embeddings[keep_positions]

    results = {
        "environment": collect_environment(),
        "config": {
            "image_embeddings": args.image_embeddings,
            "valid_metadata": args.valid_metadata,
            "kb_text_embeddings": args.kb_text_embeddings,
            "entity_kb": args.entity_kb,
            "topk": args.topk,
            "indexes": args.indexes,
        },
        "dataset": {
            "queries": int(query_vectors.shape[0]),
            "entities": int(kb_vectors.shape[0]),
            "embedding_dim": int(kb_vectors.shape[1]),
        },
        "indexes": {},
    }

    for index_type in args.indexes:
        metrics = benchmark_index(
            index_type=index_type,
            kb_vectors=kb_vectors,
            query_vectors=query_vectors,
            gold=gold,
            topk=args.topk,
        )
        results["indexes"][index_type] = metrics
        print(
            f"{index_type}: R@1={metrics['recall@1']:.4f}, "
            f"R@10={metrics['recall@10']:.4f}, "
            f"latency={metrics['latency_ms']:.2f} ms, "
            f"size={metrics['index_size_mb']:.2f} MB"
        )

    if "flat" in results["indexes"]:
        baseline = results["indexes"]["flat"]
        base_r1 = baseline["recall@1"]
        base_r10 = baseline["recall@10"]
        base_size = baseline["index_size_mb"]
        base_latency = baseline["latency_ms"]
        for name, metrics in results["indexes"].items():
            metrics["recall@1_drop_vs_flat"] = float(base_r1 - metrics["recall@1"])
            metrics["recall@10_drop_vs_flat"] = float(base_r10 - metrics["recall@10"])
            metrics["size_saving_mb_vs_flat"] = float(base_size - metrics["index_size_mb"])
            metrics["compression_ratio_vs_flat"] = float(base_size / max(metrics["index_size_mb"], 1e-12))
            metrics["latency_delta_ms_vs_flat"] = float(metrics["latency_ms"] - base_latency)
            metrics["speedup_vs_flat"] = float(base_latency / max(metrics["latency_ms"], 1e-12))
            metrics["qps_ratio_vs_flat"] = float(metrics["queries_per_second"] / max(baseline["queries_per_second"], 1e-12))

    labels = list(results["indexes"].keys())
    size_mb = [results["indexes"][k]["index_size_mb"] for k in labels]
    latency_ms = [results["indexes"][k]["latency_ms"] for k in labels]
    recall1 = [results["indexes"][k]["recall@1"] for k in labels]
    results["pareto_frontiers"] = {
        "size_vs_recall@1": pareto_frontier(labels, size_mb, recall1, minimize_x=True),
        "latency_vs_recall@1": pareto_frontier(labels, latency_ms, recall1, minimize_x=True),
    }

    if args.plot_dir:
        plot_paths = create_plots(results, args.plot_dir)
        results["plots"] = plot_paths

    if args.out_json:
        out_path = Path(args.out_json)
        csv_path = str(out_path.with_suffix(".csv"))
        results["tradeoff_csv"] = export_tradeoff_csv(results, csv_path)

        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(f"Saved results to {out_path}")


if __name__ == "__main__":
    main()
