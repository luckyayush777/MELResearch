import argparse
import csv
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def run_command(cmd: List[str]) -> None:
    print("\n>>", " ".join(cmd))
    proc = subprocess.run(cmd)
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed with exit code {proc.returncode}: {' '.join(cmd)}")


def load_json(path: Path) -> Dict[str, object]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def create_embedding_plots(results: Dict[str, object], out_dir: Path) -> List[str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    kb_results = results["kb"]
    labels = list(kb_results.keys())
    recall10 = [kb_results[label]["recall@10"] for label in labels]
    recall100 = [kb_results[label]["recall@100"] for label in labels]
    latency = [kb_results[label]["latency_ms"] for label in labels]
    text_storage = [kb_results[label]["text_storage_mb"] for label in labels]
    mrr = [kb_results[label]["mrr"] for label in labels]

    created: List[str] = []

    plt.figure(figsize=(8, 5))
    for x, y, label in zip(text_storage, recall100, labels):
        plt.scatter([x], [y], s=140)
        plt.annotate(label, (x, y), textcoords="offset points", xytext=(5, 5))
    plt.xlabel("Text embedding storage (MB)")
    plt.ylabel("Recall@100")
    plt.title("Official WikiMEL Embedding Quantization: Recall@100 vs Storage")
    plt.grid(alpha=0.25)
    path1 = out_dir / "embedding_recall100_vs_storage.png"
    plt.tight_layout()
    plt.savefig(path1, dpi=160)
    plt.close()
    created.append(str(path1))

    plt.figure(figsize=(8, 5))
    for x, y, label in zip(latency, recall10, labels):
        plt.scatter([x], [y], s=140)
        plt.annotate(label, (x, y), textcoords="offset points", xytext=(5, 5))
    plt.xlabel("Latency (ms)")
    plt.ylabel("Recall@10")
    plt.title("Official WikiMEL Embedding Quantization: Recall@10 vs Latency")
    plt.grid(alpha=0.25)
    path2 = out_dir / "embedding_recall10_vs_latency.png"
    plt.tight_layout()
    plt.savefig(path2, dpi=160)
    plt.close()
    created.append(str(path2))

    plt.figure(figsize=(8, 5))
    plt.bar(labels, mrr)
    plt.ylabel("MRR")
    plt.title("Official WikiMEL Embedding Quantization: MRR")
    plt.grid(axis="y", alpha=0.2)
    path3 = out_dir / "embedding_mrr.png"
    plt.tight_layout()
    plt.savefig(path3, dpi=160)
    plt.close()
    created.append(str(path3))

    return created


def build_summary(
    baseline: Dict[str, object],
    embed_results: Dict[str, object],
    faiss_results: Dict[str, object],
) -> Dict[str, object]:
    base_metrics = baseline["metrics"]
    embedding_summary = {}
    for mode, metrics in embed_results["kb"].items():
        embedding_summary[mode] = {
            "recall@1": metrics["recall@1"],
            "recall@10": metrics["recall@10"],
            "recall@100": metrics["recall@100"],
            "mrr": metrics["mrr"],
            "latency_ms": metrics["latency_ms"],
            "queries_per_second": metrics["queries_per_second"],
            "text_storage_mb": metrics["text_storage_mb"],
            "delta_recall@1": metrics["recall@1"] - base_metrics["recall@1"],
            "delta_recall@10": metrics["recall@10"] - base_metrics["recall@10"],
            "delta_recall@100": metrics["recall@100"] - base_metrics["recall@100"],
            "delta_mrr": metrics["mrr"] - base_metrics["mrr"],
        }

    faiss_summary = {}
    for index_name, metrics in faiss_results["indexes"].items():
        faiss_summary[index_name] = {
            "recall@1": metrics["recall@1"],
            "recall@10": metrics["recall@10"],
            "recall@100": metrics.get("recall@100"),
            "mrr@k": metrics["mrr@k"],
            "latency_ms": metrics["latency_ms"],
            "queries_per_second": metrics["queries_per_second"],
            "index_size_mb": metrics["index_size_mb"],
            "delta_recall@1": metrics["recall@1"] - base_metrics["recall@1"],
            "delta_recall@10": metrics["recall@10"] - base_metrics["recall@10"],
            "delta_recall@100": (
                metrics.get("recall@100", base_metrics["recall@100"]) - base_metrics["recall@100"]
                if metrics.get("recall@100") is not None
                else None
            ),
        }

    return {
        "baseline": {
            "recall@1": base_metrics["recall@1"],
            "recall@10": base_metrics["recall@10"],
            "recall@100": base_metrics["recall@100"],
            "mrr": base_metrics["mrr"],
            "latency_ms": base_metrics["latency_ms"],
            "queries_per_second": base_metrics["queries_per_second"],
            "candidate_storage_mb": base_metrics["candidate_storage_mb"],
        },
        "embedding_quantization": embedding_summary,
        "faiss_index_quantization": faiss_summary,
    }


def export_embedding_csv(path: Path, summary: Dict[str, object]) -> str:
    rows = []
    for mode, metrics in summary["embedding_quantization"].items():
        row = {"mode": mode}
        row.update(metrics)
        rows.append(row)
    fieldnames = list(rows[0].keys()) if rows else ["mode"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return str(path)


def choose_recommendations(summary: Dict[str, object]) -> Dict[str, object]:
    base = summary["baseline"]
    safe_modes = []
    for mode, metrics in summary["embedding_quantization"].items():
        if metrics["delta_recall@100"] >= -0.005 and metrics["text_storage_mb"] < base["candidate_storage_mb"]:
            safe_modes.append(mode)

    safe_indexes = []
    for name, metrics in summary["faiss_index_quantization"].items():
        if metrics["delta_recall@10"] >= -0.01 and metrics["index_size_mb"] < base["candidate_storage_mb"]:
            safe_indexes.append(name)

    return {
        "quality_preserving_embedding_modes": safe_modes,
        "quality_preserving_indexes": safe_indexes,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the official WikiMEL quantization suite on cached artifacts.")
    parser.add_argument("--artifact-root", default=os.path.join("data", "wikimel_official_test_v1"))
    parser.add_argument("--baseline-json", default=os.path.join("benchmarks", "wikimel_official_test_v1", "baseline_results.json"))
    parser.add_argument("--out-dir", default=os.path.join("benchmarks", "wikimel_official_test_v1", "quant_suite"))
    parser.add_argument("--quant-modes", nargs="+", default=["fp32", "fp16", "int8"])
    parser.add_argument("--faiss-indexes", nargs="+", default=["flat", "sq8", "pq_m32_nbits6", "pq_m32_nbits8", "pq_m64_nbits6", "pq_m64_nbits4", "opq_pq_m32_nbits6", "opq_pq_m32_nbits8"])
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--faiss-topk", type=int, default=100)
    args = parser.parse_args()

    artifact_root = Path(args.artifact_root)
    emb_dir = artifact_root / "embeddings"
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    image_embeddings = emb_dir / "image_embeddings.npy"
    kb_text_embeddings = emb_dir / "kb_text_embeddings.npy"
    valid_metadata = emb_dir / "valid_metadata.json"
    entity_kb = artifact_root / "entity_kb.json"
    baseline_json = Path(args.baseline_json)

    required = [image_embeddings, kb_text_embeddings, valid_metadata, entity_kb, baseline_json]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing required official WikiMEL artifacts:\n" + "\n".join(missing))

    quant_json = out_dir / "embedding_quantization_results.json"
    quant_cmd = [
        sys.executable,
        "benchmark_quantization.py",
        "--skip-paired",
        "--image-embeddings",
        str(image_embeddings),
        "--valid-metadata",
        str(valid_metadata),
        "--entity-kb",
        str(entity_kb),
        "--kb-text-embeddings",
        str(kb_text_embeddings),
        "--quant-modes",
        *args.quant_modes,
        "--repeats",
        str(args.repeats),
        "--warmup",
        str(args.warmup),
        "--out-json",
        str(quant_json),
    ]
    run_command(quant_cmd)

    faiss_json = out_dir / "faiss_pq_results.json"
    faiss_plot_dir = out_dir / "faiss_pq_plots"
    faiss_cmd = [
        sys.executable,
        "benchmark_faiss_pq.py",
        "--image-embeddings",
        str(image_embeddings),
        "--valid-metadata",
        str(valid_metadata),
        "--kb-text-embeddings",
        str(kb_text_embeddings),
        "--entity-kb",
        str(entity_kb),
        "--topk",
        str(args.faiss_topk),
        "--indexes",
        *args.faiss_indexes,
        "--out-json",
        str(faiss_json),
        "--plot-dir",
        str(faiss_plot_dir),
    ]
    run_command(faiss_cmd)

    baseline = load_json(baseline_json)
    embed_results = load_json(quant_json)
    faiss_results = load_json(faiss_json)

    summary = build_summary(baseline, embed_results, faiss_results)
    summary["recommendations"] = choose_recommendations(summary)
    summary["embedding_plots"] = create_embedding_plots(embed_results, out_dir / "embedding_plots")
    summary["embedding_csv"] = export_embedding_csv(out_dir / "embedding_quantization_summary.csv", summary)

    summary_json = out_dir / "suite_summary.json"
    with summary_json.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\nOfficial WikiMEL quantization suite completed.")
    print(f"Results root: {out_dir}")
    print("Quality-preserving embedding modes:", ", ".join(summary["recommendations"]["quality_preserving_embedding_modes"]) or "none")
    print("Quality-preserving indexes:", ", ".join(summary["recommendations"]["quality_preserving_indexes"]) or "none")


if __name__ == "__main__":
    main()
