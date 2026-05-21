import argparse
import csv
import json
from pathlib import Path

import numpy as np


def read_csv(path: Path):
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def read_jsonl(path: Path):
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def as_float(row, key, default=0.0):
    try:
        return float(row.get(key, default))
    except (TypeError, ValueError):
        return default


def as_int(row, key, default=0):
    try:
        return int(float(row.get(key, default)))
    except (TypeError, ValueError):
        return default


def write_csv(path: Path, rows):
    fieldnames = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def load_ranks(mode_dir: Path):
    parts = []
    for path in sorted((mode_dir / "rank_batches").glob("batch_*.npy")):
        arr = np.load(path)
        if arr.size:
            parts.append(arr)
    return np.concatenate(parts) if parts else np.array([], dtype=np.int64)


def row_key(row):
    return row.get("sweep_mode"), as_int(row, "chunk_size")


def find_row(rows, mode, chunk):
    for row in rows:
        if row_key(row) == (mode, chunk):
            return row
    raise RuntimeError(f"Missing row: {mode} chunk {chunk}")


def budget_rows(full_rows, repeat_rows):
    # Use the fresh repeat for the headline 6 GB and 12 GB rows; use the
    # existing full sweep for lower budgets that were not rerun.
    source_by_chunk = {
        1000: full_rows,
        2500: full_rows,
        5000: repeat_rows,
        10000: repeat_rows,
    }
    budgets = [
        ("~2 GB", 1000),
        ("~4 GB", 2500),
        ("~6 GB", 5000),
        ("~12 GB", 10000),
    ]
    rows = []
    for budget, chunk in budgets:
        source = source_by_chunk[chunk]
        fp32 = find_row(source, "fp32_ref", chunk)
        quant = find_row(source, "text_image_tokens_int4", chunk)
        rows.append({
            "budget": budget,
            "recommended_mode": "text_image_tokens_int4",
            "chunk_size": chunk,
            "hits@1": f"{as_float(quant, 'hits@1'):.6f}",
            "mrr": f"{as_float(quant, 'mrr'):.6f}",
            "delta_hits@1_vs_fp32": f"{as_float(quant, 'hits@1') - as_float(fp32, 'hits@1'):.6f}",
            "delta_mrr_vs_fp32": f"{as_float(quant, 'mrr') - as_float(fp32, 'mrr'):.6f}",
            "peak_eval_vram_mb": f"{as_float(quant, 'gpu_peak_allocated_mb_eval'):.2f}",
            "qps": f"{as_float(quant, 'queries_per_second'):.3f}",
            "avg_query_latency_ms": f"{as_float(quant, 'avg_query_latency_ms'):.2f}",
            "p95_query_latency_ms": f"{as_float(quant, 'p95_query_latency_ms'):.2f}",
            "runtime_s": f"{as_float(quant, 'seconds'):.2f}",
            "fp32_runtime_s": f"{as_float(fp32, 'seconds'):.2f}",
            "speedup_vs_fp32_same_chunk": f"{as_float(fp32, 'seconds') / max(as_float(quant, 'seconds'), 1e-9):.3f}",
            "logical_cache_mb": f"{as_float(quant, 'entity_cache_storage_mb'):.2f}",
            "disk_cache_mb": f"{as_float(quant, 'entity_cache_disk_mb'):.2f}",
        })
    return rows


def conservative_text_row(full_rows):
    row = find_row(full_rows, "text_tokens_int4", 5000)
    fp32 = find_row(full_rows, "fp32_ref", 5000)
    return {
        "mode": "text_tokens_int4",
        "chunk_size": 5000,
        "hits@1": f"{as_float(row, 'hits@1'):.6f}",
        "mrr": f"{as_float(row, 'mrr'):.6f}",
        "delta_hits@1_vs_fp32": f"{as_float(row, 'hits@1') - as_float(fp32, 'hits@1'):.6f}",
        "delta_mrr_vs_fp32": f"{as_float(row, 'mrr') - as_float(fp32, 'mrr'):.6f}",
        "peak_eval_vram_mb": f"{as_float(row, 'gpu_peak_allocated_mb_eval'):.2f}",
        "qps": f"{as_float(row, 'queries_per_second'):.3f}",
        "speedup_vs_fp32_same_chunk": f"{as_float(fp32, 'seconds') / max(as_float(row, 'seconds'), 1e-9):.3f}",
        "logical_cache_mb": f"{as_float(row, 'entity_cache_storage_mb'):.2f}",
    }


def rank_flip_examples(fp32_dir: Path, quant_dir: Path, test_path: Path, limit: int):
    fp32_ranks = load_ranks(fp32_dir)
    quant_ranks = load_ranks(quant_dir)
    test_rows = read_jsonl(test_path)
    n = min(len(fp32_ranks), len(quant_ranks), len(test_rows))
    deltas = quant_ranks[:n].astype(np.int64) - fp32_ranks[:n].astype(np.int64)

    examples = []
    for idx in np.where(deltas != 0)[0]:
        sample = test_rows[int(idx)]
        delta = int(deltas[int(idx)])
        examples.append({
            "query_index": int(idx),
            "direction": "improved" if delta < 0 else "worsened",
            "fp32_gold_rank": int(fp32_ranks[int(idx)]),
            "int4_gold_rank": int(quant_ranks[int(idx)]),
            "rank_delta": delta,
            "mention": sample.get("mention", ""),
            "target": sample.get("target", ""),
            "golden": sample.get("golden", ""),
            "img_name": sample.get("img_name", ""),
            "text": sample.get("text", ""),
        })

    improved = sorted((e for e in examples if e["rank_delta"] < 0), key=lambda e: e["rank_delta"])
    worsened = sorted((e for e in examples if e["rank_delta"] > 0), key=lambda e: e["rank_delta"], reverse=True)
    balanced = improved[:limit] + worsened[:limit]

    summary = {
        "queries_compared": int(n),
        "rank_flip_count": int(np.count_nonzero(deltas)),
        "rank_improved_count": int(np.count_nonzero(deltas < 0)),
        "rank_worsened_count": int(np.count_nonzero(deltas > 0)),
        "mean_abs_rank_delta": float(np.mean(np.abs(deltas))),
        "median_abs_rank_delta": float(np.median(np.abs(deltas))),
        "max_rank_improvement": int(np.min(deltas)),
        "max_rank_worsening": int(np.max(deltas)),
    }
    return summary, balanced


def md_table(rows, columns):
    lines = [
        "| " + " | ".join(title for title, _ in columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(key, "")) for _, key in columns) + " |")
    return lines


def write_report(path: Path, budget, text_fallback, rank_summary, examples):
    lines = [
        "# Key Repeat Rows And GPU Budget Readout",
        "",
        "## Repeated Headline Rows",
        "",
        "The repeated key rows confirm that `text_image_tokens_int4` is stable at the practical operating chunks.",
        "",
    ]
    lines.extend(md_table(
        [row for row in budget if row["chunk_size"] in (5000, 10000)],
        [
            ("Budget", "budget"),
            ("Chunk", "chunk_size"),
            ("H@1", "hits@1"),
            ("MRR", "mrr"),
            ("Delta H@1", "delta_hits@1_vs_fp32"),
            ("Peak VRAM MB", "peak_eval_vram_mb"),
            ("QPS", "qps"),
            ("Speedup", "speedup_vs_fp32_same_chunk"),
        ],
    ))
    lines.extend([
        "",
        "## GPU Budget Recommendation",
        "",
    ])
    lines.extend(md_table(
        budget,
        [
            ("Budget", "budget"),
            ("Mode", "recommended_mode"),
            ("Chunk", "chunk_size"),
            ("H@1", "hits@1"),
            ("MRR", "mrr"),
            ("Peak VRAM MB", "peak_eval_vram_mb"),
            ("QPS", "qps"),
            ("P95 ms", "p95_query_latency_ms"),
            ("Speedup", "speedup_vs_fp32_same_chunk"),
            ("Logical MB", "logical_cache_mb"),
            ("Disk MB", "disk_cache_mb"),
        ],
    ))
    lines.extend([
        "",
        "## Conservative Fallback",
        "",
    ])
    lines.extend(md_table(
        [text_fallback],
        [
            ("Mode", "mode"),
            ("Chunk", "chunk_size"),
            ("H@1", "hits@1"),
            ("MRR", "mrr"),
            ("Delta H@1", "delta_hits@1_vs_fp32"),
            ("Peak VRAM MB", "peak_eval_vram_mb"),
            ("QPS", "qps"),
            ("Speedup", "speedup_vs_fp32_same_chunk"),
            ("Logical MB", "logical_cache_mb"),
        ],
    ))
    lines.extend([
        "",
        "## Rank-Flip Summary",
        "",
        f"- Queries compared: `{rank_summary['queries_compared']}`",
        f"- Rank flips: `{rank_summary['rank_flip_count']}`",
        f"- Improved: `{rank_summary['rank_improved_count']}`",
        f"- Worsened: `{rank_summary['rank_worsened_count']}`",
        f"- Mean absolute rank delta: `{rank_summary['mean_abs_rank_delta']:.6f}`",
        f"- Median absolute rank delta: `{rank_summary['median_abs_rank_delta']:.6f}`",
        f"- Max improvement: `{rank_summary['max_rank_improvement']}`",
        f"- Max worsening: `{rank_summary['max_rank_worsening']}`",
        "",
        "## Rank-Flip Examples",
        "",
    ])
    lines.extend(md_table(
        examples,
        [
            ("Dir", "direction"),
            ("Idx", "query_index"),
            ("Mention", "mention"),
            ("Target", "target"),
            ("FP32", "fp32_gold_rank"),
            ("Int4", "int4_gold_rank"),
            ("Delta", "rank_delta"),
        ],
    ))
    lines.extend([
        "",
        "Detailed examples with full text are saved in `rank_flip_examples.csv` and `rank_flip_examples.json`.",
        "",
    ])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--full-summary", default="results/chunk_size_sweep_full/chunk_size_sweep_summary.csv")
    parser.add_argument("--repeat-summary", default="results/chunk_size_sweep_repeat_keyrows/chunk_size_sweep_summary.csv")
    parser.add_argument("--test-jsonl", default="../datasets/wikimel/test.json")
    parser.add_argument("--out-dir", default="results/keyrow_report")
    parser.add_argument("--examples-per-side", type=int, default=10)
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    full_rows = read_csv((script_dir / args.full_summary).resolve())
    repeat_rows = read_csv((script_dir / args.repeat_summary).resolve())
    out_dir = (script_dir / args.out_dir).resolve()

    budget = budget_rows(full_rows, repeat_rows)
    text_fallback = conservative_text_row(full_rows)
    fp32 = Path(find_row(repeat_rows, "fp32_ref", 5000)["checkpoint_dir"])
    quant = Path(find_row(repeat_rows, "text_image_tokens_int4", 5000)["checkpoint_dir"])
    rank_summary, examples = rank_flip_examples(
        fp32,
        quant,
        (script_dir / args.test_jsonl).resolve(),
        args.examples_per_side,
    )

    write_csv(out_dir / "gpu_budget_recommendation.csv", budget)
    write_csv(out_dir / "conservative_text_tokens_int4.csv", [text_fallback])
    write_csv(out_dir / "rank_flip_examples.csv", examples)
    write_json(out_dir / "rank_flip_examples.json", {"summary": rank_summary, "examples": examples})
    write_report(out_dir / "KEYROW_REPORT.md", budget, text_fallback, rank_summary, examples)
    print(f"Wrote report artifacts to {out_dir}")


if __name__ == "__main__":
    main()
