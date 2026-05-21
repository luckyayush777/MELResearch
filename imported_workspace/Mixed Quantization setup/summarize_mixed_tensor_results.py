import argparse
import csv
from pathlib import Path


def as_float(row, key, default=0.0):
    try:
        return float(row.get(key, default))
    except (TypeError, ValueError):
        return default


def pct(value):
    return f"{value * 100.0:.2f}%"


def main():
    parser = argparse.ArgumentParser(description="Create a report for explicit mixed tensor quantization runs.")
    parser.add_argument("--summary-csv", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    summary_csv = (script_dir / args.summary_csv).resolve()
    out_path = (script_dir / args.out).resolve()

    with summary_csv.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))

    baseline = next((row for row in rows if row.get("text_tokens_mode") == "fp32" and row.get("image_patch_tokens_mode") == "fp32"), None)
    if baseline is None:
        raise RuntimeError("No fp32/fp32 baseline row found.")

    lines = [
        "# Mixed Text + Image Token Quantization",
        "",
        "This compares fp32 cache tensors against quantizing both `text_tokens` and `image_patch_tokens` while keeping CLS tensors fp32.",
        "",
        "## Comparison",
        "",
        "| Mode | Text tokens | Image patches | H@1 | MRR | H@1 delta | MRR delta | Speedup | Cache ratio | Text ratio | Image ratio | Disk MB | Transfer s | TGLU s | Peak VRAM | Rank flips |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]

    for row in rows:
        lines.append(
            "| "
            f"{row.get('mode', '')} | "
            f"{row.get('text_tokens_mode', '')} | "
            f"{row.get('image_patch_tokens_mode', '')} | "
            f"{as_float(row, 'hits@1'):.6f} | "
            f"{as_float(row, 'mrr'):.6f} | "
            f"{as_float(row, 'delta_hits@1_vs_fp32'):.6f} | "
            f"{as_float(row, 'delta_mrr_vs_fp32'):.6f} | "
            f"{as_float(row, 'speedup_vs_fp32', 1.0):.3f}x | "
            f"{pct(as_float(row, 'cache_size_ratio_vs_fp32', 1.0))} | "
            f"{pct(as_float(row, 'text_tokens_size_ratio_vs_fp32', 1.0))} | "
            f"{pct(as_float(row, 'image_patch_tokens_size_ratio_vs_fp32', 1.0))} | "
            f"{as_float(row, 'entity_cache_disk_mb'):.2f} | "
            f"{as_float(row, 'entity_transfer_s'):.2f} | "
            f"{as_float(row, 'tglu_s'):.2f} | "
            f"{as_float(row, 'gpu_peak_allocated_mb_eval'):.2f} | "
            f"{int(as_float(row, 'rank_flip_count'))} |"
        )

    lines.extend([
        "",
        "## Readout",
        "",
        "The mixed int4 run is useful if the small H@1/MRR deltas are acceptable for the deployment story.",
        "Peak VRAM remains dominated by matcher/TGLU allocation at the current chunk size, so chunk-size or matcher changes are still needed for a real smaller-GPU proof.",
    ])

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
