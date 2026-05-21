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


def main() -> None:
    parser = argparse.ArgumentParser(description="Create an attribution report for a target tensor quantization run.")
    parser.add_argument("--summary-csv", required=True)
    parser.add_argument("--target-tensor", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    summary_csv = (script_dir / args.summary_csv).resolve()
    out_path = (script_dir / args.out).resolve()
    target_tensor = args.target_tensor
    target_mode_key = f"{target_tensor}_mode"
    target_cache_key = f"entity_cache_{target_tensor}_mb"

    with summary_csv.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))

    baseline = next((row for row in rows if row.get(target_mode_key) == "fp32"), None)
    if baseline is None:
        raise RuntimeError(f"No fp32 baseline row found using key {target_mode_key}.")

    lines = [
        f"# Entity {target_tensor} Quantization Attribution",
        "",
        f"This report compares changing only `{target_tensor}` while keeping the other cached tensors fixed.",
        "",
        "## Baseline",
        "",
        f"- H@1: `{as_float(baseline, 'hits@1'):.6f}`",
        f"- MRR: `{as_float(baseline, 'mrr'):.6f}`",
        f"- Evaluation seconds: `{as_float(baseline, 'seconds'):.2f}`",
        f"- Peak eval VRAM: `{as_float(baseline, 'gpu_peak_allocated_mb_eval'):.2f} MB allocated`, `{as_float(baseline, 'gpu_peak_reserved_mb_eval'):.2f} MB reserved`",
        f"- Entity cache logical size: `{as_float(baseline, 'entity_cache_storage_mb'):.2f} MB`",
        f"- Target tensor logical size: `{as_float(baseline, target_cache_key):.2f} MB`",
        f"- Entity cache disk size: `{as_float(baseline, 'entity_cache_disk_mb'):.2f} MB`",
        "",
        "## Mode Comparison",
        "",
        "| Mode | H@1 delta | MRR delta | Speedup | Cache ratio | Target ratio | Peak eval VRAM | QPS | Transfer s | VDLU s | CMFU s | Rank flips |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]

    for row in rows:
        mode = row.get(target_mode_key, row.get("mode", "unknown"))
        lines.append(
            "| "
            f"{mode} | "
            f"{as_float(row, 'delta_hits@1_vs_fp32'):.6f} | "
            f"{as_float(row, 'delta_mrr_vs_fp32'):.6f} | "
            f"{as_float(row, 'speedup_vs_fp32', 1.0):.3f}x | "
            f"{pct(as_float(row, 'cache_size_ratio_vs_fp32', 1.0))} | "
            f"{pct(as_float(row, f'{target_tensor}_size_ratio_vs_fp32', 1.0))} | "
            f"{as_float(row, 'gpu_peak_allocated_mb_eval'):.2f} | "
            f"{as_float(row, 'queries_per_second'):.3f} | "
            f"{as_float(row, 'entity_transfer_s'):.2f} | "
            f"{as_float(row, 'vdlu_s'):.2f} | "
            f"{as_float(row, 'cmfu_s'):.2f} | "
            f"{int(as_float(row, 'rank_flip_count'))} |"
        )

    lines.extend([
        "",
        "## Deployment Readout",
        "",
        "Use this table to decide whether the target tensor is a useful compression point:",
        "",
        "```text",
        "Accuracy: H@1/MRR deltas vs fp32",
        "Storage: logical cache ratio, target tensor ratio, disk MB",
        "Runtime: speedup, QPS, transfer/VDLU/CMFU movement",
        "Memory: peak eval VRAM allocated/reserved",
        "Stability: rank flips, rank improved/worsened counts",
        "```",
    ])

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
