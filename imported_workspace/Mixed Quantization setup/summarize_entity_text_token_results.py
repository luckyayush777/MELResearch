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
    parser = argparse.ArgumentParser(description="Create an attribution report for entity_text_tokens quantization.")
    parser.add_argument("--summary-csv", default="results/entity_text_tokens_full/entity_text_token_quantization_summary.csv")
    parser.add_argument("--out", default="results/entity_text_tokens_full/attribution_report.md")
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    summary_csv = (script_dir / args.summary_csv).resolve()
    out_path = (script_dir / args.out).resolve()

    with summary_csv.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))

    baseline = next((row for row in rows if row.get("entity_text_tokens_mode") == "fp32"), None)
    if baseline is None:
        raise RuntimeError("No fp32 baseline row found in the summary CSV.")

    lines = [
        "# Entity Text Token Quantization Attribution",
        "",
        "This report attributes gains from changing only `entity_text_tokens` while keeping the other cached tensors fixed.",
        "",
        "## Baseline",
        "",
        f"- H@1: `{as_float(baseline, 'hits@1'):.6f}`",
        f"- MRR: `{as_float(baseline, 'mrr'):.6f}`",
        f"- Evaluation seconds: `{as_float(baseline, 'seconds'):.2f}`",
        f"- Entity cache size: `{as_float(baseline, 'entity_cache_storage_mb'):.2f} MB`",
        f"- Entity text-token size: `{as_float(baseline, 'entity_cache_text_tokens_mb'):.2f} MB`",
        "",
        "## Mode Comparison",
        "",
        "| Mode | H@1 delta | MRR delta | Speedup | Cache ratio | Text-token ratio | Main time movement |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]

    base_transfer = as_float(baseline, "entity_transfer_s")
    base_tglu = as_float(baseline, "tglu_s")
    base_total = as_float(baseline, "seconds")

    for row in rows:
        mode = row.get("entity_text_tokens_mode", row.get("mode", "unknown"))
        transfer_delta = as_float(row, "entity_transfer_s") - base_transfer
        tglu_delta = as_float(row, "tglu_s") - base_tglu
        total_delta = as_float(row, "seconds") - base_total
        movement = (
            f"transfer {transfer_delta:.2f}s, "
            f"TGLU {tglu_delta:.2f}s, "
            f"total {total_delta:.2f}s"
        )
        lines.append(
            "| "
            f"{mode} | "
            f"{as_float(row, 'delta_hits@1_vs_fp32'):.6f} | "
            f"{as_float(row, 'delta_mrr_vs_fp32'):.6f} | "
            f"{as_float(row, 'speedup_vs_fp32', 1.0):.3f}x | "
            f"{pct(as_float(row, 'cache_size_ratio_vs_fp32', 1.0))} | "
            f"{pct(as_float(row, 'text_token_size_ratio_vs_fp32', 1.0))} | "
            f"{movement} |"
        )

    lines.extend([
        "",
        "## Attribution Checklist",
        "",
        "Use the strongest non-fp32 row that preserves H@1/MRR.",
        "",
        "Claim structure:",
        "",
        "```text",
        "Only entity_text_tokens were reduced; text_cls, image_cls, and image_patch_tokens stayed fixed.",
        "The entity_text_tokens cache shrank by X%, reducing the total entity cache by Y%.",
        "The entity_transfer_s and/or TGLU subprocess changed by Z seconds.",
        "Mention encoder and ranking time did not explain the gain.",
        "Accuracy changed by delta H@1 = A and delta MRR = B.",
        "```",
    ])

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
