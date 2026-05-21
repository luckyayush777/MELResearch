import argparse
import csv
import json
from pathlib import Path


def read_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, payload):
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def normalize_row(row):
    rename = {
        "entity_cache_entity_cache_disk_mb": "entity_cache_disk_mb",
        "entity_cache_entity_cache_shard_count": "entity_cache_shard_count",
        "entity_cache_entity_cache_build_peak_gpu_allocated_mb": "gpu_peak_allocated_mb_entity_cache_build",
        "entity_cache_entity_cache_build_peak_gpu_reserved_mb": "gpu_peak_reserved_mb_entity_cache_build",
        "entity_cache_entity_cache_load_peak_gpu_allocated_mb": "gpu_peak_allocated_mb_entity_cache_load",
        "entity_cache_entity_cache_load_peak_gpu_reserved_mb": "gpu_peak_reserved_mb_entity_cache_load",
    }
    for old, new in rename.items():
        if old in row and new not in row:
            row[new] = row[old]
    peak_candidates = [
        "gpu_peak_allocated_mb_eval",
        "gpu_peak_allocated_mb_mention_encoder",
        "gpu_peak_allocated_mb_entity_chunk_transfer",
        "gpu_peak_allocated_mb_matcher",
        "gpu_peak_allocated_mb_layernorm",
        "gpu_peak_allocated_mb_tglu",
        "gpu_peak_allocated_mb_vdlu",
        "gpu_peak_allocated_mb_cmfu",
    ]
    row["gpu_peak_allocated_mb_eval"] = max(float(row.get(key, 0.0) or 0.0) for key in peak_candidates)
    row["gpu_peak_allocated_mb_total_run"] = max(
        float(row.get("gpu_peak_allocated_mb_eval", 0.0) or 0.0),
        float(row.get("gpu_peak_allocated_mb_entity_cache_build", 0.0) or 0.0),
        float(row.get("gpu_peak_allocated_mb_entity_cache_load", 0.0) or 0.0),
    )
    row["gpu_peak_reserved_mb_total_run"] = max(
        float(row.get("gpu_peak_reserved_mb_eval", 0.0) or 0.0),
        float(row.get("gpu_peak_reserved_mb_entity_cache_build", 0.0) or 0.0),
        float(row.get("gpu_peak_reserved_mb_entity_cache_load", 0.0) or 0.0),
    )
    return row


def write_csv(path: Path, rows):
    fieldnames = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description="Normalize quantization result JSON/CSV keys after a run.")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--summary-json", required=True)
    parser.add_argument("--summary-csv", required=True)
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    out_dir = (script_dir / args.out_dir).resolve()
    summary_json = (script_dir / args.summary_json).resolve()
    summary_csv = (script_dir / args.summary_csv).resolve()

    rows = []
    for result_path in sorted(out_dir.glob("*/result.json")):
        row = normalize_row(read_json(result_path))
        write_json(result_path, row)
        rows.append(row)

    if summary_json.exists():
        details = read_json(summary_json)
    else:
        details = {}
    details["results"] = rows
    write_json(summary_json, details)
    write_csv(summary_csv, rows)
    print(f"Normalized {len(rows)} rows")


if __name__ == "__main__":
    main()
