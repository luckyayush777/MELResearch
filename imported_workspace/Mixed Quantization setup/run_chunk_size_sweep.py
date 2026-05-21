import argparse
import csv
import json
import sys
from pathlib import Path

import torch
from omegaconf import OmegaConf

import evaluate_entity_text_tokens as evaluator


TENSOR_NAMES = evaluator.TENSOR_NAMES


def read_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def write_csv(path: Path, rows):
    if not rows:
        return
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


def load_existing_cache(cache_dir: Path, tensor_modes, device: torch.device):
    manifest_path = cache_dir / "manifest.json"
    manifest = read_json(manifest_path)
    parts = [[], [], [], []]
    for rel_path in manifest["shards"]:
        shard = torch.load(str(cache_dir / rel_path), map_location="cpu")
        for idx, tensor in enumerate(shard["tensors"]):
            parts[idx].append(tensor)
    return tuple(evaluator.concat_cache_parts(piece, tensor_modes[idx]) for idx, piece in enumerate(parts))


def row_modes(mode_name, tensor_modes):
    row = {"sweep_mode": mode_name}
    for name, mode in zip(TENSOR_NAMES, tensor_modes):
        row[f"{name}_mode"] = mode
    row["entity_text_tokens_mode"] = row["text_tokens_mode"]
    return row


def add_deltas(rows):
    baselines = {int(row["chunk_size"]): row for row in rows if row.get("sweep_mode") == "fp32_ref" and row.get("run_success") is True}
    for row in rows:
        base = baselines.get(int(row.get("chunk_size", 0)))
        if not base:
            continue
        for key in ("hits@1", "hits@3", "hits@10", "hits@20", "mrr", "mr", "seconds", "queries_per_second", "gpu_peak_allocated_mb_eval"):
            if key in row and key in base:
                row[f"delta_{key}_vs_fp32_same_chunk"] = float(row[key]) - float(base[key])
        if float(row.get("seconds", 0.0)) > 0:
            row["speedup_vs_fp32_same_chunk"] = float(base["seconds"]) / float(row["seconds"])
    return rows


def main():
    parser = argparse.ArgumentParser(description="Run chunk-size sweep using existing quantized entity caches.")
    parser.add_argument("--repo-root", default="..")
    parser.add_argument("--mimic-root", default="../ExpSetup/external/MIMIC_reproduction")
    parser.add_argument("--config", default="../ExpSetup/external/MIMIC_reproduction/config/wikimel_noleak.yaml")
    parser.add_argument("--checkpoint", default="../ExpSetup/external/MIMIC_reproduction/runs/WikiMEL_noleak/version_4/checkpoints/final.ckpt")
    parser.add_argument("--split", choices=["dev", "test"], default="test")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--chunk-sizes", nargs="+", type=int, default=[1000, 2500, 5000, 10000, 20000, 40000])
    parser.add_argument("--out-dir", default="results/chunk_size_sweep")
    parser.add_argument("--limit-queries", type=int, default=0)
    parser.add_argument("--modes", nargs="+", default=["fp32_ref", "text_tokens_int4", "text_image_tokens_int4"])
    parser.add_argument(
        "--matcher-tile-size",
        type=int,
        default=0,
        help="Optional entity subtile size for TGLU inside each chunk. Use to test fixed-chunk peak VRAM reduction.",
    )
    parser.add_argument(
        "--entity-subtile-size",
        type=int,
        default=0,
        help="Optional entity subtile size for transfer/dequantization and full matcher scoring inside each chunk.",
    )
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    repo_root = (script_dir / args.repo_root).resolve()
    mimic_root = (script_dir / args.mimic_root).resolve()
    evaluator.add_repo_paths(repo_root, mimic_root)

    from codes.utils.dataset import DataModuleForMIMIC

    config_path = (script_dir / args.config).resolve()
    checkpoint_path = (script_dir / args.checkpoint).resolve()
    out_dir = (script_dir / args.out_dir).resolve()
    device = torch.device(args.device)
    env_meta = evaluator.environment_metadata(device)

    cfg = evaluator.resolve_config_paths(OmegaConf.load(config_path), mimic_root)
    data_module = DataModuleForMIMIC(cfg)
    model = evaluator.load_model(cfg, checkpoint_path, device)
    eval_loader = data_module.test_dataloader() if args.split == "test" else data_module.val_dataloader()

    mode_specs = {
        "fp32_ref": {
            "cache_dir": script_dir / "results/text_image_tokens_int4_metrics/fp32_ref/entity_cache",
            "tensor_modes": ("fp32", "fp32", "fp32", "fp32"),
        },
        "text_tokens_int4": {
            "cache_dir": script_dir / "results/entity_text_tokens_full/entity_text_tokens_int4__other_fp32/entity_cache",
            "tensor_modes": ("fp32", "fp32", "int4", "fp32"),
        },
        "text_image_tokens_int4": {
            "cache_dir": script_dir / "results/text_image_tokens_int4_metrics/text_image_tokens_int4/entity_cache",
            "tensor_modes": ("fp32", "fp32", "int4", "int4"),
        },
    }

    rows = []
    details = {
        **env_meta,
        "checkpoint": str(checkpoint_path),
        "config": str(config_path),
        "split": args.split,
        "chunk_sizes": args.chunk_sizes,
        "modes": args.modes,
        "matcher_tile_size": int(args.matcher_tile_size or 0),
        "entity_subtile_size": int(args.entity_subtile_size or 0),
    }

    for mode_name in args.modes:
        spec = mode_specs[mode_name]
        cache_dir = spec["cache_dir"].resolve()
        tensor_modes = spec["tensor_modes"]
        if not cache_dir.exists():
            raise RuntimeError(f"Missing cache for {mode_name}: {cache_dir}")
        print(f"Loading cache for {mode_name}: {cache_dir}")
        entity_cache = load_existing_cache(cache_dir, tensor_modes, device)
        manifest = read_json(cache_dir / "manifest.json")
        disk = evaluator.disk_usage_mb(cache_dir)

        for chunk_size in args.chunk_sizes:
            label = f"{mode_name}_chunk_{chunk_size}"
            if args.entity_subtile_size and args.entity_subtile_size > 0:
                label = f"{label}_entity_tile_{args.entity_subtile_size}"
            if args.matcher_tile_size and args.matcher_tile_size > 0:
                label = f"{label}_tglu_tile_{args.matcher_tile_size}"
            mode_dir = out_dir / label
            result_path = mode_dir / "result.json"
            if result_path.exists():
                row = read_json(result_path)
                rows.append(row)
                print(f"Reusing {result_path}")
                continue
            try:
                metrics = evaluator.evaluate_split_with_metrics(
                    model=model,
                    dataloader=eval_loader,
                    entity_cache=entity_cache,
                    device=device,
                    label=label,
                    mode_dir=mode_dir,
                    chunk_size=chunk_size,
                    limit_queries=args.limit_queries,
                    matcher_tile_size=args.matcher_tile_size,
                    entity_subtile_size=args.entity_subtile_size,
                )
                row = {
                    **env_meta,
                    **metrics,
                    **row_modes(mode_name, tensor_modes),
                    "run_success": True,
                    "failed_stage": "",
                    "exception_type": "",
                    "exception_message": "",
                    "oom_boolean": False,
                    "checkpoint_path": str(checkpoint_path),
                    "config_path": str(config_path),
                    "split": args.split,
                    "num_entities": int(manifest.get("entities", evaluator.cache_tensor_length(entity_cache[0]))),
                    "chunk_size": chunk_size,
                    "eval_batch_size": int(getattr(cfg.data, "eval_batch_size", 0) or 0),
                    "entity_cache_storage_mb": sum(int(v) for v in manifest.get("tensor_storage_bytes", {}).values()) / evaluator.BYTES_PER_MB,
                    "entity_cache_disk_mb": disk["entity_cache_disk_mb"],
                    "entity_cache_shard_count": disk["entity_cache_shard_count"],
                }
            except Exception as exc:
                row = {
                    **env_meta,
                    **row_modes(mode_name, tensor_modes),
                    "mode": label,
                    "sweep_mode": mode_name,
                    "run_success": False,
                    "failed_stage": "evaluate",
                    "exception_type": type(exc).__name__,
                    "exception_message": str(exc),
                    "oom_boolean": "out of memory" in str(exc).lower(),
                    "checkpoint_path": str(checkpoint_path),
                    "config_path": str(config_path),
                    "split": args.split,
                    "chunk_size": chunk_size,
                    "matcher_tile_size": int(args.matcher_tile_size or 0),
                    "entity_subtile_size": int(args.entity_subtile_size or 0),
                }
            rows.append(row)
            rows = add_deltas(rows)
            write_json(result_path, row)
            details["results"] = rows
            write_json(out_dir / "chunk_size_sweep_results.json", details)
            write_csv(out_dir / "chunk_size_sweep_summary.csv", rows)
            print(json.dumps(row, indent=2))

    rows = add_deltas(rows)
    details["results"] = rows
    write_json(out_dir / "chunk_size_sweep_results.json", details)
    write_csv(out_dir / "chunk_size_sweep_summary.csv", rows)
    print(f"Saved chunk sweep to {out_dir}")


if __name__ == "__main__":
    sys.exit(main())
