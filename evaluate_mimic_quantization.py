import argparse
import csv
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from omegaconf import OmegaConf
from tqdm import tqdm


def add_mimic_to_path(mimic_root: Path) -> None:
    mimic_root = mimic_root.resolve()
    if str(mimic_root) not in sys.path:
        sys.path.insert(0, str(mimic_root))


def resolve_config_paths(args, mimic_root: Path):
    for key in (
        "kb_img_folder",
        "mention_img_folder",
        "qid2id",
        "entity",
        "train_file",
        "dev_file",
        "test_file",
    ):
        value = getattr(args.data, key)
        if isinstance(value, str) and value and not os.path.isabs(value):
            setattr(args.data, key, str(mimic_root / value))
    return args


def tensor_storage_bytes(tensor: torch.Tensor, mode: str) -> int:
    elements = int(tensor.numel())
    if mode == "fp32":
        return elements * 4
    if mode == "fp16":
        return elements * 2
    if mode == "int8":
        vectors = int(np.prod(tensor.shape[:-1])) if tensor.ndim > 1 else int(tensor.shape[0])
        return elements + vectors * 4
    raise ValueError(f"Unsupported mode: {mode}")


def quantize_tensor(tensor: torch.Tensor, mode: str) -> torch.Tensor:
    if mode == "fp32":
        return tensor.float()
    if mode == "fp16":
        return tensor.half().float()
    if mode == "int8":
        x = tensor.float()
        max_abs = x.abs().amax(dim=-1, keepdim=True)
        scales = torch.clamp(max_abs / 127.0, min=1e-12)
        q = torch.clamp(torch.round(x / scales), -127, 127).to(torch.int8)
        return q.float() * scales
    raise ValueError(f"Unsupported mode: {mode}")


def quantize_pack(tensors: Tuple[torch.Tensor, ...], mode: str) -> Tuple[Tuple[torch.Tensor, ...], int]:
    storage = sum(tensor_storage_bytes(t, mode) for t in tensors)
    return tuple(quantize_tensor(t, mode) for t in tensors), storage


def write_json(path: Path, payload: Dict[str, object]) -> None:
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    tmp_path.replace(path)


def read_json(path: Path) -> Dict[str, object]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def retrieval_metrics(ranks: np.ndarray) -> Dict[str, float]:
    ranks = ranks.astype(np.int64)
    return {
        "queries": int(ranks.shape[0]),
        "hits@1": float(np.mean(ranks <= 1)),
        "hits@3": float(np.mean(ranks <= 3)),
        "hits@5": float(np.mean(ranks <= 5)),
        "hits@10": float(np.mean(ranks <= 10)),
        "hits@20": float(np.mean(ranks <= 20)),
        "mr": float(np.mean(ranks)),
        "mrr": float(np.mean(1.0 / ranks)),
    }


def build_entity_cache(model, dataloader, device: torch.device, mode: str, limit_entities: int = 0):
    outputs: List[List[torch.Tensor]] = [[], [], [], []]
    seen = 0
    start = time.perf_counter()
    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"Entity cache ({mode})", total=len(dataloader)):
            if limit_entities and seen >= limit_entities:
                break
            if limit_entities:
                remaining = limit_entities - seen
                batch = {k: v[:remaining] for k, v in batch.items()}
            batch = {k: v.to(device) for k, v in batch.items()}
            encoded = model.encoder(**batch)
            quantized, _ = quantize_pack(tuple(t.detach().cpu() for t in encoded), mode)
            for idx, tensor in enumerate(quantized):
                outputs[idx].append(tensor)
            seen += int(encoded[0].shape[0])

    cache = tuple(torch.cat(parts, dim=0) for parts in outputs)
    storage_bytes = sum(tensor_storage_bytes(tensor, mode) for tensor in cache)
    return cache, {
        "entities": int(cache[0].shape[0]),
        "storage_mb": storage_bytes / (1024.0 * 1024.0),
        "seconds": time.perf_counter() - start,
    }


def build_or_load_entity_cache(
    model,
    dataloader,
    device: torch.device,
    mode: str,
    mode_dir: Path,
    limit_entities: int = 0,
):
    entity_dir = mode_dir / "entity_cache"
    shard_dir = entity_dir / "shards"
    manifest_path = entity_dir / "manifest.json"
    shard_dir.mkdir(parents=True, exist_ok=True)

    start = time.perf_counter()
    manifest = read_json(manifest_path) if manifest_path.exists() else {
        "mode": mode,
        "limit_entities": limit_entities,
        "complete": False,
        "shards": [],
        "entities": 0,
    }

    if manifest.get("complete") and int(manifest.get("limit_entities", 0)) == int(limit_entities):
        print(f"Reusing complete entity cache for {mode}: {manifest_path}")
    else:
        seen = int(manifest.get("entities", 0))
        start_batch = len(manifest.get("shards", []))
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(dataloader, desc=f"Entity cache ({mode})", total=len(dataloader))):
                if batch_idx < start_batch:
                    continue
                if limit_entities and seen >= limit_entities:
                    break
                if limit_entities:
                    remaining = limit_entities - seen
                    batch = {k: v[:remaining] for k, v in batch.items()}
                batch = {k: v.to(device) for k, v in batch.items()}
                encoded = model.encoder(**batch)
                quantized, _ = quantize_pack(tuple(t.detach().cpu() for t in encoded), mode)
                shard_path = shard_dir / f"shard_{batch_idx:05d}.pt"
                torch.save({"batch_idx": batch_idx, "tensors": quantized}, shard_path)
                seen += int(quantized[0].shape[0])
                manifest["shards"].append(str(shard_path.relative_to(entity_dir)))
                manifest["entities"] = seen
                manifest["complete"] = False
                write_json(manifest_path, manifest)

        manifest["complete"] = True
        write_json(manifest_path, manifest)

    parts: List[List[torch.Tensor]] = [[], [], [], []]
    for rel_path in tqdm(manifest["shards"], desc=f"Load entity cache ({mode})"):
        shard = torch.load(str(entity_dir / rel_path), map_location="cpu")
        for idx, tensor in enumerate(shard["tensors"]):
            parts[idx].append(tensor)
    cache = tuple(torch.cat(piece, dim=0) for piece in parts)
    storage_bytes = sum(tensor_storage_bytes(tensor, mode) for tensor in cache)
    meta = {
        "entities": int(cache[0].shape[0]),
        "storage_mb": storage_bytes / (1024.0 * 1024.0),
        "seconds": time.perf_counter() - start,
        "manifest": str(manifest_path),
        "shards": len(manifest["shards"]),
    }
    return cache, meta


def evaluate_split(model, dataloader, entity_cache, device: torch.device, mode: str, chunk_size: int, limit_queries: int = 0):
    ranks = []
    skipped = 0
    seen_queries = 0
    entity_count = int(entity_cache[0].shape[0])
    start = time.perf_counter()
    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"Evaluate ({mode})", total=len(dataloader)):
            if limit_queries and seen_queries >= limit_queries:
                break
            if limit_queries:
                remaining = limit_queries - seen_queries
                batch = {k: v[:remaining] for k, v in batch.items()}
            seen_queries += int(batch["answer"].shape[0])
            answer = batch.pop("answer")
            keep = answer < entity_count
            if not bool(keep.any()):
                skipped += int(answer.numel())
                continue
            skipped += int((~keep).sum().item())
            answer = answer[keep].to(device)
            batch = {k: v[keep].to(device) for k, v in batch.items()}

            mention_pack, mention_storage = quantize_pack(tuple(t.detach().cpu() for t in model.encoder(**batch)), mode)
            mention_pack = tuple(t.to(device) for t in mention_pack)

            scores = []
            for start_pos in range(0, entity_count, chunk_size):
                end_pos = min(start_pos + chunk_size, entity_count)
                ent_chunk = tuple(t[start_pos:end_pos].to(device) for t in entity_cache)
                chunk_score, _ = model.matcher(
                    ent_chunk[0],
                    ent_chunk[2],
                    mention_pack[0],
                    mention_pack[2],
                    ent_chunk[1],
                    ent_chunk[3],
                    mention_pack[1],
                    mention_pack[3],
                )
                scores.append(chunk_score)

            scores_tensor = torch.cat(scores, dim=-1)
            rank = torch.argsort(torch.argsort(scores_tensor, dim=-1, descending=True), dim=-1) + 1
            ranks.append(rank[torch.arange(answer.shape[0], device=device), answer].cpu().numpy())

    if not ranks:
        raise RuntimeError("No queries were evaluated. Increase --entity-limit or remove it.")
    ranks_np = np.concatenate(ranks)
    metrics = retrieval_metrics(ranks_np)
    metrics["skipped_queries"] = int(skipped)
    metrics["seconds"] = time.perf_counter() - start
    metrics["mode"] = mode
    return metrics


def evaluate_split_checkpointed(
    model,
    dataloader,
    entity_cache,
    device: torch.device,
    mode: str,
    mode_dir: Path,
    chunk_size: int,
    limit_queries: int = 0,
):
    ranks_dir = mode_dir / "rank_batches"
    ranks_dir.mkdir(parents=True, exist_ok=True)
    entity_count = int(entity_cache[0].shape[0])
    skipped = 0
    seen_queries = 0
    start = time.perf_counter()

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc=f"Evaluate ({mode})", total=len(dataloader))):
            rank_path = ranks_dir / f"batch_{batch_idx:05d}.npy"
            skip_path = ranks_dir / f"batch_{batch_idx:05d}.json"
            if rank_path.exists() and skip_path.exists():
                batch_meta = read_json(skip_path)
                seen_queries += int(batch_meta.get("seen_queries", 0))
                skipped += int(batch_meta.get("skipped_queries", 0))
                continue
            if limit_queries and seen_queries >= limit_queries:
                break
            if limit_queries:
                remaining = limit_queries - seen_queries
                batch = {k: v[:remaining] for k, v in batch.items()}
            current_seen = int(batch["answer"].shape[0])
            seen_queries += current_seen
            answer = batch.pop("answer")
            keep = answer < entity_count
            current_skipped = int((~keep).sum().item())
            skipped += current_skipped
            if not bool(keep.any()):
                np.save(rank_path, np.array([], dtype=np.int64))
                write_json(skip_path, {"seen_queries": current_seen, "skipped_queries": current_skipped})
                continue

            answer = answer[keep].to(device)
            batch = {k: v[keep].to(device) for k, v in batch.items()}
            mention_pack, _ = quantize_pack(tuple(t.detach().cpu() for t in model.encoder(**batch)), mode)
            mention_pack = tuple(t.to(device) for t in mention_pack)

            scores = []
            for start_pos in range(0, entity_count, chunk_size):
                end_pos = min(start_pos + chunk_size, entity_count)
                ent_chunk = tuple(t[start_pos:end_pos].to(device) for t in entity_cache)
                chunk_score, _ = model.matcher(
                    ent_chunk[0],
                    ent_chunk[2],
                    mention_pack[0],
                    mention_pack[2],
                    ent_chunk[1],
                    ent_chunk[3],
                    mention_pack[1],
                    mention_pack[3],
                )
                scores.append(chunk_score)

            scores_tensor = torch.cat(scores, dim=-1)
            rank = torch.argsort(torch.argsort(scores_tensor, dim=-1, descending=True), dim=-1) + 1
            batch_ranks = rank[torch.arange(answer.shape[0], device=device), answer].cpu().numpy()
            np.save(rank_path, batch_ranks)
            write_json(skip_path, {"seen_queries": current_seen, "skipped_queries": current_skipped})

    rank_parts = [np.load(path) for path in sorted(ranks_dir.glob("batch_*.npy"))]
    ranks_np = np.concatenate([part for part in rank_parts if part.size]) if rank_parts else np.array([], dtype=np.int64)
    if ranks_np.size == 0:
        raise RuntimeError("No queries were evaluated. Increase --entity-limit or remove it.")
    metrics = retrieval_metrics(ranks_np)
    metrics["skipped_queries"] = int(skipped)
    metrics["seconds"] = time.perf_counter() - start
    metrics["mode"] = mode
    metrics["checkpoint_dir"] = str(mode_dir)
    return metrics


def load_model(args, checkpoint: Path, device: torch.device):
    from codes.model.lightning_mimic import LightningForMIMIC

    model = LightningForMIMIC(args)
    ckpt = torch.load(str(checkpoint), map_location="cpu", weights_only=False)
    state_dict = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()
    return model


def write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate representation quantization on a trained MIMIC WikiMEL checkpoint.")
    parser.add_argument("--mimic-root", default="external/MIMIC_reproduction")
    parser.add_argument("--config", default="external/MIMIC_reproduction/config/wikimel.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", choices=["dev", "test"], default="test")
    parser.add_argument("--modes", nargs="+", default=["fp32", "fp16", "int8"])
    parser.add_argument("--out-dir", default="benchmarks/mimic_wikimel_quantization")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--chunk-size", type=int, default=20000)
    parser.add_argument("--limit-queries", type=int, default=0)
    parser.add_argument("--entity-limit", type=int, default=0)
    parser.add_argument("--force", action="store_true", help="Recompute completed mode results.")
    args_cli = parser.parse_args()

    mimic_root = Path(args_cli.mimic_root)
    add_mimic_to_path(mimic_root)

    from codes.utils.dataset import DataModuleForMIMIC

    cfg = resolve_config_paths(OmegaConf.load(args_cli.config), mimic_root)
    cfg.data.eval_chunk_size = args_cli.chunk_size
    device = torch.device(args_cli.device)
    out_dir = Path(args_cli.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    data_module = DataModuleForMIMIC(cfg)
    model = load_model(cfg, Path(args_cli.checkpoint), device)
    eval_loader = data_module.test_dataloader() if args_cli.split == "test" else data_module.val_dataloader()

    rows = []
    details = {
        "checkpoint": str(Path(args_cli.checkpoint).resolve()),
        "config": str(Path(args_cli.config).resolve()),
        "split": args_cli.split,
        "device": str(device),
        "chunk_size": args_cli.chunk_size,
        "limit_queries": args_cli.limit_queries,
        "entity_limit": args_cli.entity_limit,
        "modes": args_cli.modes,
    }

    for mode in args_cli.modes:
        mode_dir = out_dir / mode
        mode_dir.mkdir(parents=True, exist_ok=True)
        mode_result_path = mode_dir / "result.json"
        if mode_result_path.exists() and not args_cli.force:
            row = read_json(mode_result_path)
            rows.append(row)
            print(f"Reusing completed result for {mode}: {mode_result_path}")
            continue

        entity_cache, cache_meta = build_or_load_entity_cache(
            model=model,
            dataloader=data_module.entity_dataloader(),
            device=device,
            mode=mode,
            mode_dir=mode_dir,
            limit_entities=args_cli.entity_limit,
        )
        metrics = evaluate_split_checkpointed(
            model=model,
            dataloader=eval_loader,
            entity_cache=entity_cache,
            device=device,
            mode=mode,
            mode_dir=mode_dir,
            chunk_size=args_cli.chunk_size,
            limit_queries=args_cli.limit_queries,
        )
        row = {**metrics, **{f"entity_cache_{k}": v for k, v in cache_meta.items()}}
        rows.append(row)
        write_json(mode_result_path, row)
        details["results"] = rows
        write_json(out_dir / "mimic_quantization_results.json", details)
        write_csv(out_dir / "mimic_quantization_summary.csv", rows)
        print(json.dumps(row, indent=2))

    details["results"] = rows
    write_json(out_dir / "mimic_quantization_results.json", details)
    write_csv(out_dir / "mimic_quantization_summary.csv", rows)
    print(f"Saved results to {out_dir}")


if __name__ == "__main__":
    main()
