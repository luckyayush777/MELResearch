import argparse
import csv
import datetime as dt
import json
import math
import os
import platform
import sys
import time
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
from omegaconf import OmegaConf
from tqdm import tqdm


TENSOR_NAMES = ("text_cls", "image_cls", "text_tokens", "image_patch_tokens")
CACHE_FORMAT_VERSION = 2
BYTES_PER_MB = 1024.0 * 1024.0
QUANT_ERROR_SAMPLE_VECTORS = 2048
LEGACY_TARGET_ALIASES = {
    "entity_text_tokens": "text_tokens",
    "entity_image_patch_tokens": "image_patch_tokens",
}


def add_repo_paths(repo_root: Path, mimic_root: Path) -> None:
    for path in (repo_root.resolve(), mimic_root.resolve()):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))


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


def parse_mode_bits(mode: str) -> int:
    if mode == "fp32":
        return 32
    if mode == "fp16":
        return 16
    if mode.startswith("int"):
        bits = int(mode[3:])
        if bits < 2 or bits > 8:
            raise ValueError(f"Only int2 through int8 are supported, got {mode}")
        return bits
    raise ValueError(f"Unsupported quantization mode: {mode}")


def tensor_storage_bytes(tensor: torch.Tensor, mode: str) -> int:
    elements = int(tensor.numel())
    bits = parse_mode_bits(mode)
    if bits == 32:
        return elements * 4
    if bits == 16:
        return elements * 2
    vectors = int(np.prod(tensor.shape[:-1])) if tensor.ndim > 1 else int(tensor.shape[0])
    packed_values = math.ceil(elements * bits / 8.0)
    scale_bytes = vectors * 4
    return packed_values + scale_bytes


def cpu_rss_mb() -> float:
    try:
        import psutil

        return float(psutil.Process(os.getpid()).memory_info().rss) / BYTES_PER_MB
    except Exception:
        return 0.0


def cuda_memory_snapshot(device: torch.device, prefix: str = "") -> Dict[str, float]:
    if device.type != "cuda" or not torch.cuda.is_available():
        return {}
    return {
        f"{prefix}gpu_allocated_mb": torch.cuda.memory_allocated(device) / BYTES_PER_MB,
        f"{prefix}gpu_reserved_mb": torch.cuda.memory_reserved(device) / BYTES_PER_MB,
        f"{prefix}gpu_peak_allocated_mb": torch.cuda.max_memory_allocated(device) / BYTES_PER_MB,
        f"{prefix}gpu_peak_reserved_mb": torch.cuda.max_memory_reserved(device) / BYTES_PER_MB,
    }


def reset_cuda_peak(device: torch.device) -> None:
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)


def disk_usage_mb(path: Path) -> Dict[str, float]:
    if not path.exists():
        return {
            "entity_cache_disk_mb": 0.0,
            "entity_cache_shard_count": 0,
            "avg_shard_disk_mb": 0.0,
            "max_shard_disk_mb": 0.0,
            "min_shard_disk_mb": 0.0,
        }
    shard_sizes = [p.stat().st_size / BYTES_PER_MB for p in sorted((path / "shards").glob("*.pt"))]
    total = sum(p.stat().st_size for p in path.rglob("*") if p.is_file()) / BYTES_PER_MB
    return {
        "entity_cache_disk_mb": total,
        "entity_cache_shard_count": len(shard_sizes),
        "avg_shard_disk_mb": float(np.mean(shard_sizes)) if shard_sizes else 0.0,
        "max_shard_disk_mb": float(np.max(shard_sizes)) if shard_sizes else 0.0,
        "min_shard_disk_mb": float(np.min(shard_sizes)) if shard_sizes else 0.0,
    }


def environment_metadata(device: torch.device) -> Dict[str, object]:
    meta: Dict[str, object] = {
        "timestamp": dt.datetime.now(dt.timezone.utc).isoformat(),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "pytorch_version": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_version": torch.version.cuda,
        "cache_format_version": CACHE_FORMAT_VERSION,
        "git_status_note": "git unavailable",
    }
    if device.type == "cuda" and torch.cuda.is_available():
        props = torch.cuda.get_device_properties(device)
        meta.update({
            "device_name": props.name,
            "gpu_total_memory_mb": props.total_memory / BYTES_PER_MB,
            "cuda_device_index": int(device.index or torch.cuda.current_device()),
        })
    else:
        meta.update({"device_name": str(device), "gpu_total_memory_mb": 0.0})
    return meta


def quantization_error_stats(tensor: torch.Tensor, mode: str) -> Dict[str, float]:
    if not mode.startswith("int"):
        return {}
    bits = parse_mode_bits(mode)
    qmax = (2 ** (bits - 1)) - 1
    x = tensor.detach().float().cpu()
    vector_shape = x.shape[:-1] if x.ndim > 1 else (x.shape[0],)
    vectors = int(np.prod(vector_shape))
    flat = x.reshape(vectors, -1)
    sample = flat[: min(vectors, QUANT_ERROR_SAMPLE_VECTORS)]
    max_abs = sample.abs().amax(dim=-1, keepdim=True)
    scales = torch.clamp(max_abs / float(qmax), min=1e-12)
    q = torch.clamp(torch.round(sample / scales), -qmax, qmax).to(torch.int8)
    deq = q.float() * scales
    error = deq - sample
    sample_norm = torch.linalg.vector_norm(sample)
    return {
        "sample_vectors": int(sample.shape[0]),
        "mean_abs_error": float(error.abs().mean().item()),
        "max_abs_error": float(error.abs().max().item()),
        "relative_l2_error": float((torch.linalg.vector_norm(error) / torch.clamp(sample_norm, min=1e-12)).item()),
        "cosine_similarity_mean": float(torch.nn.functional.cosine_similarity(sample, deq, dim=-1).mean().item()),
        "cosine_similarity_std": float(torch.nn.functional.cosine_similarity(sample, deq, dim=-1).std(unbiased=False).item()),
        "sign_flip_rate": float(((sample.sign() != deq.sign()) & (sample != 0)).float().mean().item()),
        "zero_rate": float((q == 0).float().mean().item()),
        "saturation_rate": float((q.abs() == qmax).float().mean().item()),
        "scale_mean": float(scales.mean().item()),
        "scale_std": float(scales.std(unbiased=False).item()),
        "scale_min": float(scales.min().item()),
        "scale_max": float(scales.max().item()),
    }


def update_quant_error_aggregate(aggregate: Dict[str, Dict[str, float]], name: str, stats: Dict[str, float]) -> None:
    if not stats:
        return
    current = aggregate.setdefault(name, {"sample_vectors": 0})
    old_n = float(current.get("sample_vectors", 0))
    new_n = float(stats.get("sample_vectors", 0))
    total_n = old_n + new_n
    if total_n <= 0:
        return
    weighted_keys = (
        "mean_abs_error",
        "relative_l2_error",
        "cosine_similarity_mean",
        "cosine_similarity_std",
        "sign_flip_rate",
        "zero_rate",
        "saturation_rate",
        "scale_mean",
        "scale_std",
    )
    for key in weighted_keys:
        current[key] = ((float(current.get(key, 0.0)) * old_n) + (float(stats.get(key, 0.0)) * new_n)) / total_n
    for key in ("max_abs_error", "scale_max"):
        current[key] = max(float(current.get(key, 0.0)), float(stats.get(key, 0.0)))
    if "scale_min" in stats:
        current["scale_min"] = float(stats["scale_min"]) if old_n == 0 else min(float(current.get("scale_min", stats["scale_min"])), float(stats["scale_min"]))
    current["sample_vectors"] = int(total_n)


def quantize_tensor(tensor: torch.Tensor, mode: str) -> torch.Tensor:
    if mode == "fp32":
        return tensor.float()
    if mode == "fp16":
        return tensor.half().float()

    bits = parse_mode_bits(mode)
    qmax = (2 ** (bits - 1)) - 1
    x = tensor.float()
    max_abs = x.abs().amax(dim=-1, keepdim=True)
    scales = torch.clamp(max_abs / float(qmax), min=1e-12)
    q = torch.clamp(torch.round(x / scales), -qmax, qmax).to(torch.int8)
    return q.float() * scales


def quantize_tensor_packed(tensor: torch.Tensor, mode: str) -> Dict[str, object]:
    bits = parse_mode_bits(mode)
    if bits in (16, 32):
        dtype_tensor = tensor.half() if bits == 16 else tensor.float()
        return {"format": "plain", "mode": mode, "tensor": dtype_tensor.cpu()}

    qmax = (2 ** (bits - 1)) - 1
    x = tensor.float().cpu()
    max_abs = x.abs().amax(dim=-1, keepdim=True)
    scales = torch.clamp(max_abs / float(qmax), min=1e-12).half()
    q = torch.clamp(torch.round(x / scales.float()), -qmax, qmax).to(torch.int8)
    return {
        "format": "symmetric_per_vector",
        "mode": mode,
        "q": q,
        "scale": scales,
        "shape": list(tensor.shape),
    }


def dequantize_packed_tensor(packed: Dict[str, object], device: torch.device, start: int = None, end: int = None) -> torch.Tensor:
    if packed["format"] == "plain":
        tensor = packed["tensor"]
        if start is not None:
            tensor = tensor[start:end]
        return tensor.to(device).float()

    q = packed["q"]
    scale = packed["scale"]
    if start is not None:
        q = q[start:end]
        scale = scale[start:end]
    return q.to(device).float() * scale.to(device).float()


def cache_tensor_length(tensor_or_packed) -> int:
    if isinstance(tensor_or_packed, dict):
        if tensor_or_packed["format"] == "plain":
            return int(tensor_or_packed["tensor"].shape[0])
        return int(tensor_or_packed["q"].shape[0])
    return int(tensor_or_packed.shape[0])


def cache_tensor_to_device(tensor_or_packed, device: torch.device, start: int, end: int) -> torch.Tensor:
    if isinstance(tensor_or_packed, dict):
        return dequantize_packed_tensor(tensor_or_packed, device, start, end)
    return tensor_or_packed[start:end].to(device)


def concat_cache_parts(parts: List[object], mode: str):
    if not parts:
        raise RuntimeError("No cache parts were loaded.")
    first = parts[0]
    if not isinstance(first, dict):
        return torch.cat(parts, dim=0)
    if first["format"] == "plain":
        return {
            "format": "plain",
            "mode": mode,
            "tensor": torch.cat([part["tensor"] for part in parts], dim=0),
        }
    return {
        "format": "symmetric_per_vector",
        "mode": mode,
        "q": torch.cat([part["q"] for part in parts], dim=0),
        "scale": torch.cat([part["scale"] for part in parts], dim=0),
        "shape": [sum(int(part["q"].shape[0]) for part in parts), *first["shape"][1:]],
    }


def quantize_pack(tensors: Sequence[torch.Tensor], modes: Sequence[str]) -> Tuple[Tuple[torch.Tensor, ...], Dict[str, object]]:
    quantized = []
    tensor_storage = {}
    tensor_storage_mb = {}
    for name, tensor, mode in zip(TENSOR_NAMES, tensors, modes):
        storage = tensor_storage_bytes(tensor, mode)
        tensor_storage[name] = storage
        tensor_storage_mb[f"{name}_mb"] = storage / (1024.0 * 1024.0)
        quantized.append(quantize_tensor(tensor, mode))
    total_storage = int(sum(tensor_storage.values()))
    return tuple(quantized), {
        "storage_bytes": total_storage,
        "storage_mb": total_storage / (1024.0 * 1024.0),
        "tensor_storage_bytes": tensor_storage,
        "tensor_storage_mb": tensor_storage_mb,
    }


def pack_entity_cache_tensors(tensors: Sequence[torch.Tensor], modes: Sequence[str]) -> Tuple[Tuple[object, ...], Dict[str, object]]:
    packed = []
    tensor_storage = {}
    tensor_storage_mb = {}
    quant_error = {}
    for idx, (name, tensor, mode) in enumerate(zip(TENSOR_NAMES, tensors, modes)):
        storage = tensor_storage_bytes(tensor, mode)
        tensor_storage[name] = storage
        tensor_storage_mb[f"{name}_mb"] = storage / (1024.0 * 1024.0)
        if mode.startswith("int"):
            quant_error[name] = quantization_error_stats(tensor, mode)
            packed.append(quantize_tensor_packed(tensor, mode))
        else:
            packed.append(quantize_tensor(tensor, mode).cpu())
    total_storage = int(sum(tensor_storage.values()))
    return tuple(packed), {
        "storage_bytes": total_storage,
        "storage_mb": total_storage / (1024.0 * 1024.0),
        "tensor_storage_bytes": tensor_storage,
        "tensor_storage_mb": tensor_storage_mb,
        "quant_error": quant_error,
    }


def write_json(path: Path, payload: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    tmp_path.replace(path)


def read_json(path: Path) -> Dict[str, object]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


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


def load_model(args, checkpoint: Path, device: torch.device):
    from codes.model.lightning_mimic import LightningForMIMIC

    model = LightningForMIMIC(args)
    ckpt = torch.load(str(checkpoint), map_location="cpu", weights_only=False)
    state_dict = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()
    return model


def normalize_target_tensor(target_tensor: str) -> str:
    target_tensor = LEGACY_TARGET_ALIASES.get(target_tensor, target_tensor)
    if target_tensor not in TENSOR_NAMES:
        raise ValueError(f"Unsupported target tensor: {target_tensor}. Choose one of {', '.join(TENSOR_NAMES)}")
    return target_tensor


def setup_modes(target_tensor: str, target_mode: str, other_mode: str) -> Tuple[str, Tuple[str, str, str, str]]:
    modes = [other_mode, other_mode, other_mode, other_mode]
    target_tensor = normalize_target_tensor(target_tensor)
    modes[TENSOR_NAMES.index(target_tensor)] = target_mode
    label = f"entity_{target_tensor}_{target_mode}__other_{other_mode}"
    return label, tuple(modes)


def parse_mode_spec(spec: str) -> Tuple[str, Tuple[str, str, str, str]]:
    if ":" not in spec:
        raise ValueError(f"Mode spec must look like label:text_tokens=int4,image_patch_tokens=int4, got {spec}")
    label, body = spec.split(":", 1)
    modes = {name: "fp32" for name in TENSOR_NAMES}
    for item in body.split(","):
        if not item.strip():
            continue
        if "=" not in item:
            raise ValueError(f"Mode spec item must look like tensor=mode, got {item}")
        key, value = [part.strip() for part in item.split("=", 1)]
        key = normalize_target_tensor(key)
        parse_mode_bits(value)
        modes[key] = value
    return label.strip(), tuple(modes[name] for name in TENSOR_NAMES)


def build_or_load_entity_cache(
    model,
    dataloader,
    device: torch.device,
    label: str,
    tensor_modes: Sequence[str],
    mode_dir: Path,
    limit_entities: int,
):
    entity_dir = mode_dir / "entity_cache"
    shard_dir = entity_dir / "shards"
    manifest_path = entity_dir / "manifest.json"
    shard_dir.mkdir(parents=True, exist_ok=True)

    cpu_start = cpu_rss_mb()
    cpu_peak = cpu_start
    start = time.perf_counter()
    reset_cuda_peak(device)
    manifest = read_json(manifest_path) if manifest_path.exists() else {
        "cache_format_version": CACHE_FORMAT_VERSION,
        "label": label,
        "tensor_modes": dict(zip(TENSOR_NAMES, tensor_modes)),
        "limit_entities": limit_entities,
        "complete": False,
        "shards": [],
        "entities": 0,
        "tensor_storage_bytes": {name: 0 for name in TENSOR_NAMES},
        "quant_error": {},
    }

    expected_modes = dict(zip(TENSOR_NAMES, tensor_modes))
    if manifest.get("tensor_modes") != expected_modes:
        raise RuntimeError(f"Existing cache has different tensor modes: {manifest_path}")
    if int(manifest.get("cache_format_version", 1)) != CACHE_FORMAT_VERSION:
        raise RuntimeError(f"Existing cache uses an old format; remove it before resuming: {manifest_path}")

    if manifest.get("complete") and int(manifest.get("limit_entities", 0)) == int(limit_entities):
        print(f"Reusing complete entity cache for {label}: {manifest_path}")
    else:
        seen = int(manifest.get("entities", 0))
        start_batch = len(manifest.get("shards", []))
        encode_s = 0.0
        quantize_s = float(manifest.get("quantize_s", 0.0))
        save_s = float(manifest.get("save_s", 0.0))

        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(dataloader, desc=f"Entity cache ({label})", total=len(dataloader))):
                if batch_idx < start_batch:
                    continue
                if limit_entities and seen >= limit_entities:
                    break
                if limit_entities:
                    remaining = limit_entities - seen
                    batch = {k: v[:remaining] for k, v in batch.items()}

                batch = {k: v.to(device) for k, v in batch.items()}
                t0 = time.perf_counter()
                encoded = model.encoder(**batch)
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                encode_s += time.perf_counter() - t0

                t0 = time.perf_counter()
                quantized, storage_meta = pack_entity_cache_tensors(tuple(t.detach().cpu() for t in encoded), tensor_modes)
                quantize_s += time.perf_counter() - t0

                shard_path = shard_dir / f"shard_{batch_idx:05d}.pt"
                tmp_shard_path = shard_path.with_suffix(shard_path.suffix + ".tmp")
                t0 = time.perf_counter()
                torch.save({"batch_idx": batch_idx, "tensors": quantized}, tmp_shard_path)
                tmp_shard_path.replace(shard_path)
                save_s += time.perf_counter() - t0

                seen += int(quantized[0].shape[0])
                manifest["shards"].append(str(shard_path.relative_to(entity_dir)))
                manifest["entities"] = seen
                manifest["complete"] = False
                manifest["encode_s"] = encode_s
                manifest["quantize_s"] = quantize_s
                manifest["save_s"] = save_s
                for name, bytes_value in storage_meta["tensor_storage_bytes"].items():
                    manifest["tensor_storage_bytes"][name] = int(manifest["tensor_storage_bytes"].get(name, 0)) + int(bytes_value)
                for name, stats in storage_meta.get("quant_error", {}).items():
                    update_quant_error_aggregate(manifest.setdefault("quant_error", {}), name, stats)
                cpu_peak = max(cpu_peak, cpu_rss_mb())
                write_json(manifest_path, manifest)

        manifest["complete"] = True
        manifest["entity_cache_peak_cpu_rss_mb"] = cpu_peak
        manifest.update(cuda_memory_snapshot(device, "entity_cache_build_"))
        write_json(manifest_path, manifest)

    reset_cuda_peak(device)
    load_start = time.perf_counter()
    parts: List[List[object]] = [[], [], [], []]
    for rel_path in tqdm(manifest["shards"], desc=f"Load entity cache ({label})"):
        shard = torch.load(str(entity_dir / rel_path), map_location="cpu")
        for idx, tensor in enumerate(shard["tensors"]):
            parts[idx].append(tensor)
        cpu_peak = max(cpu_peak, cpu_rss_mb())
    cache = tuple(concat_cache_parts(piece, tensor_modes[idx]) for idx, piece in enumerate(parts))
    load_s = time.perf_counter() - load_start
    cpu_after_load = cpu_rss_mb()
    cpu_peak = max(cpu_peak, cpu_after_load)

    total_storage_bytes = int(sum(manifest["tensor_storage_bytes"].values()))
    tensor_storage_mb = {
        f"{name}_mb": int(bytes_value) / (1024.0 * 1024.0)
        for name, bytes_value in manifest["tensor_storage_bytes"].items()
    }
    meta = {
        "entities": cache_tensor_length(cache[0]),
        "storage_mb": total_storage_bytes / (1024.0 * 1024.0),
        "seconds": time.perf_counter() - start,
        "encode_s": float(manifest.get("encode_s", 0.0)),
        "quantize_s": float(manifest.get("quantize_s", 0.0)),
        "save_s": float(manifest.get("save_s", 0.0)),
        "load_s": load_s,
        "manifest": str(manifest_path),
        "shards": len(manifest["shards"]),
        "cpu_rss_mb_start": cpu_start,
        "cpu_rss_mb_after_entity_cache_load": cpu_after_load,
        "cpu_rss_mb_peak": max(cpu_peak, float(manifest.get("entity_cache_peak_cpu_rss_mb", 0.0))),
        "entity_cache_build_peak_gpu_allocated_mb": float(manifest.get("entity_cache_build_gpu_peak_allocated_mb", 0.0)),
        "entity_cache_build_peak_gpu_reserved_mb": float(manifest.get("entity_cache_build_gpu_peak_reserved_mb", 0.0)),
        "entity_cache_load_peak_gpu_allocated_mb": cuda_memory_snapshot(device, "entity_cache_load_").get("entity_cache_load_gpu_peak_allocated_mb", 0.0),
        "entity_cache_load_peak_gpu_reserved_mb": cuda_memory_snapshot(device, "entity_cache_load_").get("entity_cache_load_gpu_peak_reserved_mb", 0.0),
        **tensor_storage_mb,
        **disk_usage_mb(entity_dir),
    }
    for tensor_name, stats in manifest.get("quant_error", {}).items():
        for key, value in stats.items():
            meta[f"{tensor_name}_quant_{key}"] = value
    return cache, meta


def timed_matcher(model, ent_chunk, mention_pack, device: torch.device, matcher_tile_size: int = 0):
    timings = {"layernorm_s": 0.0, "tglu_s": 0.0, "vdlu_s": 0.0, "cmfu_s": 0.0}
    peaks = {
        "gpu_peak_allocated_mb_layernorm": 0.0,
        "gpu_peak_allocated_mb_tglu": 0.0,
        "gpu_peak_allocated_mb_vdlu": 0.0,
        "gpu_peak_allocated_mb_cmfu": 0.0,
    }

    reset_cuda_peak(device)
    t0 = time.perf_counter()
    entity_text_cls = model.matcher.text_cls_layernorm(ent_chunk[0])
    mention_text_cls = model.matcher.text_cls_layernorm(mention_pack[0])
    entity_text_tokens = model.matcher.text_tokens_layernorm(ent_chunk[2])
    mention_text_tokens = model.matcher.text_tokens_layernorm(mention_pack[2])
    entity_image_cls = model.matcher.image_cls_layernorm(ent_chunk[1])
    mention_image_cls = model.matcher.image_cls_layernorm(mention_pack[1])
    entity_image_tokens = model.matcher.image_tokens_layernorm(ent_chunk[3])
    mention_image_tokens = model.matcher.image_tokens_layernorm(mention_pack[3])
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        peaks["gpu_peak_allocated_mb_layernorm"] = torch.cuda.max_memory_allocated(device) / BYTES_PER_MB
    timings["layernorm_s"] += time.perf_counter() - t0

    reset_cuda_peak(device)
    t0 = time.perf_counter()
    if matcher_tile_size and matcher_tile_size > 0 and entity_text_cls.shape[0] > matcher_tile_size:
        text_score_parts = []
        for tile_start in range(0, entity_text_cls.shape[0], matcher_tile_size):
            tile_end = min(tile_start + matcher_tile_size, entity_text_cls.shape[0])
            text_score_parts.append(model.matcher.tglu(
                entity_text_cls[tile_start:tile_end],
                entity_text_tokens[tile_start:tile_end],
                mention_text_cls,
                mention_text_tokens,
            ))
        text_score = torch.cat(text_score_parts, dim=-1)
    else:
        text_score = model.matcher.tglu(entity_text_cls, entity_text_tokens, mention_text_cls, mention_text_tokens)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        peaks["gpu_peak_allocated_mb_tglu"] = torch.cuda.max_memory_allocated(device) / BYTES_PER_MB
    timings["tglu_s"] += time.perf_counter() - t0

    reset_cuda_peak(device)
    t0 = time.perf_counter()
    image_score = model.matcher.vdlu(entity_image_cls, entity_image_tokens, mention_image_cls, mention_image_tokens)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        peaks["gpu_peak_allocated_mb_vdlu"] = torch.cuda.max_memory_allocated(device) / BYTES_PER_MB
    timings["vdlu_s"] += time.perf_counter() - t0

    reset_cuda_peak(device)
    t0 = time.perf_counter()
    image_text_score = model.matcher.cmfu(entity_text_cls, entity_image_tokens, mention_text_cls, mention_image_tokens)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        peaks["gpu_peak_allocated_mb_cmfu"] = torch.cuda.max_memory_allocated(device) / BYTES_PER_MB
    timings["cmfu_s"] += time.perf_counter() - t0

    return (text_score + image_score + image_text_score) / 3.0, timings, peaks


def evaluate_split_with_metrics(
    model,
    dataloader,
    entity_cache,
    device: torch.device,
    label: str,
    mode_dir: Path,
    chunk_size: int,
    limit_queries: int,
    matcher_tile_size: int = 0,
    entity_subtile_size: int = 0,
):
    ranks_dir = mode_dir / "rank_batches"
    batch_metrics_dir = mode_dir / "batch_metrics"
    ranks_dir.mkdir(parents=True, exist_ok=True)
    batch_metrics_dir.mkdir(parents=True, exist_ok=True)

    entity_count = cache_tensor_length(entity_cache[0])
    skipped = 0
    seen_queries = 0
    start = time.perf_counter()
    timer_totals = {
        "mention_to_device_s": 0.0,
        "mention_encoder_s": 0.0,
        "entity_transfer_s": 0.0,
        "layernorm_s": 0.0,
        "tglu_s": 0.0,
        "vdlu_s": 0.0,
        "cmfu_s": 0.0,
        "score_concat_s": 0.0,
        "rank_sort_s": 0.0,
    }
    chunk_count = 0
    entity_subtile_count = 0
    batch_latencies = []
    query_latencies = []
    peak_metrics = {
        "gpu_peak_allocated_mb_eval": 0.0,
        "gpu_peak_reserved_mb_eval": 0.0,
        "gpu_peak_allocated_mb_mention_encoder": 0.0,
        "gpu_peak_allocated_mb_entity_chunk_transfer": 0.0,
        "gpu_peak_allocated_mb_matcher": 0.0,
        "gpu_peak_allocated_mb_layernorm": 0.0,
        "gpu_peak_allocated_mb_tglu": 0.0,
        "gpu_peak_allocated_mb_vdlu": 0.0,
        "gpu_peak_allocated_mb_cmfu": 0.0,
    }
    reset_cuda_peak(device)

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc=f"Evaluate ({label})", total=len(dataloader))):
            rank_path = ranks_dir / f"batch_{batch_idx:05d}.npy"
            meta_path = batch_metrics_dir / f"batch_{batch_idx:05d}.json"
            if rank_path.exists() and meta_path.exists():
                meta = read_json(meta_path)
                seen_queries += int(meta.get("seen_queries", 0))
                skipped += int(meta.get("skipped_queries", 0))
                chunk_count += int(meta.get("chunks", 0))
                entity_subtile_count += int(meta.get("entity_subtiles", meta.get("chunks", 0)))
                for key in timer_totals:
                    timer_totals[key] += float(meta.get(key, 0.0))
                if float(meta.get("batch_total_s", 0.0)) > 0:
                    batch_latencies.append(float(meta.get("batch_total_s", 0.0)))
                batch_queries = int(meta.get("batch_queries", 0))
                if batch_queries and float(meta.get("batch_total_s", 0.0)) > 0:
                    query_latencies.extend([float(meta["batch_total_s"]) / batch_queries] * batch_queries)
                for key in peak_metrics:
                    peak_metrics[key] = max(peak_metrics[key], float(meta.get(key, 0.0)))
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
                write_json(meta_path, {"seen_queries": current_seen, "skipped_queries": current_skipped, "batch_queries": 0, "chunks": 0, "batch_total_s": 0.0})
                continue

            batch_start = time.perf_counter()
            valid_queries = int(keep.sum().item())
            answer = answer[keep].to(device)
            t0 = time.perf_counter()
            batch = {k: v[keep].to(device) for k, v in batch.items()}
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            batch_timer = {key: 0.0 for key in timer_totals}
            batch_timer["mention_to_device_s"] += time.perf_counter() - t0

            reset_cuda_peak(device)
            t0 = time.perf_counter()
            mention_pack = tuple(t.detach() for t in model.encoder(**batch))
            if device.type == "cuda":
                torch.cuda.synchronize(device)
                peak_metrics["gpu_peak_allocated_mb_mention_encoder"] = max(peak_metrics["gpu_peak_allocated_mb_mention_encoder"], torch.cuda.max_memory_allocated(device) / BYTES_PER_MB)
            batch_timer["mention_encoder_s"] += time.perf_counter() - t0

            scores = []
            batch_chunks = 0
            batch_entity_subtiles = 0
            for start_pos in range(0, entity_count, chunk_size):
                end_pos = min(start_pos + chunk_size, entity_count)
                subtile_size = int(entity_subtile_size or 0)
                if subtile_size <= 0:
                    subtile_size = end_pos - start_pos
                for subtile_start in range(start_pos, end_pos, subtile_size):
                    subtile_end = min(subtile_start + subtile_size, end_pos)
                    reset_cuda_peak(device)
                    t0 = time.perf_counter()
                    ent_subtile = tuple(cache_tensor_to_device(t, device, subtile_start, subtile_end) for t in entity_cache)
                    if device.type == "cuda":
                        torch.cuda.synchronize(device)
                        peak_metrics["gpu_peak_allocated_mb_entity_chunk_transfer"] = max(peak_metrics["gpu_peak_allocated_mb_entity_chunk_transfer"], torch.cuda.max_memory_allocated(device) / BYTES_PER_MB)
                    batch_timer["entity_transfer_s"] += time.perf_counter() - t0

                    chunk_score, matcher_timers, matcher_peaks = timed_matcher(
                        model,
                        ent_subtile,
                        mention_pack,
                        device,
                        matcher_tile_size=matcher_tile_size,
                    )
                    for key, value in matcher_timers.items():
                        batch_timer[key] += value
                    for key, value in matcher_peaks.items():
                        peak_metrics[key] = max(peak_metrics[key], value)
                        peak_metrics["gpu_peak_allocated_mb_matcher"] = max(peak_metrics["gpu_peak_allocated_mb_matcher"], value)
                    scores.append(chunk_score)
                    batch_entity_subtiles += 1
                batch_chunks += 1

            t0 = time.perf_counter()
            scores_tensor = torch.cat(scores, dim=-1)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            batch_timer["score_concat_s"] += time.perf_counter() - t0

            t0 = time.perf_counter()
            rank = torch.argsort(torch.argsort(scores_tensor, dim=-1, descending=True), dim=-1) + 1
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            batch_timer["rank_sort_s"] += time.perf_counter() - t0

            batch_ranks = rank[torch.arange(answer.shape[0], device=device), answer].cpu().numpy()
            np.save(rank_path, batch_ranks)
            for key in timer_totals:
                timer_totals[key] += batch_timer[key]
            chunk_count += batch_chunks
            entity_subtile_count += batch_entity_subtiles
            batch_total_s = time.perf_counter() - batch_start
            batch_latencies.append(batch_total_s)
            query_latencies.extend([batch_total_s / max(valid_queries, 1)] * valid_queries)
            if device.type == "cuda":
                peak_metrics["gpu_peak_allocated_mb_eval"] = max(peak_metrics["gpu_peak_allocated_mb_eval"], torch.cuda.max_memory_allocated(device) / BYTES_PER_MB)
                peak_metrics["gpu_peak_reserved_mb_eval"] = max(peak_metrics["gpu_peak_reserved_mb_eval"], torch.cuda.max_memory_reserved(device) / BYTES_PER_MB)
            write_json(meta_path, {
                "seen_queries": current_seen,
                "skipped_queries": current_skipped,
                "batch_queries": valid_queries,
                "chunks": batch_chunks,
                "entity_subtiles": batch_entity_subtiles,
                "batch_total_s": batch_total_s,
                "batch_peak_gpu_allocated_mb": peak_metrics["gpu_peak_allocated_mb_eval"],
                "batch_peak_gpu_reserved_mb": peak_metrics["gpu_peak_reserved_mb_eval"],
                **batch_timer,
                **peak_metrics,
            })

    rank_parts = [np.load(path) for path in sorted(ranks_dir.glob("batch_*.npy"))]
    nonempty_rank_parts = [part for part in rank_parts if part.size]
    ranks_np = np.concatenate(nonempty_rank_parts) if nonempty_rank_parts else np.array([], dtype=np.int64)
    if ranks_np.size == 0:
        raise RuntimeError("No queries were evaluated. Increase --entity-limit or remove it.")

    metrics = retrieval_metrics(ranks_np)
    total_s = time.perf_counter() - start
    timed_s = sum(timer_totals.values())
    batch_lat_np = np.array(batch_latencies, dtype=np.float64) if batch_latencies else np.array([0.0])
    query_lat_ms = np.array(query_latencies, dtype=np.float64) * 1000.0 if query_latencies else np.array([0.0])
    query_entity_pairs = int(metrics["queries"]) * int(entity_count)
    peak_metrics["gpu_peak_allocated_mb_eval"] = max(
        peak_metrics["gpu_peak_allocated_mb_eval"],
        peak_metrics["gpu_peak_allocated_mb_mention_encoder"],
        peak_metrics["gpu_peak_allocated_mb_entity_chunk_transfer"],
        peak_metrics["gpu_peak_allocated_mb_matcher"],
    )
    metrics.update({
        "mode": label,
        "checkpoint_dir": str(mode_dir),
        "skipped_queries": int(skipped),
        "seconds": total_s,
        "total_eval_seconds": total_s,
        "chunks": int(chunk_count),
        "matcher_tile_size": int(matcher_tile_size or 0),
        "entity_subtile_size": int(entity_subtile_size or 0),
        "entity_subtiles": int(entity_subtile_count),
        "timed_subprocess_s": timed_s,
        "unattributed_overhead_s": max(0.0, total_s - timed_s),
        "avg_query_latency_ms": float(np.mean(query_lat_ms)),
        "median_query_latency_ms": float(np.median(query_lat_ms)),
        "p90_query_latency_ms": float(np.percentile(query_lat_ms, 90)),
        "p95_query_latency_ms": float(np.percentile(query_lat_ms, 95)),
        "p99_query_latency_ms": float(np.percentile(query_lat_ms, 99)),
        "min_query_latency_ms": float(np.min(query_lat_ms)),
        "max_query_latency_ms": float(np.max(query_lat_ms)),
        "avg_batch_latency_s": float(np.mean(batch_lat_np)),
        "median_batch_latency_s": float(np.median(batch_lat_np)),
        "p90_batch_latency_s": float(np.percentile(batch_lat_np, 90)),
        "p95_batch_latency_s": float(np.percentile(batch_lat_np, 95)),
        "p99_batch_latency_s": float(np.percentile(batch_lat_np, 99)),
        "queries_per_second": float(metrics["queries"] / total_s) if total_s else 0.0,
        "entities_scored_per_second": float(query_entity_pairs / total_s) if total_s else 0.0,
        "query_entity_pairs_per_second": float(query_entity_pairs / total_s) if total_s else 0.0,
        "batches_per_second": float(len(batch_latencies) / total_s) if total_s else 0.0,
        "chunks_per_second": float(chunk_count / total_s) if total_s else 0.0,
        "entity_subtiles_per_second": float(entity_subtile_count / total_s) if total_s else 0.0,
        **timer_totals,
        **peak_metrics,
    })
    for key, value in timer_totals.items():
        metrics[f"{key}_pct"] = (value / total_s) if total_s else 0.0
    return metrics


def add_baseline_deltas(rows: List[Dict[str, object]], target_tensor: str = "text_tokens") -> List[Dict[str, object]]:
    target_mode_key = f"{target_tensor}_mode"
    baseline = next((row for row in rows if row.get(target_mode_key) == "fp32"), None)
    if baseline is None:
        return rows

    target_cache_key = f"entity_cache_{target_tensor}_mb"
    for row in rows:
        for key in ("hits@1", "hits@3", "hits@10", "hits@20", "mrr", "mr", "seconds", "entity_cache_storage_mb", target_cache_key):
            if key in row and key in baseline:
                row[f"delta_{key}_vs_fp32"] = float(row[key]) - float(baseline[key])
        if "seconds" in row and "seconds" in baseline and float(row["seconds"]) > 0:
            row["speedup_vs_fp32"] = float(baseline["seconds"]) / float(row["seconds"])
        if "entity_cache_storage_mb" in row and float(baseline.get("entity_cache_storage_mb", 0.0)) > 0:
            row["cache_size_ratio_vs_fp32"] = float(row["entity_cache_storage_mb"]) / float(baseline["entity_cache_storage_mb"])
        if target_cache_key in row and float(baseline.get(target_cache_key, 0.0)) > 0:
            row[f"{target_tensor}_size_ratio_vs_fp32"] = float(row[target_cache_key]) / float(baseline[target_cache_key])
    return rows


def load_rank_batches(mode_dir: Path) -> np.ndarray:
    rank_dir = mode_dir / "rank_batches"
    parts = [np.load(path) for path in sorted(rank_dir.glob("batch_*.npy"))]
    nonempty = [part for part in parts if part.size]
    return np.concatenate(nonempty) if nonempty else np.array([], dtype=np.int64)


def add_rank_stability(rows: List[Dict[str, object]], target_tensor: str = "text_tokens") -> List[Dict[str, object]]:
    target_mode_key = f"{target_tensor}_mode"
    baseline = next((row for row in rows if row.get(target_mode_key) == "fp32"), None)
    if baseline is None or "checkpoint_dir" not in baseline:
        return rows
    baseline_ranks = load_rank_batches(Path(str(baseline["checkpoint_dir"])))
    if baseline_ranks.size == 0:
        return rows
    for row in rows:
        if "checkpoint_dir" not in row:
            continue
        ranks = load_rank_batches(Path(str(row["checkpoint_dir"])))
        n = min(int(baseline_ranks.size), int(ranks.size))
        if n == 0:
            continue
        delta = ranks[:n].astype(np.int64) - baseline_ranks[:n].astype(np.int64)
        abs_delta = np.abs(delta)
        row.update({
            "rank_flip_count": int(np.count_nonzero(delta)),
            "rank_improved_count": int(np.count_nonzero(delta < 0)),
            "rank_worsened_count": int(np.count_nonzero(delta > 0)),
            "mean_abs_rank_delta": float(np.mean(abs_delta)),
            "median_abs_rank_delta": float(np.median(abs_delta)),
            "gold_rank_delta_mean": float(np.mean(delta)),
            "gold_rank_delta_median": float(np.median(delta)),
            "gold_rank_delta_p95": float(np.percentile(delta, 95)),
            "gold_rank_delta_max": int(np.max(delta)),
            "max_rank_worsening": int(np.max(delta)),
            "max_rank_improvement": int(np.min(delta)),
        })
    return rows


def add_fp32_deltas(rows: List[Dict[str, object]]) -> List[Dict[str, object]]:
    baseline = next((
        row for row in rows
        if all(row.get(f"{name}_mode") == "fp32" for name in TENSOR_NAMES)
    ), None)
    if baseline is None:
        return rows
    for row in rows:
        for key in (
            "hits@1", "hits@3", "hits@10", "hits@20", "mrr", "mr", "seconds",
            "entity_cache_storage_mb", "entity_cache_text_tokens_mb", "entity_cache_image_patch_tokens_mb",
        ):
            if key in row and key in baseline:
                row[f"delta_{key}_vs_fp32"] = float(row[key]) - float(baseline[key])
        if "seconds" in row and "seconds" in baseline and float(row["seconds"]) > 0:
            row["speedup_vs_fp32"] = float(baseline["seconds"]) / float(row["seconds"])
        for tensor_name in TENSOR_NAMES:
            key = f"entity_cache_{tensor_name}_mb"
            if key in row and float(baseline.get(key, 0.0)) > 0:
                row[f"{tensor_name}_size_ratio_vs_fp32"] = float(row[key]) / float(baseline[key])
        if "entity_cache_storage_mb" in row and float(baseline.get("entity_cache_storage_mb", 0.0)) > 0:
            row["cache_size_ratio_vs_fp32"] = float(row["entity_cache_storage_mb"]) / float(baseline["entity_cache_storage_mb"])
    return rows


def add_fp32_rank_stability(rows: List[Dict[str, object]]) -> List[Dict[str, object]]:
    baseline = next((
        row for row in rows
        if all(row.get(f"{name}_mode") == "fp32" for name in TENSOR_NAMES)
    ), None)
    if baseline is None or "checkpoint_dir" not in baseline:
        return rows
    baseline_ranks = load_rank_batches(Path(str(baseline["checkpoint_dir"])))
    if baseline_ranks.size == 0:
        return rows
    for row in rows:
        if "checkpoint_dir" not in row:
            continue
        ranks = load_rank_batches(Path(str(row["checkpoint_dir"])))
        n = min(int(baseline_ranks.size), int(ranks.size))
        if n == 0:
            continue
        delta = ranks[:n].astype(np.int64) - baseline_ranks[:n].astype(np.int64)
        abs_delta = np.abs(delta)
        row.update({
            "rank_flip_count": int(np.count_nonzero(delta)),
            "rank_improved_count": int(np.count_nonzero(delta < 0)),
            "rank_worsened_count": int(np.count_nonzero(delta > 0)),
            "mean_abs_rank_delta": float(np.mean(abs_delta)),
            "median_abs_rank_delta": float(np.median(abs_delta)),
            "gold_rank_delta_mean": float(np.mean(delta)),
            "gold_rank_delta_median": float(np.median(delta)),
            "gold_rank_delta_p95": float(np.percentile(delta, 95)),
            "gold_rank_delta_max": int(np.max(delta)),
            "max_rank_worsening": int(np.max(delta)),
            "max_rank_improvement": int(np.min(delta)),
        })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate mixed quantization for one entity cache tensor.")
    parser.add_argument("--repo-root", default="..")
    parser.add_argument("--mimic-root", default="../ExpSetup/external/MIMIC_reproduction")
    parser.add_argument("--config", default="../ExpSetup/external/MIMIC_reproduction/config/wikimel_noleak.yaml")
    parser.add_argument("--checkpoint", default="../ExpSetup/external/MIMIC_reproduction/runs/WikiMEL_noleak/version_4/checkpoints/final.ckpt")
    parser.add_argument("--split", choices=["dev", "test"], default="test")
    parser.add_argument("--entity-text-token-modes", nargs="+", default=["fp32", "int8", "int6", "int4", "int3", "int2"])
    parser.add_argument("--target-tensor", choices=TENSOR_NAMES, default="text_tokens")
    parser.add_argument("--target-modes", nargs="+")
    parser.add_argument(
        "--mode-specs",
        nargs="+",
        help="Explicit runs as label:tensor=mode,tensor=mode. Unspecified tensors default to fp32.",
    )
    parser.add_argument("--other-mode", default="fp32")
    parser.add_argument("--out-dir", default="results/entity_text_tokens")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--chunk-size", type=int, default=20000)
    parser.add_argument(
        "--matcher-tile-size",
        type=int,
        default=0,
        help="Optional entity subtile size for TGLU inside each eval chunk. Keeps chunk-size fixed but lowers matcher peak VRAM.",
    )
    parser.add_argument(
        "--entity-subtile-size",
        type=int,
        default=0,
        help="Optional entity subtile size for transfer/dequantization and full matcher scoring inside each eval chunk.",
    )
    parser.add_argument("--limit-queries", type=int, default=0)
    parser.add_argument("--entity-limit", type=int, default=0)
    parser.add_argument("--force", action="store_true")
    args_cli = parser.parse_args()
    target_tensor = normalize_target_tensor(args_cli.target_tensor)
    target_modes = args_cli.target_modes or args_cli.entity_text_token_modes
    explicit_specs = [parse_mode_spec(spec) for spec in args_cli.mode_specs] if args_cli.mode_specs else []

    script_dir = Path(__file__).resolve().parent
    repo_root = (script_dir / args_cli.repo_root).resolve()
    mimic_root = (script_dir / args_cli.mimic_root).resolve()
    add_repo_paths(repo_root, mimic_root)

    from codes.utils.dataset import DataModuleForMIMIC

    config_path = (script_dir / args_cli.config).resolve()
    checkpoint_path = (script_dir / args_cli.checkpoint).resolve()
    out_dir = (script_dir / args_cli.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = resolve_config_paths(OmegaConf.load(config_path), mimic_root)
    cfg.data.eval_chunk_size = args_cli.chunk_size
    device = torch.device(args_cli.device)
    env_meta = environment_metadata(device)

    data_module = DataModuleForMIMIC(cfg)
    model = load_model(cfg, checkpoint_path, device)
    eval_loader = data_module.test_dataloader() if args_cli.split == "test" else data_module.val_dataloader()

    rows = []
    details = {
        "checkpoint": str(checkpoint_path),
        "config": str(config_path),
        "split": args_cli.split,
        "device": str(device),
        "chunk_size": args_cli.chunk_size,
        "matcher_tile_size": int(args_cli.matcher_tile_size or 0),
        "entity_subtile_size": int(args_cli.entity_subtile_size or 0),
        "limit_queries": args_cli.limit_queries,
        "entity_limit": args_cli.entity_limit,
        "other_mode": args_cli.other_mode,
        "target_tensor": target_tensor,
        "target_modes": target_modes,
        "mode_specs": args_cli.mode_specs or [],
        "entity_text_token_modes": target_modes if target_tensor == "text_tokens" else args_cli.entity_text_token_modes,
        **env_meta,
    }

    summary_stem = "mixed_tensor_quantization" if explicit_specs else ("entity_text_token_quantization" if target_tensor == "text_tokens" else f"entity_{target_tensor}_quantization")
    run_specs = explicit_specs or [setup_modes(target_tensor, target_mode, args_cli.other_mode) for target_mode in target_modes]

    for label, tensor_modes in run_specs:
        mode_by_name = dict(zip(TENSOR_NAMES, tensor_modes))
        mode_dir = out_dir / label
        mode_dir.mkdir(parents=True, exist_ok=True)
        result_path = mode_dir / "result.json"

        if result_path.exists() and not args_cli.force:
            row = read_json(result_path)
            rows.append(row)
            print(f"Reusing completed result for {label}: {result_path}")
            continue

        try:
            entity_cache, cache_meta = build_or_load_entity_cache(
                model=model,
                dataloader=data_module.entity_dataloader(),
                device=device,
                label=label,
                tensor_modes=tensor_modes,
                mode_dir=mode_dir,
                limit_entities=args_cli.entity_limit,
            )
            metrics = evaluate_split_with_metrics(
                model=model,
                dataloader=eval_loader,
                entity_cache=entity_cache,
                device=device,
                label=label,
                mode_dir=mode_dir,
                chunk_size=args_cli.chunk_size,
                limit_queries=args_cli.limit_queries,
                matcher_tile_size=args_cli.matcher_tile_size,
                entity_subtile_size=args_cli.entity_subtile_size,
            )
            row = {
                **env_meta,
                **metrics,
                "run_success": True,
                "failed_stage": "",
                "exception_type": "",
                "exception_message": "",
                "oom_boolean": False,
                "checkpoint_path": str(checkpoint_path),
                "config_path": str(config_path),
                "split": args_cli.split,
                "num_entities": int(cache_meta.get("entities", 0)),
                "chunk_size": args_cli.chunk_size,
                "eval_batch_size": int(getattr(cfg.data, "eval_batch_size", 0) or 0),
                "text_cls_mode": mode_by_name["text_cls"],
                "image_cls_mode": mode_by_name["image_cls"],
                "text_tokens_mode": mode_by_name["text_tokens"],
                "image_patch_tokens_mode": mode_by_name["image_patch_tokens"],
                "entity_text_tokens_mode": mode_by_name["text_tokens"],
                "other_mode": args_cli.other_mode,
                **{(key if key.startswith("entity_cache_") else f"entity_cache_{key}"): value for key, value in cache_meta.items()},
            }
        except Exception as exc:
            row = {
                **env_meta,
                "mode": label,
                "checkpoint_dir": str(mode_dir),
                "run_success": False,
                "failed_stage": "unknown",
                "exception_type": type(exc).__name__,
                "exception_message": str(exc),
                "oom_boolean": "out of memory" in str(exc).lower(),
                "checkpoint_path": str(checkpoint_path),
                "config_path": str(config_path),
                "split": args_cli.split,
                "chunk_size": args_cli.chunk_size,
                "matcher_tile_size": int(args_cli.matcher_tile_size or 0),
                "entity_subtile_size": int(args_cli.entity_subtile_size or 0),
                "text_cls_mode": mode_by_name["text_cls"],
                "image_cls_mode": mode_by_name["image_cls"],
                "text_tokens_mode": mode_by_name["text_tokens"],
                "image_patch_tokens_mode": mode_by_name["image_patch_tokens"],
                "entity_text_tokens_mode": mode_by_name["text_tokens"],
                "other_mode": args_cli.other_mode,
            }
        rows.append(row)
        rows = add_fp32_deltas(rows) if explicit_specs else add_baseline_deltas(rows, target_tensor)
        rows = add_fp32_rank_stability(rows) if explicit_specs else add_rank_stability(rows, target_tensor)
        write_json(result_path, row)
        details["results"] = rows
        write_json(out_dir / f"{summary_stem}_results.json", details)
        write_csv(out_dir / f"{summary_stem}_summary.csv", rows)
        print(json.dumps(row, indent=2))

    rows = add_fp32_deltas(rows) if explicit_specs else add_baseline_deltas(rows, target_tensor)
    rows = add_fp32_rank_stability(rows) if explicit_specs else add_rank_stability(rows, target_tensor)
    details["results"] = rows
    write_json(out_dir / f"{summary_stem}_results.json", details)
    write_csv(out_dir / f"{summary_stem}_summary.csv", rows)
    print(f"Saved results to {out_dir}")


if __name__ == "__main__":
    main()
