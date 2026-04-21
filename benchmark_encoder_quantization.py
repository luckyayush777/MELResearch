import argparse
import json
import os
import platform
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

try:
    import psutil
except ImportError:
    psutil = None


DEFAULT_VALID_META = os.path.join("data", "embeddings", "valid_metadata.json")
DEFAULT_IMAGE_DIR = os.path.join("data", "images")
DEFAULT_PAIRED_TEXT_EMB = os.path.join("data", "embeddings", "text_embeddings.npy")
DEFAULT_KB_TEXT_EMB = os.path.join("data", "embeddings", "kb_text_embeddings.npy")
DEFAULT_ENTITY_KB = os.path.join("data", "entity_kb.json")


def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def bytes_to_mb(n_bytes: int) -> float:
    return float(n_bytes) / (1024.0 * 1024.0)


def process_memory_snapshot() -> Dict[str, int]:
    if psutil is None:
        return {}
    info = psutil.Process(os.getpid()).memory_info()
    snapshot = {"rss_bytes": int(getattr(info, "rss", 0))}
    for attr in ("peak_wset", "wset", "vms", "peak_pagefile", "pagefile"):
        if hasattr(info, attr):
            snapshot[f"{attr}_bytes"] = int(getattr(info, attr))
    return snapshot


def collect_environment() -> Dict[str, object]:
    env = {
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "processor": platform.processor(),
    }
    try:
        import torch

        env["torch_version"] = torch.__version__
        env["cuda_available"] = bool(torch.cuda.is_available())
        if torch.cuda.is_available():
            env["gpu_name"] = torch.cuda.get_device_name(0)
    except Exception as exc:
        env["torch_info_error"] = f"{type(exc).__name__}: {exc}"
    if psutil is not None:
        env["psutil_version"] = psutil.__version__
    return env


def normalize_rows(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    return x / norms


def compute_ranks(sim: np.ndarray, gold_indices: np.ndarray) -> np.ndarray:
    gold_scores = sim[np.arange(sim.shape[0]), gold_indices][:, None]
    return 1 + (sim > gold_scores).sum(axis=1)


def retrieval_metrics(ranks: np.ndarray) -> Dict[str, float]:
    return {
        "queries": int(ranks.shape[0]),
        "recall@1": float(np.mean(ranks <= 1)),
        "recall@5": float(np.mean(ranks <= 5)),
        "recall@10": float(np.mean(ranks <= 10)),
        "mrr": float(np.mean(1.0 / ranks)),
        "median_rank": float(np.median(ranks)),
        "mean_rank": float(np.mean(ranks)),
    }


def kb_gold_indices(valid_meta: List[Dict[str, str]], entity_kb: List[Dict[str, str]]) -> Tuple[np.ndarray, np.ndarray]:
    title_to_index = {row["title"]: idx for idx, row in enumerate(entity_kb)}
    keep_positions = []
    gold_indices = []
    for idx, item in enumerate(valid_meta):
        title = item["title"]
        if title in title_to_index:
            keep_positions.append(idx)
            gold_indices.append(title_to_index[title])
    return np.array(keep_positions, dtype=np.int64), np.array(gold_indices, dtype=np.int64)


def build_visual_mode(mode: str, model_name: str, pretrained: str):
    import open_clip
    import torch

    model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
    model.eval()

    if mode == "gpu_fp32":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available for gpu_fp32 mode.")
        visual = model.visual.to("cuda", dtype=torch.float32).eval()
        return visual, preprocess, {"device": "cuda", "dtype": "float32", "batch_size": 256}

    if mode == "gpu_fp16":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available for gpu_fp16 mode.")
        visual = model.visual.to("cuda", dtype=torch.float16).eval()
        return visual, preprocess, {"device": "cuda", "dtype": "float16", "batch_size": 256}

    if mode == "cpu_fp32":
        visual = model.visual.to("cpu", dtype=torch.float32).eval()
        return visual, preprocess, {"device": "cpu", "dtype": "float32", "batch_size": 64}

    if mode == "cpu_int8_dynamic":
        visual = model.visual.to("cpu", dtype=torch.float32).eval()
        visual = torch.ao.quantization.quantize_dynamic(visual, {torch.nn.Linear}, dtype=torch.qint8)
        return visual, preprocess, {"device": "cpu", "dtype": "int8_dynamic", "batch_size": 64}

    raise ValueError(f"Unsupported mode: {mode}")


def encode_images_for_mode(
    mode: str,
    model_name: str,
    pretrained: str,
    valid_meta: List[Dict[str, str]],
    image_dir: str,
) -> Tuple[np.ndarray, Dict[str, object]]:
    import torch

    visual, preprocess, mode_info = build_visual_mode(mode, model_name, pretrained)
    device = torch.device(mode_info["device"])
    batch_size = int(mode_info["batch_size"])
    embeddings = []

    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    peak_rss_bytes = 0
    peak_wset_bytes = 0
    start_time = time.perf_counter()

    for start in range(0, len(valid_meta), batch_size):
        items = valid_meta[start : start + batch_size]
        batch = []
        for item in items:
            img_path = os.path.join(image_dir, item["image_file"])
            tensor = preprocess(Image.open(img_path).convert("RGB"))
            batch.append(tensor)
        batch_tensor = torch.stack(batch)
        if device.type == "cuda":
            batch_tensor = batch_tensor.to(device=device, dtype=torch.float16 if mode == "gpu_fp16" else torch.float32)
        else:
            batch_tensor = batch_tensor.to(device=device, dtype=torch.float32)

        before = process_memory_snapshot()
        with torch.no_grad():
            feats = visual(batch_tensor)
            feats = feats / feats.norm(dim=-1, keepdim=True)
        after = process_memory_snapshot()
        embeddings.append(feats.detach().cpu().float().numpy())
        peak_rss_bytes = max(peak_rss_bytes, int(before.get("rss_bytes", 0)), int(after.get("rss_bytes", 0)))
        peak_wset_bytes = max(
            peak_wset_bytes,
            int(before.get("peak_wset_bytes", 0)),
            int(after.get("peak_wset_bytes", 0)),
        )

    total_seconds = time.perf_counter() - start_time
    embeddings_np = np.vstack(embeddings).astype(np.float32)
    metadata = {
        "mode": mode,
        "device": mode_info["device"],
        "dtype": mode_info["dtype"],
        "batch_size": batch_size,
        "images_encoded": int(embeddings_np.shape[0]),
        "embedding_dim": int(embeddings_np.shape[1]),
        "encoding_time_seconds": total_seconds,
        "latency_ms": total_seconds * 1000.0,
        "per_image_latency_ms": (total_seconds * 1000.0) / max(int(embeddings_np.shape[0]), 1),
        "images_per_second": float(int(embeddings_np.shape[0]) / max(total_seconds, 1e-12)),
        "peak_rss_mb": bytes_to_mb(peak_rss_bytes) if peak_rss_bytes else None,
        "peak_working_set_mb": bytes_to_mb(peak_wset_bytes) if peak_wset_bytes else None,
    }
    if device.type == "cuda":
        metadata["gpu_peak_memory_mb"] = bytes_to_mb(int(torch.cuda.max_memory_allocated()))

    return embeddings_np, metadata


def evaluate_mode(
    image_embeddings: np.ndarray,
    paired_text_embeddings: np.ndarray,
    kb_text_embeddings: np.ndarray,
    kb_keep_positions: np.ndarray,
    kb_gold_indices: np.ndarray,
) -> Dict[str, object]:
    paired_sim = normalize_rows(image_embeddings) @ normalize_rows(paired_text_embeddings).T
    paired_gold = np.arange(image_embeddings.shape[0], dtype=np.int64)
    paired_ranks = compute_ranks(paired_sim, paired_gold)

    kb_image_embeddings = image_embeddings[kb_keep_positions]
    kb_sim = normalize_rows(kb_image_embeddings) @ normalize_rows(kb_text_embeddings).T
    kb_ranks = compute_ranks(kb_sim, kb_gold_indices)

    return {
        "paired": retrieval_metrics(paired_ranks),
        "kb": retrieval_metrics(kb_ranks),
    }


def create_plots(results: Dict[str, object], out_dir: str) -> List[str]:
    os.makedirs(out_dir, exist_ok=True)
    modes = list(results["modes"].keys())
    labels = modes

    kb_r1 = [results["modes"][m]["retrieval"]["kb"]["recall@1"] for m in modes]
    paired_r1 = [results["modes"][m]["retrieval"]["paired"]["recall@1"] for m in modes]
    per_image_ms = [results["modes"][m]["encoding"]["per_image_latency_ms"] for m in modes]
    images_per_second = [results["modes"][m]["encoding"]["images_per_second"] for m in modes]

    created = []

    plt.figure(figsize=(10, 5))
    x = np.arange(len(labels))
    width = 0.35
    plt.bar(x - width / 2, paired_r1, width=width, label="Paired R@1")
    plt.bar(x + width / 2, kb_r1, width=width, label="KB R@1")
    plt.xticks(x, labels, rotation=20)
    plt.ylim(0.0, max(max(kb_r1), max(paired_r1)) * 1.2)
    plt.ylabel("Recall@1")
    plt.title("Encoder Quantization Accuracy")
    plt.legend()
    plt.tight_layout()
    path1 = os.path.join(out_dir, "encoder_quantization_accuracy.png")
    plt.savefig(path1, dpi=160)
    plt.close()
    created.append(path1)

    plt.figure(figsize=(10, 5))
    plt.bar(labels, per_image_ms)
    plt.ylabel("Per-image latency (ms)")
    plt.title("Encoder Quantization Latency")
    plt.xticks(rotation=20)
    plt.tight_layout()
    path2 = os.path.join(out_dir, "encoder_quantization_latency.png")
    plt.savefig(path2, dpi=160)
    plt.close()
    created.append(path2)

    plt.figure(figsize=(10, 5))
    plt.bar(labels, images_per_second)
    plt.ylabel("Images / second")
    plt.title("Encoder Quantization Throughput")
    plt.xticks(rotation=20)
    plt.tight_layout()
    path3 = os.path.join(out_dir, "encoder_quantization_throughput.png")
    plt.savefig(path3, dpi=160)
    plt.close()
    created.append(path3)

    return created


def main():
    parser = argparse.ArgumentParser(description="Benchmark image encoder quantization for multimodal entity linking.")
    parser.add_argument("--valid-metadata", default=DEFAULT_VALID_META)
    parser.add_argument("--image-dir", default=DEFAULT_IMAGE_DIR)
    parser.add_argument("--paired-text-embeddings", default=DEFAULT_PAIRED_TEXT_EMB)
    parser.add_argument("--kb-text-embeddings", default=DEFAULT_KB_TEXT_EMB)
    parser.add_argument("--entity-kb", default=DEFAULT_ENTITY_KB)
    parser.add_argument("--model-name", default="ViT-B-32")
    parser.add_argument("--pretrained", default="openai")
    parser.add_argument(
        "--modes",
        nargs="+",
        default=["gpu_fp32", "gpu_fp16", "cpu_fp32", "cpu_int8_dynamic"],
    )
    parser.add_argument("--max-queries", type=int, default=0, help="Optional cap on the number of valid query images.")
    parser.add_argument("--out-json", default="")
    parser.add_argument("--plot-dir", default="")
    args = parser.parse_args()

    valid_meta = load_json(args.valid_metadata)
    paired_text_embeddings = np.load(args.paired_text_embeddings).astype(np.float32)
    kb_text_embeddings = np.load(args.kb_text_embeddings).astype(np.float32)
    entity_kb = load_json(args.entity_kb)

    if args.max_queries > 0:
        valid_meta = valid_meta[: args.max_queries]
        paired_text_embeddings = paired_text_embeddings[: args.max_queries]

    if len(valid_meta) != paired_text_embeddings.shape[0]:
        raise ValueError("valid_metadata and paired_text_embeddings length mismatch.")

    kb_keep_positions, kb_gold = kb_gold_indices(valid_meta, entity_kb)

    results = {
        "environment": collect_environment(),
        "config": {
            "valid_metadata": args.valid_metadata,
            "image_dir": args.image_dir,
            "paired_text_embeddings": args.paired_text_embeddings,
            "kb_text_embeddings": args.kb_text_embeddings,
            "entity_kb": args.entity_kb,
            "model_name": args.model_name,
            "pretrained": args.pretrained,
            "modes": args.modes,
            "max_queries": args.max_queries,
        },
        "dataset": {
            "paired_queries": len(valid_meta),
            "kb_queries": int(kb_keep_positions.shape[0]),
            "kb_entities": len(entity_kb),
        },
        "modes": {},
    }

    for mode in args.modes:
        embeddings, encoding_meta = encode_images_for_mode(
            mode=mode,
            model_name=args.model_name,
            pretrained=args.pretrained,
            valid_meta=valid_meta,
            image_dir=args.image_dir,
        )
        retrieval = evaluate_mode(
            image_embeddings=embeddings,
            paired_text_embeddings=paired_text_embeddings,
            kb_text_embeddings=kb_text_embeddings,
            kb_keep_positions=kb_keep_positions,
            kb_gold_indices=kb_gold,
        )
        results["modes"][mode] = {
            "encoding": encoding_meta,
            "retrieval": retrieval,
        }
        print(
            f"{mode}: paired R@1={retrieval['paired']['recall@1']:.4f}, "
            f"kb R@1={retrieval['kb']['recall@1']:.4f}, "
            f"encode={encoding_meta['images_per_second']:.2f} img/s"
        )

    if args.plot_dir:
        plot_paths = create_plots(results, args.plot_dir)
        results["plots"] = plot_paths

    if args.out_json:
        out_path = Path(args.out_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(f"Saved results to {out_path}")


if __name__ == "__main__":
    main()
