import argparse
import json
import os
import platform
import sys
import time
from typing import Dict, List, Tuple

import numpy as np

try:
    import psutil
except ImportError:
    psutil = None


DEFAULT_IMAGE_EMB = os.path.join("data", "embeddings", "image_embeddings.npy")
DEFAULT_TEXT_EMB = os.path.join("data", "embeddings", "text_embeddings.npy")
DEFAULT_VALID_META = os.path.join("data", "embeddings", "valid_metadata.json")
DEFAULT_ENTITY_KB = os.path.join("data", "entity_kb.json")


def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def quantize_fp16(x: np.ndarray) -> Tuple[np.ndarray, Dict[str, float]]:
    q = x.astype(np.float16).astype(np.float32)
    return q, {"storage_bytes": x.size * np.dtype(np.float16).itemsize}


def quantize_int8_rowwise(x: np.ndarray) -> Tuple[np.ndarray, Dict[str, float]]:
    # Row-wise symmetric quantization is simple and works well for normalized embeddings.
    max_abs = np.max(np.abs(x), axis=1, keepdims=True)
    scales = np.maximum(max_abs / 127.0, 1e-12).astype(np.float32)
    q = np.clip(np.round(x / scales), -127, 127).astype(np.int8)
    deq = (q.astype(np.float32) * scales).astype(np.float32)
    storage_bytes = q.size * np.dtype(np.int8).itemsize + scales.size * np.dtype(np.float32).itemsize
    return deq, {"storage_bytes": int(storage_bytes)}


def normalize_rows(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    return x / norms


def apply_quantization(x: np.ndarray, mode: str) -> Tuple[np.ndarray, Dict[str, float]]:
    if mode == "fp32":
        return x.astype(np.float32, copy=False), {"storage_bytes": int(x.size * np.dtype(np.float32).itemsize)}
    if mode == "fp16":
        return quantize_fp16(x)
    if mode == "int8":
        return quantize_int8_rowwise(x)
    raise ValueError(f"Unsupported quantization mode: {mode}")


def compute_ranks(sim: np.ndarray, gold_indices: np.ndarray) -> np.ndarray:
    gold_scores = sim[np.arange(sim.shape[0]), gold_indices][:, None]
    # Rank = 1 + number of candidates with a strictly higher score.
    return 1 + (sim > gold_scores).sum(axis=1)


def retrieval_metrics(ranks: np.ndarray) -> Dict[str, float]:
    ranks = ranks.astype(np.int64)
    return {
        "queries": int(ranks.shape[0]),
        "recall@1": float(np.mean(ranks <= 1)),
        "recall@5": float(np.mean(ranks <= 5)),
        "recall@10": float(np.mean(ranks <= 10)),
        "recall@20": float(np.mean(ranks <= 20)),
        "recall@50": float(np.mean(ranks <= 50)),
        "recall@100": float(np.mean(ranks <= 100)),
        "mrr": float(np.mean(1.0 / ranks)),
        "median_rank": float(np.median(ranks)),
        "mean_rank": float(np.mean(ranks)),
    }


def format_bytes(n_bytes: int) -> str:
    units = ["B", "KB", "MB", "GB"]
    size = float(n_bytes)
    for unit in units:
        if size < 1024 or unit == units[-1]:
            return f"{size:.2f} {unit}"
        size /= 1024.0
    return f"{n_bytes} B"


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


def subset_gold_indices(valid_meta: List[Dict[str, str]]) -> np.ndarray:
    return np.arange(len(valid_meta), dtype=np.int64)


def kb_gold_indices(valid_meta: List[Dict[str, str]], entity_kb: List[Dict[str, str]]) -> Tuple[np.ndarray, List[Dict[str, str]]]:
    entity_to_index = {}
    title_to_index = {}
    for idx, row in enumerate(entity_kb):
        entity_id = row.get("entity_id", "")
        title = row.get("title", "")
        if entity_id:
            entity_to_index[entity_id] = idx
        if title:
            title_to_index[title] = idx
    gold = []
    filtered_meta = []
    for item in valid_meta:
        entity_id = item.get("entity_id", "")
        title = item.get("title", "")
        if entity_id and entity_id in entity_to_index:
            gold.append(entity_to_index[entity_id])
            filtered_meta.append(item)
        elif title in title_to_index:
            gold.append(title_to_index[title])
            filtered_meta.append(item)
    return np.array(gold, dtype=np.int64), filtered_meta


def encode_kb_texts(entity_kb_path: str, model_name: str, pretrained: str, batch_size: int) -> np.ndarray:
    import torch
    import open_clip

    entity_kb = load_json(entity_kb_path)
    texts = []
    for row in entity_kb:
        title = row.get("title", "")
        desc = row.get("description", "")[:200]
        texts.append(f"{title}. {desc}" if desc else title)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _, _ = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
    tokenizer = open_clip.get_tokenizer(model_name)
    model = model.to(device).eval()

    outputs = []
    start_time = time.perf_counter()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        tokens = tokenizer(batch).to(device)
        with torch.no_grad():
            feats = model.encode_text(tokens)
            feats = feats / feats.norm(dim=-1, keepdim=True)
        outputs.append(feats.cpu().float().numpy())
    embeddings = np.vstack(outputs)
    metadata = {
        "encoding_time_seconds": time.perf_counter() - start_time,
        "rows": len(texts),
        "embedding_dim": int(embeddings.shape[1]),
    }
    if torch.cuda.is_available():
        metadata["gpu_peak_memory_mb"] = bytes_to_mb(int(torch.cuda.max_memory_allocated()))
    return embeddings, metadata


def profile_retrieval(
    image_embeddings: np.ndarray,
    candidate_text_embeddings: np.ndarray,
    gold_indices: np.ndarray,
    repeats: int,
    warmup: int,
) -> Tuple[np.ndarray, Dict[str, float]]:
    sim = None
    ranks = None

    for _ in range(max(0, warmup)):
        sim = image_embeddings @ candidate_text_embeddings.T
        ranks = compute_ranks(sim, gold_indices)

    total_latencies_ms = []
    similarity_latencies_ms = []
    ranking_latencies_ms = []
    peak_rss_bytes = 0
    peak_wset_bytes = 0

    for _ in range(max(1, repeats)):
        before = process_memory_snapshot()
        t0 = time.perf_counter()
        sim = image_embeddings @ candidate_text_embeddings.T
        t1 = time.perf_counter()
        ranks = compute_ranks(sim, gold_indices)
        t2 = time.perf_counter()
        after = process_memory_snapshot()

        total_latencies_ms.append((t2 - t0) * 1000.0)
        similarity_latencies_ms.append((t1 - t0) * 1000.0)
        ranking_latencies_ms.append((t2 - t1) * 1000.0)
        peak_rss_bytes = max(
            peak_rss_bytes,
            int(before.get("rss_bytes", 0)),
            int(after.get("rss_bytes", 0)),
        )
        peak_wset_bytes = max(
            peak_wset_bytes,
            int(before.get("peak_wset_bytes", 0)),
            int(after.get("peak_wset_bytes", 0)),
        )

    assert sim is not None and ranks is not None
    avg_total_ms = float(np.mean(total_latencies_ms))
    query_count = int(image_embeddings.shape[0])

    profile = {
        "latency_ms": avg_total_ms,
        "similarity_latency_ms": float(np.mean(similarity_latencies_ms)),
        "ranking_latency_ms": float(np.mean(ranking_latencies_ms)),
        "latency_ms_std": float(np.std(total_latencies_ms)),
        "per_query_latency_ms": avg_total_ms / max(query_count, 1),
        "queries_per_second": float(query_count / max(avg_total_ms / 1000.0, 1e-12)),
        "similarity_matrix_bytes": int(sim.nbytes),
        "similarity_matrix_mb": bytes_to_mb(int(sim.nbytes)),
        "peak_rss_mb": bytes_to_mb(peak_rss_bytes) if peak_rss_bytes else None,
        "peak_working_set_mb": bytes_to_mb(peak_wset_bytes) if peak_wset_bytes else None,
        "repeats": int(max(1, repeats)),
        "warmup": int(max(0, warmup)),
    }
    return ranks, profile


def run_benchmark(
    image_embeddings: np.ndarray,
    candidate_text_embeddings: np.ndarray,
    gold_indices: np.ndarray,
    quant_modes: List[str],
    repeats: int,
    warmup: int,
) -> Dict[str, Dict[str, float]]:
    results = {}

    for mode in quant_modes:
        q_img, img_info = apply_quantization(image_embeddings, mode)
        q_txt, txt_info = apply_quantization(candidate_text_embeddings, mode)
        q_img = normalize_rows(q_img)
        q_txt = normalize_rows(q_txt)

        ranks, perf = profile_retrieval(
            image_embeddings=q_img,
            candidate_text_embeddings=q_txt,
            gold_indices=gold_indices,
            repeats=repeats,
            warmup=warmup,
        )
        metrics = retrieval_metrics(ranks)
        metrics["image_storage"] = format_bytes(int(img_info["storage_bytes"]))
        metrics["text_storage"] = format_bytes(int(txt_info["storage_bytes"]))
        metrics["image_storage_mb"] = bytes_to_mb(int(img_info["storage_bytes"]))
        metrics["text_storage_mb"] = bytes_to_mb(int(txt_info["storage_bytes"]))
        metrics.update(perf)
        results[mode] = metrics

    return results


def print_result_table(title: str, results: Dict[str, Dict[str, float]]) -> None:
    print(f"\n{title}")
    print("-" * len(title))
    header = (
        f"{'mode':<6} {'R@1':>8} {'R@10':>8} {'R@100':>8} {'MRR':>8} "
        f"{'MedR':>8} {'Lat(ms)':>10} {'QPS':>10} {'ImgMem':>12} {'TxtMem':>12}"
    )
    print(header)
    for mode, row in results.items():
        print(
            f"{mode:<6} "
            f"{row['recall@1']:>8.4f} "
            f"{row['recall@10']:>8.4f} "
            f"{row['recall@100']:>8.4f} "
            f"{row['mrr']:>8.4f} "
            f"{row['median_rank']:>8.1f} "
            f"{row['latency_ms']:>10.2f} "
            f"{row['queries_per_second']:>10.1f} "
            f"{row['image_storage']:>12} "
            f"{row['text_storage']:>12}"
        )


def main():
    parser = argparse.ArgumentParser(description="Benchmark embedding quantization for multimodal entity linking.")
    parser.add_argument("--image-embeddings", default=DEFAULT_IMAGE_EMB)
    parser.add_argument("--text-embeddings", default=DEFAULT_TEXT_EMB)
    parser.add_argument("--valid-metadata", default=DEFAULT_VALID_META)
    parser.add_argument("--entity-kb", default=DEFAULT_ENTITY_KB)
    parser.add_argument("--quant-modes", nargs="+", default=["fp32", "fp16", "int8"])
    parser.add_argument("--skip-paired", action="store_true", help="Skip the paired image-to-text benchmark.")
    parser.add_argument("--skip-kb", action="store_true", help="Skip the full-KB entity-linking benchmark.")
    parser.add_argument("--kb-text-embeddings", default="", help="Optional .npy file with precomputed KB text embeddings.")
    parser.add_argument("--save-kb-text-embeddings", default="", help="Optional path to save computed KB text embeddings.")
    parser.add_argument("--out-json", default="", help="Optional path to save benchmark results as JSON.")
    parser.add_argument("--repeats", type=int, default=5, help="Number of timed retrieval repeats per quantization mode.")
    parser.add_argument("--warmup", type=int, default=1, help="Number of untimed warmup retrieval runs per quantization mode.")
    parser.add_argument("--model-name", default="ViT-B-32")
    parser.add_argument("--pretrained", default="openai")
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()

    valid_meta = load_json(args.valid_metadata)
    image_embeddings = np.load(args.image_embeddings).astype(np.float32)

    if len(valid_meta) != image_embeddings.shape[0]:
        raise ValueError(
            f"Length mismatch: valid_metadata has {len(valid_meta)} rows but image embeddings has {image_embeddings.shape[0]}"
        )

    all_results = {
        "environment": collect_environment(),
        "artifacts": {
            "image_embeddings": args.image_embeddings,
            "paired_text_embeddings": args.text_embeddings,
            "valid_metadata": args.valid_metadata,
            "entity_kb": args.entity_kb,
            "kb_text_embeddings": args.kb_text_embeddings or args.save_kb_text_embeddings or "",
        },
        "profiling": {
            "repeats": args.repeats,
            "warmup": args.warmup,
        },
    }

    if not args.skip_paired:
        paired_text_embeddings = np.load(args.text_embeddings).astype(np.float32)
        if paired_text_embeddings.shape[0] != image_embeddings.shape[0]:
            raise ValueError("Image and paired text embeddings must have the same number of rows.")
        paired_results = run_benchmark(
            image_embeddings=image_embeddings,
            candidate_text_embeddings=paired_text_embeddings,
            gold_indices=subset_gold_indices(valid_meta),
            quant_modes=args.quant_modes,
            repeats=args.repeats,
            warmup=args.warmup,
        )
        all_results["paired"] = paired_results
        print_result_table("Paired Retrieval Benchmark", paired_results)

    if not args.skip_kb:
        entity_kb = load_json(args.entity_kb)
        gold_indices, filtered_meta = kb_gold_indices(valid_meta, entity_kb)
        kb_titles = {row["title"] for row in entity_kb}
        kb_entity_ids = {row.get("entity_id", "") for row in entity_kb}
        keep_mask = np.array(
            [
                (item.get("entity_id", "") in kb_entity_ids) or (item["title"] in kb_titles)
                for item in valid_meta
            ],
            dtype=bool,
        )
        filtered_images = image_embeddings[keep_mask]

        if args.kb_text_embeddings:
            kb_text_embeddings = np.load(args.kb_text_embeddings).astype(np.float32)
            kb_encoding_meta = {"source": "precomputed", "path": args.kb_text_embeddings}
        else:
            kb_text_embeddings, kb_encoding_meta = encode_kb_texts(
                entity_kb_path=args.entity_kb,
                model_name=args.model_name,
                pretrained=args.pretrained,
                batch_size=args.batch_size,
            )
            kb_text_embeddings = kb_text_embeddings.astype(np.float32)
            if args.save_kb_text_embeddings:
                np.save(args.save_kb_text_embeddings, kb_text_embeddings)
                kb_encoding_meta["saved_to"] = args.save_kb_text_embeddings

        if kb_text_embeddings.shape[0] != len(entity_kb):
            raise ValueError("KB text embeddings row count does not match entity_kb length.")
        if filtered_images.shape[0] != gold_indices.shape[0]:
            raise ValueError("Filtered image count does not match gold index count.")

        kb_results = run_benchmark(
            image_embeddings=filtered_images,
            candidate_text_embeddings=kb_text_embeddings,
            gold_indices=gold_indices,
            quant_modes=args.quant_modes,
            repeats=args.repeats,
            warmup=args.warmup,
        )
        all_results["kb"] = kb_results
        all_results["kb_metadata"] = {
            "entity_count": len(entity_kb),
            "query_count": len(filtered_meta),
            "missing_query_titles": len(valid_meta) - len(filtered_meta),
            "encoding": kb_encoding_meta,
        }
        print(f"\nKB queries evaluated: {len(filtered_meta)} / {len(valid_meta)}")
        print_result_table("Full-KB Entity Linking Benchmark", kb_results)

    if args.out_json:
        with open(args.out_json, "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=2)
        print(f"\nSaved benchmark results to {args.out_json}")


if __name__ == "__main__":
    main()
