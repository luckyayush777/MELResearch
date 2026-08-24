"""Development-only confidence diagnostics for the frozen compact checkpoint.

Computes fused, text-only, and vision-only full-KB retrieval, plus a horizontal
flip stability check.  It deliberately never opens the frozen test split.
"""

import argparse
import gc
import json
import sys
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from pytorch_lightning.utilities import move_data_to_device

FAST_MEL_ROOT = Path(__file__).resolve().parent / "fastmel"
sys.path.insert(0, str(FAST_MEL_ROOT))
from codes.model.lightning import Lightning, cosine_similarity
from codes.utils.dataset import DataModuleForMIMIC


def encode(model, batch, mode):
    batch = {key: value.clone() if torch.is_tensor(value) else value for key, value in batch.items()}
    if mode == "text":
        batch["empty_img_flag"] = torch.ones_like(batch["empty_img_flag"], dtype=torch.bool)
    elif mode == "vision":
        batch["attention_mask"] = torch.zeros_like(batch["attention_mask"])
    elif mode == "flip":
        batch["pixel_values"] = torch.flip(batch["pixel_values"], dims=[3])
    empty_img_flag = batch.pop("empty_img_flag")
    return model.encoder(**batch, empty_img_flag=empty_img_flag)


@torch.inference_mode()
def make_cache(model, loader, device, mode):
    cache = None
    offset = 0
    for batch in loader:
        vectors = encode(model, move_data_to_device(batch, device), mode).cpu()
        if cache is None:
            cache = torch.empty((len(loader.dataset), vectors.shape[1]), dtype=vectors.dtype)
        cache[offset:offset + len(vectors)] = vectors
        offset += len(vectors)
        del vectors
    return cache


@torch.inference_mode()
def score_queries(model, loader, cache, device, mode, topk):
    cache = cache.to(device)
    all_top_scores, all_top_ids, all_gold_scores, all_answers = [], [], [], []
    for batch in loader:
        batch = move_data_to_device(batch, device)
        answers = batch.pop("answer")
        batch.pop("candidates", None)
        query = encode(model, batch, mode)
        scores = cosine_similarity(query, cache)
        values, ids = scores.topk(topk, dim=1)
        all_top_scores.append(values.cpu())
        all_top_ids.append(ids.cpu())
        all_gold_scores.append(scores.gather(1, answers[:, None]).squeeze(1).cpu())
        all_answers.append(answers.cpu())
    return tuple(torch.cat(x).numpy() for x in (all_top_scores, all_top_ids, all_gold_scores, all_answers))


def distribution(scores):
    # Softmax over the retrieved top-K only: a shape statistic, not calibrated probability.
    probs = torch.softmax(torch.tensor(scores, dtype=torch.float64), dim=1).numpy()
    entropy = -(probs * np.log(np.clip(probs, 1e-12, None))).sum(axis=1)
    return {
        "top1_minus_top2": (scores[:, 0] - scores[:, 1]).tolist(),
        "top1_minus_topk": (scores[:, 0] - scores[:, -1]).tolist(),
        "topk_score_std": scores.std(axis=1).tolist(),
        "topk_normalized_entropy": (entropy / np.log(scores.shape[1])).tolist(),
        "top1_topk_softmax_probability": probs[:, 0].tolist(),
    }


def summary(values):
    values = np.asarray(values, dtype=float)
    return {"mean": float(values.mean()), "median": float(np.median(values)),
            "p10": float(np.quantile(values, .1)), "p90": float(np.quantile(values, .9))}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="fastmel/config/wikimel_clean_sanitized_visual_fp32_debug.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--topk", type=int, default=20)
    parser.add_argument("--out", default="experiments/wikimel_clean_compact_vs_mimic_v1/compact_dev_confidence_seed43.json")
    args = parser.parse_args()
    if args.topk < 2:
        raise ValueError("--topk must be at least 2")

    config = OmegaConf.load(args.config)
    data = DataModuleForMIMIC(config)
    # This is a dev-only evaluator.  Keeping pre-tokenized training/test rows and
    # raw KB dictionaries serves no purpose here and pushes a Windows process
    # over its memory limit while constructing the full entity cache.
    del data.train_data, data.test_data, data.raw_kb_entity, data.kb_id2entity
    gc.collect()
    model = Lightning(config)
    state = torch.load(args.checkpoint, map_location="cpu")["state_dict"]
    model.load_state_dict(state, strict=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()

    # Retaining three 109,976-vector fp32 caches at once is unnecessary and can
    # exceed host memory on Windows. Build, score, and release each cache.
    results = {}
    for mode in ("fused", "text", "vision"):
        print(f"Building {mode} entity cache", flush=True)
        cache = make_cache(model, data.entity_dataloader(), device, mode)
        print(f"Scoring {mode} development queries", flush=True)
        results[mode] = score_queries(model, data.val_dataloader(), cache, device, mode, args.topk)
        del cache
        gc.collect()
        torch.cuda.empty_cache()
    print("Building fused entity cache for flip stability", flush=True)
    cache = make_cache(model, data.entity_dataloader(), device, "fused")
    print("Scoring horizontally flipped development queries", flush=True)
    results["flip"] = score_queries(model, data.val_dataloader(), cache, device, "flip", args.topk)
    entity_count = len(cache)
    del cache
    gc.collect()
    torch.cuda.empty_cache()
    fused_scores, fused_ids, fused_gold, answers = results["fused"]
    text_scores, text_ids, text_gold, _ = results["text"]
    vision_scores, vision_ids, vision_gold, _ = results["vision"]
    flip_scores, flip_ids, flip_gold, _ = results["flip"]

    agreement = {
        "fused_text_top1": (fused_ids[:, 0] == text_ids[:, 0]),
        "fused_vision_top1": (fused_ids[:, 0] == vision_ids[:, 0]),
        "text_vision_top1": (text_ids[:, 0] == vision_ids[:, 0]),
        "all_three_top1": (fused_ids[:, 0] == text_ids[:, 0]) & (fused_ids[:, 0] == vision_ids[:, 0]),
    }
    stable = fused_ids[:, 0] == flip_ids[:, 0]
    payload = {
        "schema_version": 1,
        "split": "development_only",
        "checkpoint": str(Path(args.checkpoint)),
        "queries": int(len(answers)), "entities": int(entity_count), "topk": args.topk,
        "definitions": {
            "text": "image tokens masked for query and entities",
            "vision": "text tokens masked for query and entities",
            "flip_stability": "fused entity cache; query image horizontally flipped",
            "topk_distribution": "softmax/entropy over top-K retrieved scores only; not a calibrated probability",
        },
        "aggregate": {
            "fused_top1_accuracy": float((fused_ids[:, 0] == answers).mean()),
            "text_top1_accuracy": float((text_ids[:, 0] == answers).mean()),
            "vision_top1_accuracy": float((vision_ids[:, 0] == answers).mean()),
            "flip_top1_accuracy": float((flip_ids[:, 0] == answers).mean()),
            "agreement_rates": {key: float(value.mean()) for key, value in agreement.items()},
            "flip_top1_stability": float(stable.mean()),
            "fused_margin_summary": {key: summary(value) for key, value in distribution(fused_scores).items()},
            "gold_in_topk": {mode: float((ids == answers[:, None]).any(axis=1).mean())
                             for mode, (_, ids, _, _) in results.items()},
        },
        "per_query": {
            "answer_index": answers.tolist(),
            "fused_top_ids": fused_ids.tolist(), "text_top_ids": text_ids.tolist(), "vision_top_ids": vision_ids.tolist(), "flip_top_ids": flip_ids.tolist(),
            "fused_top_scores": fused_scores.tolist(), "text_top_scores": text_scores.tolist(), "vision_top_scores": vision_scores.tolist(), "flip_top_scores": flip_scores.tolist(),
            "fused_gold_score": fused_gold.tolist(), "text_gold_score": text_gold.tolist(), "vision_gold_score": vision_gold.tolist(), "flip_gold_score": flip_gold.tolist(),
            "fused_distribution": distribution(fused_scores),
            "fused_text_top1_agree": agreement["fused_text_top1"].tolist(),
            "fused_vision_top1_agree": agreement["fused_vision_top1"].tolist(),
            "text_vision_top1_agree": agreement["text_vision_top1"].tolist(),
            "all_modalities_top1_agree": agreement["all_three_top1"].tolist(),
            "flip_top1_stable": stable.tolist(),
        },
    }
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload["aggregate"], indent=2))
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
