"""
Geometry-aware quantization experiment for multimodal entity linking.

This module uses paired CLIP text/image embeddings and measures how
quantization changes:
- cross-modal retrieval quality
- local neighborhood structure
- pairwise cosine geometry
- per-point drift in a shared 2D projection

The resulting plots are meant to make geometric distortion visible, not just
report top-line accuracy.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from quantization_utils import fake_quantize

try:
    import matplotlib

    matplotlib.use("Agg")
except ImportError:
    matplotlib = None

logger = logging.getLogger(__name__)

CONDITIONS: Tuple[str, ...] = ("text_only", "image_only", "both")


@dataclass
class GeometryExperimentConfig:
    text_embeddings_path: str
    image_embeddings_path: str
    metadata_path: str
    output_dir: str
    bit_widths: List[int]
    subset_size: Optional[int] = 1000
    seed: int = 42
    neighbor_k: int = 10
    pair_sample_size: int = 20000
    drift_plot_size: int = 200
    drift_plot_bits: int = 4
    drift_plot_condition: str = "both"
    renormalize: bool = True
    symmetric: bool = True


def normalize_rows(embeddings: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.clip(norms, 1e-12, None)
    return embeddings / norms


def quantize_embeddings(
    embeddings: np.ndarray,
    bits: int,
    symmetric: bool = True,
    renormalize: bool = True,
) -> Tuple[np.ndarray, Dict[str, float]]:
    tensor = torch.from_numpy(embeddings.astype(np.float32, copy=False))
    quantized = fake_quantize(tensor, bits=bits, symmetric=symmetric).cpu().numpy().astype(np.float32)

    baseline_norms = np.linalg.norm(embeddings, axis=1)
    raw_norms = np.linalg.norm(quantized, axis=1)
    if renormalize:
        quantized = normalize_rows(quantized)

    metrics = {
        "mean_abs_norm_error": float(np.mean(np.abs(raw_norms - baseline_norms))),
        "max_abs_norm_error": float(np.max(np.abs(raw_norms - baseline_norms))),
        "mean_component_abs_error": float(np.mean(np.abs(embeddings - quantized))),
    }
    return quantized, metrics


def _safe_corrcoef(x: np.ndarray, y: np.ndarray) -> float:
    if x.size == 0 or y.size == 0:
        return 0.0
    x_std = float(np.std(x))
    y_std = float(np.std(y))
    if x_std < 1e-12 or y_std < 1e-12:
        return 1.0
    return float(np.corrcoef(x, y)[0, 1])


def sample_index_pairs(
    n_items: int,
    sample_size: int,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    if n_items <= 1:
        return np.array([], dtype=np.int64), np.array([], dtype=np.int64)

    idx_a = rng.integers(0, n_items, size=sample_size)
    idx_b = rng.integers(0, n_items, size=sample_size)
    valid = idx_a != idx_b
    return idx_a[valid], idx_b[valid]


def compute_retrieval_metrics(similarities: np.ndarray) -> Dict[str, float]:
    if similarities.size == 0:
        return {
            "recall@1": 0.0,
            "recall@5": 0.0,
            "recall@10": 0.0,
            "mrr": 0.0,
            "mean_rank": 0.0,
            "median_rank": 0.0,
            "diag_similarity_mean": 0.0,
            "diag_similarity_std": 0.0,
            "margin_mean": 0.0,
        }

    diagonal = np.diag(similarities)
    ranks = 1 + (similarities > diagonal[:, None]).sum(axis=1)

    hardest_negative = similarities.copy()
    np.fill_diagonal(hardest_negative, -np.inf)
    margins = diagonal - hardest_negative.max(axis=1)

    return {
        "recall@1": float(np.mean(ranks <= 1)),
        "recall@5": float(np.mean(ranks <= 5)),
        "recall@10": float(np.mean(ranks <= 10)),
        "mrr": float(np.mean(1.0 / ranks)),
        "mean_rank": float(np.mean(ranks)),
        "median_rank": float(np.median(ranks)),
        "diag_similarity_mean": float(np.mean(diagonal)),
        "diag_similarity_std": float(np.std(diagonal)),
        "margin_mean": float(np.mean(margins)),
    }


def compute_rank_change_metrics(
    baseline_similarities: np.ndarray,
    quantized_similarities: np.ndarray,
) -> Dict[str, float]:
    base_diag = np.diag(baseline_similarities)
    quant_diag = np.diag(quantized_similarities)

    base_ranks = 1 + (baseline_similarities > base_diag[:, None]).sum(axis=1)
    quant_ranks = 1 + (quantized_similarities > quant_diag[:, None]).sum(axis=1)

    return {
        "mean_rank_delta": float(np.mean(quant_ranks - base_ranks)),
        "mean_abs_rank_delta": float(np.mean(np.abs(quant_ranks - base_ranks))),
        "top1_flip_rate": float(np.mean((base_ranks == 1) != (quant_ranks == 1))),
        "top10_exit_rate": float(np.mean((base_ranks <= 10) & (quant_ranks > 10))),
    }


def compute_neighbor_overlap(
    baseline_embeddings: np.ndarray,
    quantized_embeddings: np.ndarray,
    neighbor_k: int,
) -> float:
    if baseline_embeddings.shape[0] <= 1:
        return 1.0

    k = min(neighbor_k, baseline_embeddings.shape[0] - 1)
    base_sims = baseline_embeddings @ baseline_embeddings.T
    quant_sims = quantized_embeddings @ quantized_embeddings.T

    np.fill_diagonal(base_sims, -np.inf)
    np.fill_diagonal(quant_sims, -np.inf)

    base_neighbors = np.argpartition(-base_sims, kth=np.arange(k), axis=1)[:, :k]
    quant_neighbors = np.argpartition(-quant_sims, kth=np.arange(k), axis=1)[:, :k]

    overlaps = []
    for base_row, quant_row in zip(base_neighbors, quant_neighbors):
        overlaps.append(len(set(base_row.tolist()).intersection(set(quant_row.tolist()))) / k)
    return float(np.mean(overlaps))


def compute_geometry_metrics(
    baseline_embeddings: np.ndarray,
    quantized_embeddings: np.ndarray,
    neighbor_k: int,
    pair_sample_size: int,
    rng: np.random.Generator,
) -> Dict[str, float]:
    self_alignment = np.sum(baseline_embeddings * quantized_embeddings, axis=1)

    idx_a, idx_b = sample_index_pairs(
        n_items=baseline_embeddings.shape[0],
        sample_size=pair_sample_size,
        rng=rng,
    )
    baseline_pairs = np.sum(baseline_embeddings[idx_a] * baseline_embeddings[idx_b], axis=1)
    quantized_pairs = np.sum(quantized_embeddings[idx_a] * quantized_embeddings[idx_b], axis=1)

    return {
        "self_alignment_mean": float(np.mean(self_alignment)),
        "self_alignment_p10": float(np.percentile(self_alignment, 10)),
        "self_alignment_min": float(np.min(self_alignment)),
        "neighbor_overlap@k": compute_neighbor_overlap(
            baseline_embeddings,
            quantized_embeddings,
            neighbor_k=neighbor_k,
        ),
        "pairwise_cosine_corr": _safe_corrcoef(baseline_pairs, quantized_pairs),
        "pairwise_cosine_mae": float(np.mean(np.abs(baseline_pairs - quantized_pairs))),
    }


def compute_cross_modal_geometry(
    baseline_text: np.ndarray,
    baseline_image: np.ndarray,
    quantized_text: np.ndarray,
    quantized_image: np.ndarray,
    pair_sample_size: int,
    rng: np.random.Generator,
) -> Dict[str, float]:
    sample_size = min(pair_sample_size, baseline_text.shape[0] * baseline_image.shape[0])
    idx_img = rng.integers(0, baseline_image.shape[0], size=sample_size)
    idx_txt = rng.integers(0, baseline_text.shape[0], size=sample_size)

    baseline_cross = np.sum(baseline_image[idx_img] * baseline_text[idx_txt], axis=1)
    quantized_cross = np.sum(quantized_image[idx_img] * quantized_text[idx_txt], axis=1)

    diagonal_baseline = np.sum(baseline_image * baseline_text, axis=1)
    diagonal_quantized = np.sum(quantized_image * quantized_text, axis=1)

    return {
        "cross_modal_pairwise_corr": _safe_corrcoef(baseline_cross, quantized_cross),
        "cross_modal_pairwise_mae": float(np.mean(np.abs(baseline_cross - quantized_cross))),
        "paired_similarity_delta_mean": float(np.mean(diagonal_quantized - diagonal_baseline)),
        "paired_similarity_delta_abs_mean": float(np.mean(np.abs(diagonal_quantized - diagonal_baseline))),
    }


def _project_pca(
    baseline_text: np.ndarray,
    baseline_image: np.ndarray,
    quantized_text: np.ndarray,
    quantized_image: np.ndarray,
) -> Dict[str, np.ndarray]:
    baseline_stack = np.concatenate([baseline_text, baseline_image], axis=0)
    mean = baseline_stack.mean(axis=0, keepdims=True)
    centered = baseline_stack - mean

    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    basis = vt[:2].T

    def project(x: np.ndarray) -> np.ndarray:
        return (x - mean) @ basis

    return {
        "baseline_text": project(baseline_text),
        "baseline_image": project(baseline_image),
        "quantized_text": project(quantized_text),
        "quantized_image": project(quantized_image),
    }


def _ensure_output_dir(output_dir: str) -> Path:
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_geometry_inputs(
    text_embeddings_path: str,
    image_embeddings_path: str,
    metadata_path: str,
    subset_size: Optional[int],
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, List[Dict], np.ndarray]:
    text = np.load(text_embeddings_path).astype(np.float32)
    image = np.load(image_embeddings_path).astype(np.float32)
    with open(metadata_path, "r", encoding="utf-8") as handle:
        metadata = json.load(handle)

    if text.shape != image.shape:
        raise ValueError(f"Text/image shapes do not match: {text.shape} vs {image.shape}")
    if len(metadata) != text.shape[0]:
        raise ValueError(f"Metadata size {len(metadata)} does not match embedding rows {text.shape[0]}")

    text = normalize_rows(text)
    image = normalize_rows(image)

    indices = np.arange(text.shape[0])
    if subset_size is not None and subset_size < text.shape[0]:
        rng = np.random.default_rng(seed)
        indices = np.sort(rng.choice(indices, size=subset_size, replace=False))
        text = text[indices]
        image = image[indices]
        metadata = [metadata[i] for i in indices.tolist()]

    return text, image, metadata, indices


def collect_flip_examples(
    baseline_similarities: np.ndarray,
    quantized_similarities: np.ndarray,
    metadata: Sequence[Dict],
    top_k: int = 20,
) -> List[Dict[str, object]]:
    baseline_diag = np.diag(baseline_similarities)
    quantized_diag = np.diag(quantized_similarities)

    baseline_ranks = 1 + (baseline_similarities > baseline_diag[:, None]).sum(axis=1)
    quantized_ranks = 1 + (quantized_similarities > quantized_diag[:, None]).sum(axis=1)

    deltas = quantized_ranks - baseline_ranks
    order = np.argsort(-np.abs(deltas))[:top_k]

    examples = []
    for idx in order.tolist():
        item = metadata[idx]
        examples.append(
            {
                "index": int(idx),
                "title": item.get("title", f"item_{idx}"),
                "image_file": item.get("image_file"),
                "baseline_rank": int(baseline_ranks[idx]),
                "quantized_rank": int(quantized_ranks[idx]),
                "rank_delta": int(deltas[idx]),
                "baseline_pair_similarity": float(baseline_diag[idx]),
                "quantized_pair_similarity": float(quantized_diag[idx]),
            }
        )
    return examples


def generate_retrieval_plot(results: Dict, output_dir: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib is not installed; skipping retrieval plot")
        return

    baseline_i2t = results["baseline"]["image_to_text_retrieval"]
    baseline_t2i = results["baseline"]["text_to_image_retrieval"]
    bits = sorted((int(bit) for bit in results["conditions"]["both"].keys()), reverse=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, direction_key, baseline_metrics, title in [
        (axes[0], "image_to_text_retrieval", baseline_i2t, "Image -> Text Retrieval"),
        (axes[1], "text_to_image_retrieval", baseline_t2i, "Text -> Image Retrieval"),
    ]:
        x_values = [32] + bits
        for condition in CONDITIONS:
            y_values = [baseline_metrics["recall@1"]]
            y_values.extend(results["conditions"][condition][str(bit)][direction_key]["recall@1"] for bit in bits)
            ax.plot(x_values, y_values, marker="o", linewidth=2, label=condition.replace("_", " "))

        ax.set_title(title)
        ax.set_xlabel("Bit Width")
        ax.set_ylabel("Recall@1")
        ax.grid(True, alpha=0.3)
        ax.invert_xaxis()
        ax.legend()

    plt.tight_layout()
    plt.savefig(output_dir / "retrieval_vs_bits.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def generate_geometry_plot(results: Dict, output_dir: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib is not installed; skipping geometry plot")
        return

    bits = sorted((int(bit) for bit in results["conditions"]["both"].keys()), reverse=True)
    both_results = results["conditions"]["both"]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    metric_specs = [
        ("self_alignment_mean", "Self Alignment"),
        ("neighbor_overlap@k", "Neighbor Overlap"),
        ("pairwise_cosine_corr", "Pairwise Cosine Corr."),
    ]

    for ax, (metric_name, title) in zip(axes, metric_specs):
        text_values = [both_results[str(bit)]["text_geometry"][metric_name] for bit in bits]
        image_values = [both_results[str(bit)]["image_geometry"][metric_name] for bit in bits]

        ax.plot(bits, text_values, marker="o", linewidth=2, label="text")
        ax.plot(bits, image_values, marker="s", linewidth=2, label="image")
        ax.set_title(title)
        ax.set_xlabel("Bit Width")
        ax.grid(True, alpha=0.3)
        ax.invert_xaxis()
        ax.legend()

    plt.tight_layout()
    plt.savefig(output_dir / "geometry_vs_bits.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def generate_similarity_scatter(
    baseline_text: np.ndarray,
    baseline_image: np.ndarray,
    quantized_text: np.ndarray,
    quantized_image: np.ndarray,
    output_path: Path,
    sample_size: int,
    seed: int,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib is not installed; skipping similarity scatter")
        return

    rng = np.random.default_rng(seed)
    idx_img = rng.integers(0, baseline_image.shape[0], size=sample_size)
    idx_txt = rng.integers(0, baseline_text.shape[0], size=sample_size)

    baseline_scores = np.sum(baseline_image[idx_img] * baseline_text[idx_txt], axis=1)
    quantized_scores = np.sum(quantized_image[idx_img] * quantized_text[idx_txt], axis=1)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(baseline_scores, quantized_scores, s=8, alpha=0.25)
    low = float(min(baseline_scores.min(), quantized_scores.min()))
    high = float(max(baseline_scores.max(), quantized_scores.max()))
    ax.plot([low, high], [low, high], color="black", linestyle="--", linewidth=1)
    ax.set_xlabel("Baseline cross-modal cosine")
    ax.set_ylabel("Quantized cross-modal cosine")
    ax.set_title("Cross-modal Geometry Preservation")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def generate_drift_plot(
    baseline_text: np.ndarray,
    baseline_image: np.ndarray,
    quantized_text: np.ndarray,
    quantized_image: np.ndarray,
    output_path: Path,
    seed: int,
    drift_plot_size: int,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib is not installed; skipping drift plot")
        return

    projections = _project_pca(baseline_text, baseline_image, quantized_text, quantized_image)
    rng = np.random.default_rng(seed)
    sample_size = min(drift_plot_size, baseline_text.shape[0])
    indices = np.sort(rng.choice(np.arange(baseline_text.shape[0]), size=sample_size, replace=False))

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=True, sharey=True)
    panel_specs = [
        (
            axes[0],
            "Text embedding drift",
            projections["baseline_text"],
            projections["quantized_text"],
            "#1f77b4",
        ),
        (
            axes[1],
            "Image embedding drift",
            projections["baseline_image"],
            projections["quantized_image"],
            "#d62728",
        ),
    ]

    for ax, title, base_proj, quant_proj, color in panel_specs:
        ax.scatter(base_proj[:, 0], base_proj[:, 1], s=8, alpha=0.15, color="gray")
        ax.quiver(
            base_proj[indices, 0],
            base_proj[indices, 1],
            quant_proj[indices, 0] - base_proj[indices, 0],
            quant_proj[indices, 1] - base_proj[indices, 1],
            angles="xy",
            scale_units="xy",
            scale=1,
            width=0.003,
            alpha=0.65,
            color=color,
        )
        ax.scatter(quant_proj[indices, 0], quant_proj[indices, 1], s=15, alpha=0.8, color=color)
        ax.set_title(title)
        ax.grid(True, alpha=0.25)
        ax.set_xlabel("PC1")

    axes[0].set_ylabel("PC2")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def generate_summary_report(results: Dict, output_path: Path) -> None:
    baseline_i2t = results["baseline"]["image_to_text_retrieval"]
    baseline_t2i = results["baseline"]["text_to_image_retrieval"]

    lines = [
        "# Quantization Geometry Report",
        "",
        "## Baseline",
        "",
        f"- Image -> Text Recall@1: {baseline_i2t['recall@1']:.4f}",
        f"- Text -> Image Recall@1: {baseline_t2i['recall@1']:.4f}",
        f"- Baseline paired similarity mean: {baseline_i2t['diag_similarity_mean']:.4f}",
        "",
        "## Bit Sweep Summary",
        "",
    ]

    target_bits = "4" if "4" in results["conditions"]["both"] else next(iter(results["conditions"]["both"]))
    for condition in CONDITIONS:
        entry = results["conditions"][condition][target_bits]
        lines.append(
            f"- {condition.replace('_', ' ')} at {target_bits}-bit: "
            f"I->T R@1={entry['image_to_text_retrieval']['recall@1']:.4f}, "
            f"T->I R@1={entry['text_to_image_retrieval']['recall@1']:.4f}, "
            f"text self-align={entry['text_geometry']['self_alignment_mean']:.4f}, "
            f"image self-align={entry['image_geometry']['self_alignment_mean']:.4f}"
        )

    output_path.write_text("\n".join(lines), encoding="utf-8")


def run_geometry_quantization_experiment(config: GeometryExperimentConfig) -> Dict:
    output_dir = _ensure_output_dir(config.output_dir)
    logger.info("Loading paired text/image embeddings for geometry experiment")

    baseline_text, baseline_image, metadata, selected_indices = load_geometry_inputs(
        text_embeddings_path=config.text_embeddings_path,
        image_embeddings_path=config.image_embeddings_path,
        metadata_path=config.metadata_path,
        subset_size=config.subset_size,
        seed=config.seed,
    )

    baseline_i2t = baseline_image @ baseline_text.T
    baseline_t2i = baseline_text @ baseline_image.T

    results: Dict[str, object] = {
        "config": {
            "text_embeddings_path": config.text_embeddings_path,
            "image_embeddings_path": config.image_embeddings_path,
            "metadata_path": config.metadata_path,
            "output_dir": config.output_dir,
            "bit_widths": config.bit_widths,
            "subset_size": config.subset_size,
            "seed": config.seed,
            "neighbor_k": config.neighbor_k,
            "pair_sample_size": config.pair_sample_size,
            "drift_plot_bits": config.drift_plot_bits,
            "drift_plot_condition": config.drift_plot_condition,
            "renormalize": config.renormalize,
            "symmetric": config.symmetric,
        },
        "selected_indices": selected_indices.tolist(),
        "baseline": {
            "image_to_text_retrieval": compute_retrieval_metrics(baseline_i2t),
            "text_to_image_retrieval": compute_retrieval_metrics(baseline_t2i),
        },
        "conditions": {condition: {} for condition in CONDITIONS},
    }

    drift_payload: Optional[Tuple[np.ndarray, np.ndarray, str, int]] = None

    for condition in CONDITIONS:
        logger.info("Running condition: %s", condition)
        for bits in config.bit_widths:
            logger.info("Evaluating %s at %s-bit", condition, bits)
            rng = np.random.default_rng(config.seed + bits + len(condition))

            quant_text = baseline_text.copy()
            quant_image = baseline_image.copy()

            quantization_stats = {
                "text": {
                    "mean_abs_norm_error": 0.0,
                    "max_abs_norm_error": 0.0,
                    "mean_component_abs_error": 0.0,
                },
                "image": {
                    "mean_abs_norm_error": 0.0,
                    "max_abs_norm_error": 0.0,
                    "mean_component_abs_error": 0.0,
                },
            }

            if condition in {"text_only", "both"}:
                quant_text, quantization_stats["text"] = quantize_embeddings(
                    baseline_text,
                    bits=bits,
                    symmetric=config.symmetric,
                    renormalize=config.renormalize,
                )
            if condition in {"image_only", "both"}:
                quant_image, quantization_stats["image"] = quantize_embeddings(
                    baseline_image,
                    bits=bits,
                    symmetric=config.symmetric,
                    renormalize=config.renormalize,
                )

            image_to_text = quant_image @ quant_text.T
            text_to_image = quant_text @ quant_image.T

            condition_result = {
                "bits": bits,
                "condition": condition,
                "quantization_stats": quantization_stats,
                "text_geometry": compute_geometry_metrics(
                    baseline_embeddings=baseline_text,
                    quantized_embeddings=quant_text,
                    neighbor_k=config.neighbor_k,
                    pair_sample_size=config.pair_sample_size,
                    rng=rng,
                ),
                "image_geometry": compute_geometry_metrics(
                    baseline_embeddings=baseline_image,
                    quantized_embeddings=quant_image,
                    neighbor_k=config.neighbor_k,
                    pair_sample_size=config.pair_sample_size,
                    rng=rng,
                ),
                "cross_modal_geometry": compute_cross_modal_geometry(
                    baseline_text=baseline_text,
                    baseline_image=baseline_image,
                    quantized_text=quant_text,
                    quantized_image=quant_image,
                    pair_sample_size=config.pair_sample_size,
                    rng=rng,
                ),
                "image_to_text_retrieval": compute_retrieval_metrics(image_to_text),
                "text_to_image_retrieval": compute_retrieval_metrics(text_to_image),
                "image_to_text_rank_change": compute_rank_change_metrics(baseline_i2t, image_to_text),
                "text_to_image_rank_change": compute_rank_change_metrics(baseline_t2i, text_to_image),
            }

            results["conditions"][condition][str(bits)] = condition_result

            if condition == config.drift_plot_condition and bits == config.drift_plot_bits:
                drift_payload = (quant_text, quant_image, condition, bits)

    if drift_payload is not None:
        quant_text, quant_image, drift_condition, drift_bits = drift_payload
        generate_drift_plot(
            baseline_text=baseline_text,
            baseline_image=baseline_image,
            quantized_text=quant_text,
            quantized_image=quant_image,
            output_path=output_dir / f"drift_{drift_condition}_{drift_bits}bit.png",
            seed=config.seed,
            drift_plot_size=config.drift_plot_size,
        )
        generate_similarity_scatter(
            baseline_text=baseline_text,
            baseline_image=baseline_image,
            quantized_text=quant_text,
            quantized_image=quant_image,
            output_path=output_dir / f"similarity_scatter_{drift_condition}_{drift_bits}bit.png",
            sample_size=config.pair_sample_size,
            seed=config.seed,
        )

        image_to_text = quant_image @ quant_text.T
        flips = collect_flip_examples(
            baseline_similarities=baseline_i2t,
            quantized_similarities=image_to_text,
            metadata=metadata,
        )
        with open(output_dir / f"rank_flip_examples_{drift_condition}_{drift_bits}bit.json", "w", encoding="utf-8") as handle:
            json.dump(flips, handle, indent=2)

    generate_retrieval_plot(results, output_dir)
    generate_geometry_plot(results, output_dir)
    generate_summary_report(results, output_dir / "geometry_report.md")

    with open(output_dir / "geometry_results.json", "w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)

    return results
