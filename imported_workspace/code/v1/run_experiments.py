#!/usr/bin/env python3
"""
CLIP Quantization Experiments - Main Runner
============================================

Example usage and quick-start scripts.

Usage:
    # Quick test with small subset
    python run_experiments.py --quick-test
    
    # Full bit width sweep
    python run_experiments.py --experiment bit-sweep
    
    # Layer importance analysis
    python run_experiments.py --experiment layer-importance
    
    # Full suite
    python run_experiments.py --experiment full
"""

import argparse
import json
import logging
from pathlib import Path
import sys

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from experiment_harness import (
    ExperimentConfig, 
    ExperimentRunner,
    generate_accuracy_vs_quantization_plot,
    generate_layer_importance_report
)
from fast_iteration import (
    ProgressiveEvaluator,
    QuickVisualizer,
    Timer,
    ExperimentTimer,
    ExperimentCheckpointer,
    generate_bit_sweep_configs,
    generate_layer_ablation_configs
)
from quantization_utils import (
    LayerSensitivityAnalyzer,
    estimate_memory_savings,
    benchmark_inference_speed
)
from geometry_experiment import (
    GeometryExperimentConfig,
    run_geometry_quantization_experiment,
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def run_quick_test(config: ExperimentConfig) -> dict:
    """
    Quick sanity check with minimal data.
    Good for verifying the pipeline works.
    """
    logger.info("Running quick test...")
    
    # Force small subset
    config.subset_size = 100
    
    runner = ExperimentRunner(config)
    
    # Test baseline
    with Timer("Baseline evaluation"):
        baseline = runner.evaluate_baseline()
    
    logger.info(f"Baseline results: {baseline}")
    
    return {"baseline": baseline, "status": "quick_test_passed"}


def run_bit_sweep(config: ExperimentConfig) -> dict:
    """
    Run bit width sweep experiment.
    Tests accuracy at different quantization levels.
    """
    logger.info("Running bit width sweep...")
    
    timer = ExperimentTimer()
    runner = ExperimentRunner(config)
    
    timer.start_phase("baseline")
    baseline = runner.evaluate_baseline()
    timer.end_phase()
    
    timer.start_phase("dynamic_quantization")
    dynamic_results = runner.run_bit_width_sweep("dynamic")
    timer.end_phase()
    
    # Generate plots
    output_dir = Path(config.cache_dir) / "outputs"
    output_dir.mkdir(exist_ok=True)
    
    # Extract accuracy values for plotting
    accuracy_by_bits = {}
    for bits, data in dynamic_results.items():
        if bits == "baseline":
            accuracy_by_bits[32] = data["metrics"]["accuracy@1"]
        elif isinstance(bits, int) and "metrics" in data:
            accuracy_by_bits[bits] = data["metrics"]["accuracy@1"]
    
    QuickVisualizer.plot_accuracy_curve(
        accuracy_by_bits,
        title="CLIP Entity Linking: Accuracy vs Bit Width",
        save_path=str(output_dir / "bit_sweep_accuracy.png")
    )
    
    # Summary table
    summary = QuickVisualizer.create_summary_table(dynamic_results)
    logger.info(f"\n{summary}")
    
    return {
        "baseline": baseline,
        "dynamic_results": dynamic_results,
        "timing": timer.get_summary()
    }


def run_layer_importance(config: ExperimentConfig) -> dict:
    """
    Analyze which layers are most important for accuracy.
    Helps decide what to keep in higher precision.
    """
    logger.info("Running layer importance analysis...")
    
    runner = ExperimentRunner(config)
    
    # Get layer groups
    layer_groups = runner.model.get_layer_groups()
    logger.info(f"Layer groups found: {list(layer_groups.keys())}")
    
    # Run importance analysis
    with Timer("Layer importance analysis"):
        results = runner.run_layer_importance_analysis(bits=8)
    
    # Generate report
    report = generate_layer_importance_report(results)
    
    output_dir = Path(config.cache_dir) / "outputs"
    output_dir.mkdir(exist_ok=True)
    
    with open(output_dir / "layer_importance.md", "w") as f:
        f.write(report)
    
    logger.info(f"\n{report}")
    
    # Compute importance scores for visualization
    importance_scores = {}
    for group_name, data in results.items():
        if group_name != "baseline" and "degradation" in data:
            importance_scores[group_name] = data["degradation"].get("accuracy@1", 0)
    
    QuickVisualizer.plot_layer_importance(
        importance_scores,
        save_path=str(output_dir / "layer_importance.png")
    )
    
    return results


def run_progressive_evaluation(config: ExperimentConfig) -> dict:
    """
    Use progressive evaluation for faster iteration.
    Evaluates on increasingly larger subsets until convergence.
    """
    logger.info("Running progressive evaluation...")
    
    # Don't use config subset - let progressive evaluator handle it
    config.subset_size = None
    runner = ExperimentRunner(config)
    
    progressive = ProgressiveEvaluator(
        subset_sizes=[100, 500, 1000, 2500],
        min_improvement=0.005
    )
    
    # Define evaluation function
    def eval_fn(indices):
        # Create temporary subset
        original_mentions = runner.dataset.mentions
        runner.dataset.mentions = [original_mentions[i] for i in indices]
        
        try:
            entity_embs = runner._get_entity_embeddings()
            mention_embs = runner._get_mention_embeddings()
            predictions = runner._predict_entities(mention_embs, entity_embs)
            gold = [m.gold_entity_id for m in runner.dataset.mentions]
            
            from experiment_harness import EntityLinkingMetrics
            return EntityLinkingMetrics.compute_all(predictions, gold)
        finally:
            runner.dataset.mentions = original_mentions
    
    results = progressive.evaluate_progressive(
        eval_fn,
        total_samples=len(runner.dataset),
        metric_key="accuracy@1"
    )
    
    logger.info(f"Converged at {results['converged_at_size']} samples")
    logger.info(f"Final metrics: {results['final_metrics']}")
    
    return results


def run_full_suite(config: ExperimentConfig) -> dict:
    """
    Run complete experiment suite.
    """
    logger.info("Running full experiment suite...")
    
    checkpointer = ExperimentCheckpointer(Path(config.cache_dir) / "checkpoints")
    
    results = {}
    
    # Check for existing results
    completed = checkpointer.get_completed_experiments()
    logger.info(f"Previously completed: {completed}")
    
    # 1. Baseline
    if "baseline" not in completed:
        with Timer("Baseline"):
            runner = ExperimentRunner(config)
            results["baseline"] = runner.evaluate_baseline()
            checkpointer.save_checkpoint("baseline", {
                "status": "completed",
                "results": results["baseline"]
            })
    else:
        results["baseline"] = checkpointer.load_checkpoint("baseline")["results"]
    
    # 2. Bit sweep
    if "bit_sweep" not in completed:
        with Timer("Bit sweep"):
            results["bit_sweep"] = run_bit_sweep(config)
            checkpointer.save_checkpoint("bit_sweep", {
                "status": "completed",
                "results": results["bit_sweep"]
            })
    else:
        results["bit_sweep"] = checkpointer.load_checkpoint("bit_sweep")["results"]
    
    # 3. Layer importance
    if "layer_importance" not in completed:
        with Timer("Layer importance"):
            results["layer_importance"] = run_layer_importance(config)
            checkpointer.save_checkpoint("layer_importance", {
                "status": "completed",
                "results": results["layer_importance"]
            })
    else:
        results["layer_importance"] = checkpointer.load_checkpoint("layer_importance")["results"]
    
    # Save final results
    output_dir = Path(config.cache_dir) / "outputs"
    output_dir.mkdir(exist_ok=True)
    
    with open(output_dir / "full_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    
    return results


def run_benchmarks(config: ExperimentConfig) -> dict:
    """
    Run performance benchmarks (speed, memory).
    """
    logger.info("Running benchmarks...")
    
    from transformers import CLIPModel
    import torch
    
    model = CLIPModel.from_pretrained(config.clip_model_name)
    model.to(config.device)
    model.eval()
    
    results = {}
    
    # Memory estimates
    for bits in [32, 16, 8, 4]:
        results[f"{bits}bit_memory"] = estimate_memory_savings(model, bits)
    
    # Speed benchmarks (text encoder)
    text_input_shape = (32, 77)  # batch_size, seq_len
    results["text_encoder_speed"] = benchmark_inference_speed(
        model.text_model,
        text_input_shape,
        num_iterations=50
    )
    
    # Speed benchmarks (vision encoder)
    vision_input_shape = (32, 3, 224, 224)  # batch_size, channels, height, width
    results["vision_encoder_speed"] = benchmark_inference_speed(
        model.vision_model,
        vision_input_shape,
        num_iterations=50
    )
    
    logger.info(f"Benchmark results: {json.dumps(results, indent=2)}")
    
    return results


def run_geometry_experiment(args) -> dict:
    """
    Run an embedding-space geometry study across text-only, image-only,
    and joint quantization conditions.
    """
    logger.info("Running geometry-aware quantization experiment...")

    bit_widths = [int(part.strip()) for part in args.geometry_bits.split(",") if part.strip()]
    geometry_config = GeometryExperimentConfig(
        text_embeddings_path=args.text_embeddings,
        image_embeddings_path=args.image_embeddings,
        metadata_path=args.metadata_path,
        output_dir=args.output_dir,
        bit_widths=bit_widths,
        subset_size=args.geometry_subset_size,
        seed=args.geometry_seed,
        neighbor_k=args.geometry_neighbor_k,
        pair_sample_size=args.geometry_pair_sample_size,
        drift_plot_size=args.geometry_drift_plot_size,
        drift_plot_bits=args.geometry_drift_bits,
        drift_plot_condition=args.geometry_drift_condition,
        renormalize=not args.geometry_no_renorm,
    )
    return run_geometry_quantization_experiment(geometry_config)


def main():
    parser = argparse.ArgumentParser(
        description="CLIP Quantization Experiments for Entity Linking"
    )
    
    parser.add_argument(
        "--experiment", 
        choices=["quick-test", "bit-sweep", "layer-importance", 
                 "progressive", "full", "benchmark", "geometry"],
        default="quick-test",
        help="Type of experiment to run"
    )
    
    parser.add_argument(
        "--data-path",
        default="./data/wikidiverse",
        help="Path to WikiDiverse dataset"
    )
    
    parser.add_argument(
        "--cache-dir",
        default="./cache",
        help="Directory for caching"
    )
    
    parser.add_argument(
        "--subset-size",
        type=int,
        default=None,
        help="Subset size for faster iteration"
    )
    
    parser.add_argument(
        "--model",
        default="openai/clip-vit-base-patch32",
        help="CLIP model to use"
    )
    
    parser.add_argument(
        "--device",
        default="cuda" if __import__("torch").cuda.is_available() else "cpu",
        help="Device to use"
    )
    
    parser.add_argument(
        "--output-dir",
        default="./results",
        help="Output directory for results"
    )

    default_embedding_root = (Path(__file__).resolve().parents[2] / "ExpSetup" / "data" / "embeddings")

    parser.add_argument(
        "--text-embeddings",
        default=str(default_embedding_root / "text_embeddings.npy"),
        help="Path to paired text embeddings for the geometry experiment"
    )

    parser.add_argument(
        "--image-embeddings",
        default=str(default_embedding_root / "image_embeddings.npy"),
        help="Path to paired image embeddings for the geometry experiment"
    )

    parser.add_argument(
        "--metadata-path",
        default=str(default_embedding_root / "valid_metadata.json"),
        help="Path to metadata aligned with the paired embeddings"
    )

    parser.add_argument(
        "--geometry-bits",
        default="8,6,4,3,2",
        help="Comma-separated bit widths for the geometry experiment"
    )

    parser.add_argument(
        "--geometry-subset-size",
        type=int,
        default=1000,
        help="Subset size for the geometry experiment"
    )

    parser.add_argument(
        "--geometry-pair-sample-size",
        type=int,
        default=20000,
        help="Random pair count used for pairwise geometry statistics"
    )

    parser.add_argument(
        "--geometry-neighbor-k",
        type=int,
        default=10,
        help="Neighborhood size for local geometry overlap"
    )

    parser.add_argument(
        "--geometry-seed",
        type=int,
        default=42,
        help="Random seed for the geometry experiment"
    )

    parser.add_argument(
        "--geometry-drift-bits",
        type=int,
        default=4,
        help="Bit width to visualize in the 2D drift plot"
    )

    parser.add_argument(
        "--geometry-drift-condition",
        choices=["text_only", "image_only", "both"],
        default="both",
        help="Condition to visualize in the 2D drift plot"
    )

    parser.add_argument(
        "--geometry-drift-plot-size",
        type=int,
        default=200,
        help="Number of points to annotate with drift arrows"
    )

    parser.add_argument(
        "--geometry-no-renorm",
        action="store_true",
        help="Disable post-quantization L2 renormalization"
    )
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.experiment == "geometry":
        results = run_geometry_experiment(args)
    else:
        # Create config for model-based experiments.
        config = ExperimentConfig(
            clip_model_name=args.model,
            device=args.device,
            wikidiverse_path=args.data_path,
            cache_dir=args.cache_dir,
            subset_size=args.subset_size
        )

        if args.experiment == "quick-test":
            results = run_quick_test(config)
        elif args.experiment == "bit-sweep":
            results = run_bit_sweep(config)
        elif args.experiment == "layer-importance":
            results = run_layer_importance(config)
        elif args.experiment == "progressive":
            results = run_progressive_evaluation(config)
        elif args.experiment == "full":
            results = run_full_suite(config)
        elif args.experiment == "benchmark":
            results = run_benchmarks(config)
        else:
            raise ValueError(f"Unknown experiment: {args.experiment}")
    
    # Save results
    with open(output_dir / f"{args.experiment}_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    
    logger.info(f"Results saved to {output_dir}")


if __name__ == "__main__":
    main()
