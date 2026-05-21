"""
Fast Iteration Utilities
========================

Tools for rapid experimentation with minimal overhead:
- Subset sampling strategies
- Progressive evaluation
- Early stopping for experiments
- Parallel experiment execution
- Quick visualization
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from typing import Dict, List, Optional, Tuple, Callable, Any, Iterator
from dataclasses import dataclass
import numpy as np
import json
import time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import multiprocessing as mp
from functools import partial
import logging
import hashlib

logger = logging.getLogger(__name__)


# =============================================================================
# Smart Subset Sampling
# =============================================================================

class SmartSubsetSampler:
    """
    Intelligent subset sampling that maintains distribution properties.
    """
    
    @staticmethod
    def stratified_sample(dataset, labels: List, subset_size: int) -> List[int]:
        """Sample maintaining class distribution."""
        from collections import Counter
        
        label_counts = Counter(labels)
        samples_per_class = {
            label: max(1, int(subset_size * count / len(labels)))
            for label, count in label_counts.items()
        }
        
        indices_by_label = {}
        for idx, label in enumerate(labels):
            if label not in indices_by_label:
                indices_by_label[label] = []
            indices_by_label[label].append(idx)
        
        selected_indices = []
        for label, indices in indices_by_label.items():
            n_samples = min(samples_per_class.get(label, 1), len(indices))
            selected_indices.extend(np.random.choice(indices, n_samples, replace=False))
        
        return selected_indices[:subset_size]
    
    @staticmethod
    def difficulty_sample(dataset, scores: List[float], subset_size: int,
                          difficulty: str = "mixed") -> List[int]:
        """
        Sample based on difficulty scores (e.g., model confidence).
        
        difficulty: "easy", "hard", or "mixed"
        """
        sorted_indices = np.argsort(scores)
        
        if difficulty == "easy":
            return sorted_indices[-subset_size:].tolist()
        elif difficulty == "hard":
            return sorted_indices[:subset_size].tolist()
        else:  # mixed
            # Sample evenly from easy, medium, hard
            n_per_bucket = subset_size // 3
            hard = sorted_indices[:n_per_bucket]
            easy = sorted_indices[-n_per_bucket:]
            mid_start = len(sorted_indices) // 2 - n_per_bucket // 2
            medium = sorted_indices[mid_start:mid_start + n_per_bucket]
            return np.concatenate([hard, medium, easy]).tolist()
    
    @staticmethod
    def diversity_sample(embeddings: torch.Tensor, subset_size: int) -> List[int]:
        """Sample to maximize diversity using k-means++ initialization."""
        n = embeddings.shape[0]
        selected = [np.random.randint(n)]
        
        for _ in range(subset_size - 1):
            # Compute distances to nearest selected point
            dists = torch.cdist(embeddings, embeddings[selected]).min(dim=1)[0]
            
            # Sample proportional to squared distance
            probs = (dists ** 2) / (dists ** 2).sum()
            next_idx = np.random.choice(n, p=probs.numpy())
            selected.append(next_idx)
        
        return selected


# =============================================================================
# Progressive Evaluation
# =============================================================================

class ProgressiveEvaluator:
    """
    Evaluate on progressively larger subsets with early stopping.
    Quickly identifies if an experiment is worth continuing.
    """
    
    def __init__(self, 
                 subset_sizes: List[int] = [100, 500, 1000, 5000],
                 confidence_threshold: float = 0.95,
                 min_improvement: float = 0.001):
        self.subset_sizes = subset_sizes
        self.confidence_threshold = confidence_threshold
        self.min_improvement = min_improvement
        
    def evaluate_progressive(self, 
                            eval_fn: Callable[[List[int]], Dict[str, float]],
                            total_samples: int,
                            metric_key: str = "accuracy@1") -> Dict[str, Any]:
        """
        Progressively evaluate on larger subsets.
        
        Args:
            eval_fn: Function that takes indices and returns metrics dict
            total_samples: Total number of samples available
            metric_key: Primary metric to track
            
        Returns:
            Dict with final metrics and convergence info
        """
        results_history = []
        
        for size in self.subset_sizes:
            if size > total_samples:
                size = total_samples
            
            # Sample indices
            indices = np.random.choice(total_samples, size, replace=False).tolist()
            
            # Evaluate
            start_time = time.time()
            metrics = eval_fn(indices)
            eval_time = time.time() - start_time
            
            results_history.append({
                "subset_size": size,
                "metrics": metrics,
                "eval_time": eval_time
            })
            
            logger.info(f"Subset {size}: {metric_key}={metrics.get(metric_key, 0):.4f} "
                       f"(took {eval_time:.2f}s)")
            
            # Check convergence
            if len(results_history) >= 2:
                prev_metric = results_history[-2]["metrics"].get(metric_key, 0)
                curr_metric = metrics.get(metric_key, 0)
                
                # If metric is stable, we can estimate final performance
                if abs(curr_metric - prev_metric) < self.min_improvement:
                    logger.info(f"Converged at subset size {size}")
                    break
            
            if size >= total_samples:
                break
        
        # Estimate final performance with confidence interval
        final_metrics = results_history[-1]["metrics"]
        
        return {
            "final_metrics": final_metrics,
            "convergence_history": results_history,
            "converged_at_size": results_history[-1]["subset_size"],
            "total_eval_time": sum(r["eval_time"] for r in results_history)
        }


# =============================================================================
# Early Stopping for Experiments
# =============================================================================

class ExperimentEarlyStopper:
    """
    Early stopping for experiments based on intermediate results.
    """
    
    def __init__(self, 
                 baseline_metrics: Dict[str, float],
                 max_degradation: float = 0.1,
                 patience: int = 2):
        self.baseline_metrics = baseline_metrics
        self.max_degradation = max_degradation
        self.patience = patience
        self.bad_results_count = 0
        
    def should_stop(self, current_metrics: Dict[str, float], 
                   primary_metric: str = "accuracy@1") -> Tuple[bool, str]:
        """
        Check if experiment should be stopped early.
        
        Returns:
            (should_stop, reason)
        """
        baseline_value = self.baseline_metrics.get(primary_metric, 0)
        current_value = current_metrics.get(primary_metric, 0)
        
        degradation = (baseline_value - current_value) / max(baseline_value, 1e-8)
        
        if degradation > self.max_degradation:
            self.bad_results_count += 1
            if self.bad_results_count >= self.patience:
                return True, f"Degradation {degradation:.2%} exceeds threshold"
        else:
            self.bad_results_count = 0
        
        return False, ""
    
    def reset(self):
        """Reset counter for new experiment."""
        self.bad_results_count = 0


# =============================================================================
# Parallel Experiment Execution
# =============================================================================

@dataclass
class ExperimentTask:
    """Represents a single experiment to run."""
    experiment_id: str
    config: Dict[str, Any]
    priority: int = 0  # Higher = run first


class ParallelExperimentRunner:
    """
    Run multiple experiments in parallel.
    """
    
    def __init__(self, 
                 max_workers: int = None,
                 use_gpu_queue: bool = True):
        self.max_workers = max_workers or mp.cpu_count()
        self.use_gpu_queue = use_gpu_queue
        self.results: Dict[str, Any] = {}
        
    def run_experiments(self, 
                        tasks: List[ExperimentTask],
                        experiment_fn: Callable[[Dict], Dict],
                        progress_callback: Optional[Callable] = None) -> Dict[str, Any]:
        """
        Run experiments in parallel.
        
        Args:
            tasks: List of experiment tasks
            experiment_fn: Function that takes config and returns results
            progress_callback: Optional callback for progress updates
            
        Returns:
            Dict mapping experiment_id to results
        """
        # Sort by priority
        tasks = sorted(tasks, key=lambda t: -t.priority)
        
        if self.use_gpu_queue and torch.cuda.is_available():
            # GPU experiments need sequential access to GPU
            # Use threading for I/O parallelism
            return self._run_gpu_experiments(tasks, experiment_fn, progress_callback)
        else:
            # CPU experiments can run in parallel
            return self._run_cpu_experiments(tasks, experiment_fn, progress_callback)
    
    def _run_gpu_experiments(self, tasks, experiment_fn, progress_callback):
        """Run experiments that need GPU sequentially with async I/O."""
        results = {}
        
        for i, task in enumerate(tasks):
            logger.info(f"Running experiment {task.experiment_id} ({i+1}/{len(tasks)})")
            
            try:
                result = experiment_fn(task.config)
                results[task.experiment_id] = {"status": "success", "result": result}
            except Exception as e:
                results[task.experiment_id] = {"status": "error", "error": str(e)}
            
            if progress_callback:
                progress_callback(i + 1, len(tasks), task.experiment_id)
        
        return results
    
    def _run_cpu_experiments(self, tasks, experiment_fn, progress_callback):
        """Run CPU experiments in parallel."""
        results = {}
        
        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_task = {
                executor.submit(experiment_fn, task.config): task
                for task in tasks
            }
            
            completed = 0
            for future in as_completed(future_to_task):
                task = future_to_task[future]
                completed += 1
                
                try:
                    result = future.result()
                    results[task.experiment_id] = {"status": "success", "result": result}
                except Exception as e:
                    results[task.experiment_id] = {"status": "error", "error": str(e)}
                
                if progress_callback:
                    progress_callback(completed, len(tasks), task.experiment_id)
        
        return results


# =============================================================================
# Quick Visualization
# =============================================================================

class QuickVisualizer:
    """
    Fast plotting utilities for experiment monitoring.
    """
    
    @staticmethod
    def plot_accuracy_curve(results: Dict[int, float], 
                           title: str = "Accuracy vs Bit Width",
                           save_path: Optional[str] = None):
        """Plot accuracy vs quantization bit width."""
        try:
            import matplotlib.pyplot as plt
            
            bits = sorted(results.keys())
            accs = [results[b] for b in bits]
            
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.plot(bits, accs, 'bo-', linewidth=2, markersize=8)
            ax.set_xlabel('Bit Width')
            ax.set_ylabel('Accuracy')
            ax.set_title(title)
            ax.grid(True, alpha=0.3)
            ax.invert_xaxis()
            
            if save_path:
                plt.savefig(save_path, dpi=100, bbox_inches='tight')
            plt.close()
            
            return fig
        except ImportError:
            logger.warning("matplotlib not available")
            return None
    
    @staticmethod
    def plot_layer_importance(importance_scores: Dict[str, float],
                             top_k: int = 15,
                             save_path: Optional[str] = None):
        """Plot layer importance scores."""
        try:
            import matplotlib.pyplot as plt
            
            # Sort and take top k
            sorted_items = sorted(importance_scores.items(), 
                                 key=lambda x: x[1], reverse=True)[:top_k]
            layers = [item[0].split('.')[-1] for item in sorted_items]
            scores = [item[1] for item in sorted_items]
            
            fig, ax = plt.subplots(figsize=(10, 6))
            bars = ax.barh(range(len(layers)), scores)
            ax.set_yticks(range(len(layers)))
            ax.set_yticklabels(layers)
            ax.set_xlabel('Importance Score')
            ax.set_title('Layer Importance for Quantization')
            ax.invert_yaxis()
            
            if save_path:
                plt.savefig(save_path, dpi=100, bbox_inches='tight')
            plt.close()
            
            return fig
        except ImportError:
            logger.warning("matplotlib not available")
            return None
    
    @staticmethod
    def plot_pareto_frontier(results: List[Dict],
                            x_key: str = "model_size_mb",
                            y_key: str = "accuracy@1",
                            save_path: Optional[str] = None):
        """Plot Pareto frontier of accuracy vs model size."""
        try:
            import matplotlib.pyplot as plt
            
            x_vals = [r.get(x_key, 0) for r in results]
            y_vals = [r.get(y_key, 0) for r in results]
            labels = [r.get("label", f"Config {i}") for i, r in enumerate(results)]
            
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.scatter(x_vals, y_vals, s=100)
            
            for i, label in enumerate(labels):
                ax.annotate(label, (x_vals[i], y_vals[i]), 
                           textcoords="offset points", xytext=(5, 5))
            
            ax.set_xlabel(x_key)
            ax.set_ylabel(y_key)
            ax.set_title('Accuracy vs Model Size Pareto Frontier')
            ax.grid(True, alpha=0.3)
            
            if save_path:
                plt.savefig(save_path, dpi=100, bbox_inches='tight')
            plt.close()
            
            return fig
        except ImportError:
            logger.warning("matplotlib not available")
            return None
    
    @staticmethod
    def create_summary_table(results: Dict[str, Dict], 
                            metrics: List[str] = None) -> str:
        """Create markdown table summarizing results."""
        if not results:
            return "No results to display"
        
        # Determine metrics from first result
        if metrics is None:
            first_result = next(iter(results.values()))
            if "metrics" in first_result:
                metrics = list(first_result["metrics"].keys())
            else:
                metrics = list(first_result.keys())
        
        # Header
        header = "| Config | " + " | ".join(metrics) + " |"
        separator = "|" + "|".join(["---"] * (len(metrics) + 1)) + "|"
        
        rows = [header, separator]
        
        for config_name, result in results.items():
            if "metrics" in result:
                values = result["metrics"]
            else:
                values = result
            
            row = f"| {config_name} | "
            row += " | ".join(f"{values.get(m, 'N/A'):.4f}" 
                             if isinstance(values.get(m), (int, float)) 
                             else str(values.get(m, 'N/A'))
                             for m in metrics)
            row += " |"
            rows.append(row)
        
        return "\n".join(rows)


# =============================================================================
# Experiment Checkpointing
# =============================================================================

class ExperimentCheckpointer:
    """
    Save and resume experiments for long-running sweeps.
    """
    
    def __init__(self, checkpoint_dir: str):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
    def _get_checkpoint_path(self, experiment_id: str) -> Path:
        return self.checkpoint_dir / f"{experiment_id}.json"
    
    def save_checkpoint(self, experiment_id: str, state: Dict):
        """Save experiment state."""
        path = self._get_checkpoint_path(experiment_id)
        with open(path, 'w') as f:
            json.dump(state, f, indent=2, default=str)
    
    def load_checkpoint(self, experiment_id: str) -> Optional[Dict]:
        """Load experiment state if exists."""
        path = self._get_checkpoint_path(experiment_id)
        if path.exists():
            with open(path) as f:
                return json.load(f)
        return None
    
    def get_completed_experiments(self) -> List[str]:
        """Get list of completed experiment IDs."""
        completed = []
        for path in self.checkpoint_dir.glob("*.json"):
            with open(path) as f:
                state = json.load(f)
                if state.get("status") == "completed":
                    completed.append(path.stem)
        return completed
    
    def clear_checkpoints(self):
        """Clear all checkpoints."""
        for path in self.checkpoint_dir.glob("*.json"):
            path.unlink()


# =============================================================================
# Timing Utilities
# =============================================================================

class Timer:
    """Context manager for timing code blocks."""
    
    def __init__(self, name: str = ""):
        self.name = name
        self.start_time = None
        self.elapsed = None
        
    def __enter__(self):
        self.start_time = time.perf_counter()
        return self
    
    def __exit__(self, *args):
        self.elapsed = time.perf_counter() - self.start_time
        if self.name:
            logger.info(f"{self.name}: {self.elapsed:.3f}s")


class ExperimentTimer:
    """Track timing across experiment phases."""
    
    def __init__(self):
        self.timings: Dict[str, float] = {}
        self.current_phase: Optional[str] = None
        self.phase_start: Optional[float] = None
        
    def start_phase(self, phase_name: str):
        """Start timing a phase."""
        if self.current_phase:
            self.end_phase()
        self.current_phase = phase_name
        self.phase_start = time.perf_counter()
        
    def end_phase(self):
        """End current phase."""
        if self.current_phase and self.phase_start:
            elapsed = time.perf_counter() - self.phase_start
            self.timings[self.current_phase] = elapsed
        self.current_phase = None
        self.phase_start = None
        
    def get_summary(self) -> Dict[str, Any]:
        """Get timing summary."""
        total = sum(self.timings.values())
        return {
            "phases": self.timings,
            "total_seconds": total,
            "breakdown_percent": {
                k: (v / total * 100) if total > 0 else 0
                for k, v in self.timings.items()
            }
        }


# =============================================================================
# Configuration Generators
# =============================================================================

def generate_bit_sweep_configs(bits_list: List[int] = [2, 3, 4, 6, 8],
                                strategies: List[str] = ["dynamic"]) -> List[Dict]:
    """Generate configurations for bit width sweep."""
    configs = []
    for strategy in strategies:
        for bits in bits_list:
            configs.append({
                "id": f"{strategy}_{bits}bit",
                "strategy": strategy,
                "bits": bits
            })
    return configs


def generate_layer_ablation_configs(layer_groups: Dict[str, List[str]],
                                     bits: int = 4) -> List[Dict]:
    """Generate configurations for layer ablation study."""
    configs = []
    
    # Quantize each group individually
    for group_name, layers in layer_groups.items():
        configs.append({
            "id": f"only_{group_name}_{bits}bit",
            "type": "layer_selective",
            "bits": bits,
            "include_patterns": layers,
            "exclude_patterns": []
        })
    
    # Leave each group out
    all_layers = [l for layers in layer_groups.values() for l in layers]
    for group_name, layers in layer_groups.items():
        other_layers = [l for l in all_layers if l not in layers]
        configs.append({
            "id": f"exclude_{group_name}_{bits}bit",
            "type": "layer_selective",
            "bits": bits,
            "include_patterns": other_layers,
            "exclude_patterns": layers
        })
    
    return configs


def generate_mixed_precision_configs(sensitive_layers: List[str],
                                      bits_high: int = 8,
                                      bits_low: int = 4) -> List[Dict]:
    """Generate mixed-precision configurations."""
    configs = []
    
    # Different percentages of layers in high precision
    for pct in [0.1, 0.25, 0.5]:
        n_high = int(len(sensitive_layers) * pct)
        high_prec_layers = sensitive_layers[:n_high]
        
        configs.append({
            "id": f"mixed_{int(pct*100)}pct_high",
            "type": "mixed_precision",
            "high_bits": bits_high,
            "low_bits": bits_low,
            "high_precision_layers": high_prec_layers
        })
    
    return configs
