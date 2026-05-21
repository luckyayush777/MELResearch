# CLIP Quantization Harness for Multimodal Entity Linking

A fast-iteration experimentation framework for studying quantization effects on CLIP models in multimodal entity linking tasks, specifically designed for WikiDiverse.

## Design Philosophy

This harness prioritizes **iteration speed** through:

1. **Aggressive Caching**: Embeddings, quantized models, and results are cached at every level
2. **Progressive Evaluation**: Start with small subsets, scale up only when needed
3. **Early Stopping**: Abandon experiments that clearly won't work
4. **Parallel Execution**: Run multiple configurations simultaneously
5. **Checkpointing**: Resume long experiments from where they left off

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Quick sanity check (100 samples)
python run_experiments.py --experiment quick-test

# Bit width sweep with subset
python run_experiments.py --experiment bit-sweep --subset-size 1000

# Full experiment suite
python run_experiments.py --experiment full
```

## Directory Structure

```
clip_quantization_harness/
├── experiment_harness.py    # Core experimentation framework
├── quantization_utils.py    # Advanced quantization strategies
├── fast_iteration.py        # Speed optimization utilities
├── run_experiments.py       # CLI entry point
├── requirements.txt
└── README.md
```

## Key Components

### 1. Experiment Harness (`experiment_harness.py`)

**ExperimentConfig**: Central configuration
```python
config = ExperimentConfig(
    clip_model_name="openai/clip-vit-base-patch32",
    wikidiverse_path="./data/wikidiverse",
    subset_size=1000,  # Fast iteration
    quantization_bits=[8, 6, 4, 3, 2]
)
```

**ExperimentRunner**: Main orchestrator
```python
runner = ExperimentRunner(config)
baseline = runner.evaluate_baseline()
bit_sweep = runner.run_bit_width_sweep()
layer_importance = runner.run_layer_importance_analysis()
```

### 2. Quantization Strategies (`quantization_utils.py`)

| Strategy | Description | Speed | Accuracy |
|----------|-------------|-------|----------|
| `DynamicQuantization` | PyTorch dynamic quant | Fast | Good |
| `StaticQuantization` | With calibration | Medium | Better |
| `BitsAndBytesQuantization` | 4-bit/8-bit with bitsandbytes | Fast | Good |
| `GPTQQuantizer` | Hessian-based weight update | Slow | Best |
| `AWQQuantizer` | Activation-aware | Medium | Very Good |
| `LayerSelectiveQuantization` | Per-layer control | Fast | Configurable |
| `MixedPrecisionQuantizer` | Different bits per layer | Medium | Optimal |

### 3. Fast Iteration Tools (`fast_iteration.py`)

**Progressive Evaluation**: Scale up gradually
```python
evaluator = ProgressiveEvaluator(
    subset_sizes=[100, 500, 1000, 5000],
    min_improvement=0.001
)
results = evaluator.evaluate_progressive(eval_fn, total_samples)
```

**Smart Subset Sampling**: Maintain distribution
```python
# Stratified sampling
indices = SmartSubsetSampler.stratified_sample(dataset, labels, 1000)

# Diversity sampling (k-means++)
indices = SmartSubsetSampler.diversity_sample(embeddings, 1000)
```

**Parallel Experiments**: Run multiple configs
```python
tasks = [ExperimentTask(id=f"exp_{i}", config=cfg) for i, cfg in enumerate(configs)]
runner = ParallelExperimentRunner(max_workers=4)
results = runner.run_experiments(tasks, experiment_fn)
```

## Experiment Types

### 1. Bit Width Sweep
Test accuracy degradation across bit widths (32 → 8 → 4 → 2 bits):

```bash
python run_experiments.py --experiment bit-sweep
```

Outputs:
- `bit_sweep_accuracy.png`: Accuracy vs bit width curve
- Compression ratios and inference speedups

### 2. Layer Importance Analysis
Identify which CLIP layers are most sensitive to quantization:

```bash
python run_experiments.py --experiment layer-importance
```

Layer groups analyzed:
- `text_attention`: Text encoder self-attention
- `text_mlp`: Text encoder MLPs
- `vision_attention`: Vision encoder self-attention
- `vision_mlp`: Vision encoder MLPs
- `projection`: Cross-modal projection layers

### 3. Geometry Experiment
Visualize how quantization distorts the paired text-image embedding space:

```bash
python run_experiments.py --experiment geometry --output-dir ./results/geometry
```

This experiment uses precomputed paired embeddings from `ExpSetup/data/embeddings/`
and runs three conditions across bit widths:
- `text_only`: quantize only text embeddings
- `image_only`: quantize only image embeddings
- `both`: quantize both modalities

Outputs:
- `retrieval_vs_bits.png`: Recall@1 across conditions and bit widths
- `geometry_vs_bits.png`: self-alignment, neighbor overlap, pairwise cosine preservation
- `drift_both_4bit.png`: 2D PCA projection with drift arrows
- `similarity_scatter_both_4bit.png`: baseline vs quantized cross-modal cosine scatter
- `rank_flip_examples_both_4bit.json`: examples whose retrieval ranks changed the most
- `geometry_results.json`: full metrics for analysis

### 3. Mixed Precision Optimization
Find optimal per-layer bit allocation:

```python
from quantization_utils import MixedPrecisionQuantizer, MixedPrecisionConfig

config = MixedPrecisionConfig(
    default_bits=4,
    sensitive_layers_bits=8,
    sensitivity_threshold=0.1
)
quantizer = MixedPrecisionQuantizer(config)
config = quantizer.auto_configure(model, target_compression=4.0)
```

## WikiDiverse Data Format

Expected directory structure:
```
data/wikidiverse/
├── test_mentions.json
├── train_mentions.json
├── entities.json
└── images/
```

**mentions.json format**:
```json
[
  {
    "id": "mention_001",
    "mention": "Einstein",
    "context": "Albert Einstein developed the theory of relativity...",
    "image_path": "images/mention_001.jpg",
    "entity_id": "Q937",
    "candidates": ["Q937", "Q1234", "Q5678"]
  }
]
```

**entities.json format**:
```json
[
  {
    "id": "Q937",
    "name": "Albert Einstein",
    "description": "German-born theoretical physicist",
    "image_path": "images/Q937.jpg",
    "wikidata_id": "Q937",
    "aliases": ["Einstein", "A. Einstein"]
  }
]
```

## Caching Strategy

The harness uses a 3-level cache:

1. **Memory Cache**: Hot embeddings in RAM
2. **Disk Cache**: Persistent embeddings in `cache/embeddings/`
3. **Results Cache**: Experiment results in `cache/results/`

Cache invalidation is automatic based on:
- Model name changes
- Quantization config changes
- Data subset changes

To clear cache:
```python
import shutil
shutil.rmtree("./cache")
```

## Metrics

| Metric | Description |
|--------|-------------|
| `accuracy@1` | Exact match accuracy |
| `accuracy@5` | Gold entity in top 5 predictions |
| `accuracy@10` | Gold entity in top 10 predictions |
| `mrr` | Mean Reciprocal Rank |
| `ndcg@10` | Normalized DCG at 10 |

## Tips for Fast Iteration

1. **Start small**: `--subset-size 500` for initial experiments
2. **Use progressive evaluation**: Converges quickly on most experiments
3. **Cache everything**: First run is slow, subsequent runs are fast
4. **Parallelize sweeps**: Use `ParallelExperimentRunner` for hyperparameter searches
5. **Use early stopping**: `ExperimentEarlyStopper` abandons bad configs quickly

## Extending the Framework

### Adding New Quantization Strategies

```python
from experiment_harness import QuantizationStrategy

class MyQuantization(QuantizationStrategy):
    def __init__(self, my_param: int):
        self.my_param = my_param
    
    def quantize(self, model, calibration_data=None):
        # Your quantization logic
        return quantized_model
    
    def get_config_dict(self):
        return {"type": "my_quant", "my_param": self.my_param}
```

### Custom Evaluation Metrics

```python
from experiment_harness import EntityLinkingMetrics

# Add to EntityLinkingMetrics class
@staticmethod
def my_metric(predictions, gold):
    # Your metric logic
    return score
```

### Different Datasets

Subclass `WikiDiverseDataset` and implement `_load_data()`:

```python
class MyDataset(WikiDiverseDataset):
    def _load_data(self):
        # Load your data format
        self.mentions = [...]
        self.entities = {...}
```

## Citation

If you use this harness, please cite:

```bibtex
@misc{clip_quantization_harness,
  title={CLIP Quantization Harness for Multimodal Entity Linking},
  year={2024}
}
```
