# Image Patch Tokens Quantization Plan

Saved on 2026-05-19.

## Current Run

Goal:

```text
Compare entity image_patch_tokens int4 against the clean fp32 baseline.
```

Started command:

```powershell
python ".\Mixed Quantization setup\evaluate_entity_text_tokens.py" `
  --target-tensor image_patch_tokens `
  --target-modes fp32 int4 `
  --other-mode fp32 `
  --out-dir results/image_patch_tokens_fp32_int4 `
  --split test `
  --chunk-size 20000
```

Output directory:

```text
Mixed Quantization setup/results/image_patch_tokens_fp32_int4
```

Run isolation:

```text
text_cls = fp32
image_cls = fp32
text_tokens = fp32
image_patch_tokens = fp32/int4
```

Baseline:

```text
Clean no-leak WikiMEL fp32, not the old leaky checkpoint.
Checkpoint: ExpSetup/external/MIMIC_reproduction/runs/WikiMEL_noleak/version_4/checkpoints/final.ckpt
Config: ExpSetup/external/MIMIC_reproduction/config/wikimel_noleak.yaml
```

## Why This Run

The 19Feb result showed `entity_text_tokens` int4 is the strongest conservative deployment point so far:

```text
H@1 unchanged
MRR unchanged/slightly higher
text-token cache greatly reduced
entity transfer and TGLU time reduced
```

The next sensitivity question is whether the visual local-token path can also be compressed:

```text
image_patch_tokens are the second-largest fp32 cache tensor.
They feed VDLU and CMFU, so they may matter most for visually grounded or ambiguous examples.
```

## Metrics To Collect In This First Comparison

The current evaluator already captures:

```text
run mode
checkpoint/config/split
chunk size
number of entities
queries and skipped queries
H@1, H@3, H@5, H@10, H@20
MR, MRR
delta_H@1/H@3/H@10/H@20/MR/MRR_vs_fp32
total eval seconds
speedup_vs_fp32
entity cache logical MB
per-tensor logical cache MB
cache ratio vs fp32
image_patch_tokens size ratio vs fp32
mention_to_device_s
mention_encoder_s
entity_transfer_s
layernorm_s
tglu_s
vdlu_s
cmfu_s
score_concat_s
rank_sort_s
subprocess percentages
per-batch timing JSON
rank batches
```

Primary comparison table for this run:

| Mode | H@1 | MRR | Delta H@1 | Delta MRR | Eval seconds | Speedup | Total cache MB | Image patch MB | Transfer s | VDLU s | CMFU s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| fp32 | | | | | | | | | | | |
| int4 image_patch_tokens | | | | | | | | | | | |

## Gaps To Add Before The Final Smaller-GPU Claim

The metrics plan requires several deployment metrics that are not fully covered by the current evaluator yet:

```text
GPU peak allocated/reserved memory by phase
CPU RSS start/load/peak
actual disk artifact size
query latency percentiles
batch latency percentiles
queries/sec and query-entity-pairs/sec
rank stability and rank-flip examples
quantization error metrics for image_patch_tokens
chunk-size feasibility sweep
smaller-GPU memory-budget simulation
failure/OOM fields
dataset-slice breakdowns
```

Add these in this order:

1. Add environment/run metadata to each result row: timestamp, device name, CUDA/PyTorch versions, GPU total memory, checkpoint, config, split, tensor modes, cache format version, git status note.
2. Add memory tracking around entity cache build, cache load, mention encoder, entity chunk transfer, matcher, and total eval.
3. Add throughput and latency summaries from existing batch JSON: QPS, query-entity pairs/sec, average/p50/p95/p99 query latency, and batch latency percentiles.
4. Add actual disk-size scanner for each mode directory and append `entity_cache_disk_mb`, shard count, min/avg/max shard MB.
5. Add rank-stability comparison against fp32 rank batches: top1 same rate, top-k overlap if top-k indices are saved, rank improved/worsened counts, rank-flip examples.
6. Add quantization-error sampling for image_patch_tokens: MAE, relative L2, cosine similarity, saturation, zero rate, scale statistics.
7. Run chunk sweeps at 1000, 2500, 5000, 10000, 20000, and 40000 for fp32 and the best image-patch mode.
8. Run memory-budget simulations or direct smaller-GPU runs after peak-memory instrumentation is reliable.

## Decision Criteria

Treat `image_patch_tokens` int4 as promising if:

```text
H@1 and MRR stay within noise of fp32
image_patch_tokens logical cache shrinks substantially
entity_transfer_s decreases
VDLU and/or CMFU time does not regress enough to erase the transfer gain
rank-worsening examples are rare or explainable
```

If int4 is unstable, run:

```text
image_patch_tokens fp32 int8 int6 int4 int3 int2
```

and use int8/int6 as the conservative visual-token compression point.
