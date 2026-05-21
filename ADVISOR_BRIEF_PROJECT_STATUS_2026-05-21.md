# Advisor Brief: Clean WikiMEL Quantization and Low-Memory Inference

Date: 2026-05-21

## Executive Summary

This project investigates whether a strong multimodal entity linking model can be compressed and scheduled for smaller-GPU deployment without losing retrieval quality. The current work uses a MIMIC-style WikiMEL setup and focuses on entity-cache quantization, matcher tiling, and low-memory inference scheduling.

The main result is that the project now has a clean no-leak WikiMEL baseline and a practical deployment path:

| Setting | H@1 | MRR | Peak eval VRAM | QPS | Main value |
|---|---:|---:|---:|---:|---|
| Clean fp32 reference | 0.741629 | 0.818908 | 5851.28 MB | 9.074 | Valid baseline |
| Mixed int4 cache, no tiling | 0.741379 | 0.818733 | 5851.15 MB | 17.086 | Storage and speed win |
| Mixed int4 cache + TGLU tile 250 | 0.741379 | 0.818733 | 1922.29 MB | 17.444 | Best throughput/memory balance |
| Mixed int4 cache + entity subtile 500 | 0.741379 | 0.818733 | 1160.72 MB | 12.927 | Practical sub-1.2 GB point |
| Mixed int4 cache + entity subtile 250 | 0.741379 | 0.818733 | 901.35 MB | 10.379 | Lowest measured memory |

The safest current headline is:

> On clean no-leak WikiMEL, mixed 4-bit quantization of entity-side text and image local-token caches preserves retrieval quality, reduces logical cache size to about 14.9% of fp32, improves throughput by about 1.8x, and with TGLU tiling lowers peak eval VRAM from about 5.85 GB to about 1.92 GB at chunk size 5000.

## Project Goal

The goal is to build a quantization framework for multimodal entity linking that reports both retrieval quality and deployment cost. The framework measures:

- retrieval metrics: H@1, H@3, H@5, H@10, H@20, MRR, mean rank;
- compression metrics: logical cache size, disk cache size, per-tensor footprint;
- system metrics: runtime, QPS, transfer time, matcher time, and peak GPU allocation;
- scheduling metrics: entity chunk size, TGLU tile size, and entity subtile size.

## Dataset and Baseline Process

The initial reproduced WikiMEL result was rejected because the data build leaked information across splits. The leakage audit found:

| Leakage source | Old leaky count |
|---|---:|
| Train/test exact duplicate rows | 1213 / 5349 test rows |
| Train/test mention+sentence overlap | 1212 / 5349 test rows |
| Train/test image overlap | 983 / 5349 test rows |
| Gold entity image leakage | 5349 / 5349 test rows |

This explained the earlier inflated result of approximately 96.65 H@1.

A clean no-leak dataset was rebuilt at:

```text
external/MIMIC_reproduction/data/WikiMEL_noleak
```

The clean audit result is:

| Leakage check | Count |
|---|---:|
| Train/dev/test exact overlap | 0 |
| Train/dev/test mention+sentence overlap | 0 |
| Train/dev/test image overlap | 0 |
| Gold entity image leakage | 0 |

Clean split sizes:

| Split/item | Count |
|---|---:|
| Train mentions | 18,092 |
| Dev mentions | 2,083 |
| Test mentions | 4,002 |
| Candidate entities | 109,976 |

All current claims use the clean no-leak checkpoint:

```text
external/MIMIC_reproduction/runs/WikiMEL_noleak/version_4/checkpoints/final.ckpt
```

## Experimental Process

The work was organized into four phases.

### 1. Clean Baseline and Representation Quantization

The first phase evaluated the clean MIMIC-style model with fp32, fp16, and simulated int8 entity-cache representations.

| Mode | H@1 | H@10 | H@20 | MRR | Entity cache |
|---|---:|---:|---:|---:|---:|
| fp32 | 0.741379 | 0.954773 | 0.974013 | 0.818783 | 10860.67 MB |
| fp16 | 0.741379 | 0.954773 | 0.974013 | 0.818784 | 5430.33 MB |
| int8 | 0.742129 | 0.953773 | 0.974263 | 0.819176 | 2753.76 MB |

This established that entity-cache compression can preserve retrieval quality on the clean split.

### 2. Per-Tensor Mixed Quantization

The entity cache was decomposed into four tensors:

| Tensor | fp32 size |
|---|---:|
| text_cls | 214.80 MB |
| image_cls | 40.27 MB |
| text_tokens | 8591.88 MB |
| image_patch_tokens | 2013.72 MB |
| total | 10860.67 MB |

The largest target was `text_tokens`; the second target was `image_patch_tokens`. The strongest deployment configuration keeps CLS tensors high precision and quantizes local token tensors to int4:

```text
text_cls = fp32
image_cls = fp32
text_tokens = int4
image_patch_tokens = int4
```

Cache footprint:

| Configuration | Logical cache | Disk cache | Text-token cache | Image-patch cache |
|---|---:|---:|---:|---:|
| fp32 | 10860.67 MB | 10860.88 MB | 8591.88 MB | 2013.72 MB |
| mixed text+image int4 | 1618.53 MB | 2925.63 MB | 1090.77 MB | 272.69 MB |

This reduces the logical entity cache to about 14.9% of the fp32 cache.

### 3. Chunk-Size Sweep

The next phase varied the number of candidate entities scored per chunk. This showed that quantization alone reduces cache size and transfer time, but does not reduce fixed-chunk peak VRAM because the matcher still materializes large dequantized tensors.

Key repeated rows:

| Mode | Chunk | H@1 | MRR | Peak eval VRAM | Runtime | QPS | Speedup |
|---|---:|---:|---:|---:|---:|---:|---:|
| fp32_ref | 5000 | 0.741629 | 0.818908 | 5851.28 MB | 440.92 s | 9.074 | 1.000x |
| mixed int4 | 5000 | 0.741379 | 0.818733 | 5851.15 MB | 234.23 s | 17.086 | 1.882x |
| fp32_ref | 10000 | 0.741629 | 0.818908 | 11063.21 MB | 433.62 s | 9.229 | 1.000x |
| mixed int4 | 10000 | 0.741379 | 0.818733 | 11063.21 MB | 231.30 s | 17.302 | 1.875x |

Interpretation: mixed int4 is stable and fast, but at the same chunk size the peak memory bottleneck is matcher-side computation, not stored cache size.

### 4. TGLU Matcher Tiling and Entity Subtiling

Profiling showed that the TGLU text global-local unit dominated peak eval memory. TGLU creates large temporary attention tensors over entity text tokens and mention text tokens. An optional matcher tile size was added to score the entity dimension in smaller TGLU blocks while preserving the same ranking result.

TGLU tiling results:

| Mode | Chunk | TGLU tile | H@1 | MRR | Peak eval VRAM | Runtime | QPS |
|---|---:|---:|---:|---:|---:|---:|---:|
| fp32_ref | 5000 | 1000 | 0.741629 | 0.818908 | 2470.60 MB | 442.52 s | 9.044 |
| mixed int4 | 5000 | 1000 | 0.741379 | 0.818733 | 2470.84 MB | 234.46 s | 17.069 |
| fp32_ref | 10000 | 1000 | 0.741629 | 0.818908 | 3458.94 MB | 435.89 s | 9.181 |
| mixed int4 | 10000 | 1000 | 0.741379 | 0.818733 | 3458.94 MB | 232.52 s | 17.212 |

For chunk 5000, a tile sweep found the best balanced row:

| TGLU tile | H@1 | MRR | Peak eval VRAM | Runtime | QPS |
|---:|---:|---:|---:|---:|---:|
| no tile | 0.741379 | 0.818733 | 5851.15 MB | 234.23 s | 17.086 |
| 250 | 0.741379 | 0.818733 | 1922.29 MB | 229.42 s | 17.444 |
| 500 | 0.741379 | 0.818733 | 2049.06 MB | 235.40 s | 17.001 |
| 1000 | 0.741379 | 0.818733 | 2470.84 MB | 234.46 s | 17.069 |
| 2000 | 0.741379 | 0.818733 | 3315.92 MB | 234.89 s | 17.038 |

After TGLU tiling, entity transfer/dequantization became the next memory floor. Entity-subtile transfer was added to slice the packed cache before GPU transfer:

| Path | Entity subtile | H@1 | MRR | Peak eval VRAM | Runtime | QPS |
|---|---:|---:|---:|---:|---:|---:|
| no tiling | 0 | 0.741379 | 0.818733 | 5851.15 MB | 234.23 s | 17.086 |
| TGLU tile 250 | 0 | 0.741379 | 0.818733 | 1922.29 MB | 229.42 s | 17.444 |
| full entity-subtile | 500 | 0.741379 | 0.818733 | 1160.72 MB | 309.58 s | 12.927 |
| full entity-subtile | 250 | 0.741379 | 0.818733 | 901.35 MB | 385.59 s | 10.379 |

This gives two deployment modes:

- TGLU-only tiling is the best throughput/memory balance.
- Entity-subtile transfer is the lowest-memory path, with a runtime tradeoff.

## Rank Stability

Rank movement between fp32 and mixed int4 at chunk size 5000 was small and balanced:

| Metric | Value |
|---|---:|
| Queries compared | 4002 |
| Rank flips | 114 |
| Improved | 55 |
| Worsened | 59 |
| Mean absolute rank delta | 0.266867 |
| Median absolute rank delta | 0.000000 |

This supports the claim that average retrieval behavior is stable, with most queries unchanged and improvements/worsenings nearly balanced.

## Full-Model Weight Quantization Status

Weight quantization was explored as an extension, but it is not the current headline result.

Initial dequantized int8/int4 linear-weight smoke tests did not show immediate accuracy collapse on small slices, but those tests did not use real compressed GPU kernels.

Real bitsandbytes CUDA tests showed a more nuanced result:

| Scope | Weight mode | Queries | H@1 | MRR | Peak eval VRAM | QPS | Interpretation |
|---|---|---:|---:|---:|---:|---:|---|
| all `nn.Linear` | bnb_int8 | 20 | 0.000 | 0.000024 | 1593.78 MB | 7.760 | Accuracy collapse |
| MIMIC head only | bnb_int8 | 20 | 0.950 | 0.975000 | 1944.69 MB | 7.606 | No collapse on tiny slice |
| MIMIC head only | bnb_int8 | 200 | 0.965 | 0.978750 | 1970.85 MB | 9.552 | Viable diagnostic, slower |

Conclusion: naive full-model low-bit replacement is unsafe. Targeted model-weight quantization may be useful later, but current evidence favors entity-cache quantization plus matcher scheduling.

## Main Results to Present

1. The old leaky 96.65 H@1 result was invalidated and replaced with a clean no-leak baseline.
2. Clean fp32 baseline on 4002 test queries and 109,976 entities is approximately H@1 0.7416 and MRR 0.8189.
3. Mixed int4 quantization of entity-side local token caches preserves H@1/MRR within about 0.00025 H@1 and 0.00018 MRR.
4. Mixed int4 reduces logical entity-cache size from 10860.67 MB to 1618.53 MB, about 14.9% of fp32.
5. At chunk size 5000, mixed int4 improves throughput from about 9.07 QPS to about 17.09 QPS.
6. TGLU tiling solves the fixed-chunk peak-memory issue, reducing peak eval VRAM from 5851.15 MB to 1922.29 MB at chunk 5000 with no accuracy loss.
7. Entity-subtile transfer pushes memory lower, down to 901.35 MB peak eval VRAM, while preserving H@1/MRR but reducing QPS to 10.379.

## Shortfalls and Caveats

- The clean baseline H@1 is lower than the original published MIMIC WikiMEL target, but it is valid under the no-leak data audit. The prior higher local result should not be used.
- Peak memory is measured using PyTorch CUDA instrumentation on an RTX 4090, not by enforcing hard memory caps on smaller GPUs.
- The sub-8-bit cache sizes are logical packed sizes; actual disk cache is larger because practical files also store integer codes, scales, and metadata.
- Quantization alone does not reduce fixed-chunk peak VRAM; matcher tiling is required for that claim.
- The entity-subtile path reduces memory the most but repeats more matcher work, so it is a memory-first setting rather than the fastest setting.
- The chunk size 20000 mixed-int4 row was pathological and should be excluded from recommendations.
- Weight quantization is exploratory. Full `nn.Linear` bitsandbytes int8 collapsed accuracy, while targeted MIMIC-head quantization needs more calibration and full-test validation.
- More qualitative analysis is needed to explain which entity types or ambiguity cases are most affected by quantization.

## Recommended Next Steps

1. Lock a canonical final table with five rows: fp32 reference, mixed int4 no tiling, mixed int4 TGLU tile 250, entity subtile 500, and entity subtile 250.
2. Run exact rank-equivalence checks for TGLU-tiled and entity-subtiled paths against the non-tiled mixed-int4 path.
3. Validate the low-memory claims under hard memory caps or on smaller GPUs, especially 2 GB, 4 GB, 6 GB, and 8 GB budgets.
4. Optimize entity-subtile runtime by reusing mention-side projections and reducing repeated VDLU/CMFU work.
5. Add slice-level error analysis: entity type, ambiguity level, visual-vs-text dependency, and rank-flip examples.
6. Treat model-weight quantization as a future extension after calibration, not as the central result.

## Current Best Recommendation

For advisor discussion and near-term reporting, use this as the main operating point:

```text
text_image_tokens_int4, chunk_size=5000, matcher_tile_size=250
```

It is the best balance of accuracy, speed, and memory:

| Metric | Value |
|---|---:|
| H@1 | 0.741379 |
| MRR | 0.818733 |
| Logical cache | 1618.53 MB |
| Peak eval VRAM | 1922.29 MB |
| Runtime | 229.42 s |
| QPS | 17.444 |

The concise research story is:

> Clean the benchmark, profile the valid baseline, quantize the large entity-side token caches, and schedule matcher computation for the target GPU budget. The current implementation preserves WikiMEL retrieval quality while substantially reducing cache size, transfer cost, runtime, and peak evaluation memory.
