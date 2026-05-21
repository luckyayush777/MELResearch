# Final Status Report - WikiMEL No-Leak Quantization and Tiled Matching

Saved on 2026-05-20.

## Executive Summary

The project now has a clean, no-leak MIMIC WikiMEL baseline and a strong smaller-GPU deployment story.

The old leaky WikiMEL result of roughly `96.65 H@1` has been retired. The valid clean no-leak baseline is approximately:

| Metric | Clean fp32 baseline |
|---|---:|
| H@1 | 0.741379 to 0.741629 |
| MRR | 0.818783 to 0.818908 |
| Queries | 4002 |
| Entities | 109976 |

The best current result depends on the deployment goal:

| Goal | Recommended setting | Peak eval VRAM | QPS | H@1 | MRR |
|---|---|---:|---:|---:|---:|
| Best throughput/memory balance | `text_image_tokens_int4`, `chunk_size=5000`, `matcher_tile_size=250` | 1922.29 MB | 17.444 | 0.741379 | 0.818733 |
| Practical sub-1.2 GB setting | `text_image_tokens_int4`, `chunk_size=5000`, `entity_subtile_size=500` | 1160.72 MB | 12.927 | 0.741379 | 0.818733 |
| Lowest measured memory | `text_image_tokens_int4`, `chunk_size=5000`, `entity_subtile_size=250` | 901.35 MB | 10.379 | 0.741379 | 0.818733 |
| Conservative scientific fallback | `text_tokens_int4`, `chunk_size=5000` | 5850.26 MB | 14.559 | 0.741629 | 0.818910 |

The core finding is now stronger than the original quantization result:

```text
Mixed int4 cache quantization reduces cache size, disk footprint, transfer time, and runtime.
TGLU matcher tiling reduces fixed-chunk peak VRAM with almost no runtime cost.
Full entity-subtile transfer/dequantization pushes memory even lower by slicing the packed cache before moving it to GPU, with an expected runtime tradeoff.
```

## Source Reports And Artifacts Read

Narrative reports:

```text
18 MayResult.md
19MayReport.md
19MayGPUMemoryFindings.md
20 MayResult.md
tilingResults20May.md
SmallerCheapGPU_MetricsPlan.md
Mixed Quantization setup/TILED_MATCHER_INVESTIGATION.md
Mixed Quantization setup/results/keyrow_report/KEYROW_REPORT.md
```

Primary aggregate results:

```text
ExpSetup/benchmarks/mimic_wikimel_noleak_quantization/mimic_quantization_summary.csv
Mixed Quantization setup/results/entity_text_tokens_full/entity_text_token_quantization_summary.csv
Mixed Quantization setup/results/chunk_size_sweep_full/chunk_size_sweep_summary.csv
Mixed Quantization setup/results/chunk_size_sweep_repeat_keyrows/chunk_size_sweep_summary.csv
Mixed Quantization setup/results/keyrow_report/gpu_budget_recommendation.csv
Mixed Quantization setup/results/keyrow_report/conservative_text_tokens_int4.csv
Mixed Quantization setup/results/tiled_matcher_chunk5000_tile_sweep_summary.csv
Mixed Quantization setup/results/entity_subtile_chunk5000_summary.csv
```

## Clean Dataset And Baseline Status

The initial high WikiMEL score was invalid because the dataset had leakage:

- exact train/test duplicate rows;
- mention, sentence, and image overlaps across splits;
- every test mention image appeared in the gold entity image list.

The fixed clean dataset is:

```text
ExpSetup/external/MIMIC_reproduction/data/WikiMEL_noleak
```

Clean config:

```text
ExpSetup/external/MIMIC_reproduction/config/wikimel_noleak.yaml
```

Clean checkpoint:

```text
ExpSetup/external/MIMIC_reproduction/runs/WikiMEL_noleak/version_4/checkpoints/final.ckpt
```

Leakage audit for `WikiMEL_noleak`:

| Leakage check | Count |
|---|---:|
| Train/dev/test exact overlap | 0 |
| Train/dev/test mention+sentence overlap | 0 |
| Train/dev/test image overlap | 0 |
| Gold entity image leakage | 0 |

This clean checkpoint is the only valid checkpoint for the current results. The old leaky checkpoint should not be used for comparisons.

## Quantization Findings

### Early Clean Quantization

The first clean quantization pass showed fp16 and simulated int8 preserved accuracy while reducing representation size:

| Mode | H@1 | H@10 | H@20 | MRR | Entity cache |
|---|---:|---:|---:|---:|---:|
| fp32 | 0.741379 | 0.954773 | 0.974013 | 0.818783 | 10860.67 MB |
| fp16 | 0.741379 | 0.954773 | 0.974013 | 0.818784 | 5430.33 MB |
| int8 | 0.742129 | 0.953773 | 0.974263 | 0.819176 | 2753.76 MB |

### Entity Text-Token Quantization

`entity_text_tokens` are the largest entity-cache tensor and feed the expensive TGLU matcher path.

Fp32 cache composition:

| Tensor | Size |
|---|---:|
| text_cls | 214.80 MB |
| image_cls | 40.27 MB |
| text_tokens | 8591.88 MB |
| image_patch_tokens | 2013.72 MB |
| total | 10860.67 MB |

Text-token-only quantization results:

| Mode | H@1 | MRR | Speedup | Total logical cache | Text-token cache |
|---|---:|---:|---:|---:|---:|
| fp32 | 0.741379 | 0.818783 | 1.00x | 10860.67 MB | 8591.88 MB |
| int8 | 0.741379 | 0.818783 | 1.09x | 4433.54 MB | 2164.75 MB |
| int6 | 0.741379 | 0.818783 | 1.10x | 3896.55 MB | 1627.76 MB |
| int4 | 0.741379 | 0.818785 | 1.77x | 3359.56 MB | 1090.77 MB |
| int3 | 0.741629 | 0.818859 | 1.90x | 3091.06 MB | 822.27 MB |
| int2 | 0.741129 | 0.818744 | 1.89x | 2822.57 MB | 553.77 MB |

The conservative result is `entity_text_tokens=int4`: it preserves H@1/MRR while reducing total logical cache from `10860.67 MB` to `3359.56 MB` and speeding evaluation from `1378.72 s` to `778.41 s`.

### Mixed Text+Image Token Int4

The strongest deployment cache configuration is:

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
| text_image_tokens_int4 | 1618.53 MB | 2925.63 MB | 1090.77 MB | 272.69 MB |

Mixed int4 reduces logical cache to about `14.9%` of fp32.

## Chunk-Size Sweep Status

Before matcher tiling, peak eval VRAM was controlled mainly by `chunk_size`, not by cache quantization. At a fixed chunk, fp32 and int4 used nearly identical peak VRAM because the matcher still materialized large dequantized tensors.

Full sweep summary for usable rows:

| Mode | Chunk | H@1 | MRR | Peak eval VRAM | Runtime | QPS | Speedup vs fp32 same chunk |
|---|---:|---:|---:|---:|---:|---:|---:|
| fp32_ref | 1000 | 0.741629 | 0.818908 | 1680.02 MB | 476.85 s | 8.393 | 1.000x |
| text_image_tokens_int4 | 1000 | 0.741379 | 0.818733 | 1680.08 MB | 265.41 s | 15.078 | 1.797x |
| fp32_ref | 2500 | 0.741629 | 0.818908 | 3244.45 MB | 441.58 s | 9.063 | 1.000x |
| text_image_tokens_int4 | 2500 | 0.741379 | 0.818733 | 3243.63 MB | 251.31 s | 15.925 | 1.757x |
| fp32_ref | 5000 | 0.741629 | 0.818908 | 5850.26 MB | 436.03 s | 9.178 | 1.000x |
| text_image_tokens_int4 | 5000 | 0.741379 | 0.818733 | 5850.26 MB | 237.58 s | 16.845 | 1.835x |
| fp32_ref | 10000 | 0.741629 | 0.818908 | 11064.15 MB | 434.16 s | 9.218 | 1.000x |
| text_image_tokens_int4 | 10000 | 0.741379 | 0.818733 | 11064.74 MB | 238.20 s | 16.801 | 1.823x |

The `chunk_size=20000` mixed row is not recommended. It reached `21491.13 MB` peak and regressed badly to `2558.70 s` runtime.

## Repeated Key Rows

The key deployment rows were repeated for `fp32_ref` and `text_image_tokens_int4` at chunks `5000` and `10000`.

| Mode | Chunk | H@1 | MRR | Peak eval VRAM | Runtime | QPS | Speedup |
|---|---:|---:|---:|---:|---:|---:|---:|
| fp32_ref | 5000 | 0.741629 | 0.818908 | 5851.28 MB | 440.92 s | 9.074 | 1.000x |
| text_image_tokens_int4 | 5000 | 0.741379 | 0.818733 | 5851.15 MB | 234.23 s | 17.086 | 1.882x |
| fp32_ref | 10000 | 0.741629 | 0.818908 | 11063.21 MB | 433.62 s | 9.229 | 1.000x |
| text_image_tokens_int4 | 10000 | 0.741379 | 0.818733 | 11063.21 MB | 231.30 s | 17.302 | 1.875x |

This confirmed that mixed int4 is stable and fast, but also confirmed that quantization alone does not reduce fixed-chunk peak VRAM.

## Rank Stability

Rank movement between `fp32_ref_chunk_5000` and `text_image_tokens_int4_chunk_5000` was small and balanced:

| Metric | Value |
|---|---:|
| Queries compared | 4002 |
| Rank flips | 114 |
| Improved | 55 |
| Worsened | 59 |
| Mean absolute rank delta | 0.266867 |
| Median absolute rank delta | 0.000000 |
| Max improvement | -86 |
| Max worsening | 210 |

Interpretation: the average retrieval metrics are stable, improvements and regressions are nearly balanced, and most queries do not move.

## TGLU Matcher Tiling

The fixed-chunk peak memory bottleneck was TGLU. In the repeated key rows, peak eval allocation equaled the TGLU peak:

| Mode | Chunk | Peak eval VRAM | Peak TGLU |
|---|---:|---:|---:|
| fp32_ref | 5000 | 5851.28 MB | 5851.28 MB |
| text_image_tokens_int4 | 5000 | 5851.15 MB | 5851.15 MB |
| fp32_ref | 10000 | 11063.21 MB | 11063.21 MB |
| text_image_tokens_int4 | 10000 | 11063.21 MB | 11063.21 MB |

An optional `--matcher-tile-size` was added. It keeps the external chunk fixed, but splits only the entity dimension inside TGLU.

Full key rows with `matcher_tile_size=1000`:

| Mode | Chunk | TGLU tile | H@1 | MRR | Peak eval VRAM | Runtime | QPS | Speedup vs tiled fp32 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| fp32_ref | 5000 | 1000 | 0.741629 | 0.818908 | 2470.60 MB | 442.52 s | 9.044 | 1.000x |
| text_image_tokens_int4 | 5000 | 1000 | 0.741379 | 0.818733 | 2470.84 MB | 234.46 s | 17.069 | 1.887x |
| fp32_ref | 10000 | 1000 | 0.741629 | 0.818908 | 3458.94 MB | 435.89 s | 9.181 | 1.000x |
| text_image_tokens_int4 | 10000 | 1000 | 0.741379 | 0.818733 | 3458.94 MB | 232.52 s | 17.212 | 1.875x |

Peak reduction:

| Mode | Chunk | Non-tiled peak | Tiled peak | Relative reduction |
|---|---:|---:|---:|---:|
| text_image_tokens_int4 | 5000 | 5851.15 MB | 2470.84 MB | 57.8% |
| text_image_tokens_int4 | 10000 | 11063.21 MB | 3458.94 MB | 68.7% |

## Chunk 5000 TGLU Tile Sweep

The chunk `5000` tile sweep found that TGLU-only tiling reduces memory until the entity transfer/dequantization floor dominates.

| TGLU tile | H@1 | MRR | Peak eval VRAM | Transfer peak | Runtime | QPS | Reduction vs no tile |
|---:|---:|---:|---:|---:|---:|---:|---:|
| no tile | 0.741379 | 0.818733 | 5851.15 MB | 1921.78 MB | 234.23 s | 17.086 | 0.0% |
| 250 | 0.741379 | 0.818733 | 1922.29 MB | 1922.29 MB | 229.42 s | 17.444 | 67.1% |
| 500 | 0.741379 | 0.818733 | 2049.06 MB | 1921.84 MB | 235.40 s | 17.001 | 65.0% |
| 1000 | 0.741379 | 0.818733 | 2470.84 MB | 1922.46 MB | 234.46 s | 17.069 | 57.8% |
| 2000 | 0.741379 | 0.818733 | 3315.92 MB | 1921.84 MB | 234.89 s | 17.038 | 43.3% |

Best balance from this sweep:

```text
text_image_tokens_int4, chunk_size=5000, matcher_tile_size=250
```

It preserves accuracy, keeps the best observed QPS among these rows, and lowers peak eval allocation to about `1.92 GB`.

## Full Entity-Subtile Transfer/Dequantization

After TGLU-only tiling, the next floor was moving/dequantizing the full 5000-entity cache chunk. The evaluator now supports:

```text
--entity-subtile-size
```

This slices the packed cache before transfer to GPU, while preserving the outer `chunk_size=5000` scheduling/reporting unit.

Entity-subtile results:

| Path | Entity subtile | TGLU tile | H@1 | MRR | Peak eval VRAM | Transfer peak | Runtime | QPS | Entity subtiles |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| no tiling | 0 | 0 | 0.741379 | 0.818733 | 5851.15 MB | 1921.78 MB | 234.23 s | 17.086 | 4422 |
| TGLU-only tiling | 0 | 250 | 0.741379 | 0.818733 | 1922.29 MB | 1922.29 MB | 229.42 s | 17.444 | 4422 |
| full entity-subtile | 500 | 0 | 0.741379 | 0.818733 | 1160.72 MB | 764.44 MB | 309.58 s | 12.927 | 44220 |
| full entity-subtile | 250 | 0 | 0.741379 | 0.818733 | 901.35 MB | 699.97 MB | 385.59 s | 10.379 | 88440 |

Memory reduction vs no tiling:

| Configuration | Peak eval VRAM | Reduction |
|---|---:|---:|
| no tiling | 5851.15 MB | 0.0% |
| TGLU-only tile 250 | 1922.29 MB | 67.1% |
| entity subtile 500 | 1160.72 MB | 80.2% |
| entity subtile 250 | 901.35 MB | 84.6% |

The entity-subtile path is now the lowest-memory path, but it repeats more full-matcher work:

| Configuration | TGLU time | VDLU time | CMFU time | Entity transfer | Runtime | QPS |
|---|---:|---:|---:|---:|---:|---:|
| no tiling | 94.61 s | 8.59 s | 4.12 s | 89.75 s | 234.23 s | 17.086 |
| TGLU-only tile 250 | 91.55 s | 8.77 s | 4.43 s | 89.82 s | 229.42 s | 17.444 |
| entity subtile 500 | 105.68 s | 29.96 s | 21.99 s | 104.43 s | 309.58 s | 12.927 |
| entity subtile 250 | 109.95 s | 55.65 s | 42.09 s | 121.29 s | 385.59 s | 10.379 |

Interpretation:

```text
TGLU-only tiling is the best throughput/memory balance.
Full entity-subtile transfer/dequantization is the lowest-memory path.
```

## Current Claims That Are Supported

Strong supported claim:

```text
On clean no-leak WikiMEL, mixed 4-bit quantization of entity-side text and image local-token caches preserves retrieval quality while reducing logical cache size to about 14.9% of fp32 and improving throughput by roughly 1.8x at practical chunk sizes.
```

Updated fixed-chunk memory claim:

```text
At chunk_size=5000, adding TGLU matcher tiling reduces peak eval allocation from about 5.85 GB to as low as 1.92 GB with unchanged H@1/MRR and no throughput loss.
```

Lowest-memory claim:

```text
At chunk_size=5000, slicing the packed cache before GPU transfer with entity_subtile_size=250 lowers peak eval allocation to about 0.90 GB while preserving H@1/MRR, at the cost of lower throughput.
```

Claims to avoid:

```text
Quantization alone reduces fixed-chunk peak VRAM.
The system is proven on every small GPU.
The sub-8-bit artifact size equals the logical packed representation size.
```

## Current Caveats

- Peak memory is measured with PyTorch CUDA instrumentation on an RTX 4090, not with enforced hard memory caps.
- The clean no-leak setup is valid, but direct target-device tests are still needed for deployment claims.
- Sub-8-bit cache-size columns are logical packed sizes; actual disk artifacts may be larger because integer codes and scales are stored in practical containers.
- `int3` and `int2` are interesting but should be repeated before becoming central claims.
- Full entity-subtile transfer/dequantization currently repeats VDLU/CMFU and other matcher work, so it is a memory-first path rather than a throughput-first path.
- The `chunk_size=20000` mixed row is pathological and should be excluded from recommendations.

## Recommended Next Steps

### 1. Lock The Current Best Rows

Create one final canonical CSV/report table with these rows:

| Label | Mode | Chunk | Matcher tile | Entity subtile |
|---|---|---:|---:|---:|
| fp32 reference | fp32_ref | 5000 | 0 | 0 |
| mixed int4 no tiling | text_image_tokens_int4 | 5000 | 0 | 0 |
| best balanced | text_image_tokens_int4 | 5000 | 250 | 0 |
| practical low memory | text_image_tokens_int4 | 5000 | 0 | 500 |
| lowest memory | text_image_tokens_int4 | 5000 | 0 | 250 |

This makes the story clean for slides/paper/reporting.

### 2. Run Rank Equivalence Checks For Tiled Paths

Compare rank batches for:

```text
non-tiled mixed int4
TGLU-only tile 250
entity_subtile_size 500
entity_subtile_size 250
```

Expected result: identical H@1/MRR and likely identical or near-identical ranks. Document exact rank equality or the number of rank movements.

### 3. Run Enforced-Memory Validation

Use a smaller GPU or enforce memory caps to validate the memory-budget story.

Priority budgets:

```text
2 GB
4 GB
6 GB
8 GB
12 GB
```

The most valuable validation is proving that:

```text
matcher_tile_size=250 works under a roughly 2 GB budget
entity_subtile_size=500 works under a roughly 1.2 GB budget
entity_subtile_size=250 works under a roughly 1 GB budget
```

### 4. Optimize Entity-Subtile Runtime

The entity-subtile implementation proves the memory floor can be broken. The next engineering challenge is reducing repeated matcher overhead.

Promising directions:

- reuse mention-side projections across entity subtiles;
- avoid recomputing VDLU/CMFU components that can be shared;
- keep transfer/dequantization subtiled while batching matcher calls more efficiently;
- evaluate a hybrid setting such as `entity_subtile_size=500` plus internal TGLU tiling only if it reduces peaks without much extra overhead.

### 5. Add Slice And Error Analysis

For paper-quality evidence, add:

- entity-type breakdowns;
- ambiguous-name buckets;
- visually grounded vs text-dominant examples;
- quantization error metrics per tensor;
- rank-flip examples with full query context.

This will explain where quantization is harmless, where it helps, and where it hurts.

### 6. Package The Framework Story

The current framework pitch should be:

```text
Profile the clean baseline.
Quantize selected entity-cache tensors.
Measure accuracy, cache size, transfer cost, throughput, and peak memory.
Choose chunk/tile/subtile scheduling for the target GPU budget.
Report the best configuration under an accuracy-loss constraint.
```

The current best demonstration of that framework is:

```text
text_image_tokens_int4 + TGLU tiling for the balanced path
text_image_tokens_int4 + entity-subtile transfer/dequantization for the lowest-memory path
```

## Bottom Line

The work has moved from a simple quantization result to a stronger deployment framework:

```text
Mixed int4 cache compression gives the speed/storage win.
TGLU tiling gives the fixed-chunk peak-memory win.
Entity-subtile transfer/dequantization gives the sub-1 GB memory path.
```

The safest headline operating point is:

```text
text_image_tokens_int4, chunk_size=5000, matcher_tile_size=250
```

It preserves clean WikiMEL retrieval quality, reduces logical cache to about `14.9%` of fp32, cuts peak eval VRAM from about `5.85 GB` to about `1.92 GB`, and keeps throughput high at `17.444 QPS`.
