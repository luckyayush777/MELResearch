# 19 May GPU Memory Findings

## Purpose

This report summarizes the GPU memory and runtime findings from the chunk-size sweep for the WikiMEL no-leak multimodal entity linking quantization experiments. The immediate goal was to find the strongest short-term deployment story: lower hardware requirements and faster evaluation without meaningful accuracy loss.

The key result is that quantization alone does not reduce peak matcher VRAM at a fixed chunk size, because the current evaluation path still dequantizes into the matcher/TGLU computation. The real practical win is quantization plus chunk-size control: compressed caches reduce storage and transfer cost, while chunk size sets the peak GPU allocation.

## Inputs

- Metrics plan: `SmallerCheapGPU_MetricsPlan.md`
- Sweep output: `Mixed Quantization setup/results/chunk_size_sweep_full/chunk_size_sweep_summary.csv`
- Full sweep JSON: `Mixed Quantization setup/results/chunk_size_sweep_full/chunk_size_sweep_results.json`
- Logs:
  - `Mixed Quantization setup/results/chunk_size_sweep_full/run.err.log`
  - `Mixed Quantization setup/results/chunk_size_sweep_full/run_no40000.err.log`

The 40000 chunk-size run was intentionally stopped because it was already too slow and the 20000 row had proven the large-chunk behavior.

## Main Finding

The best deployment point from this sweep is:

| Mode | Chunk | H@1 | MRR | Peak eval VRAM | Runtime | QPS | Speedup vs fp32 same chunk |
|---|---:|---:|---:|---:|---:|---:|---:|
| text_image_tokens_int4 | 5000 | 0.741379 | 0.818733 | 5850.26 MB | 237.58 s | 16.845 | 1.835x |

This is the strongest short-term framework result so far: text+image int4 keeps accuracy effectively unchanged while running under roughly 6 GB peak allocated eval VRAM and improving throughput by about 1.84x versus the fp32 row at the same chunk size.

## Accuracy Summary

For chunk sizes 1000, 2500, 5000, and 10000:

| Mode | Accuracy behavior |
|---|---|
| fp32_ref | H@1 0.741629, MRR 0.818908 |
| text_tokens_int4 | H@1 0.741629, MRR 0.818910 |
| text_image_tokens_int4 | H@1 0.741379, MRR 0.818733 |

The text-only int4 run shows no H@1 loss and a tiny positive MRR movement, which is measurement-level noise. The mixed text+image int4 run loses only 0.000250 H@1 and 0.000175 MRR against fp32 at the same chunk size. That is practically negligible for the short-term sales/framework claim.

At chunk size 20000, fp32 and int4 rows use a slightly different fp32 reference accuracy profile:

| Mode | Chunk | H@1 | MRR |
|---|---:|---:|---:|
| fp32_ref | 20000 | 0.741379 | 0.818783 |
| text_tokens_int4 | 20000 | 0.741379 | 0.818785 |
| text_image_tokens_int4 | 20000 | 0.741129 | 0.818608 |

The 20000 mixed row should not be treated as the recommended operating point because its runtime is anomalous.

## Memory And Runtime Sweep

### fp32 Reference

| Chunk | Peak eval VRAM | Runtime | QPS | Avg query latency | P95 query latency | Transfer | TGLU |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1000 | 1680.02 MB | 476.85 s | 8.393 | 112.63 ms | 118.72 ms | 304.24 s | 100.02 s |
| 2500 | 3244.45 MB | 441.58 s | 9.063 | 104.17 ms | 106.82 ms | 294.46 s | 95.46 s |
| 5000 | 5850.26 MB | 436.03 s | 9.178 | 102.78 ms | 103.71 ms | 293.97 s | 93.18 s |
| 10000 | 11064.15 MB | 434.16 s | 9.218 | 102.27 ms | 105.44 ms | 292.38 s | 92.96 s |
| 20000 | 21491.26 MB | 588.53 s | 6.800 | 271.12 ms | 1299.44 ms | 293.92 s | 767.42 s |

The fp32 rows show the expected memory trend: larger chunks increase peak GPU allocation. Runtime improves from chunk 1000 to 10000, then becomes worse at 20000 because the matcher/TGLU work gets too heavy.

### Text Tokens Int4

| Chunk | H@1 | MRR | Peak eval VRAM | Runtime | QPS | Transfer | TGLU | Speedup |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1000 | 0.741629 | 0.818910 | 1680.08 MB | 303.29 s | 13.195 | 135.27 s | 98.11 s | 1.572x |
| 2500 | 0.741629 | 0.818910 | 3243.63 MB | 275.09 s | 14.548 | 129.39 s | 94.53 s | 1.605x |
| 5000 | 0.741629 | 0.818910 | 5850.26 MB | 274.88 s | 14.559 | 130.24 s | 94.42 s | 1.586x |
| 10000 | 0.741629 | 0.818910 | 11064.74 MB | 279.75 s | 14.305 | 126.33 s | 104.30 s | 1.552x |
| 20000 | 0.741379 | 0.818785 | 21491.13 MB | 480.45 s | 8.330 | 127.42 s | 304.10 s | 1.225x |

Text-token int4 is a very clean result. It gives a large transfer-time reduction and a stable 1.55x to 1.61x speedup over fp32 for the usable chunk range, with no accuracy loss.

### Text And Image Tokens Int4

| Chunk | H@1 | MRR | Peak eval VRAM | Runtime | QPS | Transfer | TGLU | Speedup |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1000 | 0.741379 | 0.818733 | 1680.08 MB | 265.41 s | 15.078 | 97.89 s | 98.88 s | 1.797x |
| 2500 | 0.741379 | 0.818733 | 3243.63 MB | 251.31 s | 15.925 | 92.04 s | 105.27 s | 1.757x |
| 5000 | 0.741379 | 0.818733 | 5850.26 MB | 237.58 s | 16.845 | 89.06 s | 98.71 s | 1.835x |
| 10000 | 0.741379 | 0.818733 | 11064.74 MB | 238.20 s | 16.801 | 89.80 s | 99.56 s | 1.823x |
| 20000 | 0.741129 | 0.818608 | 21491.13 MB | 2558.70 s | 1.564 | 90.17 s | 2420.28 s | 0.230x |

The mixed text+image int4 result is the strongest hardware story. It has the lowest transfer cost and highest throughput in the usable range. The 20000 row is pathological and should be excluded from deployment recommendations.

## Cache And Storage Reduction

The earlier mixed quantization run measured cache footprint directly:

| Configuration | Logical cache | Disk cache | Text-token cache | Image-patch cache |
|---|---:|---:|---:|---:|
| fp32 | 10860.67 MB | 10860.88 MB | 8591.88 MB | 2013.72 MB |
| text_image_tokens_int4 | 1618.53 MB | 2925.63 MB | 1090.77 MB | 272.69 MB |

This means mixed int4 reduces logical cache footprint to about 14.90% of fp32. Image-patch cache drops to 13.54% of fp32, and text-token cache drops to 12.70% of fp32.

That explains the major transfer-time improvement:

| Mode | Transfer at chunk 5000 |
|---|---:|
| fp32_ref | 293.97 s |
| text_tokens_int4 | 130.24 s |
| text_image_tokens_int4 | 89.06 s |

## Hardware Budget Recommendations

| Hardware budget | Recommended mode | Chunk | Peak eval VRAM | Expected behavior |
|---|---|---:|---:|---|
| Around 2 GB | text_image_tokens_int4 | 1000 | 1680.08 MB | Safest low-memory setting; still 1.80x faster than fp32 same chunk |
| Around 4 GB | text_image_tokens_int4 | 2500 | 3243.63 MB | Good low-memory throughput at 15.93 QPS |
| Around 6 GB | text_image_tokens_int4 | 5000 | 5850.26 MB | Best overall operating point |
| Around 12 GB | text_image_tokens_int4 | 10000 | 11064.74 MB | Similar speed to chunk 5000, but uses much more VRAM |
| Around 24 GB | Avoid chunk 20000 for mixed | 20000 | 21491.13 MB | Runtime regresses badly; not worth it |

The important framework message is not simply "quantize and it becomes smaller." It is: quantization reduces cache/storage/transfer, then the framework selects a chunk size that fits the target GPU memory budget.

## Why Peak VRAM Does Not Drop At Fixed Chunk Size

At the same chunk size, fp32 and int4 have almost identical peak allocated eval VRAM. This suggests the current peak is dominated by matcher-side tensors and TGLU computation after dequantization, not by the stored cache representation.

So the current implementation gives maximum gains in:

- cache footprint
- disk footprint
- host-to-device/cache transfer time
- throughput
- ability to use smaller chunk sizes while keeping runtime acceptable

It does not yet give maximum gains in:

- peak matcher allocation at a fixed chunk size
- model compute memory during the TGLU-heavy section

To reduce fixed-chunk peak VRAM further, the next technical step would be quantized or tiled matcher computation that avoids fully materializing large dequantized tensors.

## Interpretation For The Short-Term Framework

This is enough to support a practical framework pitch:

1. The framework profiles a baseline.
2. It quantizes selected cache tensors.
3. It measures accuracy deltas, cache size, transfer time, throughput, and peak memory.
4. It recommends the best configuration for a GPU memory budget.

For this workload, the recommended output is text+image int4 with chunk size 5000. It keeps H@1 effectively unchanged, lowers logical cache to about 15% of fp32, lowers transfer time from 293.97 s to 89.06 s at the 5000 chunk, and runs at about 1.84x the fp32 throughput under roughly 6 GB peak eval allocation.

## Caveats

- The 20000 mixed row is anomalously slow and should not be used as a deployment recommendation.
- The 40000 row was intentionally stopped and is not part of the final sweep.
- Peak eval VRAM is reported from the run instrumentation. It is a strong feasibility signal, but a hard deployment claim should be confirmed on the target GPU with memory limits enforced.
- The current system still dequantizes for matcher computation, so fixed-chunk peak VRAM is not expected to fall much yet.

## Next Steps

1. Repeat the recommended rows, especially text+image int4 at chunks 5000 and 10000, to confirm stability.
2. Add an automatic hardware-budget recommender to the framework.
3. Run a true constrained-memory test on a smaller GPU or with enforced memory caps.
4. Investigate tiled or quantized matcher computation to attack peak VRAM directly.
5. Capture per-query rank movement examples for paper-quality analysis.
