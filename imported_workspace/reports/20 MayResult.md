# 20 May Result

Saved on 2026-05-20.

## Goal

Confirm the practical smaller-GPU result for clean no-leak MIMIC WikiMEL inference, then package the evidence into a concise deployment report.

This report uses only the clean no-leak checkpoint:

```text
ExpSetup/external/MIMIC_reproduction/runs/WikiMEL_noleak/version_4/checkpoints/final.ckpt
```

Old leaky checkpoint was not used.

## What Was Done

The key rows were repeated for:

```text
fp32_ref
text_image_tokens_int4
```

at chunk sizes:

```text
5000
10000
```

New repeat output:

```text
Mixed Quantization setup/results/chunk_size_sweep_repeat_keyrows/chunk_size_sweep_summary.csv
Mixed Quantization setup/results/chunk_size_sweep_repeat_keyrows/chunk_size_sweep_results.json
```

A reusable report generator was added:

```text
Mixed Quantization setup/build_keyrow_report.py
```

Generated report artifacts:

```text
Mixed Quantization setup/results/keyrow_report/KEYROW_REPORT.md
Mixed Quantization setup/results/keyrow_report/gpu_budget_recommendation.csv
Mixed Quantization setup/results/keyrow_report/conservative_text_tokens_int4.csv
Mixed Quantization setup/results/keyrow_report/rank_flip_examples.csv
Mixed Quantization setup/results/keyrow_report/rank_flip_examples.json
```

## Fresh Repeat Results

The repeated key rows confirm that `text_image_tokens_int4` is stable at the practical operating chunks.

| Mode | Chunk | H@1 | MRR | Delta H@1 vs fp32 | Peak eval VRAM | Runtime | QPS | Speedup vs fp32 same chunk |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| text_image_tokens_int4 | 5000 | `0.741379` | `0.818733` | `-0.000250` | `5851.15 MB` | `234.23 s` | `17.086` | `1.882x` |
| text_image_tokens_int4 | 10000 | `0.741379` | `0.818733` | `-0.000250` | `11063.21 MB` | `231.30 s` | `17.302` | `1.875x` |

The earlier full sweep had already shown the same behavior. The fresh repeat strengthens the result and makes the `chunk_size=5000` row a solid headline deployment point.

## GPU Budget Recommendation

| Hardware budget | Recommended mode | Chunk | H@1 | MRR | Peak eval VRAM | QPS | P95 query latency | Speedup | Logical cache | Disk cache |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Around 2 GB | text_image_tokens_int4 | 1000 | `0.741379` | `0.818733` | `1680.08 MB` | `15.078` | `61.00 ms` | `1.797x` | `1618.53 MB` | `2925.63 MB` |
| Around 4 GB | text_image_tokens_int4 | 2500 | `0.741379` | `0.818733` | `3243.63 MB` | `15.925` | `54.15 ms` | `1.757x` | `1618.53 MB` | `2925.63 MB` |
| Around 6 GB | text_image_tokens_int4 | 5000 | `0.741379` | `0.818733` | `5851.15 MB` | `17.086` | `53.58 ms` | `1.882x` | `1618.53 MB` | `2925.63 MB` |
| Around 12 GB | text_image_tokens_int4 | 10000 | `0.741379` | `0.818733` | `11063.21 MB` | `17.302` | `52.74 ms` | `1.875x` | `1618.53 MB` | `2925.63 MB` |

Best current recommendation:

```text
text_image_tokens_int4, chunk_size=5000
```

Reason:

- Fits under roughly `6 GB` peak eval allocation.
- Keeps H@1 effectively unchanged.
- Gives the strongest practical speedup in the repeated rows.
- Keeps logical entity cache at about `14.9%` of fp32.

## Conservative Fallback

If the mixed text+image int4 row feels too aggressive, the safer fallback is:

```text
text_tokens_int4, chunk_size=5000
```

| Mode | Chunk | H@1 | MRR | Delta H@1 vs fp32 | Peak eval VRAM | QPS | Speedup | Logical cache |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| text_tokens_int4 | 5000 | `0.741629` | `0.818910` | `0.000000` | `5850.26 MB` | `14.559` | `1.586x` | `3359.56 MB` |

This is the strongest conservative scientific result because it preserves accuracy with no H@1 loss while still giving a large cache and runtime improvement.

## Rank-Flip Summary

Rank stability was computed from the fresh repeated `chunk_size=5000` rank batches:

```text
fp32_ref_chunk_5000
text_image_tokens_int4_chunk_5000
```

Summary:

| Metric | Value |
|---|---:|
| Queries compared | `4002` |
| Rank flips | `114` |
| Improved | `55` |
| Worsened | `59` |
| Mean absolute rank delta | `0.266867` |
| Median absolute rank delta | `0.000000` |
| Max improvement | `-86` |
| Max worsening | `210` |

The rank movement is balanced: improvements and worsenings are nearly even, and the median rank movement is zero.

Example rank movements:

| Direction | Query idx | Mention | Target | fp32 rank | int4 rank | Delta |
|---|---:|---|---|---:|---:|---:|
| improved | 2440 | Julie London | Julie London | 2486 | 2400 | -86 |
| improved | 809 | Wilberforce | William Wilberforce | 10004 | 9930 | -74 |
| improved | 1293 | Atahualpa | Atahualpa | 25697 | 25623 | -74 |
| worsened | 27 | Brown | Sherrod Brown | 57086 | 57296 | 210 |
| worsened | 2657 | Edna Purviance | Edna Purviance | 13226 | 13352 | 126 |
| worsened | 2372 | Hu Shih | Hu Shih | 11651 | 11771 | 120 |

Full examples:

```text
Mixed Quantization setup/results/keyrow_report/rank_flip_examples.csv
Mixed Quantization setup/results/keyrow_report/rank_flip_examples.json
```

## Main Claim

The strongest current claim is:

```text
On clean no-leak WikiMEL, mixed 4-bit quantization of entity-side text and image local-token caches preserves retrieval quality while reducing logical cache size to about 14.9% of fp32 and enabling a faster low-memory operating point around 6 GB peak eval VRAM.
```

Use this deployment framing:

```text
Quantization reduces cache size and transfer cost. Chunk-size selection controls peak eval VRAM. Together, mixed int4 plus chunk-size control moves MIMIC-style multimodal entity linking toward smaller, cheaper GPU deployment.
```

Do not claim:

```text
Quantization alone reduces fixed-chunk peak VRAM.
```

Reason:

```text
At the same chunk size, fp32 and int4 have nearly identical peak eval allocation because the current matcher/TGLU path still materializes large dequantized tensors during scoring.
```

## Recommended Next Step

The next technical step is rank-flip analysis in a separate pass:

```text
inspect improved/worsened examples
group flips by entity type or ambiguity
identify whether visual cases behave differently from text-dominant cases
```

After that, the next engineering step is:

```text
tiled or quantized matcher computation
```

That is the path to reducing peak VRAM at a fixed chunk size, not just reducing cache size and transfer time.
