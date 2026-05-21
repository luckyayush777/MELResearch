# Tiled Matcher Investigation

Saved on 2026-05-20.

## Context From Existing Results

The current reports show that mixed int4 cache quantization is already useful for cache size, disk size, transfer time, and throughput. The missing fixed-chunk win is peak eval VRAM:

| Mode | Chunk | Peak eval VRAM | Peak matcher | Peak TGLU |
|---|---:|---:|---:|---:|
| fp32_ref | 5000 | 5851.28 MB | 5851.28 MB | 5851.28 MB |
| text_image_tokens_int4 | 5000 | 5851.15 MB | 5851.15 MB | 5851.15 MB |
| text_image_tokens_int4 | 10000 | 11063.21 MB | 11063.21 MB | 11063.21 MB |

This confirms the bottleneck: at fixed chunk size, peak memory is dominated by the matcher-side TGLU computation after cache tensors have been moved/dequantized, not by the packed cache itself.

## Code Change

Added an optional `--matcher-tile-size` to:

```text
Mixed Quantization setup/evaluate_entity_text_tokens.py
Mixed Quantization setup/run_chunk_size_sweep.py
```

The new path keeps the external `chunk_size` unchanged, but scores the TGLU text matcher in entity subtiles. For example:

```powershell
python "Mixed Quantization setup/run_chunk_size_sweep.py" `
  --modes text_image_tokens_int4 `
  --chunk-sizes 5000 `
  --matcher-tile-size 1000 `
  --out-dir results/tiled_matcher_chunk5000_tile1000
```

The output now records:

```text
matcher_tile_size
gpu_peak_allocated_mb_eval
gpu_peak_allocated_mb_matcher
gpu_peak_allocated_mb_tglu
gpu_peak_allocated_mb_entity_chunk_transfer
tglu_s
seconds
queries_per_second
hits@1
mrr
```

## Smoke Result

A 20-query smoke run was executed against the existing `text_image_tokens_int4` cache at `chunk_size=5000`.

| Run | Queries | Chunk | TGLU tile | H@1 | MRR | Peak eval VRAM | Peak TGLU | TGLU time | Runtime |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline smoke | 20 | 5000 | 0 | 0.950000 | 0.966667 | 5826.42 MB | 5826.42 MB | 0.558 s | 1.733 s |
| tiled smoke | 20 | 5000 | 1000 | 0.950000 | 0.966667 | 2445.43 MB | 2445.43 MB | 0.503 s | 1.714 s |

Smoke artifacts:

```text
Mixed Quantization setup/results/tiled_matcher_smoke_baseline/chunk_size_sweep_summary.csv
Mixed Quantization setup/results/tiled_matcher_smoke/chunk_size_sweep_summary.csv
```

This is not a full test-set result, but it verifies the mechanism and instrumentation. On the same first eval batch, TGLU tiling reduced fixed-chunk peak allocation by about 58.0% while preserving the sampled ranks.

## Full Key-Row Result

The full clean no-leak test rows were run with `--matcher-tile-size 1000`:

```text
Mixed Quantization setup/results/tiled_matcher_keyrows_tile1000/chunk_size_sweep_summary.csv
Mixed Quantization setup/results/tiled_matcher_keyrows_tile1000/chunk_size_sweep_results.json
```

| Mode | Chunk | TGLU tile | H@1 | MRR | Peak eval VRAM | Peak TGLU | Runtime | QPS | Speedup vs tiled fp32 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| fp32_ref | 5000 | 1000 | 0.741629 | 0.818908 | 2470.60 MB | 2470.60 MB | 442.52 s | 9.044 | 1.000x |
| text_image_tokens_int4 | 5000 | 1000 | 0.741379 | 0.818733 | 2470.84 MB | 2470.84 MB | 234.46 s | 17.069 | 1.887x |
| fp32_ref | 10000 | 1000 | 0.741629 | 0.818908 | 3458.94 MB | 3458.94 MB | 435.89 s | 9.181 | 1.000x |
| text_image_tokens_int4 | 10000 | 1000 | 0.741379 | 0.818733 | 3458.94 MB | 3458.94 MB | 232.52 s | 17.212 | 1.875x |

Compared with the prior non-tiled repeated rows:

| Mode | Chunk | Non-tiled peak | Tiled peak | Peak reduction |
|---|---:|---:|---:|---:|
| fp32_ref | 5000 | 5851.28 MB | 2470.60 MB | 57.8% |
| text_image_tokens_int4 | 5000 | 5851.15 MB | 2470.84 MB | 57.8% |
| fp32_ref | 10000 | 11063.21 MB | 3458.94 MB | 68.7% |
| text_image_tokens_int4 | 10000 | 11063.21 MB | 3458.94 MB | 68.7% |

Accuracy is unchanged relative to the existing full rows. Runtime is also essentially unchanged:

| Mode | Chunk | Prior runtime | Tiled runtime |
|---|---:|---:|---:|
| fp32_ref | 5000 | 440.92 s | 442.52 s |
| text_image_tokens_int4 | 5000 | 234.23 s | 234.46 s |
| fp32_ref | 10000 | 433.62 s | 435.89 s |
| text_image_tokens_int4 | 10000 | 231.30 s | 232.52 s |

New strongest fixed-chunk claim:

```text
Tiling the TGLU matcher with a 1000-entity subtile reduces fixed-chunk peak eval allocation from about 5.85 GB to about 2.47 GB at chunk 5000, and from about 11.06 GB to about 3.46 GB at chunk 10000, while preserving the same clean WikiMEL H@1/MRR and maintaining the mixed-int4 throughput gain.
```

## Interpretation

TGLU is the correct first attack point because its peak equals the full eval peak in the existing key rows. Tiling is also low risk because it does not change the mathematical scorer; it only changes how many entities pass through TGLU at once.

This first implementation still transfers/dequantizes the whole outer entity chunk before scoring. That leaves `gpu_peak_allocated_mb_entity_chunk_transfer` around 1.9 GB in the smoke run. The next reduction would tile the whole entity matcher transfer path, so text/image tensors are moved and scored per subtile while the outer chunk remains the scheduling unit.

## Next Runs

Optional tile-size sweep:

```powershell
python "Mixed Quantization setup/run_chunk_size_sweep.py" `
  --modes text_image_tokens_int4 `
  --chunk-sizes 5000 `
  --matcher-tile-size 500 `
  --out-dir results/tiled_matcher_chunk5000_tile500
```

Capture the same columns as the current reports, plus `matcher_tile_size`, `gpu_peak_allocated_mb_tglu`, `gpu_peak_allocated_mb_entity_chunk_transfer`, and `tglu_s`.
