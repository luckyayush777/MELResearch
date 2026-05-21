# Tiling Results 20 May

Saved on 2026-05-20.

## Goal

Investigate whether tiled matcher computation can reduce peak GPU memory at a fixed external chunk size.

The earlier quantization results showed a strong storage and transfer story, but not a fixed-chunk peak VRAM story. At the same chunk size, fp32 and int4 had almost identical peak eval allocation because the current matcher path still materialized large dequantized tensors during scoring.

The specific question for this run was:

```text
Can we keep chunk_size fixed at 5000 or 10000, preserve the same ranking outputs, and lower peak matcher VRAM by tiling the expensive TGLU computation?
```

## Context From Previous Results

The previous strongest deployment row was:

```text
text_image_tokens_int4, chunk_size=5000
```

It preserved clean WikiMEL accuracy while reducing cache footprint and transfer time:

| Metric | Value |
|---|---:|
| H@1 | 0.741379 |
| MRR | 0.818733 |
| Logical cache | 1618.53 MB |
| Disk cache | 2925.63 MB |
| Peak eval VRAM | 5851.15 MB |
| Runtime | 234.23 s |
| QPS | 17.086 |
| Speedup vs fp32 same chunk | 1.882x |

The caveat was important:

```text
Quantization reduced cache size and transfer cost, but not fixed-chunk peak eval VRAM.
```

The repeated key rows showed why:

| Mode | Chunk | Peak eval VRAM | Peak matcher | Peak TGLU |
|---|---:|---:|---:|---:|
| fp32_ref | 5000 | 5851.28 MB | 5851.28 MB | 5851.28 MB |
| text_image_tokens_int4 | 5000 | 5851.15 MB | 5851.15 MB | 5851.15 MB |
| fp32_ref | 10000 | 11063.21 MB | 11063.21 MB | 11063.21 MB |
| text_image_tokens_int4 | 10000 | 11063.21 MB | 11063.21 MB | 11063.21 MB |

Peak eval allocation was exactly the TGLU peak in these rows. That made TGLU the right first target.

## Why TGLU Is The Bottleneck

The inference path is:

```text
build entity cache once
for each query batch:
  encode mention
  for each entity chunk:
    move entity chunk to GPU
    run matcher:
      TGLU text global-local unit
      VDLU vision dual unit
      CMFU cross-modal fusion unit
  concatenate scores
  rank all entities
```

The TGLU path computes pairwise local text alignment between entity text tokens and mention text tokens. For an entity chunk, it creates attention tensors shaped roughly like:

```text
[num_entities_in_chunk, batch_size, entity_seq_len, mention_seq_len]
```

So increasing `chunk_size` directly increases the largest temporary tensors inside TGLU. The cache may be int4 on CPU/disk, but once the chunk is dequantized and sent into the matcher, TGLU still creates fp32-style intermediate tensors.

That is why the old fixed-chunk peaks were nearly identical for fp32 and int4.

## Implementation

Added an optional matcher tiling parameter:

```text
--matcher-tile-size
```

Files changed:

```text
Mixed Quantization setup/evaluate_entity_text_tokens.py
Mixed Quantization setup/run_chunk_size_sweep.py
```

The new path keeps the external `chunk_size` unchanged, but splits the entity dimension only inside TGLU.

Conceptually:

```text
before:
  TGLU(entity_chunk_size=5000)

after:
  TGLU(entity_tile_size=1000)
  TGLU(entity_tile_size=1000)
  TGLU(entity_tile_size=1000)
  TGLU(entity_tile_size=1000)
  TGLU(entity_tile_size=1000)
  concatenate TGLU scores
```

The rest of the matcher still sees the same outer entity chunk. The final score tensor shape is unchanged, and ranking is unchanged.

This is low risk because it does not approximate or quantize the TGLU math. It only changes the scheduling of the entity dimension.

## Command

The full key-row run used the clean no-leak checkpoint:

```text
ExpSetup/external/MIMIC_reproduction/runs/WikiMEL_noleak/version_4/checkpoints/final.ckpt
```

Command:

```powershell
python "Mixed Quantization setup/run_chunk_size_sweep.py" `
  --modes fp32_ref text_image_tokens_int4 `
  --chunk-sizes 5000 10000 `
  --matcher-tile-size 1000 `
  --out-dir results/tiled_matcher_keyrows_tile1000
```

Artifacts:

```text
Mixed Quantization setup/results/tiled_matcher_keyrows_tile1000/chunk_size_sweep_summary.csv
Mixed Quantization setup/results/tiled_matcher_keyrows_tile1000/chunk_size_sweep_results.json
```

Environment:

| Field | Value |
|---|---|
| Device | NVIDIA GeForce RTX 4090 |
| GPU memory | 24563.5 MB |
| PyTorch | 2.1.2+cu121 |
| CUDA | 12.1 |
| Split | test |
| Queries | 4002 |
| Entities | 109976 |
| Eval batch size | 20 |
| Matcher tile size | 1000 |

## Full Results

| Mode | Chunk | TGLU tile | H@1 | MRR | Peak eval VRAM | Peak TGLU | Runtime | QPS | Speedup vs tiled fp32 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| fp32_ref | 5000 | 1000 | 0.741629 | 0.818908 | 2470.60 MB | 2470.60 MB | 442.52 s | 9.044 | 1.000x |
| text_image_tokens_int4 | 5000 | 1000 | 0.741379 | 0.818733 | 2470.84 MB | 2470.84 MB | 234.46 s | 17.069 | 1.887x |
| fp32_ref | 10000 | 1000 | 0.741629 | 0.818908 | 3458.94 MB | 3458.94 MB | 435.89 s | 9.181 | 1.000x |
| text_image_tokens_int4 | 10000 | 1000 | 0.741379 | 0.818733 | 3458.94 MB | 3458.94 MB | 232.52 s | 17.212 | 1.875x |

## Peak VRAM Reduction

Tiling directly attacks the previously unresolved fixed-chunk peak VRAM problem.

| Mode | Chunk | Non-tiled peak | Tiled peak | Absolute reduction | Relative reduction |
|---|---:|---:|---:|---:|---:|
| fp32_ref | 5000 | 5851.28 MB | 2470.60 MB | 3380.68 MB | 57.8% |
| text_image_tokens_int4 | 5000 | 5851.15 MB | 2470.84 MB | 3380.31 MB | 57.8% |
| fp32_ref | 10000 | 11063.21 MB | 3458.94 MB | 7604.27 MB | 68.7% |
| text_image_tokens_int4 | 10000 | 11063.21 MB | 3458.94 MB | 7604.27 MB | 68.7% |

This is the main result.

Before tiling, `chunk_size=10000` required about `11.06 GB` peak eval allocation. With TGLU tile size `1000`, the same outer chunk size requires about `3.46 GB`.

Before tiling, `chunk_size=5000` required about `5.85 GB`. With TGLU tile size `1000`, the same outer chunk size requires about `2.47 GB`.

## Accuracy Preservation

Tiling did not change the retrieval metrics.

| Mode | Chunk | Non-tiled H@1 | Tiled H@1 | Non-tiled MRR | Tiled MRR |
|---|---:|---:|---:|---:|---:|
| fp32_ref | 5000 | 0.741629 | 0.741629 | 0.818908 | 0.818908 |
| text_image_tokens_int4 | 5000 | 0.741379 | 0.741379 | 0.818733 | 0.818733 |
| fp32_ref | 10000 | 0.741629 | 0.741629 | 0.818908 | 0.818908 |
| text_image_tokens_int4 | 10000 | 0.741379 | 0.741379 | 0.818733 | 0.818733 |

This matches expectation. Tiling is mathematically equivalent to scoring the full chunk in one pass, then concatenating the same TGLU score blocks.

## Runtime Impact

Runtime stayed effectively unchanged.

| Mode | Chunk | Non-tiled runtime | Tiled runtime | Runtime change |
|---|---:|---:|---:|---:|
| fp32_ref | 5000 | 440.92 s | 442.52 s | +1.60 s |
| text_image_tokens_int4 | 5000 | 234.23 s | 234.46 s | +0.23 s |
| fp32_ref | 10000 | 433.62 s | 435.89 s | +2.27 s |
| text_image_tokens_int4 | 10000 | 231.30 s | 232.52 s | +1.22 s |

The runtime cost is negligible relative to the memory reduction. The mixed int4 rows keep the earlier throughput advantage:

| Chunk | fp32 tiled QPS | mixed int4 tiled QPS | Speedup |
|---:|---:|---:|---:|
| 5000 | 9.044 | 17.069 | 1.887x |
| 10000 | 9.181 | 17.212 | 1.875x |

## Timing Breakdown

The tiled mixed-int4 rows still spend most time in entity transfer and TGLU:

| Mode | Chunk | Entity transfer | TGLU | VDLU | CMFU | Runtime |
|---|---:|---:|---:|---:|---:|---:|
| text_image_tokens_int4 | 5000 | 89.66 s | 95.67 s | 8.85 s | 4.71 s | 234.46 s |
| text_image_tokens_int4 | 10000 | 88.39 s | 95.13 s | 10.39 s | 3.86 s | 232.52 s |

The TGLU time did not meaningfully increase from tiling:

| Mode | Chunk | Non-tiled TGLU | Tiled TGLU |
|---|---:|---:|---:|
| text_image_tokens_int4 | 5000 | 94.61 s | 95.67 s |
| text_image_tokens_int4 | 10000 | 93.99 s | 95.13 s |

This is useful: the tile size `1000` is small enough to reduce peak memory, but not so small that kernel overhead dominates runtime.

## Memory Component Breakdown

For `text_image_tokens_int4` with TGLU tile size `1000`:

| Chunk | Entity transfer peak | LayerNorm peak | TGLU peak | VDLU peak | CMFU peak | Eval peak |
|---:|---:|---:|---:|---:|---:|---:|
| 5000 | 1922.46 MB | 1626.40 MB | 2470.84 MB | 1832.57 MB | 1727.90 MB | 2470.84 MB |
| 10000 | 3211.05 MB | 2616.21 MB | 3458.94 MB | 3027.33 MB | 2817.12 MB | 3458.94 MB |

TGLU remains the peak stage, but it is now bounded by the tile size instead of the full chunk size. At chunk `10000`, transfer and non-TGLU matcher stages become much closer to TGLU, which means further reductions may need whole-matcher or transfer tiling rather than TGLU-only tiling.

## Updated Deployment Story

Before this run, the best honest claim was:

```text
Quantization reduces cache size and transfer cost. Chunk-size selection controls peak eval VRAM.
```

After this run, the stronger claim is:

```text
Quantization reduces cache size and transfer cost, while TGLU matcher tiling reduces peak eval VRAM at a fixed chunk size. Together, mixed int4 cache compression plus tiled matcher scoring preserves clean WikiMEL H@1/MRR and moves the chunk-5000 operating point from about 5.85 GB peak eval allocation to about 2.47 GB.
```

Recommended headline configuration:

```text
text_image_tokens_int4, chunk_size=5000, matcher_tile_size=1000
```

Why:

- H@1 and MRR match the prior mixed-int4 full result.
- Peak eval VRAM falls to about `2.47 GB`.
- QPS remains high at `17.069`.
- Speedup remains `1.887x` versus tiled fp32 at the same chunk.
- Logical cache remains `1618.53 MB`, about `14.9%` of fp32.

## Recommended Hardware-Budget Rows

The tiled results change the memory budget story:

| Hardware budget | Recommended mode | Chunk | Tile | H@1 | MRR | Peak eval VRAM | QPS |
|---|---|---:|---:|---:|---:|---:|---:|
| Around 3 GB | text_image_tokens_int4 | 5000 | 1000 | 0.741379 | 0.818733 | 2470.84 MB | 17.069 |
| Around 4 GB | text_image_tokens_int4 | 10000 | 1000 | 0.741379 | 0.818733 | 3458.94 MB | 17.212 |

The old recommendation needed about `6 GB` for chunk `5000`. With tiling, chunk `5000` fits under roughly `2.5 GB` peak allocated eval VRAM.

## Chunk 5000 Tile-Size Sweep

After the key-row run, a tile-size sweep was run for:

```text
mode = text_image_tokens_int4
chunk_size = 5000
split = test
queries = 4002
```

Sweep artifacts:

```text
Mixed Quantization setup/results/tiled_matcher_chunk5000_tile_sweep_summary.csv
Mixed Quantization setup/results/tiled_matcher_chunk5000_tile250/chunk_size_sweep_summary.csv
Mixed Quantization setup/results/tiled_matcher_chunk5000_tile500/chunk_size_sweep_summary.csv
Mixed Quantization setup/results/tiled_matcher_keyrows_tile1000/chunk_size_sweep_summary.csv
Mixed Quantization setup/results/tiled_matcher_chunk5000_tile2000/chunk_size_sweep_summary.csv
```

Results:

| TGLU tile | H@1 | MRR | Peak eval VRAM | Peak TGLU | Entity transfer peak | Runtime | QPS | Reduction vs no tile |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| no tile | 0.741379 | 0.818733 | 5851.15 MB | 5851.15 MB | 1921.78 MB | 234.23 s | 17.086 | 0.0% |
| 250 | 0.741379 | 0.818733 | 1922.29 MB | 1839.01 MB | 1922.29 MB | 229.42 s | 17.444 | 67.1% |
| 500 | 0.741379 | 0.818733 | 2049.06 MB | 2049.06 MB | 1921.84 MB | 235.40 s | 17.001 | 65.0% |
| 1000 | 0.741379 | 0.818733 | 2470.84 MB | 2470.84 MB | 1922.46 MB | 234.46 s | 17.069 | 57.8% |
| 2000 | 0.741379 | 0.818733 | 3315.92 MB | 3315.92 MB | 1921.84 MB | 234.89 s | 17.038 | 43.3% |

The sweep shows that accuracy is invariant across tile sizes, as expected. Runtime is also effectively flat. Peak memory follows the tile size until it hits the entity transfer/dequantization floor.

The `250` row is the lowest measured peak at chunk `5000`: `1922.29 MB`. At that point, the eval peak is no longer TGLU; it is entity transfer/dequantization. This is strong evidence that TGLU-only tiling has reached its practical floor for the current implementation.

Updated recommendation:

| Goal | Recommended tile | Reason |
|---|---:|---|
| Lowest measured peak at chunk 5000 | 250 | Reaches about 1.92 GB, now limited by transfer/dequantization |
| Conservative low-memory default | 500 | About 2.05 GB with stable runtime and less sub-call fragmentation |
| Earlier key-row comparison | 1000 | Still a large reduction, easy to compare with chunk 10000 key rows |

The next memory reduction must tile entity transfer/dequantization too. Once TGLU is small enough, moving the full 5000-entity chunk to GPU becomes the limiting peak.

## Full Entity-Subtile Transfer/Dequantization

The TGLU-only sweep reached a clear floor: with `matcher_tile_size=250`, peak eval allocation was `1922.29 MB`, and the entity transfer/dequantization peak was also `1922.29 MB`. At that point, the packed int4 cache was still being sliced at the outer `chunk_size=5000`, then dequantized and moved to GPU as a full 5000-entity block before matcher scoring.

To break that floor, the evaluator now supports:

```text
--entity-subtile-size
```

This keeps the outer evaluation chunk fixed at `5000` for scheduling/reporting, but slices the packed entity cache before transfer:

```text
before:
  cache_tensor_to_device(entity_cache, start, start + 5000)
  matcher(5000 entities)

after, entity_subtile_size=500:
  cache_tensor_to_device(entity_cache, start, start + 500)
  matcher(500 entities)
  cache_tensor_to_device(entity_cache, start + 500, start + 1000)
  matcher(500 entities)
  ...
  concatenate all subtile scores for the original 5000-entity chunk
```

This means the GPU never receives the full dequantized 5000-entity cache block. It receives only the current entity subtile.

Files changed:

```text
Mixed Quantization setup/evaluate_entity_text_tokens.py
Mixed Quantization setup/run_chunk_size_sweep.py
```

The implementation records:

```text
entity_subtile_size
entity_subtiles
entity_subtiles_per_second
```

It also preserves the existing `matcher_tile_size` path. For the full entity-subtile runs below, `matcher_tile_size=0` because the full matcher already receives a small entity subtile.

### Validation Commands

Smoke test:

```powershell
python "Mixed Quantization setup/run_chunk_size_sweep.py" `
  --modes text_image_tokens_int4 `
  --chunk-sizes 5000 `
  --entity-subtile-size 250 `
  --limit-queries 20 `
  --out-dir results/entity_subtile_smoke
```

Full rows:

```powershell
python "Mixed Quantization setup/run_chunk_size_sweep.py" `
  --modes text_image_tokens_int4 `
  --chunk-sizes 5000 `
  --entity-subtile-size 250 `
  --out-dir results/entity_subtile_chunk5000_tile250

python "Mixed Quantization setup/run_chunk_size_sweep.py" `
  --modes text_image_tokens_int4 `
  --chunk-sizes 5000 `
  --entity-subtile-size 500 `
  --out-dir results/entity_subtile_chunk5000_tile500
```

Consolidated artifact:

```text
Mixed Quantization setup/results/entity_subtile_chunk5000_summary.csv
```

### Results

| Path | Entity subtile | TGLU tile | H@1 | MRR | Peak eval VRAM | Transfer peak | Runtime | QPS | Entity subtiles |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| no tiling | 0 | 0 | 0.741379 | 0.818733 | 5851.15 MB | 1921.78 MB | 234.23 s | 17.086 | 4422 |
| TGLU-only tiling | 0 | 250 | 0.741379 | 0.818733 | 1922.29 MB | 1922.29 MB | 229.42 s | 17.444 | 4422 |
| full entity-subtile | 500 | 0 | 0.741379 | 0.818733 | 1160.72 MB | 764.44 MB | 309.58 s | 12.927 | 44220 |
| full entity-subtile | 250 | 0 | 0.741379 | 0.818733 | 901.35 MB | 699.97 MB | 385.59 s | 10.379 | 88440 |

Accuracy is unchanged. The rank metrics match the existing mixed-int4 result:

```text
H@1 = 0.741379
MRR = 0.818733
```

The memory reduction is substantial:

| Configuration | Peak eval VRAM | Reduction vs no tiling |
|---|---:|---:|
| no tiling | 5851.15 MB | 0.0% |
| TGLU-only tile 250 | 1922.29 MB | 67.1% |
| entity subtile 500 | 1160.72 MB | 80.2% |
| entity subtile 250 | 901.35 MB | 84.6% |

The `entity_subtile_size=250` row is the new lowest measured memory point at `chunk_size=5000`: about `0.90 GB` peak allocated eval VRAM.

### Runtime Tradeoff

Full entity-subtile transfer/dequantization is a memory win, but it repeats more matcher work. At `entity_subtile_size=250`, the run processes `88440` entity subtiles instead of `4422` outer chunks. That increases repeated non-TGLU work:

| Configuration | TGLU time | VDLU time | CMFU time | Entity transfer | Runtime | QPS |
|---|---:|---:|---:|---:|---:|---:|
| no tiling | 94.61 s | 8.59 s | 4.12 s | 89.75 s | 234.23 s | 17.086 |
| TGLU-only tile 250 | 91.55 s | 8.77 s | 4.43 s | 89.82 s | 229.42 s | 17.444 |
| entity subtile 500 | 105.68 s | 29.96 s | 21.99 s | 104.43 s | 309.58 s | 12.927 |
| entity subtile 250 | 109.95 s | 55.65 s | 42.09 s | 121.29 s | 385.59 s | 10.379 |

This is the central tradeoff:

```text
TGLU-only tiling is the best throughput/memory balance.
Full entity-subtile transfer/dequantization is the lowest-memory path.
```

Recommended rows after this addition:

| Goal | Recommended setting | Peak eval VRAM | QPS |
|---|---|---:|---:|
| Best low-memory throughput balance | `matcher_tile_size=250` | 1922.29 MB | 17.444 |
| Practical sub-1.2 GB point | `entity_subtile_size=500` | 1160.72 MB | 12.927 |
| Lowest measured memory | `entity_subtile_size=250` | 901.35 MB | 10.379 |

The full entity-subtile implementation answers the open question from the TGLU-only sweep: yes, slicing the packed cache before GPU transfer pushes the fixed-chunk memory floor below the previous transfer/dequantization peak. The cost is repeated matcher overhead across more subtiles.

## Caveats

- These are peak allocated metrics from PyTorch CUDA instrumentation, not enforced hard memory-cap tests.
- The run used an RTX 4090. Target-device allocator behavior should be confirmed on smaller GPUs.
- TGLU-only tiling reduces matcher temporary memory, but it still moves/dequantizes the whole outer entity chunk before scoring.
- Full entity-subtile transfer/dequantization fixes that transfer floor, but currently repeats VDLU/CMFU and other per-subtile matcher work.
- At chunk `10000`, `gpu_peak_allocated_mb_entity_chunk_transfer` is already `3211.05 MB`, close to the total eval peak of `3458.94 MB`. Further fixed-chunk memory reductions will likely require tiling entity transfer/dequantization and the full matcher path.

## Next Steps

1. Run an enforced-memory test on a smaller GPU or with a memory cap.

2. Compare rank batches between non-tiled, TGLU-only tiled, and entity-subtile rows to document exact rank equivalence.

3. Optimize the entity-subtile path so mention-side or shared matcher work is reused across subtiles where possible. The current low-memory implementation prioritizes memory reduction over avoiding repeated VDLU/CMFU work.

## Main Conclusion

TGLU matcher tiling solves the fixed-chunk peak VRAM problem that cache quantization alone did not solve.

At `chunk_size=5000`, mixed int4 plus TGLU tile size `1000` keeps H@1 at `0.741379`, keeps MRR at `0.818733`, keeps throughput around `17.07 QPS`, and reduces peak eval allocation from about `5.85 GB` to about `2.47 GB`.

At `chunk_size=10000`, the same tiling reduces peak eval allocation from about `11.06 GB` to about `3.46 GB`, again with unchanged accuracy and nearly unchanged runtime.

The follow-up entity-subtile implementation goes further for the chunk `5000` case: `entity_subtile_size=250` lowers peak eval allocation to about `0.90 GB`, while `entity_subtile_size=500` lands at about `1.16 GB`. These are the lowest-memory rows measured so far, with the expected throughput tradeoff from running the full matcher over more subtiles.
