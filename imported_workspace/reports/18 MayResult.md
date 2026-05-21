# 18 febResult

Saved on 2026-05-19.

## What Changed

The earlier MIMIC WikiMEL result of about `96.65 H@1` was invalid because the dataset had leakage:

- Train/test exact duplicate rows existed.
- Train/test mention, sentence, and image overlaps existed.
- Every test mention image appeared inside the gold entity `image_list`.

This was fixed by creating a clean dataset:

```text
ExpSetup/external/MIMIC_reproduction/data/WikiMEL_noleak
```

Clean config:

```text
ExpSetup/external/MIMIC_reproduction/config/wikimel_noleak.yaml
```

Audit script:

```text
ExpSetup/audit_wikimel_leakage.py
```

Audit result for `WikiMEL_noleak`:

- Train/dev/test exact overlap: `0`
- Train/dev/test mention+sentence overlap: `0`
- Train/dev/test image overlap: `0`
- Gold entity image leakage: `0`

## Clean Baseline

Clean checkpoint:

```text
ExpSetup/external/MIMIC_reproduction/runs/WikiMEL_noleak/version_4/checkpoints/final.ckpt
```

Locked baseline manifest:

```text
ExpSetup/benchmarks/mimic_wikimel_noleak_locked_baseline/manifest.json
```

Clean no-leak test metrics:

| Metric | Value |
|---|---:|
| H@1 | `0.741879` |
| H@3 | `0.879310` |
| H@5 | `0.919040` |
| H@10 | `0.954773` |
| H@20 | `0.974013` |
| MR | `40.633433` |
| MRR | `0.819056` |

This replaces the invalid leaky `96.65 H@1` result.

## Clean Quantization Results

Results file:

```text
ExpSetup/benchmarks/mimic_wikimel_noleak_quantization/mimic_quantization_summary.csv
```

| Mode | H@1 | H@10 | H@20 | MRR | Entity cache |
|---|---:|---:|---:|---:|---:|
| fp32 | `0.741379` | `0.954773` | `0.974013` | `0.818783` | `10860.67 MB` |
| fp16 | `0.741379` | `0.954773` | `0.974013` | `0.818784` | `5430.33 MB` |
| int8 | `0.742129` | `0.953773` | `0.974263` | `0.819176` | `2753.76 MB` |

Conclusion:

- `fp16` is effectively lossless and halves representation storage.
- Simulated `int8` is within noise and reduces representation storage by about 75%.
- Current `int8` is simulated/dequantized, not true end-to-end int8 kernel inference.

## Important Code Changes

Converter patched:

```text
ExpSetup/external/MIMIC_reproduction/scripts/prepare_wikimel_mimic.py
```

Added:

- `--dedupe-splits`
- `--entity-image-strategy none`

MIMIC training patched:

```text
ExpSetup/external/MIMIC_reproduction/codes/main.py
```

Key changes:

- Added `MIMIC_INIT_CKPT`.
- Added `MIMIC_RESUME_CKPT`.
- Saved full checkpoints instead of weights-only checkpoints.
- Saved a final checkpoint after training.

Optimizer patched:

```text
ExpSetup/external/MIMIC_reproduction/codes/model/lightning_mimic.py
```

Key change:

- `torch.optim.AdamW(..., foreach=False)` to avoid CUDA illegal memory access seen on the RTX 4090.

## Tomorrow: Higher Quantization

Next work should start from the clean checkpoint only:

```text
ExpSetup/external/MIMIC_reproduction/runs/WikiMEL_noleak/version_4/checkpoints/final.ckpt
```

Do not use the old leaky checkpoint:

```text
ExpSetup/benchmarks/mimic_wikimel_locked_baseline/epoch=9-step=1420.ckpt
```

Useful next directions:

1. Add lower-bit modes to `evaluate_mimic_quantization.py`, such as `int4`, `int3`, and `int2`.
2. Keep it checkpointable per mode and per batch.
3. Compare degradation against clean fp32, not the leaky baseline.
4. Consider separate quantization of:
   - text CLS
   - image CLS
   - text token tensors
   - image patch tensors
5. Consider mixed precision if uniform 4-bit or 2-bit damages H@1 too much.

Baseline to compare against tomorrow:

```text
Clean fp32 H@1 = 0.741379
Clean fp32 MRR = 0.818783
```
