# 19FebReport

Saved on 2026-05-19.

## Goal

The goal was to test whether reducing only `entity_text_tokens` in the clean MIMIC WikiMEL no-leak setup can produce meaningful cache and inference-time gains without damaging retrieval quality.

This was done inside:

```text
Mixed Quantization setup
```

The old leaky checkpoint was not used.

Clean checkpoint:

```text
ExpSetup/external/MIMIC_reproduction/runs/WikiMEL_noleak/version_4/checkpoints/final.ckpt
```

## Why Entity Text Tokens

The entity cache has four representation tensors:

```text
text_cls
image_cls
text_tokens
image_patch_tokens
```

`entity_text_tokens` are the biggest tensor and feed the expensive text global-local matcher path, `TGLU`.

In the fp32 cache:

| Tensor | Size |
|---|---:|
| text_cls | 214.80 MB |
| image_cls | 40.27 MB |
| text_tokens | 8591.88 MB |
| image_patch_tokens | 2013.72 MB |
| total | 10860.67 MB |

So `entity_text_tokens` are the strongest first target for mixed quantization.

## What Was Built

New experiment code:

```text
Mixed Quantization setup/evaluate_entity_text_tokens.py
```

Runbook:

```text
Mixed Quantization setup/RUN_ENTITY_TEXT_TOKEN_EXPERIMENT.md
```

Attribution report generator:

```text
Mixed Quantization setup/summarize_entity_text_token_results.py
```

The evaluator changes only `entity_text_tokens`; all other cached tensors stay fp32:

```text
text_cls = fp32
image_cls = fp32
text_tokens = fp32/int8/int6/int4/int3/int2
image_patch_tokens = fp32
```

## Metrics Captured

Accuracy:

```text
H@1, H@3, H@5, H@10, H@20, MR, MRR
```

Cache metrics:

```text
entity_cache_storage_mb
entity_cache_text_cls_mb
entity_cache_image_cls_mb
entity_cache_text_tokens_mb
entity_cache_image_patch_tokens_mb
cache_size_ratio_vs_fp32
text_token_size_ratio_vs_fp32
```

Runtime attribution:

```text
mention_encoder_s
entity_transfer_s
layernorm_s
tglu_s
vdlu_s
cmfu_s
score_concat_s
rank_sort_s
```

This lets the result explain where the speedup came from rather than only reporting total runtime.

## Full Run Outputs

Summary CSV:

```text
Mixed Quantization setup/results/entity_text_tokens_full/entity_text_token_quantization_summary.csv
```

Detailed JSON:

```text
Mixed Quantization setup/results/entity_text_tokens_full/entity_text_token_quantization_results.json
```

Attribution report:

```text
Mixed Quantization setup/results/entity_text_tokens_full/attribution_report.md
```

Actual artifact size:

```text
Mixed Quantization setup/results/entity_text_tokens_full/actual_disk_usage.csv
```

## Main Results

| Mode | H@1 | MRR | Speedup | Total cache | Text-token cache |
|---|---:|---:|---:|---:|---:|
| fp32 | 0.741379 | 0.818783 | 1.00x | 10860.67 MB | 8591.88 MB |
| int8 | 0.741379 | 0.818783 | 1.09x | 4433.54 MB | 2164.75 MB |
| int6 | 0.741379 | 0.818783 | 1.10x | 3896.55 MB | 1627.76 MB |
| int4 | 0.741379 | 0.818785 | 1.77x | 3359.56 MB | 1090.77 MB |
| int3 | 0.741629 | 0.818859 | 1.90x | 3091.06 MB | 822.27 MB |
| int2 | 0.741129 | 0.818744 | 1.89x | 2822.57 MB | 553.77 MB |

## Runtime Attribution

fp32:

```text
total eval seconds = 1378.72
entity_transfer_s = 293.91
TGLU_s = 1033.43
```

int4:

```text
total eval seconds = 778.41
entity_transfer_s = 127.79
TGLU_s = 601.43
```

int3:

```text
total eval seconds = 725.97
entity_transfer_s = 126.96
TGLU_s = 550.42
```

The main gains came from:

1. Smaller entity text-token cache.
2. Lower entity transfer time.
3. Lower TGLU scoring time.

Mention encoding and ranking were tiny contributors, so they do not explain the speedup.

## Best Result

The strongest conservative result is `int4` for `entity_text_tokens`.

It kept retrieval quality effectively unchanged:

```text
fp32 H@1 = 0.741379
int4 H@1 = 0.741379

fp32 MRR = 0.818783
int4 MRR = 0.818785
```

It reduced text-token cache size:

```text
8591.88 MB -> 1090.77 MB
```

It reduced total logical cache size:

```text
10860.67 MB -> 3359.56 MB
```

It improved evaluation speed:

```text
1378.72 s -> 778.41 s
speedup = 1.77x
```

## Possible Paper Claim

A possible paper-shaped claim:

```text
Entity text-token representations dominate MIMIC-style multimodal entity linking cache cost.
Quantizing only those entity-side text tokens to 4-bit preserves clean WikiMEL retrieval quality while substantially reducing cache size and inference time.
```

The result is promising because it is:

- run on the clean no-leak WikiMEL setup;
- isolated to one representation tensor;
- backed by accuracy, cache-size, and subprocess timing metrics;
- explainable through the TGLU matching path.

## Caveats

The sub-8-bit cache-size columns are logical packed representation sizes. The current experiment stores integer codes in an int8 container plus scales, so `actual_disk_usage.csv` should be used when reporting physical artifact size.

`int3` and `int2` are interesting, but should be repeated before being treated as the central claim. The safer central result is `int4`.

## Next Steps

1. Repeat `int4` and `int3` to check run-to-run stability.
2. Add an `image_patch_tokens` sweep.
3. Add mixed text+image token quantization.
4. Add rank-flip examples to show where quantization helps or hurts.
5. Add entity-type breakdowns for people, places, organizations, and visually grounded cases.
