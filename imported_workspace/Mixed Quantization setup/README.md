# Mixed Quantization Setup

Saved on 2026-05-19.

This folder captures the next setup for MIMIC WikiMEL no-leak profiling and mixed quantization. Use only the clean checkpoint:

```text
ExpSetup/external/MIMIC_reproduction/runs/WikiMEL_noleak/version_4/checkpoints/final.ckpt
```

Do not compare against the old leaky checkpoint.

## Current Clean Baseline

Source result:

```text
ExpSetup/benchmarks/mimic_wikimel_noleak_quantization/mimic_quantization_summary.csv
```

| Mode | H@1 | MRR | Test eval seconds | Entity cache seconds | Entity cache size |
|---|---:|---:|---:|---:|---:|
| fp32 | 0.741379 | 0.818783 | 1407.21 | 131.18 | 10860.67 MB |
| fp16 | 0.741379 | 0.818784 | 1402.22 | 150.89 | 5430.33 MB |
| int8 | 0.742129 | 0.819176 | 558.75 | 7.32 | 2753.76 MB |

The evaluation-time speedup in the current int8 run is from smaller cached representations and cheaper transfer/dequantized scoring behavior, not from true int8 kernels.

## A. Timing Breakdown To Measure

The code has two different timing profiles.

### Training

Training path:

```text
LightningForMIMIC.training_step
  mention encoder: MIMICEncoder(CLIP text + CLIP vision + image projection)
  entity encoder:  MIMICEncoder(CLIP text + CLIP vision + image projection)
  matcher:         MIMICMatcher(TGLU + VDLU + CMFU)
  losses:          4 cross entropy losses
  backward + AdamW
```

Expected largest training costs:

| Rank | Component | Why it is expensive | Quantization relevance |
|---:|---|---|---|
| 1 | CLIP vision encoder, mention + entity | ViT image path runs twice per training batch on 224x224 images | Expensive, but quantizing during training is riskier than representation-only quantization |
| 2 | CLIP text encoder, mention + entity | Text transformer runs twice per batch with length 40 | Sensitive to low-bit weight/activation quantization |
| 3 | Backward + AdamW optimizer state | Full model gradients and Adam states dominate training memory/time after forward | Mixed precision training helps more than post-hoc representation int8 |
| 4 | Matcher TGLU | Pairwise text-token attention between mention and entity batch | Sensitive because it directly changes text ranking logits |
| 5 | Matcher VDLU + CMFU | Vision dual scoring and text-image fusion | Sensitive when image disambiguation is needed |
| 6 | Data collator image load/resize | PIL image open + resize is done inside collator with `num_workers: 0` | Can bottleneck CPU input pipeline |

For training, the first timing experiment should separate:

1. Data loading/collation time.
2. Mention encoder forward time.
3. Entity encoder forward time.
4. Matcher forward time.
5. Loss time.
6. Backward time.
7. Optimizer step time.

### Inference / Test

Evaluation path:

```text
build entity cache once:
  entity dataloader -> MIMICEncoder -> cache 4 tensors

per query batch:
  mention MIMICEncoder
  for each entity chunk:
    move entity chunk to GPU
    MIMICMatcher(TGLU + VDLU + CMFU)
  concat scores
  rank all entities
```

Expected largest inference costs:

| Rank | Component | Why it is expensive | What to try first |
|---:|---|---|---|
| 1 | TGLU text global-local unit | Uses entity text tokens against mention text tokens for every entity chunk; text tokens are also the largest cache tensor | Quantize entity text tokens first, then mention text tokens |
| 2 | Entity chunk transfer to GPU | Every query batch moves text/image CLS and token tensors from CPU cache to GPU | Keep cache compressed and/or pre-stage larger chunks if VRAM allows |
| 3 | CMFU entity image-token projection | Reprojects entity image tokens inside every chunk scoring pass | Precompute projected image tokens or quantize image tokens |
| 4 | VDLU pairwise vision scorer | Runs two directional dual scorers for each query/entity chunk | Quantize image tokens and image CLS separately |
| 5 | Mention encoder | Runs once per query batch, much smaller than chunked all-entity matching | Keep higher precision until entity-side ablations are clear |
| 6 | Ranking sort | Sorts scores over all 109,976 entities per query | Use top-k ranking if only H@k/MRR approximations are needed carefully |

The current clean fp32 numbers show entity-cache construction is much smaller than full query scoring: about 131s to build the entity cache versus about 1407s for full evaluation. That means optimization should focus on repeated matcher scoring and entity-cache movement, not only encoder cache creation.

## B. Mixed Quantization Targets

The cached representation has four tensors:

```text
text_cls           [N, 512]
image_cls          [N, 96]
text_tokens        [N, 40, 512]
image_patch_tokens [N, about 50, 96]
```

Approximate fp32 entity-cache storage:

| Tensor | Approx size | Expected accuracy sensitivity | Expected result visibility |
|---|---:|---|---|
| text_tokens | 8.6 GB | Very high | Most stark storage win; likely visible accuracy drop at 4-bit or below |
| image_patch_tokens | 2.0 GB | Medium to high | Strong result on image-heavy examples; useful second target |
| text_cls | 0.22 GB | High | Small storage win, but can strongly affect global text score |
| image_cls | 0.04 GB | Medium | Tiny storage win; useful for sensitivity study, not compression |

Recommended mixed-precision starting matrix:

| Setup name | text_cls | text_tokens | image_cls | image_patch_tokens | Purpose |
|---|---:|---:|---:|---:|---|
| fp32_ref | 32 | 32 | 32 | 32 | Clean reference |
| all_int8 | 8 | 8 | 8 | 8 | Confirm existing lossless-ish behavior |
| text_tokens_4 | 8 | 4 | 8 | 8 | Biggest storage reduction with likely first accuracy movement |
| image_tokens_4 | 8 | 8 | 8 | 4 | Tests visual-token sensitivity |
| cls_safe_tokens_low | 8 | 4 | 8 | 4 | Aggressive cache compression while keeping CLS safer |
| text_safe_image_low | 8 | 8 | 6 | 4 | Tests whether image side can be pushed harder |
| text_low_image_safe | 8 | 4 | 8 | 8 | Tests if text tokens are the fragile bottleneck |
| extreme_4bit | 4 | 4 | 4 | 4 | Stark degradation boundary |
| extreme_3bit | 4 | 3 | 4 | 3 | Stress test |
| extreme_2bit | 4 | 2 | 4 | 2 | Expected to visibly break ranking geometry |

## Most Sensitive Layers / Tensors

Expected sensitivity order for this exact MIMIC setup:

1. `matcher.tglu` inputs, especially `entity_text_tokens` and `mention_text_tokens`.
   These drive local text alignment. Low-bit quantization here should cause the clearest H@1/MRR movement because entity linking is often text-dominant.

2. `matcher.cmfu` gate/fusion path.
   This combines text CLS with image patch tokens. Quantization can change gate behavior and cross-modal score calibration, so it is a good place to look for rank flips.

3. `text_cls`.
   It is small, but it feeds global text matching and CMFU. Do not lower it just for memory savings; lower it for sensitivity evidence.

4. `image_patch_tokens`.
   It is large enough to matter for storage and matters on visually ambiguous mentions. Expect stark per-example changes even if average H@1 moves less than text.

5. `image_cls`.
   It is the safest compression target, but because it is tiny, dramatic storage results will not come from it.

6. LayerNorm and final score averaging.
   Keep these fp32/fp16. Quantizing normalization/gating/scalar score combination is unlikely to save much and can destabilize ranking calibration.

## Where Stark Results Should Come From

The clearest storage result should come from lowering `entity_text_tokens`, because it accounts for most of the fp32 cache. Going from fp32 to int4 for text tokens alone should save roughly 6.4 GB of entity-cache storage before scale overhead.

The clearest accuracy degradation should appear when `text_tokens` go to 4-bit, 3-bit, or 2-bit. If H@1 survives text-token int4, the result is strong. If it drops sharply, mixed quantization should keep text tokens at 8-bit and push image tokens lower.

The clearest qualitative rank-flip examples should come from:

- text-token low-bit runs for aliases, people, locations, and entities with similar names;
- image-token low-bit runs for visually grounded examples where the mention image is disambiguating;
- CMFU-focused ablations where text alone is plausible but the image changes the correct entity.

## First Profiling Commands

Run these from:

```text
ExpSetup/external/MIMIC_reproduction
```

Training timing should be done on a short run first:

```powershell
$env:MIMIC_INIT_CKPT="runs/WikiMEL_noleak/version_4/checkpoints/final.ckpt"
python codes/main.py --config config/wikimel_noleak.yaml
```

Inference timing should use the existing evaluator, then add timers around:

```text
model.encoder(**batch)
model.matcher(...)
entity chunk .to(device)
torch.cat(scores)
torch.argsort(...)
```

Minimum CSV columns to collect:

```text
mode,batch_idx,encoder_s,entity_transfer_s,tglu_s,vdlu_s,cmfu_s,score_concat_s,rank_sort_s,total_s
```

For training:

```text
step,dataloader_s,mention_encoder_s,entity_encoder_s,matcher_s,loss_s,backward_s,optimizer_s,total_s
```

## Practical Recommendation

Start with representation mixed quantization, not true kernel quantization:

1. Add per-tensor modes to `ExpSetup/evaluate_mimic_quantization.py`.
2. Keep `text_cls` and `image_cls` at int8 or fp16 initially.
3. Sweep `entity_text_tokens` through int8, int6, int4, int3, int2.
4. Sweep `entity_image_patch_tokens` separately.
5. Only after the sensitivity map is clear, consider kernel-level quantization or QAT.

Best first paper-style claim to test:

```text
MIMIC WikiMEL can preserve clean H@1 with mixed quantization by keeping text/global representations higher precision while aggressively compressing visual/token caches.
```
