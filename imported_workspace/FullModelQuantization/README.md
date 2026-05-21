# Full Model Quantization Notes

Created on 2026-05-20.

## Question

Current cache/tiled-matcher experiments can take about `2-3 hours` in the worst case. Will quantizing model weights and running full-model quantization experiments be slower?

## Short Answer

Not necessarily. Weight quantization itself should usually be a one-time conversion step and should not dominate a multi-hour evaluation. The evaluation may be faster, similar, or slower depending on how quantization is implemented.

## Expected Runtime By Approach

| Approach | Likely runtime vs current experiments | Why |
|---|---:|---|
| fp16 weights | similar or faster | Native GPU support is strong; usually safe and fast. |
| int8 dynamic/static weights with real kernels | similar or faster | Can reduce weight bandwidth and compute, if the layers use optimized kernels. |
| simulated int8/int4 weights with dequantization before compute | similar or slower | Saves storage, but every forward pass may pay dequantization overhead. |
| full int4 with real kernels | potentially faster, higher setup cost | Speed depends heavily on kernel/library support and whether model layers are compatible. |
| quantized weights plus existing int4 entity cache | likely similar total runtime | Current bottleneck is still entity transfer/matcher scoring, not just model weights. |

## Why It May Not Be Much Faster

The current best result already showed that the largest deployment costs are not only model weights:

```text
entity cache size
entity transfer/dequantization
TGLU matcher temporary tensors
VDLU/CMFU repeated matcher work in entity-subtile mode
```

So quantizing weights can reduce model footprint and mention-encoder cost, but it may not dominate total runtime unless the matcher and encoder use real optimized low-bit kernels.

## Main Benefit To Test

Full model quantization could help with:

- lower model weight memory;
- lower mention-encoder memory;
- lower activation memory if supported;
- faster mention encoding;
- smaller deployment artifact;
- cleaner end-to-end low-memory deployment story.

But it has higher accuracy risk than entity-cache quantization because it changes query-side and matcher-side computation.

## Recommended Experiment Order

1. `fp16` model weights with existing mixed-int4 entity cache.
2. `int8` model weights with existing mixed-int4 entity cache.
3. Quantize only matcher linear layers.
4. Quantize only mention/query encoder.
5. Try full model quantization after the above ablations.

For every run, compare against:

```text
text_image_tokens_int4, chunk_size=5000, matcher_tile_size=250
```

Current best balanced row:

| Setting | H@1 | MRR | Peak eval VRAM | QPS |
|---|---:|---:|---:|---:|
| `text_image_tokens_int4`, `chunk_size=5000`, `matcher_tile_size=250` | 0.741379 | 0.818733 | 1922.29 MB | 17.444 |

## Practical Runtime Expectation

If the run still evaluates all `4002` queries over `109976` entities, expect the same broad runtime scale unless optimized kernels remove a real bottleneck.

Reasonable estimate:

```text
fp16/int8 weight-only tests: probably within the current 2-3 hour envelope.
simulated int4/full low-bit experiments: may be slower if dequantization happens every forward pass.
real int4 kernels: may be faster, but setup/debugging time can be much higher.
```

Start with a `--limit-queries 20` smoke test before full evaluation.
