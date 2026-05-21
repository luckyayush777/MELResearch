# Full Model Weight Quantization Smoke Results

Saved on 2026-05-20.

## Scope

Smoke tests were run inside `FullModelQuantization` to check whether full-model linear-weight quantization immediately damages retrieval accuracy.

Important limitation:

```text
No real GPU low-bit kernel backend is installed locally.
```

Checked backends:

```text
bitsandbytes = missing
auto_gptq = missing
awq = missing
optimum = missing
onnxruntime = missing
tensorrt = missing
torchao = missing
```

So this smoke test quantizes all `nn.Linear` weights and writes dequantized values back into the model for normal fp32 execution. It measures accuracy drift from quantized weights. It does not measure real low-bit kernel speed or real compressed weight memory.

## Real No-Dequantization Probe

A direct PyTorch dynamic-quantized `Linear` probe was run after questioning the dequantized smoke path.

Result:

```text
CPU dynamic quantized Linear works.
CUDA dynamic quantized Linear fails.
```

Observed CUDA failure:

```text
NotImplementedError: Could not run 'quantized::linear_dynamic' with arguments from the 'CUDA' backend.
```

Interpretation:

```text
PyTorch 2.1.2 in this environment supports dynamic quantized Linear through CPU quantized backends
such as x86/fbgemm/onednn, but not through CUDA for this model path.
```

Therefore, running the full MIMIC evaluation with true no-dequantization quantized weights on GPU is blocked without installing or integrating a GPU low-bit backend such as `bitsandbytes`, `torchao`, TensorRT, AWQ, or GPTQ-style kernels.

Running true PyTorch dynamic quantization on CPU would be a different experiment and is likely not useful for the current GPU-deployment comparison because the full MEL evaluation scores `109976` entities per query.

## bitsandbytes CUDA Backend Attempt

`bitsandbytes` was installed through the proxy. The latest `0.49.2` package pulled in a CPU-only Torch build and had to be backed out. CUDA Torch was restored to:

```text
torch = 2.1.2+cu121
CUDA = 12.1
```

Then `bitsandbytes==0.43.3` was installed with `--no-deps`, which successfully ran CUDA low-bit Linear probes on the RTX 4090:

```text
Linear8bitLt forward: ok
Linear4bit NF4 forward: ok
```

The evaluator now supports real bitsandbytes modes:

```text
--weight-modes bnb_int8
--weight-modes bnb_nf4
```

and quantization scopes:

```text
--linear-scope all
--linear-scope mimic_head
--linear-scope matcher
```

Scope definitions:

| Scope | Meaning |
|---|---|
| `all` | replace every `nn.Linear`, including CLIP text/vision backbone |
| `mimic_head` | replace MIMIC projection and matcher linear layers, excluding `encoder.clip.*` |
| `matcher` | replace only `matcher.*` linear layers |

### Real bitsandbytes Results

All rows used:

```text
cache_mode = text_image_tokens_int4
chunk_size = 5000
matcher_tile_size = 250
```

| Scope | Weight mode | Queries | H@1 | MRR | MR | Peak eval VRAM | QPS | Linear layers |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| all | bnb_int8 | 20 | 0.000 | 0.000024 | 62731.00 | 1593.78 MB | 7.760 | 163 |
| mimic_head | bnb_int8 | 20 | 0.950 | 0.975000 | 1.05 | 1944.69 MB | 7.606 | 17 |
| mimic_head | bnb_int8 | 200 | 0.965 | 0.978750 | 274.27 | 1970.85 MB | 9.552 | 17 |

Summary CSV:

```text
FullModelQuantization/bnb_real_kernel_summary.csv
```

Interpretation:

- Real bitsandbytes int8 for every `nn.Linear` collapses retrieval accuracy immediately. This is not viable as a one-shot full-model quantization approach.
- Excluding the CLIP backbone and quantizing only the MIMIC projection/matcher linear layers avoids collapse on the limited slices.
- The targeted bitsandbytes path is slower than the current best tiled baseline and does not reduce peak eval VRAM. On 200 queries it reaches `9.55 QPS`, while the non-bnb tiled path is around `16-17 QPS` on comparable limited/full settings.
- This makes real full-model low-bit kernels a negative/diagnostic result for now, not a new headline deployment result.

Practical conclusion:

```text
The current best deployment story remains mixed int4 entity-cache quantization plus matcher/transfer tiling.
Real GPU low-bit weight kernels need careful layer selection/calibration; naive full Linear replacement is unsafe.
```

Sensitive/nonlinear layers were not quantized:

```text
LayerNorm
embeddings
softmax
activations
custom non-linear ops
```

The quantized layers were the model's `nn.Linear` layers, including CLIP text/vision linear projections, MIMIC encoder projection layers, and matcher linear layers.

## Command Pattern

```powershell
$env:HF_HUB_OFFLINE='1'
python "FullModelQuantization/run_weight_quant_smoke.py" `
  --weight-modes int4 `
  --cache-mode text_image_tokens_int4 `
  --chunk-size 5000 `
  --matcher-tile-size 250 `
  --limit-queries 20 `
  --out-dir FullModelQuantization/results/weight_quant_smoke_int4 `
  --force
```

Baseline and int8 were run similarly with:

```text
--weight-modes fp32 int8
--out-dir FullModelQuantization/results/weight_quant_smoke_int8
```

## Results

All runs used:

```text
cache_mode = text_image_tokens_int4
chunk_size = 5000
matcher_tile_size = 250
limit_queries = 20
entities = 109976
```

| Weight mode | Queries | H@1 | MRR | MR | Peak eval VRAM | QPS | Quantized Linear layers | Quantized Linear params |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| fp32 | 20 | 0.95 | 0.966667 | 1.10 | 1895.90 MB | 12.980 | 0 | 0 |
| int8 | 20 | 0.95 | 0.975000 | 1.05 | 1897.50 MB | 12.800 | 163 | 123772192 |
| int4 | 20 | 1.00 | 1.000000 | 1.00 | 1895.90 MB | 13.182 | 163 | 123772192 |

Summary CSV:

```text
FullModelQuantization/weight_quant_smoke_summary.csv
```

Raw artifacts:

```text
FullModelQuantization/results/weight_quant_smoke_int8/weight_quant_smoke_summary.csv
FullModelQuantization/results/weight_quant_smoke_int4/weight_quant_smoke_summary.csv
```

## Interpretation

The smoke result is encouraging but not conclusive.

On the first 20-query slice:

- int8 linear-weight quantization did not reduce H@1.
- int4 linear-weight quantization also did not reduce H@1.
- The int4 row reached perfect H@1/MRR on this tiny slice, which should be treated as evidence that the slice is too easy/noisy, not proof that int4 full-model quantization is safe.

The current smoke path does not reduce peak VRAM because weights are dequantized back into normal tensors for execution. This is expected. The purpose was only to test accuracy sensitivity before spending time on real-kernel integration.

## Recommended Next Experiment

Run a larger partial evaluation before any full test-set run:

```powershell
$env:HF_HUB_OFFLINE='1'
python "FullModelQuantization/run_weight_quant_smoke.py" `
  --weight-modes int8 int4 `
  --cache-mode text_image_tokens_int4 `
  --chunk-size 5000 `
  --matcher-tile-size 250 `
  --limit-queries 200 `
  --out-dir FullModelQuantization/results/weight_quant_partial_200
```

Decision rule:

| Result on 200 queries | Action |
|---|---|
| H@1/MRR stable for int8 and int4 | Consider full run if time remains |
| int8 stable, int4 drops | Run full int8 only |
| both drop | Stop and report full-model quantization as future work |
| runtime/setup becomes painful | Stop; current cache quantization + tiling result is already stronger |

## Current Status

The smoke test does not reveal an immediate accuracy failure from quantizing all linear weights. The next meaningful check is `limit_queries=200`.

## Partial 200-Query Run

A larger limited run was completed after the 20-query smoke.

Command:

```powershell
$env:HF_HUB_OFFLINE='1'
python "FullModelQuantization/run_weight_quant_smoke.py" `
  --weight-modes int8 int4 `
  --cache-mode text_image_tokens_int4 `
  --chunk-size 5000 `
  --matcher-tile-size 250 `
  --limit-queries 200 `
  --out-dir FullModelQuantization/results/weight_quant_partial_200 `
  --force
```

Results:

| Weight mode | Queries | H@1 | H@3 | H@5 | H@10 | H@20 | MRR | MR | Peak eval VRAM | QPS |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| int8 | 200 | 0.970 | 0.990 | 0.995 | 0.995 | 0.995 | 0.981000 | 287.36 | 1921.69 MB | 16.754 |
| int4 | 200 | 0.975 | 0.990 | 0.990 | 0.990 | 0.990 | 0.982506 | 327.72 | 1923.28 MB | 16.782 |

Summary CSV:

```text
FullModelQuantization/weight_quant_partial_200_summary.csv
```

Raw artifacts:

```text
FullModelQuantization/results/weight_quant_partial_200/weight_quant_smoke_summary.csv
FullModelQuantization/results/weight_quant_partial_200/weight_quant_smoke_results.json
```

Interpretation:

- The 200-query partial still shows no accuracy collapse from quantizing all `nn.Linear` weights.
- `int8` and `int4` are close on this partial slice.
- Peak eval VRAM remains around `1.92 GB` because this is dequantized fp32 execution, not a real compressed-kernel run.
- This result is good enough to justify a full int8/int4 accuracy run if time remains, but it should not be reported as a real low-bit kernel deployment result.

Recommended next decision:

| Time left | Recommendation |
|---|---|
| Enough time for a full run | Run full `int8 int4` linear-weight accuracy pass with `matcher_tile_size=250`. |
| Limited time | Run `limit_queries=500` or `1000` first. |
| Need to preserve final report quality | Stop here and present this as an exploratory extension. |
