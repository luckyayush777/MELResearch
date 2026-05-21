# Entity Text Token Quantization Runbook

Use this from the repo root:

```powershell
cd "C:\Users\user\Desktop\kmpooja"
```

If Hugging Face assets need network access, use the proxy:

```powershell
$env:HTTP_PROXY="http://172.31.2.4:8080"
$env:HTTPS_PROXY="http://172.31.2.4:8080"
```

## Smoke Test

This checks that the pipeline works without committing to the full test-set runtime:

```powershell
python ".\Mixed Quantization setup\evaluate_entity_text_tokens.py" `
  --entity-text-token-modes fp32 int8 int4 `
  --entity-limit 20000 `
  --limit-queries 100 `
  --out-dir "results/entity_text_tokens_smoke"
```

## Full Sweep

This is the main experiment for showing whether shrinking only `entity_text_tokens` gives the gain:

```powershell
python ".\Mixed Quantization setup\evaluate_entity_text_tokens.py" `
  --entity-text-token-modes fp32 int8 int6 int4 int3 int2 `
  --other-mode fp32 `
  --out-dir "results/entity_text_tokens_full"
```

After the sweep finishes, generate the attribution report:

```powershell
python ".\Mixed Quantization setup\summarize_entity_text_token_results.py" `
  --summary-csv "results/entity_text_tokens_full/entity_text_token_quantization_summary.csv" `
  --out "results/entity_text_tokens_full/attribution_report.md"
```

Outputs are saved inside:

```text
Mixed Quantization setup/results/entity_text_tokens_full
```

Main summary file:

```text
Mixed Quantization setup/results/entity_text_tokens_full/entity_text_token_quantization_summary.csv
```

Detailed JSON:

```text
Mixed Quantization setup/results/entity_text_tokens_full/entity_text_token_quantization_results.json
```

Generated attribution report:

```text
Mixed Quantization setup/results/entity_text_tokens_full/attribution_report.md
```

Each mode also has:

```text
result.json
entity_cache/manifest.json
batch_metrics/*.json
rank_batches/*.npy
```

## Metrics Captured

Accuracy:

```text
hits@1,hits@3,hits@5,hits@10,hits@20,mr,mrr
```

Overall runtime:

```text
seconds,speedup_vs_fp32,timed_subprocess_s,unattributed_overhead_s
```

Subprocess timing:

```text
mention_to_device_s
mention_encoder_s
entity_transfer_s
layernorm_s
tglu_s
vdlu_s
cmfu_s
score_concat_s
rank_sort_s
```

Per-subprocess runtime share:

```text
mention_encoder_s_pct
entity_transfer_s_pct
tglu_s_pct
vdlu_s_pct
cmfu_s_pct
rank_sort_s_pct
```

Cache attribution:

```text
entity_cache_storage_mb
entity_cache_text_cls_mb
entity_cache_image_cls_mb
entity_cache_text_tokens_mb
entity_cache_image_patch_tokens_mb
cache_size_ratio_vs_fp32
text_token_size_ratio_vs_fp32
```

Quality and runtime deltas:

```text
delta_hits@1_vs_fp32
delta_mrr_vs_fp32
delta_seconds_vs_fp32
delta_entity_cache_storage_mb_vs_fp32
delta_entity_cache_text_tokens_mb_vs_fp32
```

## How To Explain The Gain

If the experiment works, the clean argument should look like this:

```text
We changed only entity_text_tokens from fp32 to intN and held text_cls, image_cls, and image_patch_tokens at fp32.
The entity_text_tokens cache shrank by X MB/Y%.
Overall entity cache shrank by X MB/Y%.
The entity_transfer_s subprocess decreased by X seconds/Y%.
The TGLU time changed by X seconds/Y%, because TGLU consumes entity_text_tokens for every entity chunk.
Accuracy changed by delta H@1 = X and delta MRR = Y.
Therefore the gain is attributable mainly to reduced entity_text_tokens storage/transfer, not to changes in mention encoding or unrelated ranking work.
```

## Best First Result To Look For

The most publishable result would be:

```text
entity_text_tokens int4 gives a large cache/transfer reduction while preserving H@1 and MRR close to fp32.
```

If int4 hurts too much, use:

```text
entity_text_tokens int8 or int6 preserve ranking, while int4/int3/int2 define the degradation boundary.
```
