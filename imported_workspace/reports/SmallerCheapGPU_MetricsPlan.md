# Smaller Cheap GPU Metrics Plan

Saved on 2026-05-19.

## End Goal

The end goal is to make MIMIC-style multimodal entity linking run on smaller, cheaper GPUs.

The experiment should not only show that quantization preserves accuracy. It should also show why the system becomes easier to deploy:

```text
lower peak GPU memory
smaller entity cache
lower transfer cost
lower latency
higher throughput
larger feasible entity chunks
same or acceptable retrieval quality
```

For now, log broadly. Later, we can separate the useful metrics from noisy ones.

## Core Question

The main deployment question is:

```text
Can mixed quantization reduce memory and latency enough to run MIMIC inference on cheaper GPUs without losing retrieval quality?
```

The current best candidate is:

```text
entity_text_tokens = int4
text_cls = fp32
image_cls = fp32
image_patch_tokens = fp32
```

## Experiment Conditions To Log

Every run should record:

```text
run_id
timestamp
git/status note if available
checkpoint path
config path
dataset split
number of entities
number of queries
chunk size
batch size
device name
CUDA version
PyTorch version
GPU total memory
driver version if available
quantization mode per tensor
cache format version
```

For tensor modes:

```text
text_cls_mode
image_cls_mode
text_tokens_mode
image_patch_tokens_mode
```

## Accuracy Metrics

Keep all current retrieval metrics:

```text
queries
skipped_queries
H@1
H@3
H@5
H@10
H@20
MR
MRR
```

Also log deltas against fp32:

```text
delta_H@1_vs_fp32
delta_H@3_vs_fp32
delta_H@10_vs_fp32
delta_H@20_vs_fp32
delta_MR_vs_fp32
delta_MRR_vs_fp32
```

Optional but useful:

```text
rank_flip_count
rank_improved_count
rank_worsened_count
mean_abs_rank_delta
median_abs_rank_delta
max_rank_worsening
max_rank_improvement
gold_score_delta_mean
gold_score_delta_std
top1_margin_mean
top1_margin_median
gold_vs_top1_margin_mean
```

## Memory Metrics

This is the most important category for the smaller-GPU story.

Log GPU memory at several phases:

```text
gpu_total_memory_mb
gpu_peak_allocated_mb_total_run
gpu_peak_reserved_mb_total_run
gpu_peak_allocated_mb_entity_cache_build
gpu_peak_reserved_mb_entity_cache_build
gpu_peak_allocated_mb_eval
gpu_peak_reserved_mb_eval
gpu_peak_allocated_mb_mention_encoder
gpu_peak_allocated_mb_entity_chunk_transfer
gpu_peak_allocated_mb_matcher
gpu_peak_allocated_mb_tglu
gpu_peak_allocated_mb_vdlu
gpu_peak_allocated_mb_cmfu
```

Useful PyTorch calls:

```python
torch.cuda.reset_peak_memory_stats()
torch.cuda.max_memory_allocated()
torch.cuda.max_memory_reserved()
torch.cuda.memory_allocated()
torch.cuda.memory_reserved()
```

Also log CPU RAM if possible:

```text
cpu_rss_mb_start
cpu_rss_mb_after_entity_cache_load
cpu_rss_mb_peak
```

## Cache Size Metrics

Log both logical and actual physical sizes.

Logical packed representation size:

```text
entity_cache_storage_mb_logical
entity_cache_text_cls_mb_logical
entity_cache_image_cls_mb_logical
entity_cache_text_tokens_mb_logical
entity_cache_image_patch_tokens_mb_logical
cache_size_ratio_vs_fp32
text_token_size_ratio_vs_fp32
```

Actual artifact size on disk:

```text
entity_cache_disk_mb
entity_cache_shard_count
avg_shard_disk_mb
max_shard_disk_mb
min_shard_disk_mb
disk_size_ratio_vs_fp32
```

In-memory loaded cache size:

```text
entity_cache_cpu_memory_mb
entity_cache_gpu_chunk_memory_mb
entity_cache_max_chunk_memory_mb
```

This distinction matters:

```text
logical packed size = theoretical compressed representation
disk size = actual saved experiment artifacts
runtime memory = what matters for cheaper GPU feasibility
```

## Latency Metrics

Log total latency:

```text
total_eval_seconds
avg_query_latency_ms
median_query_latency_ms
p90_query_latency_ms
p95_query_latency_ms
p99_query_latency_ms
min_query_latency_ms
max_query_latency_ms
```

Because evaluation runs by batches, also log batch latency:

```text
avg_batch_latency_s
median_batch_latency_s
p90_batch_latency_s
p95_batch_latency_s
p99_batch_latency_s
```

Per-query latency can be estimated as:

```text
batch_latency / number_of_queries_in_batch
```

## Throughput Metrics

Throughput is important for deployment.

Log:

```text
queries_per_second
entities_scored_per_second
query_entity_pairs_per_second
batches_per_second
chunks_per_second
```

For query-entity pairs:

```text
query_entity_pairs = queries * num_entities
query_entity_pairs_per_second = query_entity_pairs / total_eval_seconds
```

## Subprocess Timing

Keep the current subprocess timers:

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
unattributed_overhead_s
```

Also log percentages:

```text
mention_encoder_pct
entity_transfer_pct
layernorm_pct
tglu_pct
vdlu_pct
cmfu_pct
rank_sort_pct
unattributed_overhead_pct
```

Add per-batch versions:

```text
batch_idx
batch_queries
batch_total_s
batch_mention_encoder_s
batch_entity_transfer_s
batch_tglu_s
batch_vdlu_s
batch_cmfu_s
batch_rank_sort_s
batch_peak_gpu_allocated_mb
batch_peak_gpu_reserved_mb
```

## Entity Cache Build Metrics

Even if cache build is offline, log it.

```text
entity_cache_build_total_s
entity_cache_encode_s
entity_cache_quantize_s
entity_cache_save_s
entity_cache_load_s
entity_cache_entities_per_second
entity_cache_shards
entity_cache_peak_gpu_allocated_mb
entity_cache_peak_cpu_rss_mb
```

If the deployment story assumes precomputed cache, clearly separate:

```text
offline_cache_build_time
online_query_eval_time
```

## Chunk Size / Feasibility Metrics

To prove smaller GPU readiness, run chunk-size sweeps.

Log:

```text
chunk_size
max_successful_chunk_size
oom_at_chunk_size
peak_gpu_memory_mb_by_chunk_size
latency_by_chunk_size
throughput_by_chunk_size
accuracy_by_chunk_size
```

Important chunk sizes to try:

```text
1000
2500
5000
10000
20000
40000
```

The best deployment result would show:

```text
fp32 requires small chunks or fails on low VRAM
int4 allows larger chunks or avoids OOM
int4 improves latency/throughput at the same accuracy
```

## Smaller GPU Simulation Metrics

If actual smaller GPUs are unavailable, simulate constraints.

Log:

```text
gpu_memory_fraction
artificial_memory_limit_mb
run_success
failure_reason
oom_stage
max_chunk_size_under_limit
latency_under_limit
throughput_under_limit
```

Possible target GPU budgets:

```text
24 GB
16 GB
12 GB
8 GB
6 GB
4 GB
```

For each budget, log whether each mode can run:

```text
fp32
int8
int6
int4
int3
int2
```

## Failure Metrics

Failures are useful evidence.

Log:

```text
run_success
failed_stage
exception_type
exception_message
oom_boolean
last_completed_batch
last_completed_chunk
partial_results_available
resume_success
```

Common failed stages:

```text
model_load
entity_cache_build
entity_cache_load
mention_encoder
entity_transfer
matcher_tglu
matcher_vdlu
matcher_cmfu
rank_sort
result_write
```

## Quantization Error Metrics

For each tensor that is quantized:

```text
mean_abs_error
max_abs_error
relative_l2_error
cosine_similarity_mean
cosine_similarity_std
sign_flip_rate
zero_rate
saturation_rate
scale_mean
scale_std
scale_min
scale_max
```

For `entity_text_tokens`, compute error globally and per token position if possible:

```text
text_token_position_error_mean
cls_position_error
sep_position_error
padding_position_error
non_padding_position_error
```

## Ranking Stability Metrics

Quantization can preserve average H@1 while still changing rankings.

Log:

```text
top1_same_rate
top3_overlap_mean
top5_overlap_mean
top10_overlap_mean
top20_overlap_mean
gold_rank_delta_mean
gold_rank_delta_median
gold_rank_delta_p95
gold_rank_delta_max
rank_flip_examples_path
```

For examples:

```text
query_id
mention
sentence
gold_entity
fp32_rank
quantized_rank
rank_delta
fp32_top1
quantized_top1
entity_type
has_image
```

## Dataset Slice Metrics

Break down accuracy and rank stability by useful slices:

```text
entity_type
mention_length_bucket
sentence_length_bucket
has_mention_image
has_entity_image
ambiguous_name_bucket
gold_rank_fp32_bucket
people
locations
organizations
events
other
```

Useful slice metrics:

```text
slice_queries
slice_H@1
slice_MRR
slice_delta_H@1_vs_fp32
slice_delta_MRR_vs_fp32
slice_mean_rank_delta
```

## Cost-Oriented Metrics

For the cheaper GPU argument, translate technical gains into deployment framing.

Log or derive:

```text
minimum_required_gpu_memory_mb
estimated_gpu_class_supported
relative_gpu_memory_reduction
relative_latency_reduction
relative_throughput_gain
queries_per_dollar_proxy
cache_storage_cost_reduction_proxy
```

GPU classes to discuss:

```text
RTX 4090 24GB
RTX 3060 12GB
RTX 4060 Ti 16GB
T4 16GB
L4 24GB
consumer 8GB GPU
```

Do not overclaim until tested directly. Phrase as:

```text
The memory profile moves the system toward smaller GPU feasibility.
```

instead of:

```text
This proves deployment on all small GPUs.
```

## Recommended Master CSV Columns

One row per mode/run:

```text
run_id
timestamp
device_name
gpu_total_memory_mb
split
queries
num_entities
chunk_size
eval_batch_size
text_cls_mode
image_cls_mode
text_tokens_mode
image_patch_tokens_mode
H@1
H@3
H@5
H@10
H@20
MR
MRR
delta_H@1_vs_fp32
delta_MRR_vs_fp32
total_eval_seconds
speedup_vs_fp32
queries_per_second
query_entity_pairs_per_second
avg_query_latency_ms
p50_query_latency_ms
p95_query_latency_ms
p99_query_latency_ms
gpu_peak_allocated_mb_eval
gpu_peak_reserved_mb_eval
gpu_peak_allocated_mb_total
entity_cache_storage_mb_logical
entity_cache_disk_mb
entity_cache_text_tokens_mb_logical
cache_size_ratio_vs_fp32
text_token_size_ratio_vs_fp32
entity_transfer_s
tglu_s
vdlu_s
cmfu_s
rank_sort_s
mention_encoder_s
entity_transfer_pct
tglu_pct
oom_boolean
failed_stage
notes
```

## Minimum Evidence For The Smaller-GPU Claim

At minimum, produce this table:

| Mode | H@1 | MRR | Peak VRAM | Latency/query | QPS | Total cache | Max chunk size | Success on target budget |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| fp32 | | | | | | | | |
| int8 text tokens | | | | | | | | |
| int4 text tokens | | | | | | | | |
| int3 text tokens | | | | | | | | |
| int2 text tokens | | | | | | | | |

The strongest result would be:

```text
int4 preserves H@1/MRR, reduces peak VRAM, increases throughput, and allows a larger chunk size or lower memory budget than fp32.
```

## Current Best Hypothesis

Based on the 19Feb results:

```text
entity_text_tokens int4 is likely the best conservative deployment point.
```

Reason:

```text
H@1 unchanged
MRR unchanged/slightly higher
text-token cache much smaller
entity transfer time much lower
TGLU time much lower
speedup about 1.77x
```

Next proof needed:

```text
peak GPU memory and smaller-GPU feasibility.
```
