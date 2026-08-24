# Clean Compact-fp32 vs MIMIC Experiment Plan

Date: 2026-08-23

## 1. Purpose

This experiment will determine whether the compact, token-pooled fp32 model can
retain strong full-KB multimodal entity-linking accuracy while using a much
smaller persistent entity knowledge-base representation than MIMIC.

The first dataset is WikiMEL. RichpediaMEL and WikiDiverse will reuse the same
training, leakage-audit, evaluation, and measurement mechanism through
dataset-specific manifests. They are validation datasets, not separate model
mechanisms.

The experiment must answer four questions:

1. Is the dataset free of split and image leakage?
2. How does compact fp32 accuracy compare with MIMIC fp32 under an identical
   clean full-KB protocol?
3. What is the measured, serialized entity-cache compression ratio?
4. Does the compact representation improve latency, throughput, and peak memory
   as well as storage?

## 2. Current reference point

The current paper-grade MIMIC reference uses the clean WikiMEL no-leak split:

| Metric | Current clean MIMIC fp32 |
|---|---:|
| Test queries | 4,002 |
| Candidate entities | 109,976 |
| H@1 | 0.741629 |
| MRR | 0.818908 |
| Logical entity cache | 10,860.67 MB |
| Peak evaluation VRAM, chunk 5,000 | 5,851.28 MB |
| Throughput | 9.074 QPS |

The earlier 96.65 H@1 checkpoint is invalid for scientific comparison because
its dataset contained split duplicates, image overlap, and gold entity-image
leakage. It must not be used as a baseline.

## 3. Claims this experiment may support

If successful, the primary claim will be:

> Under an identical leakage-audited, full-KB WikiMEL protocol, compact fp32
> preserves competitive retrieval accuracy while reducing the persistent
> entity representation substantially relative to MIMIC fp32.

The experiment must not claim comparison with candidate-limited UniMEL results.
UniMEL's 100-candidate and 16-candidate results are a different task from the
109,976-entity full-KB evaluation used here.

## 4. Experimental principles

The comparison is valid only if both models use:

- the same train, development, and test rows;
- the same entity KB and entity-ID ordering;
- the same sanitized mention and entity images;
- the same full-KB candidate policy;
- the same metric implementation;
- the same hardware and measurement harness;
- fp32 weights and fp32 cached representations;
- model selection using development MRR only;
- no tuning against test metrics.

Architecture-specific hyperparameters may differ when required by the released
model, but every difference must be recorded.

## 5. Required dataset variants

Create the following variants for diagnosis. Only the clean visual variant is
the primary benchmark.

| ID | Split deduplication | Entity-image policy | Scientific use |
|---|---|---|---|
| `official_original` | No | Original images | Published-protocol diagnostic only |
| `dedupe_only` | Yes | Original images | Measures the effect of row deduplication; may still leak |
| `clean_no_entity_images` | Yes | No entity images | Existing clean MIMIC reference |
| `clean_sanitized_visual` | Yes | Keep only non-overlapping entity images | Primary comparison |

Do not use `official_original` or `dedupe_only` for headline claims if an audit
reports leakage.

## 6. Leakage audit

### 6.1 Inventory and manifest

For each dataset, write a manifest containing:

- dataset name and source;
- source archive/file hashes;
- preprocessing script commit;
- build timestamp;
- train/dev/test row counts;
- unique mention, image, and gold-entity counts;
- KB entity count;
- number of entity images before and after sanitization;
- entity-ID ordering checksum;
- paths to the exact split and KB files.

Save this as `dataset_manifest.json` inside the generated dataset directory.

### 6.2 Existing exact checks

Run the existing audit:

```powershell
python audit_wikimel_leakage.py `
  --data-root <PATH_TO_DATASET_ROOT>
```

The current script checks all split pairs for:

- exact row overlap;
- exact mention-plus-sentence overlap;
- exact image-path overlap;
- a mention image appearing in its gold entity's `image_list`.

Capture its output in `audit_exact.txt`.

### 6.3 Strengthen the audit

Extend the audit or add a second script that checks content rather than only raw
paths and strings.

#### Normalized text overlap

Normalize mention and context text by:

1. Unicode NFKC normalization;
2. HTML unescaping;
3. lowercasing;
4. collapsing whitespace;
5. normalizing punctuation;
6. stripping surrounding whitespace.

Check all train/dev/test pairs for:

- normalized full-row overlap;
- normalized mention-plus-context overlap;
- context-only overlap;
- duplicate rows within each split.

#### Image overlap

For every mention and entity image, compute:

- canonical normalized path;
- file size;
- SHA-256 byte hash;
- perceptual hash, such as pHash;
- decode success/failure.

Check all split pairs and the entity KB for:

- exact path matches;
- exact byte matches under different filenames;
- near-duplicate perceptual matches;
- mention image present in the gold entity image set;
- mention image present in any candidate entity image set.

Record the perceptual-distance threshold in the manifest. Manually inspect a
sample of matches close to the threshold before freezing it.

#### Label and metadata leakage

Check that:

- filenames and directory names do not encode the gold entity ID;
- generated descriptions do not contain answer IDs or candidate positions;
- the gold candidate is not always placed in a fixed list position;
- test rows were not used to train an augmentation, summarizer, or selector;
- entity-ID maps are identical across model evaluations.

Entity identities may legitimately appear across splits because entity linking
uses a shared KB. Repeated entities are not themselves leakage; repeated mention
instances or copied evidence are.

### 6.4 Build a globally sanitized visual KB

Do not solve gold-image leakage by deleting all entity images. Instead:

1. Form a global blacklist from every mention image in train, dev, and test.
2. Add exact byte-hash and confirmed perceptual duplicates to the blacklist.
3. Remove blacklisted images from every entity's image list.
4. Retain independent entity images that do not overlap with any mention image.
5. Represent entities with no remaining image using an explicit missing-image
   mask, not random pixels.
6. Freeze one sanitized KB for all splits and both models.

Using all splits to construct a removal-only blacklist is acceptable because it
does not add test information to training; it globally removes potentially
leaking evidence. Do not use test labels to select or add entity evidence.

### 6.5 Leakage acceptance gate

The primary dataset is accepted only when all of the following are zero:

- cross-split exact row overlap;
- cross-split normalized mention-plus-context overlap;
- cross-split exact image-byte overlap;
- confirmed cross-split perceptual image overlap;
- gold entity-image leakage;
- answer-ID or candidate-position leakage.

If a nonzero count remains, stop model training, inspect the cases, repair the
build, and rerun the audit. Save both the machine-readable audit JSON and a
human-readable summary.

## 7. Freeze the evaluation protocol

Before training either model, create an `evaluation_manifest.json` containing:

- test query IDs in their final order;
- gold entity ID for every query;
- ordered KB entity IDs;
- query and entity counts;
- candidate policy: full KB;
- score direction and tie handling;
- metric definitions;
- evaluation chunk size;
- hardware identifier;
- software versions and Git commit.

Use a single shared evaluator where possible. If architecture-specific scoring
requires separate evaluators, validate them on saved synthetic scores and verify
that they return identical ranks and metrics.

The primary evaluation ranks every query against all 109,976 WikiMEL entities.
Candidate-limited results may be reported separately but must be clearly labelled
and must not replace the full-KB result.

## 8. Train the fp32 baselines

### 8.1 MIMIC fp32

Train MIMIC from scratch on `clean_sanitized_visual`.

1. Use the existing clean MIMIC configuration as the starting point.
2. Change only dataset/image paths and settings required for the sanitized KB.
3. Keep fp32 training and fp32 entity caches.
4. Select the checkpoint with best development MRR.
5. Record every epoch's training loss, development H@1/H@10/MRR, wall time, and
   peak training VRAM.
6. Evaluate the test set only after the configuration is frozen.

The existing `clean_no_entity_images` result remains a modality ablation. It is
not the fairest comparator for a compact model using sanitized entity images.

### 8.2 Compact fp32

Train the compact, token-pooled model on the identical
`clean_sanitized_visual` dataset.

1. Use the same split files, KB ordering, and image set as MIMIC.
2. Confirm that each entity cache entry is the intended compact representation,
   normally one pooled vector per entity.
3. Keep model weights and cached representations in fp32.
4. Select the checkpoint using development MRR.
5. Use the same test-query order and full-KB evaluator.
6. Record the same training metrics as MIMIC.

Do not use the historical compact-model checkpoint as the clean result. It was
not trained under this frozen leakage-audited protocol.

### 8.3 Seeds

Use one seed for pipeline debugging. After the pipeline is correct, run at least
three fixed seeds for both models. Report mean, standard deviation, and individual
seed results. Pair seeds across models where possible.

## 9. Accuracy and ranking metrics

Report these metrics on development and test data:

- H@1;
- H@3;
- H@5;
- H@10;
- H@20;
- MRR;
- mean rank;
- median rank;
- 95th-percentile rank;
- number and percentage of queries evaluated;
- number of missing-gold or invalid queries.

Also save the gold rank for every query. This enables paired significance tests
and error analysis without rerunning the models.

### 9.1 Paired comparison

For compact fp32 versus MIMIC fp32, report:

- absolute H@1 and MRR difference;
- relative accuracy retention;
- queries improved, worsened, and unchanged;
- mean and median absolute rank movement;
- paired bootstrap confidence intervals over queries;
- McNemar's test for Top-1 correctness, if appropriate.

### 9.2 Slice analysis

Where metadata permits, report H@1 and MRR by:

- entity type;
- mention ambiguity or candidate-name collision count;
- mention-text length;
- entity-description length;
- mention image present/missing;
- sanitized entity image present/missing;
- text-dominant versus visually grounded cases;
- frequency of the gold entity in training.

This analysis is important because compact pooling may help by removing noise but
may hurt cases requiring fine local text or image-patch evidence.

## 10. Storage metrics

Measure rather than infer the storage claim.

For each model, report:

- number and shape of cached tensors;
- tensor dtype;
- raw tensor payload bytes: `numel * element_size`;
- logical cache size;
- actual serialized cache-directory size;
- serialization/container overhead;
- bytes per entity;
- model checkpoint size, reported separately;
- optional ANN index size, reported separately.

Use:

```text
compression_ratio = MIMIC fp32 serialized entity cache bytes
                    / compact fp32 serialized entity cache bytes
```

Do not mix model checkpoint size, entity cache size, dataset images, and ANN index
size into one number. Report them as separate components and optionally as a
clearly defined deployment total.

Verify cache completeness using entity count, entity-ID checksum, shard count,
and file hashes.

## 11. Runtime and memory metrics

Run both models on the same GPU with the same query batch size and comparable
full-KB scheduling.

Report:

- one-time entity-cache construction time;
- query encoding time;
- host-to-device transfer time;
- matcher/scoring time;
- total evaluation wall time;
- latency per query;
- QPS;
- peak CUDA memory allocated;
- peak CUDA memory reserved;
- peak process RAM;
- cache load time from disk;
- cold-cache and warm-cache results, clearly separated.

Protocol:

1. Record GPU model, driver, CUDA, PyTorch, and package versions.
2. Close unrelated GPU workloads.
3. Use at least one warm-up run.
4. Reset CUDA peak-memory statistics before every measured run.
5. Synchronize CUDA before starting and stopping timers.
6. Run at least five timed repetitions after warm-up.
7. Report mean, standard deviation, median, and raw measurements.
8. Use the same query order for every run.

The current MIMIC evaluator can produce H@1/H@3/H@5/H@10/H@20/MRR/MR and
logical cache bytes. Extend or wrap it for identical compact-model output and the
additional runtime measurements.

## 12. Required comparison tables

### 12.1 Accuracy

| Dataset | Protocol | Model | H@1 | H@3 | H@5 | H@10 | H@20 | MRR | MR |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| WikiMEL | clean full KB | MIMIC fp32 | | | | | | | |
| WikiMEL | clean full KB | compact fp32 | | | | | | | |

### 12.2 Storage

| Model | Entity representation | Raw logical cache | Serialized cache | Bytes/entity | Compression vs MIMIC |
|---|---|---:|---:|---:|---:|
| MIMIC fp32 | CLS + local text/image tokens | | | | 1.00x |
| compact fp32 | pooled entity representation | | | | |

### 12.3 Systems

| Model | Cache build | Eval runtime | ms/query | QPS | Peak allocated VRAM | Peak RAM |
|---|---:|---:|---:|---:|---:|---:|
| MIMIC fp32 | | | | | | |
| compact fp32 | | | | | | |

### 12.4 Leakage

| Check | Train-dev | Train-test | Dev-test | Gold entity-image leakage |
|---|---:|---:|---:|---:|
| Exact rows | | | | n/a |
| Normalized text | | | | n/a |
| Exact image hash | | | | |
| Confirmed perceptual duplicate | | | | |

## 13. Decision rules

Treat the experiment as successful when:

1. all primary leakage gates are zero;
2. both models evaluate the identical query set and KB ordering;
3. the compact serialized entity cache is at least 15x smaller than MIMIC fp32;
4. compact fp32 H@1 is within 1.0 absolute percentage point of MIMIC fp32, or
   exceeds it;
5. MRR and the important error slices do not show a hidden material regression;
6. runtime and memory measurements are repeatable.

Interpret other outcomes as follows:

- `>=15x` smaller and within 1 point H@1: strong main result.
- `>=15x` smaller and 1–3 points lower: viable accuracy/storage trade-off; add a
  compact reranker or distillation.
- `<15x` smaller: inspect serialization overhead and verify that local tokens are
  not still being cached.
- Large visual-slice regression: improve compact visual pooling or retain a small
  set of selected patch tokens.
- Both models far below their standard-protocol results: quantify the cost of
  cleaning; do not silently compare against leaky or candidate-limited SOTA.

## 14. RichpediaMEL and WikiDiverse

The model mechanism should not change for the other two datasets. Implement a
dataset adapter or manifest that maps dataset-specific filenames and fields into
one canonical schema:

```text
mention_id
mention_text
mention_image_list
gold_entity_id
entity_id
entity_name
entity_description
entity_image_list
```

For each additional dataset:

1. map the source data into the canonical schema;
2. run the full leakage audit independently;
3. build a globally sanitized visual KB;
4. freeze the full-KB evaluation manifest;
5. train MIMIC fp32 and compact fp32 with dataset-specific paths/counts only;
6. collect the identical accuracy, storage, runtime, and memory metrics;
7. run three seeds for the final table.

The expected differences are data plumbing, entity counts, image naming, split
sizes, and possibly text-length limits. Do not introduce dataset-specific model
mechanisms unless a documented ablation justifies them. Any unavoidable
dataset-specific hyperparameter must be disclosed.

## 15. Execution checklist

- [ ] Freeze Git commit and environment versions.
- [ ] Inventory WikiMEL source data and write source hashes.
- [ ] Run the existing exact leakage audit.
- [ ] Add normalized-text and image-hash/perceptual checks.
- [ ] Build `clean_sanitized_visual`.
- [ ] Pass all leakage acceptance gates.
- [ ] Freeze dataset and evaluation manifests.
- [ ] Smoke-test MIMIC and compact-model data loading.
- [ ] Train one debugging seed for MIMIC fp32.
- [ ] Train one debugging seed for compact fp32.
- [ ] Verify identical query IDs, gold IDs, and KB ordering.
- [ ] Serialize and validate both entity caches.
- [ ] Run full-KB development evaluation.
- [ ] Freeze checkpoint selection and configuration.
- [ ] Run the test evaluation once per final seed.
- [ ] Measure storage, latency, throughput, VRAM, and RAM.
- [ ] Save per-query ranks and raw timing repetitions.
- [ ] Complete paired significance and slice analyses.
- [ ] Repeat final runs for three seeds.
- [ ] Adapt the same pipeline to RichpediaMEL.
- [ ] Adapt the same pipeline to WikiDiverse.
- [ ] Produce final accuracy, storage, systems, and leakage tables.

## 16. Final deliverables

Store the final outputs under a versioned experiment directory containing:

```text
dataset_manifest.json
evaluation_manifest.json
audit_exact.txt
audit_detailed.json
environment.json
configs/
checkpoints_manifest.json
entity_cache_manifests/
per_query_ranks/
accuracy_results.json
storage_results.json
timing_raw.csv
systems_results.json
slice_results.csv
comparison_report.md
```

Commit code, configurations, manifests, and the final summarized tables. Do not
commit datasets, entity caches, checkpoints, raw logs, or large per-query binary
artifacts.
