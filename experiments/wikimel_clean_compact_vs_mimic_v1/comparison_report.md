# WikiMEL clean compact-fp32 vs MIMIC: experiment status

Started: 2026-08-23

## Data gate

The primary `clean_sanitized_visual` variant passes the detailed acceptance
gate. It contains 18,092 train rows, 2,083 development rows, 4,000 test rows,
and 109,976 ordered KB entities.

The historical path-only deduplication reproduced the prior 18,092 / 2,083 /
4,002 split. Content-aware normalization found two additional train/test
context collisions, so the primary test set has 4,000 rows.

| Check | Result |
|---|---:|
| Cross-split normalized mention/context overlap | 0 |
| Cross-split normalized context-only overlap | 0 |
| Cross-split exact image-byte overlap | 0 |
| Mention/entity pHash candidates at distance <= 4 after sanitization | 0 |
| Gold entity-image leakage after sanitization | 0 |
| Mention filenames containing answer IDs | 0 |

Before sanitization, the audit found 5,279 mention/entity perceptual-match
pairs, including 4,951 gold pairs. Representative gold pairs at pHash distances
0, 2, and 4 were visually confirmed as the same images with encoding, crop, or
watermark differences. The global blacklist removed 5,273 unique entity image
files, leaving 68,988 listed entity images.

## Frozen evaluation

The evaluation manifest freezes 4,000 ordered test query IDs, their gold entity
IDs, and all 109,976 ordered KB entity IDs. Candidate policy is full KB, score
direction is higher-is-better, chunk size is 5,000, and checkpoint selection is
by development MRR only.

## Model launch status

The compact/pooled loader smoke test passed with 18,092 train rows, 2,083 dev
rows, 4,000 test rows, 109,976 entities, and BM25 hard negatives. The loader now
uses deterministic zero tensors plus explicit missing-image masks. The trainer
uses fp32 and does not evaluate test after fitting.

A one-epoch compact fp32 debugging run (seed 43) completed successfully on the
RTX 4090. Its raw stdout/stderr logs are intentionally ignored by Git. Test
evaluation was disabled for this debugging run.

| Development metric | Debug result |
|---|---:|
| H@1 | 0.835814 |
| H@3 | 0.917427 |
| H@5 | 0.942391 |
| H@10 | 0.964954 |
| H@20 | 0.981277 |
| MRR | 0.884069 |
| Mean rank | 11.341815 |
| Training loss (epoch) | 0.212174 |

The run stopped cleanly at `max_epochs=1` and wrote
`epoch=0-step=283.ckpt` (608,523,983 bytes; SHA-256
`d7a809965acfa0ea692523030220063efaf4ae2a4b94af34df6e8b5c0803ed13`).
These values prove pipeline operation only; they are not final three-seed or
test-set results.

The official MIMIC source tree was restored at
`external/MIMIC_reproduction` and pinned to upstream commit
`7dc5b733d9af92cb819f7ce45caffc4112b5f150`. The clone passed `git fsck` and
was clean after checkout. A local experiment adapter now adds clean-data
compatibility, repeat-safe collation, effective missing-image masking,
deterministic CUDA startup, and test isolation. Upstream recommends Python
3.8.12 with Torch 1.11.0/CUDA 11.3, while the active experiment environment is
newer, so compatibility must be smoke-tested and recorded.

Compatibility smoke tests under the active Python 3.10/Torch 2.0 environment
show that the released MIMIC model is numerically CUDA-compatible. Model
construction, the sanitized loader inventory (using an identical single-line
qid2id shim), first collation, a finite fp32 forward/backward pass, and a real
one-batch Lightning fit all passed. The direct forward/backward used batch size
2, loss 3.465746, 1,365.32 MB peak allocated VRAM, and 1,512.00 MB peak reserved
VRAM.

All five adapter/protocol fixes are implemented. The canonical multiline map
loads all 109,976 IDs; train, evaluation, and entity collators are repeat-safe;
mention/entity image masks propagate through training and cached evaluation;
missing visual terms are excluded and text-only pairs use a text-only
denominator; test evaluation is false-by-default; and full/debug sanitized FP32
configs use the frozen 5,000-candidate chunk policy. A patched one-batch FP32
CUDA fit at the configured batch size of 64 completed with finite loss 41.2 and
8,582.84 MB peak allocated VRAM.
Detailed evidence and the reusable check are saved in
`mimic_compatibility_results.json` and `verify_mimic_fixes.py`.

The paired one-epoch MIMIC FP32 debugging run (seed 43) subsequently completed
on the same RTX 4090. It trained all 283 batches, rebuilt the complete
109,976-entity evaluation cache, ranked all 2,083 development queries, and did
not evaluate the test set.

| Development metric | MIMIC debug result |
|---|---:|
| H@1 | 0.039846 |
| H@3 | 0.093135 |
| H@5 | 0.129141 |
| H@10 | 0.200672 |
| H@20 | 0.276524 |
| MRR | 0.093252 |
| Mean rank | 371.285166 |
| Training loss (epoch) | 2.160075 |

The run stopped cleanly at `max_epochs=1` and wrote
`epoch=0-step=283.ckpt` (607,055,910 bytes; SHA-256
`08e213758f05b556c15a41320fb7c91a35a89a1e6979449b236556e78c5f7c45`).
As with the compact debugging seed, these are pipeline-validation values, not
final three-seed or test-set results. MIMIC's much lower one-epoch development
score must not be interpreted as a final model comparison because its full
training schedule is 20 epochs.

## Artifact locations

- Generated datasets and the large image-hash cache: `data/wikimel_clean_compact_vs_mimic_v1/` (Git-ignored)
- Exact audit: `audit_exact.txt`
- Detailed audit: `audit_detailed.json`
- Frozen evaluator: `evaluation_manifest.json`
- Environment inventory: `environment.json`
- Compact configs: `fastmel/config/wikimel_clean_sanitized_visual_fp32*.yaml`
- MIMIC configs: `external/MIMIC_reproduction/config/wikimel_clean_sanitized_visual_fp32*.yaml`
- MIMIC fix verification: `verify_mimic_fixes.py`
- MIMIC debug result: `mimic_debug_seed43_results.json`
