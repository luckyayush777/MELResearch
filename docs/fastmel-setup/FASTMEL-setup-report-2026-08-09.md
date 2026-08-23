# FAST-MEL Reproduction — Setup Report

**Date: 2026-08-09**

Follow-up to `FASTMEL-reproduction-notes.md` (compiled 2026-08-08). That document
was a read-only survey of the upstream repository. This one records what happened
when the setup was actually built and executed.

Source paper: *FAST-MEL: A Fast, Accurate, and Storage Efficient Solution for
Multimodal Entity Linking* (SIGIR '26, arXiv:2606.11749v1)
Code: https://github.com/to2002td-cpu/FASTMEL.git
Machine: Windows 10, RTX 4090 (24 GB, sm_89), driver 566.36, Python 3.10.6

Operational detail: `SETUP.md` in this directory is the how-to-run companion to
this report.

**Updated 2026-08-12.** The smoke test has since been re-run to completion on the
fixed source and passes end to end, including the test phase that previously died.
§5, §6 and §8 carry the new numbers; §6's runtime figures are revised downward.
Everything else stands as written on 08-09.

---

## 1. Bottom line

The environment, data, and pipeline are built and verified. Training runs and the
loss decreases on real data.

The headline finding is that **the published repository could not have produced
the paper's numbers in its released state.** Two independent defects sit directly
on the mandatory execution path:

* `codes/model/lightning.py` contained a **`SyntaxError`**, so the module could
  not be imported at all — `python codes/main.py` failed before doing anything.
* All three data collators mutate the shared dataset objects, so the code
  **cannot complete more than one epoch**. The shipped configs request 30.

Neither is a subtle numerical discrepancy; neither run completes without editing
the source. This should be weighed before investing in full training runs or
treating Table 2 as independently reproducible.

Both are fixed here, and both fixes are now verified by execution — the smoke test
runs train, validation and test end to end (§5.2).

A third defect, §2.7, blocks `num_workers > 0` but is **Windows-specific**; a Linux
reproduction using fork would not encounter it.

---

## 2. Blockers found

Seven issues had to be resolved before the pipeline would run. Only one (the
`entity_embeds` state concern) was anticipated by the 2026-08-08 notes.

| # | Issue | Severity | Found |
|---|---|---|---|
| 1 | `lightning.py:30` — incomplete ternary, a `SyntaxError` | Fatal at import | 08-09 |
| 2 | `lightning.py:10` — imports nonexistent `model_late_int` | Fatal at import | 08-09 |
| 3 | `qid2id.json` absent from repo, required by every config | Fatal at startup | 08-09 |
| 4 | `python codes/main.py` fails as documented (`PYTHONPATH`) | Fatal at startup | 08-09 |
| 5 | Collators mutate shared dataset objects | Fatal after 1 epoch | 08-09 |
| 6 | `num_entity` wrong for RichpediaMEL (160933 vs 160935) | Latent correctness risk | 08-09 |
| 7 | Bound-method `collate_fn` unpicklable under Windows spawn | Fatal with `num_workers > 0` | 08-12 |

Note the pattern: issues 5 and 7 are both invisible to static review and invisible
to a minimal smoke run. Each needed a *different* execution configuration to
surface — 5 required reaching a second pass over a dataloader, 7 required
`num_workers > 0` inside a real `Trainer.fit`.

### 2.1 `lightning.py:30` was a SyntaxError

Shipped as:

```python
self.score_function = cosine_similarity if args.score_function == "cosine"
```

A conditional expression with no `else`. Python cannot parse this, so
`codes.model.lightning` was unimportable and `main.py` could never start.
Replaced with an explicit `if/else` that raises on any non-`cosine` value —
`cosine_similarity` is the only scoring function defined anywhere in the repo.

### 2.2 `lightning.py:10` imported a module that does not exist

`from codes.model.model_late_int import LateEncoder`. `model_late_int.py` is not
in the repository — another leftover reference to the authors' private codebase,
consistent with the stray `__pycache__` entries noted on 2026-08-08. It is only
needed when `model.pooling_type` is empty (the late-interaction variant), which
no shipped config selects. Made lazy, so it fails only if that path is requested.

### 2.3 `qid2id.json` was missing entirely

Every config sets `data.qid2id: data/<Dataset>/qid2id.json`, and
`DataModuleForMIMIC.__init__` opens it before anything else — but the file is not
in the repository. This was missed by the initial survey.

It is fully derivable: `kb_entity.json` carries both `qid` (the Wikidata id the
mention files reference in `answer`/`cands`) and `id` (the dense row index).
`mimic-image-downloader/generate_qid2id.py` regenerates it. All three datasets map
onto a clean contiguous range:

| Dataset | Entities | id range | Duplicate qids |
|---|---|---|---|
| WikiMEL | 109,976 | 0..109,975 | 0 |
| RichpediaMEL | 160,935 | 0..160,934 | 0 |
| WikiDiverse | 132,460 | 0..132,459 | 0 |

The downloaded archives later turned out to ship an official `qid2id.json`. The
generated WikiMEL mapping was compared against it and is **identical**, which
independently validates the derivation.

Note `dataset.py` reads this with `json.loads(f.readline())`, so the file must be
a single line.

### 2.4 The documented run command does not work

The README's `python codes/main.py --config <path>` fails with
`ModuleNotFoundError: No module named 'codes'`. Running a script places *that
script's* directory (`codes/`) on `sys.path`, not the repo root, so
`from codes.utils.functions import ...` cannot resolve. Upstream MIMIC handled
this in `run.sh`; this repo dropped the shell script without updating the
instructions. `PYTHONPATH` must point at the repo root.

### 2.5 Collators mutate shared state — the repo cannot run 30 epochs

Found by the smoke test, and the most consequential finding.

`train_collator`, `eval_collator`, and `entity_collator` each do
`sample.pop('img_list')` (and `'sample_type'`, `'answer'`, `'candidates'`)
directly on the **shared, preprocessed** objects held in `self.train_data`,
`self.val_data`, `self.test_data`, and `self.kb_entity`.

`pop` is destructive. The first pass strips those keys permanently; the second
pass raises `KeyError: 'img_list'`. This breaks:

* epoch 2 of training (`self.train_data`)
* the second validation (`self.val_data`)
* the test phase (`self.kb_entity`, re-iterated by `_update_entity_embeds()`)

The author was aware of the hazard — lines 139 and 171 use `copy.deepcopy`
precisely to avoid it for the entity lookups — but the three collator entry
points do not.

Observed directly: the smoke run trained fine, completed a full validation pass
over all 109,976 entities, then died entering test:

```
File "codes/utils/dataset.py", line 243, in entity_collator
    img_list.append(sample.pop('img_list'))
KeyError: 'img_list'
```

Fixed by copying each sample before popping (`samples = [dict(s) for s in samples]`)
in all three collators. Verified with a direct regression test: two consecutive
passes over **all four** dataloaders (entity / val / test / train) now succeed.

### 2.6 `num_entity` wrong for RichpediaMEL

Shipped config says `160933`; the KB holds `160935`. It only worked by accident —
`_step()` computes `ceil(num_entity / eval_chunk_size)`, and with
`eval_chunk_size: 6000` the rounding happened to cover the tail. A different chunk
size would have silently truncated the candidate pool. Corrected in the local config.

### 2.7 Bound-method `collate_fn` cannot be pickled under Windows spawn

Found 2026-08-12, on the first attempt at a full run. It killed the run ~2 minutes
in, at the moment Lightning created the training dataloader's iterator:

```
AttributeError: Can't pickle local object 'FitLoop.advance.<locals>.batch_to_device'
```

All five `DataLoader`s in `dataset.py` (lines 270–302) pass a **bound method** as
`collate_fn` — `self.train_collator`, `self.eval_collator`, `self.entity_collator`.
On Windows, `num_workers > 0` means spawn, so every worker must pickle that bound
method, which pickles `self` — the entire `DataModuleForMIMIC`. By that point
Lightning has set `self.trainer`, so the pickle walks:

```
DataModuleForMIMIC.trainer -> Trainer.fit_loop -> _data_fetcher -> batch_to_device
```

and `batch_to_device` is a closure defined *inside* `FitLoop.advance`. Local
functions are not picklable. This is deterministic, not a race — it fails every
time, and it is why §5's `num_workers: 4` claim is retracted.

Fixed with a `__getstate__` on the DataModule that drops the `trainer`
back-reference before pickling. The collators only touch `self.tokenizer`,
`self.image_processor` and `self.args`, so the workers lose nothing:

```python
def __getstate__(self):
    state = self.__dict__.copy()
    state.pop('trainer', None)
    state.pop('_trainer', None)
    return state
```

Verified: with the patch, `num_workers: 4` spawns and trains on WikiMEL.

**This is also a significant speedup, not just an unblock.** `UpdateEmbed` runs at
~1.5 it/s with 4 workers against 0.37–0.41 it/s single-threaded — **~4x** — cutting
the eval phase from ~9.5 min to ~2.4 min. See §6.

Anyone reproducing this on Linux will not hit the bug at all (fork, not spawn), and
should be aware their `num_workers` baseline therefore differs from the figures here.

### Diff footprint

All source changes are confined to two files (as of 2026-08-12, blockers 1–7):

```
codes/model/lightning.py | 36 +++++++++++++++++++++++++++++++-----
codes/utils/dataset.py   | 28 ++++++++++++++++++++++++++++
2 files changed, 59 insertions(+), 5 deletions(-)
```

`git diff --stat` additionally lists two `codes/utils/__pycache__/*.pyc` files as
modified. Those are the authors' stray committed bytecode (see the 08-08 notes) and
were simply rewritten by running the code; they are not edits.

---

## 3. Environment

The repo's pinned `requirements.txt` is unusable on this machine, and not as a
matter of preference:

> **`torch==1.11.0+cu113` has no sm_89 kernels.** An RTX 4090 is sm_89. That
> wheel cannot run on this GPU under any Python version. CUDA 11.8 is the
> earliest toolkit with Ada Lovelace support.

Python 3.8 (which the `cpython-38` `.pyc` leftovers imply the authors used) would
not have helped — the CUDA constraint, not the Python version, forces the change.

| Pinned | Installed | Reason |
|---|---|---|
| `torch==1.11.0+cu113` | `torch==2.0.1+cu118` | sm_89 support |
| `pytorch-lightning==1.7.7` | `==1.9.5` | last 1.x line; validated against torch 2.0; retains `validation_epoch_end`/`test_epoch_end` and `Trainer(amp_backend=...)`, all **removed** in PL 2.0 |
| `transformers==4.18.0` | `==4.30.2` | 4.18 will not load on modern torch; 4.30.2 still exposes `CLIPProcessor.feature_extractor`, which `dataset.py` uses |
| — | `numpy==1.24.4` | torch 2.0.1 built against the numpy 1.x ABI |
| — | `setuptools<81` | setuptools 81 removed `pkg_resources`, which `lightning_fabric` imports at load time |

Verified working: `torch.cuda.is_available()` true, device reports **sm_89**, and
a real fp16 matmul executes on the GPU. The wheel ships no sm_89 cubin, but CUDA
guarantees binary compatibility within a major compute-capability generation, so
the sm_86 kernels run natively on Ada.

Model construction verified independently: 152.1M parameters, forward pass
returns the expected `(2, 512)` embedding.

---

## 4. Data acquisition

Images are **not** in the repo. They come from the MIMIC authors' three
password-protected OneDrive/SharePoint links (password `kdd2023`).

Two facts the survey could not have known, both of which changed the tooling:

* the archives are **`.tar`, not `.zip`**
* each tar contains only ~8–9 members — the images are bundled as
  **nested zips** (`kb_image.zip`, `mention_image.zip`) inside

| Archive | Size |
|---|---|
| `WikiMEL.tar` | 2.96 GiB |
| `RichpediaMEL.tar` | 2.63 GiB |
| `WikiDiverse.tar` | 1.80 GiB |
| **Total** | **~7.4 GiB** |

### Checkpointable downloader

`mimic-image-downloader/download_mimic_images.py` — stdlib only, so it runs
without the venv. Every stage is independently checkpointed to
`mimic-dataset-archives/download-checkpoint.json` (atomic writes, fsync'd);
interrupting at any point and re-running the same command resumes.

Stages: **download → verify → extract → unpack → arrange**

It clears the SharePoint ASP.NET password gate automatically and auto-retries
with re-authentication, because a multi-GB transfer routinely outlives the
`FedAuth` cookie.

Both risky paths were tested rather than assumed:

* **Range resume is byte-exact** — a file reconstructed from a seeded prefix plus
  a resumed tail produced a sha256 identical to a straight download. The server
  answers `206 Partial Content`, so resume is real.
* **Extraction resume repairs damage** — a truncated file was rebuilt, a deleted
  file restored, an intact file skipped, and a path-traversal member refused.

A `--manual` mode processes archives downloaded by hand, sharing the same
checkpoint, so the two paths are interchangeable mid-run.

### Result

| Dataset | `kb_image` | mention images |
|---|---|---|
| WikiMEL | 106,668 | 24,599 (`mention_image`) |
| RichpediaMEL | 96,073 | 15,852 (`mention_images`, plural) |
| WikiDiverse | 67,292 | 6,696 (`mention_image`) |

### Coverage verification

`choose_image()` wraps every image load in a bare `except:` and substitutes
`torch.rand((3,224,224))`. A wrong path therefore produces a **quietly bad run
rather than a crash** — nothing warns you. This is the single most dangerous
property of the codebase for a reproduction attempt.

`mimic-image-downloader/verify_image_coverage.py` guards against it, resolving
filenames with the exact rules the loader uses (including the
`split('/')[-1].split('.')[0] + '.jpg'` rename applied to mention images):

| Dataset | KB checked | KB missing | Mention checked | Mention missing |
|---|---|---|---|---|
| WikiMEL | 67,195 | 0 | 25,846 | 1 |
| RichpediaMEL | 86,769 | 0 | 15,852 | 0 |
| WikiDiverse | 58,733 | 2 | 14,988 | 0 |

**3 misses in ~270,000 lookups (0.001%).** Immaterial.

---

## 5. Smoke test results

Config both times: 1 epoch, 8 train batches, batch size 16, `num_workers: 0`
(`local-configs/wikimel_smoke.yaml`).

### 5.1 First run — 2026-08-09, before the collator fix

**What worked:** training ran on real data with loss falling monotonically
2.51 → 0.592 over 8 steps, followed by a complete entity-embedding pass over all
109,976 entities and a successful validation phase.

**What failed:** the test phase, via the collator mutation bug in §2.5.

Log: `setup-logs/smoke-wikimel.log`.

Also tested at the time: `num_workers: 4` under Windows spawn, despite `collate_fn`
being a bound method of the DataModule (which forces each worker to pickle the
preprocessed KB). No pickling failure was observed.

> **Retracted 2026-08-12.** That claim is wrong. The check was run outside
> `Trainer.fit`, before Lightning attaches itself to the DataModule. Inside a real
> fit loop, `num_workers > 0` fails every time on Windows. See §2.7.

### 5.2 Re-run — 2026-08-12, after the collator fix

Completed end to end. `Trainer.fit` stopped cleanly at `max_epochs=1`, and the
**test phase ran its second full 109,976-entity `UpdateEmbed` pass through
`entity_collator` without raising** — the exact call that produced
`KeyError: 'img_list'` on 08-09. No traceback anywhere in the log.

This confirms the §2.5 fix under real execution rather than only via the two-pass
dataloader regression test.

| | |
|---|---|
| `Val/mrr` | 0.686 |
| `Test/hits1` | 0.675 |
| `Test/hits3` | 0.725 |
| `Test/hits5` | 0.750 |
| `Test/hits10` | 0.775 |
| `Test/hits20` | 0.825 |
| `Test/mrr` | 0.711 |
| `Test/mr` | 1345.3 |

**These are a liveness signal, not a result.** The model saw 8 training steps, and
`limit_test_batches: 4` at `eval_batch_size: 20` scores only 80 mentions. `Test/mr`
of 1345 is what an almost-untrained encoder should produce. Nothing here is
comparable to Table 2.

Log: `setup-logs/smoke-wikimel-postfix.log`.

---

## 6. Runtime: budget more than the paper suggests

The paper reports 20–30 min per full run on an A100 40GB. The shipped code does
not behave that way here, and the gap is structural rather than a GPU difference.

`on_validation_start()` calls `_update_entity_embeds()`, which re-embeds the
**entire** KB — every epoch, for validation *and* test. For WikiMEL that is 215
batches of 512. `limit_val_batches` does not shrink it: capping validation to 4
batches still pays the full embedding cost first.

**This recomputation is necessary, not wasteful.** There is a single shared
`self.encoder` for both mentions and entities, nothing is frozen
(`configure_optimizers` puts every `named_parameter()` into AdamW, CLIP backbone
included), so each training epoch invalidates every stored entity embedding.
Caching across epochs would score fresh mention embeddings against stale entity
ones.

Measured (smoke config, `num_workers: 0`), across both smoke runs:

| Run | `num_workers` | `UpdateEmbed` rate | Cost per eval phase |
|---|---|---|---|
| 08-09 smoke (cold cache) | 0 | ~4.0–4.9 s/it × 215 | ~16 min |
| 08-12 smoke (warm cache) | 0 | 2.42–2.70 s/it × 215 | ~9.5 min |
| 08-12 full run | 4 | ~1.5 **it/s** × 215 | **~2.4 min** |

Training itself is negligible in the smoke configs (~1.6 it/s @ batch 16).

The third row is the one that matters, and it only became reachable after the §2.7
pickling fix. Four workers give **~4x** over single-threaded, which is consistent
with the bottleneck being the CPU-side collate path rather than the GPU.

The 08-12 run measured 9:41 for the validation pass and 9:19 for the test pass —
same machine, same config, same `num_workers: 0`, a **~40% drop** from 08-09.

The likely explanation is the OS page cache: the 08-09 run read ~2.4 GiB of
`kb_image` off disk cold, and those files were still resident on 08-12. **This has
not been verified** — treat 9.5 min as the warm-cache figure and ~16 min as the
cold-cache one, and expect the first eval phase after a reboot to be the slow kind.

Cost breakdown, from a 300-image sample of `WikiMEL/kb_image` taken on 08-09:

* images are **already pre-resized to 384×384, ~23 KB** by the MIMIC authors
* decode + LANCZOS resize costs 3.8 ms each (263 img/s single-threaded)
  → ~7 min per 110k pass
* against the 16-min cold figure that is ~44%, the remainder being CLIP
  normalization, `tokenizer.pad`, tensor stacking, and the ViT-B/32 forward.
  Note this sample was itself measured cold, so it does not decompose the 9.5-min
  warm figure — 7 min of decode inside a 9.5-min pass would leave almost nothing
  for the GPU forward, which is not credible. The warm decode cost is simply lower
  and was never re-measured.

Since nearly all of that sits inside `collate_fn`, `num_workers` is the lever —
hence `num_workers: 4` in the full configs. **Pre-resizing images on disk would
not help**; they are already small. Caching embeddings across epochs would be
incorrect.

Practical consequence: with validation every epoch, a 30-epoch run spends far
more wall-clock in `UpdateEmbed` than in training. `EarlyStopping(patience=3)` on
`Val/mrr` will usually stop well before epoch 30. To reduce further, raise
`num_workers` or set `check_val_every_n_epoch: 2`.

---

## 6a. First full run — WikiMEL, 2026-08-12

Config: `local-configs/wikimel_local.yaml`, unchanged paper hyperparameters
(CLIP ViT-B/32, lr 1e-5, batch 64, embed_dim 512, 1 hard negative), `num_workers: 4`,
run detached. Logs: `setup-logs/train-wikimel-full.{out,err}.log`.

### Course of the run

| Epoch | `Val/mrr` | Note |
|---|---|---|
| 0 | 0.907 | |
| 1 | 0.921 | +0.014 |
| 2 | 0.923 | +0.002 |
| 3 | **0.927** | best — checkpointed |
| 4–6 | no improvement | patience 1, 2, 3 |

`EarlyStopping(monitor='Val/mrr', patience=3)` stopped training after epoch 6 of a
configured 30. `main.py` calls `trainer.test(..., ckpt_path='best')`, so the test
phase ran against `epoch=3-step=1132.ckpt` (580 MB), not the final epoch-6 weights.

**Wall clock: 14:27:41 → 15:09:56, ~42 min total**, at ~5–6 min/epoch. For contrast,
the same run at `num_workers: 0` would have been roughly 8 hours; see §2.7.

### Test results

| Metric | WikiMEL |
|---|---|
| **H@1** | **88.95** |
| H@3 | 95.65 |
| H@5 | 97.16 |
| H@10 | 98.12 |
| H@20 | 98.90 |
| **MRR** | **92.56** |
| MR | 33.10 |

### What this does and does not establish

**Does:** the repaired pipeline trains to convergence on real data and produces a
coherent, strong result. Every blocker in §2 is now cleared by execution rather
than by inspection.

**Does not:** confirm reproduction of Table 2. The 86.08 figure carried in the
08-08 notes is the paper's **average H@1 across all three datasets**, not its
WikiMEL number, so 88.95 > 86.08 proves nothing on its own — WikiMEL is the
easiest of the three. A real comparison needs Table 2's per-dataset WikiMEL
figure, and an average needs RichpediaMEL and WikiDiverse run too.

Also unchanged: the hard-negative sampling still differs from the paper (§7), so
even a matching number would not mean an identical training procedure.

---

## 7. Revised assessment

The 2026-08-08 estimate was ~4–6 hours from scratch. Setup to a verified,
runnable state took roughly that, but the distribution differed:

| Step | Estimated 08-08 | Actual |
|---|---|---|
| Env setup | 30–90 min | ~45 min (plus an unforeseen `pkg_resources` pin) |
| Locate/download/lay out images | 1–3 hours | ~1 hour, fully automated |
| Debug to first successful smoke run | 30–60 min | ~1.5 hours — more defects than expected |
| Full training run | 20–30 min each | ~42 min (WikiMEL, early-stopped at epoch 6) |

The image step went faster than feared (the download is scripted and resumable);
the debugging step went slower, because the repo has more hard blockers than a
read-through revealed. Static review found 1 of the 6 issues; the other 5 only
surfaced on execution.

Still standing from the original notes, unchanged:

* hard-negative mining differs from the paper (collator samples the precomputed
  100-candidate list rather than re-mining top-10 BM25 per step)
* Table 4 ablations are **not** reproducible as-is — the `mean-attention`,
  `cross-attention`, `cross-attention-mlp`, and `learned-attention` branches
  reference classes defined nowhere in the repo

---

## 8. State and next steps

**Ready:**

* venv verified on the 4090, CLIP weights cached
* all three datasets downloaded, unpacked, coverage-verified
* four local configs written with absolute paths (3 full + 1 smoke)
* six blockers fixed; diff confined to two source files
* smoke test re-run to completion on the fixed source (08-12) — train, validation
  **and test** all pass; see §5.2
* **one full WikiMEL run completed** (08-12): H@1 88.95, MRR 92.56, ~42 min; see §6a

**Not done:**

* **no comparison against Table 2 has actually been made.** The only paper figure on
  hand is the 86.08 three-dataset H@1 average, which the WikiMEL result cannot be
  checked against. Someone needs to read the per-dataset numbers out of the paper.
* RichpediaMEL and WikiDiverse have never been run at all, not even a smoke test.
  Only the WikiMEL path is execution-verified, and no average is computable from
  one dataset.
* `num_workers` left at 4 rather than tuned — 4 is known to work and to be ~4x
  faster than 0, but 6 or 8 were never tried against RAM headroom
* the run early-stopped at epoch 6 of 30 with patience 3 and `min_delta 0.0`.
  Whether a longer schedule (larger patience, or `min_delta` tuning) would keep
  improving past `Val/mrr` 0.927 is untested.

**Suggested order:**

1. ~~Re-run the smoke test end to end~~ — done 08-12, §5.2.
2. ~~Get a real per-epoch cost~~ — done: ~5–6 min/epoch, ~42 min/run. 30 epochs is
   comfortably affordable; `check_val_every_n_epoch: 2` is not needed.
3. Pull WikiMEL's per-dataset H@1/MRR out of Table 2 and compare against §6a. This
   is the cheapest remaining step and it gates every conclusion below.
4. Run RichpediaMEL and WikiDiverse (~45 min each, configs already written), then
   compare the three-dataset average against 86.08.

Given §1, treat any divergence from the paper's numbers as a live possibility
rather than a setup error.
