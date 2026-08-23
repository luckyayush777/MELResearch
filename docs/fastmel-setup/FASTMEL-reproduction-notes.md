# FAST-MEL Reproduction Notes

Source paper: *FAST-MEL: A Fast, Accurate, and Storage Efficient Solution for Multimodal Entity Linking* (SIGIR '26, arXiv:2606.11749v1)
Code: https://github.com/to2002td-cpu/FASTMEL.git
Notes compiled: 2026-08-08

---

## 1. Repo state (read before you start)

- Created **and** last pushed the same day (2026-02-11) — this is a one-shot dump of the authors' internal research directory, not an actively maintained release. 0 stars, 0 forks, 11 commits.
- The repo contains `.DS_Store`, `.ipynb_checkpoints`, and dozens of stray `__pycache__/*.pyc` files for models that aren't in the repo at all (colbert, siglip, blip, qformer, fissfuse, reranker, m3el variants). These are leftovers from a bigger private codebase — ignore them, they can't run.
- `.pyc` files are compiled for cpython-38 → the authors likely used **Python 3.8**.

## 2. What's in the repo

```
codes/main.py                    # entry point: python codes/main.py --config <path>
codes/model/lightning.py         # pl.LightningModule wrapper (Lightning class)
codes/model/model_pooling.py     # Encoder (CLIP text/vision) + MLPWeightedPooling
codes/utils/dataset.py           # DataModuleForMIMIC (image/text loading, collators)
codes/utils/functions.py         # setup_parser() — only defines --config arg
config/wikimel_mlp_HN-512-1.yaml
config/richpediamel_mlp_HN-512-1.yaml
config/wikidiverse_mlp_HN-512-1.yaml
data/WikiMEL/          # kb_entity.json + train/dev/test w/ 100 BM25 candidates
data/RichpediaMEL/     # same structure
data/WikiDiverse/      # same structure
requirements.txt
README.md              # 3 lines total
```

**README, verbatim gist:**
1. Download KB/query images from the MIMIC repo: https://github.com/pengfei-luo/MIMIC
2. `pip install -r requirements.txt`
3. `python codes/main.py --config "path_to_config_file"`

**requirements.txt** (pinned, old):
```
torch==1.11.0+cu113
transformers==4.18.0
torchmetrics==0.11.0
tokenizers==0.12.1
pytorch-lightning==1.7.7
omegaconf==2.2.3
pillow==9.3.0
```

## 3. What's genuinely usable

- `model_pooling.py` implements the real FAST-MEL architecture: `MLPWeightedPooling` (two-layer MLP + softmax over token/patch features — matches Eq. 5–6 in the paper) and an `Encoder` wrapping CLIP's text/vision towers (`AutoModel.from_pretrained`, e.g. `openai/clip-vit-base-patch32`).
- The three config YAMLs match the paper's reported hyperparameters: CLIP ViT-B/32 backbone, lr 1e-5, batch size 64, embedding dim 512, MLP pooling, 30 epochs, 8 dataloader workers, eval batch 20, 40 max text tokens, one hard negative per instance.
- Text-side data (entity descriptions, BM25-precomputed 100-candidate lists per query, train/dev/test splits) is already included per dataset — you don't need to regenerate BM25 candidates yourself.
- `DataModuleForMIMIC` in `dataset.py` is a real, structured Lightning DataModule with separate collators for train / eval / entity-embedding-update passes, and has a fallback (random tensor) for missing images rather than crashing.

## 4. Known problems / risks

- **Images are not included.** They must be downloaded separately from the MIMIC repo and placed into whatever `kb_img_folder` / `mention_img_folder` paths the configs expect. Exact naming/layout isn't documented — you'll need to infer it from `dataset.py` (mention images get renamed to `<basename>.jpg`).
- **Hard-negative mining mismatch with the paper.** The paper says hard negatives are the top-10 BM25 candidates, resampled each step. The actual training collator in `dataset.py` does `random.sample(candidates, k=hard_neg_samples)` over the full precomputed 100-candidate list — not restricted to top-10, and not re-mined per step. Minor, but don't be surprised if your numbers don't match exactly.
- **Dead/broken pooling branches.** `Encoder.forward()` in `model_pooling.py` has code paths for `'mean-attention'`, `'cross-attention'`, `'cross-attention-mlp'`, `'learned-attention'` that reference classes (`AttentionMeanPooling`, `CrossAttentionPooling`, etc.) that are **never defined or imported anywhere in the repo**. The shipped configs select MLP pooling, so the main FAST-MEL result path should avoid this — but the Table 4 ablations (CLS-Mean, CLS-MLP, Mean Pooling, Max Pooling) are **not reproducible as-is**; you'd have to implement those pooling variants yourself.
- **Possible state bug in `lightning.py`.** `_compute_epoch_metrics()` sets `self.entity_embeds = None`, but other code paths later assume it's populated (tuple-or-tensor handling also looks inconsistent depending on `args.model.reduction`). Watch for `AttributeError`/`NoneType` failures around epoch-end evaluation.
- **Old pinned dependencies.** `torch==1.11.0+cu113` needs a CUDA 11.3-compatible driver/toolkit, which may not install cleanly on a modern CUDA 12.x machine. You'll likely want a dedicated Python 3.8 venv/conda env for this.
- No pretrained checkpoints, no license file, no CI — nothing has been validated by anyone outside the authors.

## 5. Suggested plan of attack (tomorrow)

1. **Environment first.** Create a Python 3.8 virtualenv/conda env. Try installing `requirements.txt` as-is; if the `+cu113` wheel fails on your machine, install a `torch` build matching your actual CUDA version instead (same major torch version, 1.11.x) and only fall back further if that breaks pytorch-lightning 1.7.7 compatibility.
2. **Get the images.** Go to https://github.com/pengfei-luo/MIMIC, follow its own image-download instructions, and inspect `codes/utils/dataset.py` to determine the exact folder structure/naming (`kb_img_folder`, `mention_img_folder`) before copying files in. Update the three config YAMLs with real local paths.
3. **Dry run on the smallest dataset first** (RichpediaMEL or WikiMEL, whichever has fewer entities) with a tiny `max_epochs` override to confirm the pipeline runs end-to-end before committing to a full 30-epoch run.
4. **Watch specifically for:**
   - Import/`NameError` from the undefined pooling classes (shouldn't trigger with MLP pooling, but confirm).
   - `entity_embeds` `None`/type errors at epoch-end evaluation in `lightning.py`.
   - Image path/filename mismatches (`dataset.py` expects `.jpg` after a `split('/')[-1].split('.')[0]` rename).
5. Once a smoke run works, launch full training per dataset (paper reports 20–30 min/run on a single A100 40GB) and compare H@1/MRR against Table 2 of the paper (target ~86.08 avg H@1).

## 6. Time estimate

| Step | Estimate |
|---|---|
| Env setup (Python 3.8, pinned/adjusted CUDA torch build) | 30–90 min |
| Locate, download, and lay out MIMIC images correctly | 1–3 hours |
| Debug pipeline to first successful smoke run | 30–60 min |
| Full training run per dataset (compute only) | 20–30 min each |

**Total: ~4–6 hours if starting from scratch; ~1.5–2.5 hours if you already have the WikiMEL/RichpediaMEL/WikiDiverse images and a compatible env from prior MEL work.**

## 7. Bottom line

The core FAST-MEL model is genuinely implemented and the configs match the paper, so reproducing the headline numbers is plausible with real but bounded effort. This is not a polished, turnkey release — expect to do some environment wrangling, image-path plumbing, and possibly a small code fix, not just `pip install && run`.
