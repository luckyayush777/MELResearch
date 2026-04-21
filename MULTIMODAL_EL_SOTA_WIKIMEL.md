# Multimodal EL SOTA On WikiMEL

Date: April 20, 2026

## Short Answer

Yes, you can plausibly get **close to strong WikiMEL accuracy with quantization**, but **not by quantizing the current OpenCLIP-style baseline you already ran**.

Quantization usually preserves an already strong model; it does not recover the very large gap between your current full-KB baseline and the published MEL leaders.

Your current official full-KB WikiMEL baseline in this workspace is:

- `R@1 / H@1`: `0.0970`
- `MRR`: `0.1398`

Saved at:

- `benchmarks/wikimel_official_test_v1/baseline_results.json`

By contrast, the strongest **comparable** published full-KB WikiMEL systems are around **`88-91 H@1`**.

## The Key Comparability Issue

Not all "WikiMEL SOTA" numbers mean the same thing.

There are at least three different settings in the literature:

1. **Common full-KB WikiMEL retrieval setting**
   The model ranks against a large WikiMEL candidate set from a Wikidata subset.
   This is the setting most relevant to your current benchmarking and quantization story.

2. **Candidate-limited WikiMEL**
   The model only chooses from `16` or `100` candidates.
   These numbers can be much higher, but the task is easier and not directly comparable to full-KB retrieval.

3. **Retrieve + rerank / LLM-assisted pipelines**
   These often improve accuracy further, but may use API LLMs or extra stages that are awkward to quantize.

## Best Comparable Full-KB WikiMEL Results

The cleanest common comparison table I found is in **KGMEL (SIGIR 2025)**, which reports WikiMEL results on a subset-of-Wikidata candidate KB with **`109,976` candidate entities** and gives `H@1/H@3/H@5/MRR`.

Source:

- KGMEL PDF: `https://arxiv.org/pdf/2504.15135`
- Relevant lines: Table 1 and Table 6, especially Table 6 in Appendix D

### Full-KB WikiMEL leaderboard from KGMEL's reported common setting

| Method | Year | WikiMEL H@1 | H@3 | H@5 | MRR | Notes |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| CLIP | 2021 | 83.23 | 92.10 | 94.51 | 88.23 | Strong generic pretrained baseline |
| MIMIC | 2023 | 87.98 | 95.07 | 96.37 | 91.82 | Strong retrieval-style MEL baseline |
| OT-MEL | 2024 | 88.97 | 95.63 | 96.96 | 92.59 | Better than MIMIC in this table |
| MELOV | 2024 | 88.91 | 95.61 | 96.58 | 92.32 | Very close to OT-MEL |
| M3EL | 2024 | 88.84 | 95.20 | 96.71 | 92.30 | Strong retrieval-only target for quantization |
| IIER | 2025 | 88.93 | 95.69 | 96.73 | not reported | Very strong, but MRR not shown in KGMEL table |
| KGMEL (retrieval only) | 2025 | 87.29 +/- 0.08 | 92.47 +/- 0.34 | 93.94 +/- 0.27 | 89.99 +/- 0.25 | Retrieval stage only |
| **KGMEL (+ rerank)** | **2025** | **90.58 +/- 0.25** | **95.18 +/- 0.29** | **95.87 +/- 0.24** | **93.04 +/- 0.26** | Best overall number I found in the common full-KB setting |

## What Looks Like The Current SOTA

If you care about the **best published comparable full-KB WikiMEL accuracy**, the strongest number I found is:

- **KGMEL (+ rerank), SIGIR 2025: `90.58 H@1`, `93.04 MRR`**

If you care about the **best retrieval-only style number** without an extra reranking stage, the strongest widely comparable targets in the same family are:

- **OT-MEL: `88.97 H@1`, `92.59 MRR`**
- **IIER: `88.93 H@1`**
- **MELOV: `88.91 H@1`, `92.32 MRR`**
- **M3EL: `88.84 H@1`, `92.30 MRR`**
- **MIMIC: `87.98 H@1`, `91.82 MRR`**

For a quantization paper, these retrieval-style models are much more realistic targets than a pipeline that depends on external LLM reranking.

## Important Non-Comparable WikiMEL Numbers

These are real papers and real numbers, but they are **not apples-to-apples** with your current full-KB benchmark.

### GEMEL

Source:

- GEMEL PDF: `https://arxiv.org/pdf/2306.12725`

Reported WikiMEL result:

- `82.6` Top-1 accuracy

Why it is not directly comparable:

- GEMEL reports Top-1 accuracy and uses a different experimental setup than the common full-KB retrieval tables.

### UniMEL

Sources:

- UniMEL arXiv abstract: `https://arxiv.org/abs/2407.16160`
- UniMEL HTML text: `https://ar5iv.org/html/2407.16160v2`

Reported WikiMEL results:

- `94.1` Top-1 accuracy from `100` candidate entities
- `94.2` Top-1 accuracy when aligned to a `16`-candidate comparison with GEMEL

Why it is not directly comparable:

- UniMEL explicitly reports **candidate-limited** results.
- In its table, WikiMEL is shown with `18,880` entities, not the `109,976` full-KB candidate setup used by KGMEL/MIMIC/M3EL.

### DWE+

Source:

- DWE+ PDF: `https://arxiv.org/pdf/2404.04818`

Reported WikiMEL results on the original WikiMEL setting used in that paper:

- `T@1 = 44.9`
- `T@5 = 67.2`
- `T@10 = 81.8`
- `T@20 = 93.8`

Why it is not directly comparable:

- DWE+ uses **`100` candidate entities** selected by fuzzy matching.
- It also reports a WikiMEL variant with around `17,391` entities after pruning, which is much smaller than the common `109,976`-candidate setting.

## Honest Answer To "Can Quantization Get Me To SOTA?"

### If you quantize the current local baseline

Probably **no**.

Your current baseline is roughly:

- `9.70 H@1`
- versus published comparable full-KB MEL systems at roughly `88-91 H@1`

Quantization will not close an ~`80`-point absolute accuracy gap.

### If you first reproduce a strong MEL model, then quantize it

Yes, **this is realistic**.

The right goal is:

- start from a strong unquantized MEL retrieval model around `88-89 H@1`
- quantize its encoders, embeddings, and ANN index
- retain as much of that accuracy as possible

That is a credible paper:

- "We preserve near-SOTA WikiMEL accuracy while reducing storage / memory substantially."

## Best Practical Direction For Your Paper

For a quantization paper, the most realistic target is **not** KGMEL's full retrieve+rerank pipeline, because KGMEL uses extra LLM-based stages and is harder to quantize cleanly end-to-end.

The best practical target is one of these:

1. **MIMIC**
   Strong full-KB retrieval baseline, clean encoder-style setup, good quantization target.

2. **M3EL**
   Slightly stronger than MIMIC on WikiMEL, also retrieval-oriented and quantization-friendly.

3. **OT-MEL / MELOV / IIER**
   Stronger numbers in KGMEL's comparison table, but reproduction difficulty may be higher depending on code availability.

4. **KGMEL retrieval stage only**
   Possible, but the full paper headline comes from reranking, and quantizing API-based rerank components is less convincing.

## What I Would Recommend You Do Next

If the advisor wants "accuracy close to SOTA, but more efficient", the strongest path is:

1. Reproduce **one strong full-KB retrieval model** on official WikiMEL.
2. Freeze that as the **unquantized accuracy anchor**.
3. Quantize:
   - encoder weights
   - stored entity embeddings
   - ANN / FAISS index
4. Report:
   - `H@1`, `H@3`, `H@5`, `MRR`
   - latency
   - memory
   - embedding storage
   - index size
5. Frame the result as:
   - "retain `>=99%` of strong WikiMEL retrieval accuracy at much lower memory/storage"

## Bottom Line

- **Current comparable full-KB WikiMEL SOTA I found:** `KGMEL (+ rerank)` with **`90.58 H@1`** and **`93.04 MRR`**
- **Best realistic quantization targets:** `MIMIC`, `M3EL`, or another strong retrieval-only model near `88-89 H@1`
- **Honest answer:** you can likely get **close to SOTA after quantization only if you quantize a strong MEL model first**, not by quantizing the current `9.7 H@1` OpenCLIP baseline

## Primary Sources

- KGMEL: `https://arxiv.org/pdf/2504.15135`
- MIMIC: `https://arxiv.org/pdf/2307.09721`
- M3EL: `https://arxiv.org/pdf/2412.10440`
- GEMEL: `https://arxiv.org/pdf/2306.12725`
- UniMEL: `https://arxiv.org/abs/2407.16160`
- UniMEL HTML: `https://ar5iv.org/html/2407.16160v2`
- DWE+: `https://arxiv.org/pdf/2404.04818`
