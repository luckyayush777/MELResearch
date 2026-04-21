# MEL Reproduction Targets For Quantization

Date: April 21, 2026

## Recommendation

If the goal is:

- get a **strong raw WikiMEL number first**
- keep the setup **quantization-friendly**
- stay as close as possible to **published full-KB WikiMEL results**

then the best first target is:

## 1. MIMIC

Why MIMIC is the best first target:

- It has an **official public GitHub repo**.
- It reports a **strong full-KB WikiMEL result** in the common comparison setting:
  - `H@1 = 87.98`
  - `MRR = 91.82`
- It is a **retrieval-style model**, which is much easier to quantize cleanly than a multi-stage LLM reranking system.
- Its repo has a **simple training entrypoint**:
  - `bash run.sh <gpu_id> <dataset_name>`
- It does **not** require OpenAI API keys or a complicated retrieve-rerank-generation stack.
- Newer repos explicitly build on it:
  - KGMEL says its dataset is based on the **MIMIC repository**
  - M3EL says it **refers to the code of MIMIC**

Why this matters:

- If we can reproduce MIMIC reasonably well, we immediately get a **credible strong anchor** for a quantization paper.
- After that, quantizing encoder weights, stored entity embeddings, and the ANN index becomes a much cleaner story.

## 2. M3EL

Why M3EL is the second-best target:

- It has an **official public GitHub repo**.
- It reports a slightly stronger WikiMEL result than MIMIC:
  - around `88.84 H@1` in the common table
- It is still a **retrieval-oriented model**, so it is quantization-friendly.

Why it is riskier than MIMIC:

- The repo explicitly depends on **MIMIC data/layout first**, then asks you to replace datasets with the authors' M3EL versions.
- It warns that **Transformers version affects performance**.
- It is effectively a more layered reproduction path: first MIMIC-compatible data, then M3EL-specific setup.

Bottom line:

- M3EL may give the stronger raw number.
- But it has more moving parts, so it is a worse *first* reproduction target.

## 3. DWE

Why DWE is less attractive as the first target:

- It has an **official public GitHub repo**.
- It reports strong published results.

But:

- Its repo relies on **checkpoint/preprocessed data** plus separate image downloads.
- The image download path includes **Baidu Pan**, which is often painful to reproduce from.
- More importantly, its WikiMEL evaluation setup is not the clean common full-KB setting you want to align with your current benchmark story.

Bottom line:

- Reproducible enough for experimentation.
- Not the best anchor for a "close to full-KB WikiMEL SOTA with quantization" paper.

## 4. KGMEL

Why KGMEL is not the best first target:

- It has an **official public GitHub repo** and the best published comparable number when reranking is included.
- But the repo requires:
  - `OPENAI_API_KEY`
  - `HUGGINGFACE_TOKEN`
  - `WANDB_API_KEY`
- The method includes:
  - triple generation
  - retrieval
  - reranking

This is much harder to turn into a clean quantization paper because:

- API-dependent stages are awkward to quantify and reproduce consistently.
- The strongest result comes from a **multi-stage pipeline**, not a simple retrieval encoder.

Bottom line:

- Great as a ceiling.
- Bad as the first thing to reproduce for quantization.

## Ranked Choice

1. **MIMIC**: best balance of strength, code availability, and low reproduction risk
2. **M3EL**: stronger but more fragile and layered
3. **DWE**: official code exists, but setup/comparability are weaker for your paper
4. **KGMEL**: strongest paper number, highest reproduction complexity, worst first quantization target

## Practical Paper Path

The strongest path now is:

1. Reproduce **MIMIC** on WikiMEL.
2. Freeze the unquantized result as the raw anchor.
3. Quantize:
   - model weights
   - stored entity embeddings
   - FAISS / retrieval index
4. If time allows, also reproduce **M3EL** as a stronger follow-up anchor.

## Concrete Reproduction Order

If the goal is to raise the baseline before quantization, use this order:

1. **MIMIC** first.
  - Best chance of a clean reproduction.
  - Strong enough to replace the current weak baseline with a credible number.
  - Use this as the paper-grade anchor for all later quantization runs.

2. **M3EL** second.
  - Slightly stronger than MIMIC in the common comparison table.
  - Good follow-up if MIMIC lands cleanly and you want more headroom.

3. **OT-MEL** or **MELOV** third.
  - These are strong retrieval-only comparators in the same family.
  - Useful if you want a second strong point without moving to a multi-stage pipeline.

4. **KGMEL** last.
  - Treat this as the ceiling, not the starting point.
  - Best if you want to compare against the strongest published number, but it is not the cleanest quantization baseline.

## What To Record For Each Reproduction

Keep the same evaluation protocol for every paper target:

1. `H@1`, `H@3`, `H@5`, and `MRR`.
2. Candidate KB size and whether the run is retrieval-only or retrieval + rerank.
3. Model size, embedding storage size, and ANN index size.
4. Latency, throughput, and peak memory on the same hardware.
5. Exact repo commit, dependency versions, and any dataset preprocessing steps.

## Baseline Rule

Do not quantize the current weak OpenCLIP-style baseline and expect it to approach the published MEL numbers.
Quantization is the follow-up experiment after you have a strong unquantized retrieval baseline, not a replacement for one.

## MIMIC Reproduction Steps

Use the official MIMIC repository workflow:

1. Create the environment.
  - Use Conda.
  - The repo recommends Python `3.8.12`.
  - Install the package versions from `requirements.txt` in the MIMIC repo.

2. Download the data.
  - WikiMEL and RichpediaMEL are available from MELBench.
  - WikiDiverse is available from the WikiDiverse repository.
  - The MIMIC README also links to cleaned copies of the datasets.

3. Update the data paths in the config.
  - Edit the YAML file for WikiMEL in the repo's `config/` directory.
  - Replace `YOUR_PATH` with the local dataset path.
  - Make sure the mention-image and KB-image folders match the dataset layout.

4. Run training.
  - The official entrypoint is `bash run.sh <gpu_id> <dataset_name>`.
  - For WikiMEL, use `bash run.sh 0 wikimel`.

5. Validate the result.
  - The model reports `H@1`, `H@3`, `H@5`, `MRR`, and mean rank during validation/test.
  - The official WikiMEL target to beat is `87.98 H@1` and `91.82 MRR`.

6. Lock the baseline.
  - Record the exact commit, package versions, dataset source, and config values.
  - Keep that checkpoint fixed before doing any quantization.

## Sources

- MIMIC official repo: `https://github.com/pengfei-luo/MIMIC`
- M3EL official repo: `https://github.com/zhiweihu1103/MEL-M3EL`
- DWE official repo: `https://github.com/season1blue/DWE`
- KGMEL official repo: `https://github.com/juyeonnn/KGMEL`
- KGMEL paper: `https://arxiv.org/pdf/2504.15135`
