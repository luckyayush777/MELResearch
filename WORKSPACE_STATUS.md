# Workspace Summary: WikiMEL Quantization + MIMIC Baseline Reproduction

**Last updated**: April 21, 2026

## 🟢 Completed: Local WikiMEL Quantization Baseline

**Location**: `benchmarks/wikimel_official_test_v1/quant_suite/`

### Results Generated
- **Embedding quantization**: fp32 → fp16 (50% storage) → int8 (25% storage)
  - All modes maintain accuracy on full-KB retrieval
  - Recommended for production: fp16 or int8
- **FAISS index quantization**: flat vs sq8
  - SQ8 achieves 75% compression (53.70 MB vs 214.80 MB)
  - Quality preserved: R@1 = 0.0972 (0.02 improvement vs flat)

### Artifacts Available
- `suite_summary.json` — Quantization results + recommendations
- `embedding_quantization_summary.csv` — Mode comparisons
- `faiss_pq_results.csv` — Index quantization results
- `embedding_plots/` — Accuracy vs storage tradeoff visualizations
- `faiss_pq_plots/` — FAISS index tradeoffs

### Current Baseline Metrics
- R@1: 0.0970 (9.7%)
- R@10: 0.2167 (21.67%)
- R@100: 0.4136 (41.36%)
- MRR: 0.1398

**Note**: This is a modest OpenCLIP-style baseline. The next step is to replace it with a stronger retrieval-optimized model.

---

## 🟡 In Progress: MIMIC WikiMEL Reproduction

**Location**: `external/MIMIC_reproduction/`

### What's Ready
- [x] Official MIMIC repo cloned
- [x] Setup documentation created (`SETUP_WIKIMEL.md`)
- [x] Reproduction checklist created (`REPRODUCTION_CHECKLIST.md`)
- [x] WikiMEL dataset generated in `external/MIMIC_reproduction/data/WikiMEL/`
- [x] Config paths updated for the local WikiMEL build
- [x] Image folders staged as junctions under `external/MIMIC_reproduction/data/WikiMEL/`
- [x] Dependencies installed in the workspace venv

### What Is Left

**Step 1**: Start training on a CUDA-capable machine.  
This workspace has no visible GPU, so `trainer.accelerator: 'gpu'` cannot run here.

**Step 2**: If you want a fresh download path later, reuse the converter in `external/MIMIC_reproduction/scripts/prepare_wikimel_mimic.py`.  
It can be rerun safely and prints progress while building the local WikiMEL assets.

**Step 3**: Launch training on the target GPU box.
```bash
cd external/MIMIC_reproduction
export PYTHONPATH=$(pwd)
bash run.sh 0 wikimel
```

### Expected Outcome
- Training runs for ~20 epochs (a few hours on GPU)
- Final metrics should be:
  - **H@1 ≈ 87.98** ✓ (official MIMIC paper)
  - **MRR ≈ 91.82** ✓ (official MIMIC paper)

---

## 🎯 Next Phase: Quantize the MIMIC Baseline

Once MIMIC training completes:

1. Extract the best checkpoint
2. Run the local quantization suite on the MIMIC model
3. Compare against the current weak baseline
4. Expected: 85-87 H@1 after int8 quantization (vs 87.98 unquantized)

---

## 📋 Key Files

### Quantization (Ready to use)
- [benchmarks/wikimel_official_test_v1/quant_suite/suite_summary.json](benchmarks/wikimel_official_test_v1/quant_suite/suite_summary.json)
- [MEL_REPRODUCTION_TARGETS.md](MEL_REPRODUCTION_TARGETS.md) — Ranking of baseline options

### MIMIC Reproduction (In progress)
- [external/MIMIC_reproduction/SETUP_WIKIMEL.md](external/MIMIC_reproduction/SETUP_WIKIMEL.md) — Detailed setup guide
- [external/MIMIC_reproduction/REPRODUCTION_CHECKLIST.md](external/MIMIC_reproduction/REPRODUCTION_CHECKLIST.md) — Step-by-step checklist
- [external/MIMIC_reproduction/config/wikimel.yaml](external/MIMIC_reproduction/config/wikimel.yaml) — Config template

---

## ⚙️ Technical Notes

### Campus Proxy Configuration
All package installs use the campus proxy:
```powershell
$env:HTTP_PROXY='http://172.31.2.3:8080'
$env:HTTPS_PROXY='http://172.31.2.3:8080'
$env:ALL_PROXY='http://172.31.2.3:8080'
```

### Python Environment
- Active environment: Python 3.10.6 at `C:\Users\user\AppData\Local\Programs\Python\Python310\python.exe`
- CUDA status: `torch.cuda.is_available() == True` on the RTX 4090
- Torch build: `2.1.2+cu121`
- MIMIC recommends: Python 3.8.12 (the workspace now has a newer CUDA-capable Python 3.10 setup instead)
- Dependencies: torch, transformers, pytorch-lightning, etc.

### GPU Requirements
- Recommended: 1x GPU with 24+ GB VRAM
- MIMIC training uses: batch_size=128, eval_batch_size=20 (configurable)
- Quantization: CPU-only (faiss-cpu used)

---

## 📍 Summary

| Component | Status | Location |
|-----------|--------|----------|
| **Quantization Suite** | ✅ Complete | `benchmarks/wikimel_official_test_v1/quant_suite/` |
| **MIMIC Repo** | ✅ Cloned | `external/MIMIC_reproduction/` |
| **Dependencies** | ✅ Installed | Workspace `.venv` |
| **Documentation** | ✅ Complete | `external/MIMIC_reproduction/*.md` |
| **Data** | ✅ Prepared | `external/MIMIC_reproduction/data/WikiMEL/` |

**Next action**: Run MIMIC on a CUDA-capable machine, then use that checkpoint as the stronger quantization baseline.
