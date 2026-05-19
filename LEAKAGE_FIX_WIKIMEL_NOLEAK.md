# WikiMEL Leakage Fix

Created on 2026-05-18.

## Problem Found

The previous `external/MIMIC_reproduction/data/WikiMEL` build was not paper-comparable:

- Train/test exact duplicate rows: `1213 / 5349` test rows.
- Train/test mention+sentence overlap: `1212 / 5349` test rows.
- Train/test image overlap: `983 / 5349` test rows.
- Gold entity image leakage: `5349 / 5349` test rows had the mention image inside the gold entity's `image_list`.

That explains why the reproduced MIMIC checkpoint reached `96.65 H@1`, far above the paper's `87.98 H@1`.

## Fixed Dataset

Clean dataset:

```text
external/MIMIC_reproduction/data/WikiMEL_noleak
```

Clean config:

```text
external/MIMIC_reproduction/config/wikimel_noleak.yaml
```

Build command:

```powershell
cd C:\Users\user\Desktop\kmpooja\ExpSetup
C:\Users\user\AppData\Local\Programs\Python\Python310\python.exe external\MIMIC_reproduction\scripts\prepare_wikimel_mimic.py --output-root external\MIMIC_reproduction\data\WikiMEL_noleak --dedupe-splits --entity-image-strategy none
```

Audit command:

```powershell
C:\Users\user\AppData\Local\Programs\Python\Python310\python.exe audit_wikimel_leakage.py --data-root external\MIMIC_reproduction\data\WikiMEL_noleak
```

Audit result:

- Train/dev/test exact overlap: `0`
- Train/dev/test mention+sentence overlap: `0`
- Train/dev/test image overlap: `0`
- Gold entity image leakage: `0`

## Clean Split Sizes

- Train: `18,092` mention rows
- Dev: `2,083` mention rows
- Test: `4,002` mention rows
- Candidate entities: `109,976`
- Entity image rows: `0`

## Important

The old checkpoint in `benchmarks/mimic_wikimel_locked_baseline` was trained on leaky data. It should not be used for paper-comparable claims.

Retrain with:

```powershell
cd C:\Users\user\Desktop\kmpooja\ExpSetup\external\MIMIC_reproduction
$env:PYTHONPATH=(Get-Location).Path
C:\Users\user\AppData\Local\Programs\Python\Python310\python.exe -u .\codes\main.py --config .\config\wikimel_noleak.yaml
```
