# claims-fm

A small claims foundation model for payer predictive modeling: pretrain a
~10–25M-parameter BERT-style encoder on CMS DE-SynPUF synthetic Medicare
claims sequences (masked-code modeling), then fine-tune it for two downstream
payer tasks — member next-year risk stratification and provider fraud
detection — benchmarked against tuned XGBoost baselines.

Full project spec: [SPEC.md](SPEC.md) · data: [DATA.md](DATA.md) · baseline
protocol & results: [baselines/](baselines/README.md).

## Baseline results (frozen before any transformer training — tag `v0.2-baselines`)

Test set touched once; 95% bootstrap CIs in the full reports.

| Task | Label (prevalence) | Model | AUROC | AUPRC | Payer metric |
|---|---|---|---|---|---|
| A — member risk | inpatient admission 2010 (11.6%) | XGB (calibrated) | 0.710 | 0.218 | capture@5% outreach: 12.8% |
| A — member risk | top-decile 2010 cost (10.0%) | XGB (calibrated) | 0.762 | 0.303 | capture@5% outreach: 20.5% |
| B — provider fraud | fraud flag (9.4%) | LR | 0.961 | 0.749 | precision@50: 78% |
| B — provider fraud | fraud flag (9.4%) | XGB | 0.952 | 0.711 | precision@100: 54%, recall 71% |

Honest notes: on Task B the val-selected XGB lost to plain logistic regression
on test (CIs overlap heavily — provider features are nearly linearly
separable here). Task A absolute numbers are depressed by DE-SynPUF's
synthesis (weakened code–outcome correlations); post-isotonic calibration is
excellent (ECE ≤ 0.007). Details: [results_task_a.md](baselines/results_task_a.md),
[results_task_b.md](baselines/results_task_b.md).

## Pipeline

```
make env        # uv-managed Python 3.12 environment
make download   # DE-SynPUF samples (manifest-driven, checksummed, integrity-checked)
make eda        # M0 orientation notebook
make tables     # raw CSV -> typed parquet
make sequences  # member event sequences (+ Task A windowed variant)
make vocab      # code vocabulary + tokenizer artifacts
make overlap    # Kaggle<->DE-SynPUF vocab overlap gate (>= ~60% required)
make test       # ETL invariant + leakage test suite
```

Data lives under `data/` (gitignored); provenance, checksums, and row counts
are committed in `configs/data.lock.yaml`.

## Status

- [x] M0 — scaffold & data (7 samples downloaded + checksummed, EDA notebook)
- [x] M1 — ETL, vocabulary, tokenizer (28k-token vocab; Kaggle dx overlap gate **passed at 99.9%**; leakage tests green)
- [x] M2 — baselines (LR + tuned XGBoost, both tasks, frozen at `v0.2-baselines`)
- [x] M3 — pretraining (12 epochs on a Vast 4090, val masked top-1 15.8% vs 1.4% frequency prior; probes in [reports/pretrain.md](reports/pretrain.md); **actual cost $0.52**)
- [x] M4 — Task A fine-tune & eval ([report](reports/task_a_transformer.md)): **pretraining transfers** (full FT beats from-scratch, AUPRC +0.015/+0.029) but **XGBoost wins the task** (AUROC 0.762 vs 0.715 on cost) — reported straight; **actual cost $0.32**
- [ ] M5 — Task B transfer & eval
- [ ] M6 — writeup & polish

## Budget ledger

| Item | Budgeted | Actual |
|---|---|---|
| M3 pretraining | $8 | **$0.52** (RTX 4090 @ $0.28/hr, incl. all debugging + a deliberate kill/resume drill) |
| M4 Task A | $3 | **$0.32** (6 fine-tune runs: probe/full/scratch × 2 labels) |
| M5 Task B | $3 | — |
| Reserve | $6 | — |
