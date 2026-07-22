# claims-fm

A small claims foundation model for payer predictive modeling: pretrain a
~10–25M-parameter BERT-style encoder on CMS DE-SynPUF synthetic Medicare
claims sequences (masked-code modeling), then fine-tune it for two downstream
payer tasks — member next-year risk stratification and provider fraud
detection — benchmarked against tuned XGBoost baselines.

Full project spec: [SPEC.md](SPEC.md). Dataset documentation: DATA.md (added at M1).

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
- [ ] M2 — baselines (LR + XGBoost, both tasks)
- [ ] M3 — pretraining
- [ ] M4 — Task A fine-tune & eval
- [ ] M5 — Task B transfer & eval
- [ ] M6 — writeup & polish

## Budget ledger

| Item | Budgeted | Actual |
|---|---|---|
| M3 pretraining | $8 | — |
| M4 Task A | $3 | — |
| M5 Task B | $3 | — |
| Reserve | $6 | — |
