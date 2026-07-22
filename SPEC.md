# Claims-FM — A Small Claims Foundation Model for Payer Predictive Modeling

**Purpose:** Self-assigned take-home for the Centene ML Engineer interview (predictive modeling team).
**One-line pitch:** Pretrain a small transformer on 500k+ synthetic Medicare claims histories, then fine-tune it for the two problems a payer actually pays for — care-management risk stratification and provider fraud detection — benchmarked honestly against XGBoost.

**Status:** Spec drafted 2026-07-21. Budget: ~$20 Vast.ai + local M-series Mac for ETL/baselines.

---

## 1. Narrative (the 60-second interview version)

> "Payers sit on claims sequences, not clinical notes. I treated each member's claim history as a token sequence — diagnoses, procedures, drugs, with visit and time structure — and pretrained a small BERT-style encoder on CMS's synthetic Medicare data. Then I fine-tuned it twice: once to predict next-year inpatient admission and high-cost membership (the care-management problem), and once, transferred across datasets, to flag potentially fraudulent providers from their claim patterns (the payment-integrity problem). I benchmarked everything against tuned XGBoost on engineered features, reported calibration and capacity-constrained metrics — lift at the top 5%, precision at SIU caseload — and ran the ablation that matters: does pretraining actually transfer, especially when fraud labels are scarce? Here's what I found."

The novel angle: **one pretrained claims encoder → two downstream payer tasks across two datasets.** This mirrors the "healthcare foundation model" direction (Med-BERT, CLMBR, MOTOR) at a scale one person can execute for $20.

The credibility angle: **baselines first, calibration reported, honest conclusions.** If XGBoost wins a task, say so and explain why — that reads as senior judgment, not failure.

---

## 2. Objectives

- **O1 — Claims encoder pretraining.** Pretrain a small (≈10–25M param) transformer encoder on member claim sequences from DE-SynPUF via masked-code modeling. Demonstrate learned clinical structure (embedding-space probes).
- **O2 — Task A: member risk stratification.** Fine-tune for next-year outcomes: (a) any inpatient admission, (b) top-decile total cost. This is the care-management targeting problem.
- **O3 — Task B: provider fraud detection.** Reuse the encoder on the Kaggle provider-fraud dataset (same DE-SynPUF lineage/vocabulary): pool claim/member representations per provider → fraud flag. This is the payment-integrity problem and the bridge to my security background.
- **O4 — Honest baselines.** Logistic regression + tuned XGBoost on well-engineered aggregate features for both tasks, built *before* any transformer training.
- **O5 — Evaluation that speaks payer.** Discrimination, calibration, capacity-constrained decision metrics, subgroup slices, and the transfer/data-efficiency ablations.
- **O6 — Deliverables that survive scrutiny.** Public repo, results-first README, ~1,500-word technical writeup, model card with limitations, interview defense prep.

**Non-objectives (out of scope):** clinical notes / NLP, ICD-10 modeling (data is ICD-9 era — addressed in limitations), deployment infra beyond a batch-scoring sketch, SOTA-chasing.

---

## 3. Datasets

| Dataset | Role | Size | Access |
|---|---|---|---|
| [CMS DE-SynPUF](https://www.cms.gov/data-research/statistics-trends-and-reports/medicare-claims-synthetic-public-use-files/cms-2008-2010-data-entrepreneurs-synthetic-public-use-file-de-synpuf) (2008–2010, 20 downloadable samples) | Pretraining + Task A | ~2.3M beneficiaries total; ~116k/sample. Start with 4–6 samples | Public, no credentialing. CSV zips per sample: Beneficiary Summary, Inpatient, Outpatient, Carrier, PDE |
| [Kaggle Healthcare Provider Fraud](https://www.kaggle.com/datasets/rohitrox/healthcare-provider-fraud-detection-analysis) | Task B | ~558k claims, 5,410 labeled providers, ~139k beneficiaries | Public Kaggle download |
| [CDC Social Vulnerability Index](https://www.atsdr.cdc.gov/place-health/php/svi/index.html) (county-level) | Stretch: SDOH features for Task A | Small | Public |

**Key structural fact to exploit and verify early:** the Kaggle fraud data follows the DE-SynPUF schema (ICD-9 diagnosis/procedure codes, same beneficiary chronic-condition flags). M1 must quantify token-vocabulary overlap between the two datasets. If dx-code coverage of Kaggle claims by the pretraining vocab is <~60%, revisit the transfer design before M3.

**Clean-split trick:** DE-SynPUF ships as 20 disjoint samples. Use samples 3+ for pretraining and reserve samples 1–2 exclusively for Task A fine-tune/eval. Zero-overlap separation between pretraining and evaluation members, by construction — call this out in the writeup.

---

## 4. Technical framework

- **Core:** Python 3.12+, PyTorch 2.x. Training loop: plain PyTorch or Lightning — your call. No HF `Trainer` needed (custom vocab, not text), though `transformers` layers may be reused.
- **ETL:** polars or DuckDB (carrier claims are the big files; columnar tools keep this on-Mac). Parquet as the interchange format.
- **Baselines:** scikit-learn (LR), XGBoost.
- **Tracking:** Weights & Biases free tier or TensorBoard. Every GPU run logged; config-driven (single YAML/TOML per run), seeded.
- **Compute split:** ETL, tokenization, baselines, and evaluation run locally. Vast.ai GPU (RTX 4090-class, ~$0.35–0.45/hr) only for pretraining and fine-tuning. Checkpoint to persistent storage frequently — assume instances die.
- **Repo:** public GitHub, plain conventional commits, one-command pipeline per stage (`make etl`, `make baseline`, `make pretrain`, ...), tests on ETL invariants (row counts, date ranges, no window leakage).

**Model shape (guidance, not prescription):** 4–6 layer encoder, d_model 256–384, ≈10–25M params. Input tokens = medical codes (dx/proc/drug); structure via visit-boundary tokens or segment embeddings; age/time and claim-type embeddings summed in. Masked-code modeling objective. Architecture internals (attention variant, pooling, position encoding) are implementation freedom — the spec constrains interfaces, splits, and evaluation only.

---

## 5. Task definitions (precise — leakage rules matter more than architecture)

### Task A — member next-year risk (DE-SynPUF samples 1–2)

- **Cohort:** beneficiaries with coverage in both the observation and prediction windows (define the coverage rule from beneficiary-summary fields and document it).
- **Observation window:** 2008–2009 claims → input sequence. **Prediction window:** 2010.
- **Labels:** (a) ≥1 inpatient admission in 2010; (b) 2010 total annual cost in the top decile (threshold computed on train split only).
- **Splits:** member-level 70/15/15 train/val/test, stratified on label. No member in more than one split.
- **Leakage rules:** inputs strictly from the observation window; cost threshold from train only; pretraining corpus excludes samples 1–2 entirely.

### Task B — provider fraud (Kaggle)

- **Unit:** provider. Labels exist for 5,410 providers in the train file — make your own 70/15/15 provider-level splits from it (public "test" file is unlabeled).
- **Input:** the provider's claim set — each claim encoded by the pretrained encoder (or member-history context, design choice), attention-pooled to a provider representation.
- **Leakage rules:** split by provider; note in writeup that beneficiaries can appear across providers (quantify the overlap rather than pretending it away).
- **Baseline features:** per-provider aggregates — claim counts, mix of inpatient/outpatient, reimbursement distributions, duplicate-code rates, per-beneficiary intensity, physician-role patterns.

### Ablations (the analysis that wows)

1. **Pretrained vs from-scratch** encoder, both tasks, identical architecture and budget.
2. **Label efficiency (Task B):** fine-tune on 10% / 25% / 100% of labeled providers. *This is the money chart* — real payers have very few confirmed fraud labels, and pretraining should separate from from-scratch exactly there.
3. **Transformer vs XGBoost** at equal information (same observation window), stated honestly.

---

## 6. Metrics & evaluation protocol

**Both tasks:** AUROC and AUPRC (always report prevalence next to AUPRC).

**Task A additionally:**
- **Calibration:** Brier score, reliability diagram, ECE; recalibrate (isotonic/Platt on val) if needed and show before/after. Payers act on probabilities — this section signals more domain sense than any AUROC.
- **Capacity-constrained lift:** capture rate of true admissions/high-cost members in the top 1% / 5% / 10% of predicted risk ("if care management can only outreach 5% of members, what do we catch?").
- **Subgroup slices:** AUROC + calibration by sex, age band, and race field from the beneficiary summary. Frame as health-equity due diligence (a loud Medicaid managed-care theme), not a compliance checkbox.

**Task B additionally:**
- **Precision@k** for k ≈ realistic SIU caseloads (e.g., top 50/100 providers), plus the operating point you'd actually choose and why.

**Pretraining quality probes (writeup gold):**
- Masked-code accuracy/perplexity curves.
- Nearest-neighbor probes in embedding space: pick 5–6 anchor codes (a diabetes dx, an insulin NDC, a dialysis procedure...) and show neighbors are clinically coherent. One compelling 2-D projection figure, honestly labeled as qualitative.

**Statistical hygiene:** bootstrap 95% CIs on headline metrics; single held-out test set touched once per task, at the end.

---

## 7. Milestones

Rough total: 25–40 hours of your time over 1–2 weeks; ≤ $20 GPU.

### M0 — Scaffold & data (local, ~2h)
Repo init, this spec committed, both datasets downloaded with checksums/row counts recorded, licenses noted. **Accept:** raw data present + documented; EDA notebook with 5–10 orientation plots (claims per member, code frequency long-tail, cost distribution).

### M1 — ETL, vocabulary, tokenizer (local, ~4–6h)
Member event sequences from DE-SynPUF (start: inpatient + outpatient + PDE; carrier claims optional — biggest files, add only if time allows and say so). Build code vocabulary with frequency floor; tokenizer with special tokens for visit/claim-type structure. **Quantify Kaggle↔DE-SynPUF vocab overlap.** **Accept:** parquet sequence store; DATA.md with dataset stats table; overlap ≥ ~60% of Kaggle dx tokens covered (else redesign transfer before proceeding); ETL leakage tests green.

### M2 — Baselines first (local, ~4h)
Engineered features + LR + tuned XGBoost for **both** tasks, full metric suite, frozen as `baselines/` results. **Accept:** results table committed *before any transformer training*. This ordering is itself a talking point.

### M3 — Pretraining (Vast, ~$8; ~3–6 GPU-h)
Masked-code modeling on samples 3+ (start ~4–6 samples ≈ 500–700k members). Checkpoints synced off-instance; W&B curves. **Accept:** converged loss curve; masked-code top-1 accuracy meaningfully above frequency-prior baseline; embedding probes pass the sniff test.

### M4 — Task A fine-tune & eval (Vast, ~$3)
Fine-tune (frozen-encoder linear probe *and* full fine-tune; report both), full metric suite vs baselines, from-scratch ablation. **Accept:** complete eval report incl. calibration + lift@k + slices; a defensible sentence on transformer-vs-XGBoost, whichever way it goes.

### M5 — Task B transfer & eval (Vast, ~$3)
Provider pooling head over the pretrained encoder; full metric suite vs baseline; **label-efficiency ablation** (10/25/100%); from-scratch ablation. **Accept:** eval report + the label-efficiency chart; explicit statement on whether cross-dataset pretraining transferred.

### M6 — Writeup & polish (local, ~4–6h)
- README: headline results table + the two money charts above the fold.
- ~1,500-word technical writeup (blog candidate — the theme-adaptive inline-SVG figure tooling from ft-diloco is reusable).
- Model card: intended use, training data, metrics, **limitations** (synthetic-data caveats: DE-SynPUF's synthesis weakens code–cost correlations, so absolute numbers understate real-data performance — methodology is the product; ICD-9 era; fraud labels are constructed).
- "With real Centene data, here's what changes" section: ICD-10 vocab swap (GEMs mapping or retrain), real dual-eligibility/SDOH features, monthly batch scoring, drift monitoring. **Accept:** a stranger can go from `git clone` to understanding the results in 5 minutes.

**Budget ledger:** M3 $8 · M4 $3 · M5 $3 · reserve $6. Track actuals in README.

---

## 8. Deliverables

1. Public GitHub repo (suggest `claims-fm`), one-command stages, tests, configs.
2. Results-first README (table + 2 figures above the fold).
3. Technical writeup (~1,500 words) — publishable on neumann-labs.com.
4. Model card with limitations and real-data migration section.
5. One-page PDF summary (bring to / send before the interview).
6. Optional stretch, only if everything above is done: tiny Streamlit demo (paste a synthetic member timeline → risk score + attention/attribution view).

---

## 9. What actually wows this audience (checklist)

- [ ] **Baselines before transformers**, committed first, beaten or honestly not.
- [ ] **Calibration treated as first-class** — payers consume probabilities.
- [ ] **Decision metrics in payer language** — lift@5% outreach capacity, precision@SIU caseload, not just AUROC.
- [ ] **The transfer story** — one encoder, two tasks, two datasets; label-efficiency chart for fraud.
- [ ] **Health-equity slice analysis** done thoughtfully, with discussion.
- [ ] **Limitations stated plainly** — synthetic data, ICD-9, constructed fraud labels. Naming your own caveats before they do is the strongest senior signal available.
- [ ] **Literature positioning in two sentences** — Med-BERT / CLMBR / MOTOR exist; this is the one-person, $20 version; cite, don't oversell.
- [ ] **Security-background bridge stated explicitly** in the fraud section: provider billing anomalies ≈ anomalous event streams in security telemetry.
- [ ] **Reproducibility** — seeds, configs, budget ledger, honest "what didn't work" appendix.

## 10. Risks & mitigations

| Risk | Mitigation |
|---|---|
| DE-SynPUF synthesis weakened predictive signal → transformer ≈ baseline on Task A | Expected; the deliverable is methodology + honest analysis. The fraud task (constructed labels) will show clearer separation. Say all of this out loud in the writeup. |
| Carrier-claims ETL time sink (13 line items/claim, huge files) | Start without carrier claims; add only if ahead of schedule; document the choice. |
| Kaggle↔SynPUF vocab overlap too low for transfer | Measured at M1 as a gate, before GPU spend. Fallback: dx-codes-only shared vocab. |
| Vast instance dies mid-run | Checkpoint every N steps to off-instance storage; runs sized ≤2h each; resume logic from day one. |
| Scope creep (SVI join, demo, extra ablations) | Everything in §8 item 6 and SVI is stretch-gated behind M6 completion. |

## 11. Interview defense prep (know cold before the loop)

- Why a transformer over GBM here — and when you'd still ship the GBM.
- How this migrates ICD-9 → ICD-10 (GEMs crosswalk vs retrain; code-agnostic tokenizer design).
- What changes with real member data: HIPAA posture, PHI handling, why synthetic-first was the right call for a public artifact.
- How you'd deploy: monthly batch scoring into care-management queues; recalibration cadence; drift monitoring (population shift, code-mix shift, calibration decay).
- Why AUPRC over AUROC at 10% prevalence; why calibration can matter more than either.
- The fairness slices: what you found, what you'd do about a gap, and the difference between calibration parity and equal opportunity.
- Where the fraud labels come from and why constructed labels overstate performance vs. real SIU-confirmed fraud.
- Cost of the whole thing ($20) — and what you'd do with 100× compute (scale corpus, next-event pretraining à la MOTOR, longer contexts).
