# Baselines (M2) — protocol, cohort, features

Frozen **before any transformer training** (tag `v0.2-baselines`). Results:
[results_task_a.md](results_task_a.md) · [results_task_b.md](results_task_b.md).
All numbers produced by `make baselines` (tune, train+val only) followed by a
single `make baselines-final` test pass. Config: [configs/baselines.yaml](../configs/baselines.yaml),
seed 20260721.

## Task A — member next-year risk

**Cohort waterfall** (samples 1–2; rule details in `tasks/cohort_a.py`):

| Step | Members |
|---|---|
| In 2008 beneficiary file | 232,747 |
| Present in 2008, 2009, and 2010 files | 225,599 |
| Full Part A+B coverage (HI=12, SMI=12) in 2008 & 2009 | 185,627 |
| Zero HMO months 2008–2010 | 114,041 |
| Alive entering 2010 | 114,041 |

The HMO exclusion is the big cut and is deliberate: HMO-enrolled months route
claims outside FFS files, so utilization features and 2010 outcomes would be
systematically incomplete for those members. The final "alive" step removes
nobody — members who died before 2010 have already left the 2010 file.

**Labels.** `label_ip`: ≥1 inpatient admission dated 2010 (prevalence 11.6%).
`label_cost`: 2010 total (all nine beneficiary-summary cost fields) ≥ $8,350,
the 90th percentile of the **train split only** (prevalence 10.0% by construction).

**Splits.** Member-level 70/15/15 (79,828 / 17,106 / 17,107), stratified on the
joint (ip × provisional cost bucket) key; both prevalences hold within ±0.01pp
across splits. Assignment: `data/processed/task_a_splits.parquet`,
sha256 `0df64f85e7cf45b8…` — **M4 must reuse this file.**

**Features (159).** Strictly observation-window (2008–2009) inputs, enforced by
a window guard that raises on any later-dated event (`features_a.check_window`,
tested):
demographics at 2010-01-01 (age, sex, race, ESRD, state one-hots); 11 chronic
flags + count (2009); the nine cost components per year + totals, log-totals,
and 2009−2008 trend; utilization (claims by type×year, distinct dx/px/ndc9
codes, inpatient days, active days/months); recency (days since last event /
last admission, events in final 90/180 days); counts over the top-50 3-digit
dx categories ranked by **train** prevalence (401 hypertension, 272 lipids,
V58 aftercare, 250 diabetes, …).

## Task B — provider fraud

**Splits.** Provider-level 70/15/15 (3,786 / 812 / 812), stratified on the
fraud label (9.35–9.36% everywhere). Assignment:
`data/processed/task_b_splits.parquet`, sha256 `289b206b9c00a8fc…` — **M5 must
reuse this file.**

**Beneficiary overlap (SPEC §5 caveat, quantified).** 66.1% of beneficiaries
appear under more than one provider; 79.3% of val-provider claims and 77.1% of
test-provider claims involve a beneficiary also seen under some train
provider. The prediction unit (provider) is cleanly split; member-level
information is not fully disjoint. Stated, not pretended away.

**Features (34, strictly within-provider — invariance is tested).** Volume
(claims, benes, claims/bene, active months); money (reimbursement sum/mean/
median/p90/max, IP vs OP, deductibles, zero-reimbursement rate); mix (IP claim
and revenue share, length of stay, admissions/bene); coding (distinct dx/px,
dx-per-claim, top-dx concentration, duplicate-claim rate, DRG diversity);
physician patterns (distinct attending/operating, claims/physician, missing
attending, attending=operating); panel mix (age, deceased share, chronic
burden, prior annual reimbursement); monthly burstiness.

## Modeling & evaluation

- **LR:** standardized L2 logistic regression; small grid (C × class_weight),
  selected on val AUPRC.
- **XGBoost:** `hist`, 40-draw seeded random search, early stopping on val
  AUCPR (≤2,000 trees), selected on val AUPRC.
- **Calibration (Task A):** Platt and isotonic fit on val, winner by val
  Brier; test reported before/after.
- **Label efficiency (Task B):** the selected XGB config refit on 10%/25%
  (3 seeds each) and 100% of train providers — the classical side of the M5
  comparison.
- **Test discipline:** hyperparameters, model selection, and calibrators use
  train+val only; the test split is scored exactly once, by
  `run_baselines.py --final-eval`, for all frozen models simultaneously.
- **Uncertainty:** 95% CIs via 1,000-resample unit-level bootstrap.
