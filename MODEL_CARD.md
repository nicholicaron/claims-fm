# Model card — claims-fm encoder

## Model description

6-layer pre-LN transformer encoder, d_model 320, 8 heads, FFN 1280,
16.7M parameters; weight-tied masked-code-modeling head. Inputs are summed
embeddings per event token: medical code (28,203-token vocabulary: 10,742
ICD-9 dx + 2,456 ICD-9 proc + 15,000 NDC-9 + 5 specials), claim type
(IP/OP/RX), age-at-event, study month, absolute position, and visit parity;
visits are delimited with `[VISIT]` tokens and sequences truncate
keep-most-recent at 512 positions. Downstream heads read the final-layer
`[CLS]` state.

## Intended use

Research/demonstration artifact for payer predictive-modeling workflows:
(a) member risk stratification (next-year inpatient admission; top-decile
cost) and (b) provider fraud triage, including as a frozen feature
extractor for gradient-boosting hybrids. **Not for clinical or coverage
decisions.** Trained entirely on synthetic data; nothing in this repository
has seen real member information.

## Training data

- **Pretraining:** CMS DE-SynPUF samples 3–7 (517,390 synthetic
  beneficiaries, 41.3M coded events, 2008–2010), masked-code modeling
  (15% masking, 80/10/10), 12 epochs, single RTX 4090, $0.52. Samples 1–2
  were excluded from pretraining and vocabulary by construction and
  reserved for Task A evaluation.
- **Fine-tuning:** Task A on samples 1–2 (114,041-member cohort, committed
  70/15/15 member splits); Task B on the Kaggle Healthcare Provider Fraud
  dataset (5,410 labeled providers, committed provider splits). Split
  assignments are content-hashed; hashes are verified by tests and recorded
  in the frozen metrics.

## Evaluation summary (held-out test, single pass, 95% bootstrap CIs in reports)

| Task | Best model | Key numbers |
|---|---|---|
| Member risk (admission, prev 11.6%) | XGBoost baseline | AUROC 0.710; best transformer 0.669 |
| Member risk (top-decile cost, prev 10.0%) | XGBoost baseline | AUROC 0.762; best transformer 0.715 |
| Provider fraud, 100% labels (prev 9.4%) | Hybrid XGB+embeddings | AUPRC 0.718, P@50 82%; LR 0.749 AUPRC |
| Provider fraud, 25% labels | Pretrained encoder | AUPRC 0.679 (±0.004) vs XGBoost 0.637 (±0.050) |
| Pretraining quality | — | masked top-1 15.8% vs 1.4% frequency prior; clinically coherent embedding neighborhoods |

Calibration: all deployed-style probabilities are isotonic-recalibrated on
validation; post-recalibration ECE ≤ 0.007 on Task A across models.
Subgroup slices (sex, age band, race) are reported for both model families;
the smallest race group (n=364) shows the weakest discrimination and
calibration in both.

## Limitations

1. **Synthetic data.** DE-SynPUF's synthesis weakens code–outcome and
   sequential correlations; absolute metrics understate real-data
   performance, and the measured Task A result (embeddings add nothing to
   engineered features) is a property of the synthesis, demonstrated by the
   hybrid ablation.
2. **ICD-9 era (2008–2010).** Real deployment requires an ICD-10/HCPCS
   vocabulary (GEMs-warm-started or retrained); the tokenizer is
   code-agnostic by design.
3. **Constructed fraud labels.** Kaggle's provider flags are heuristic, not
   SIU-adjudicated; fraud metrics overstate performance against confirmed
   fraud.
4. **Truncation.** 512-token cap keeps 47% of provider claims; the encoder
   never observes volume. Provider-level attention pooling over all claims
   is the known fix.
5. **Drug codes.** DE-SynPUF randomizes NDCs (940 of 15,000 RX tokens match
   the FDA directory); RX embeddings mostly encode refill repetition.

## With real payer data, what changes

ICD-10 vocabulary swap; next-visit pretraining objective over real temporal
structure; dual-eligibility/SDOH/pharmacy features; monthly batch scoring
into care-management queues with per-cycle recalibration; drift monitoring
on population mix, code mix, and calibration decay; HIPAA posture — this
public-safe synthetic-first build is exactly what you want to have proven
*before* touching PHI. First experiment on real data: the hybrid test
(frozen embeddings ⊕ engineered features), which cheaply measures whether
the sequence channel carries signal your features miss.

## Reproducibility

Config-driven and seeded end-to-end; 43-test suite covers ETL invariants,
leakage rules, masking, and resume determinism; dataset checksums and
provenance in `configs/data.lock.yaml` and `DATA.md`; budget ledger in the
README ($1.14 total). Tags: `v0.2-baselines` → `v1.0`.
