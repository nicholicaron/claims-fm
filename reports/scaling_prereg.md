# Phase 2 pre-registration: scaling study + hierarchical Task B

Committed 2026-07-23, **before any Phase 2 training**. This document freezes
the experimental design, the exhaustive list of test-set touches, the scoring
rules, and numeric predictions. `reports/scaling_results.md` will score every
prediction against what actually happened, regardless of outcome.

## Motivation

v1.0 measured three things that bear on "would a bigger model help?":

1. Task A probe / full fine-tune / from-scratch cluster within ~0.03 AUPRC —
   capacity is not the separator between these arms.
2. By the end of pretraining, val loss was improving at only ~0.006/epoch
   (6.719 → 6.703 over the final two epochs) with the LR at its cosine floor —
   diminishing returns on repeated data, i.e. data-limited rather than
   capacity-limited.
3. The hybrid ablation added at most +0.003 AUROC over engineered features on
   Task A (ip 0.7134 vs 0.7100; cost 0.7626 vs 0.7620) — the synthetic
   sequence channel is nearly empty, and no parameter count extracts signal
   that isn't in the data.

Phase 2 turns those inferences into a measured scaling curve (Track 2), and
separately fixes the one place the v1.0 model demonstrably cannot see real
information: Task B's 512-token truncation, which discards 52.8% of claims and
truncates 1,018 of 5,410 providers (Track 1).

## Track 1 — hierarchical Task B (truncation fix)

Providers are encoded as 512-token, claim-boundary-aligned chunks; for the
4,392 providers untruncated in v1.0, chunk 0 is byte-identical to the v1.0
pack row (pinned by a regression test). Chunk `[CLS]` vectors are pooled per
provider with gated attention (Ilse, Tomczak & Welling, ICML 2018 MIL pooling;
d_att 128) into one logit. Training is end-to-end with a random-K train-time
cap (K=8 chunks/provider, sampled without replacement, seeded per
(seed, epoch, provider); K=8 fully covers 98.7% of providers and ≈90% of
tokens per epoch in expectation); **evaluation always uses all chunks**
(max 59) under no_grad. Measured provider distribution (builder cost
accounting, i.e. one separator charged per claim): p50 133 tokens, p99 10
chunks, max 29,809 tokens. Modes: full, scratch (architecture-matched
control), probe (frozen encoder + pooler/head).

Known asymmetry, documented not "fixed": chunk count implicitly encodes claim
volume; the XGB baselines already have volume features, and the scratch arm
shares the architecture, so pretrained-vs-scratch stays information-equal.

## Track 2 — 2×2 scaling grid (corpus × params)

| Cell | Params | Corpus | Config |
|---|---|---|---|
| C0 | 16.7M (6L/d320/ff1280) | 5 samples, 517k members | v1.0 `best.pt` (done) |
| C1 | 16.7M | 18 samples (3–20), ~1.86M members | `configs/pretrain_17m_18s.yaml` |
| C2 | ~46.3M (10L/d512/ff2048) | 18 samples | `configs/pretrain_46m_18s.yaml` |
| C3 | ~46.3M | 5 samples | `configs/pretrain_46m_5s.yaml` |

Constants across cells: seed 20260721, optimizer/schedule/masking identical,
max_epochs 12, patience 3, max_len 512. The vocabulary is **frozen** at the
28,203 tokens built from samples 3–7; codes unseen there map to `[UNK]` (rate
reported in DATA.md; `[UNK]` is never a masking target, so it cannot inflate
masked accuracy). The pretrain val split is a per-member seeded hash with
unchanged seed and val_frac, so a member's train/val assignment is identical
in both packs. Before any Phase 2 run, `train.py` gains a cumulative
`tokens_seen` counter in `metrics.jsonl`; equal-token-budget comparisons
across cells use it directly (robust to the registered OOM fallback
16384 → 12288 → 8192).

**Fine-tune hyperparameter policy:** all downstream fine-tunes use the v1.0
config values (head_lr 1e-3; encoder_lr 3e-5 full / 3e-4 scratch; epochs,
patience, batch budget unchanged), including the 46M cells. If a 46M
fine-tune visibly fails to train (val AUPRC below the scratch arm), the LR
may be adjusted **on validation only**, and the change is recorded in
`scaling_results.md`.

## Frozen protocol (unchanged from v1.0)

- Baselines in `baselines/*.json` are never recomputed.
- Task B ranking metrics: raw sigmoid probabilities, no calibration.
- Task A: isotonic calibration fitted on val, then a single test pass;
  reported AUROC is the calibrated value (as in v1.0).
- Split contracts verified by builders: task_a `0df64f85…`, task_b `289b206b…`.
- Model selection and any hyperparameter adjustment happen on validation only.

## Registered test-set touches (exhaustive — nothing else touches test)

LE grid structure mirrors v1.0 exactly: 10% and 25% label fractions × 3 seeds,
plus a single run at 100% labels.

1. Task A full fine-tune: one test pass each for C1, C2, C3.
2. Task B single-sequence pretrained-full LE grid (structure above): test LE
   evals for C1, C2, C3.
3. Task B hierarchical at the C0 encoder: hier-full LE grid, hier-scratch LE
   grid (structure above), and hier-probe at 100% labels (single run).
4. Task B hierarchical at the best scaled cell: same three as (3).
   **Selection rule (registered):** best scaled cell = highest single-sequence
   Task B **val** AUPRC of the full-mode 100%-label run; tie-break by MLM val
   loss on the 5-sample pack's val split.

## Prediction scoring conventions

- LE comparisons at 10%/25% use the mean of 3 seeds per fraction; at 100% the
  single run (matching v1.0, whose 100% arm is also a single run).
- v1.0 reference seed spreads (frozen): 10% std ±0.018, 25% std ±0.004. P3's
  per-fraction thresholds are set at ~2× the standard error of a difference
  of two 3-seed means: 0.03 at 10%, 0.01 at 25%.
- Task B test n=812; single-run AUPRC bootstrap CIs are wide (±0.10 in v1.0),
  so 100%-label predictions are scored directionally, with point windows as
  secondary descriptors.

## Predictions (committed before results)

- **P1 — corpus scale lifts MLM.** Val masked top-1, scored on the v1.0
  5-sample pack's val split (11,869 members; identical members val in both
  packs), rises from 15.8% to 17–20% for C1; C2 ≥ C1.
- **P2 — Task A does not move.** Calibrated test AUROC within ±0.01 of the
  v1.0 transformer (ip 0.669, cost 0.715) for every cell and both labels.
  The synthesis ceiling is per-member; more members and more params do not
  repair it.
- **P3 — params alone do nothing.** C3 vs C0 on the Task B single-seq LE
  grid: |Δ 3-seed-mean AUPRC| ≤ 0.03 at 10% and ≤ 0.01 at 25% (thresholds
  from the frozen v1.0 seed spreads above); the 100% single runs reported
  descriptively. This is the direct test of the original "scale up the
  model" hypothesis.
- **P4 — scaled encoders lift Task B low-label only modestly.** 3-seed-mean
  LE AUPRC gains of at most +0.02 over the v1.0 means (0.623 at 10%, 0.679
  at 25%); at 100% labels every single-seq cell stays below XGB's 0.711.
- **P5 — the truncation fix is the real improvement.** Primary (directional):
  hier-full at the C0 encoder beats the v1.0 flat model at 100% labels
  (test AUPRC > 0.695). Secondary (point window): 0.71–0.73, i.e. into the
  XGB 0.711 / hybrid 0.718 band, with P@50 ≥ 0.80.

Interpretation rules: P2+P3+P4 confirmed ⇒ the data ceiling is *measured*,
not assumed — strengthening the v1.0 claim with a scaling curve. Any refuted
⇒ a genuinely surprising result, reported with the same prominence. P5
refuted ⇒ the truncation story was wrong; that gets reported too.

## Budget

Point estimate ≈$7 GPU total for Phase 2 (pretrains ~$3.3, downstream ~$2.1,
hierarchical ~$1.5), ceiling $10, against $20.52 Vast credit at registration.
