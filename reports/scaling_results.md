# Phase 2 results: scoring the pre-registration

Pre-registration: [scaling_prereg.md](scaling_prereg.md), committed 2026-07-23
(`0de5a40`) before any Phase 2 training. Runs executed 2026-07-23/24 on one
rented RTX 4090; every test evaluation below is one of the registered touches,
executed exactly once. Verdicts use the scoring conventions frozen in the
prereg. Nothing was re-run, re-selected, or re-thresholded after seeing test.

## Scorecard

| # | Prediction (frozen) | Outcome | Verdict |
|---|---|---|---|
| P1 | C1 masked top-1 rises 15.8% → 17–20% on the v1.0 val; C2 ≥ C1 | C1 **16.20%**, C2 **16.29%** (C3 16.10%) | **Refuted on magnitude** (rise ✓, window ✗; C2 ≥ C1 ✓) |
| P2 | Task A calibrated test AUROC within ±0.01 of v1.0 (ip 0.669 / cost 0.715), every cell, both labels | ip 0.662–0.666, cost 0.713–0.716 (max \|Δ\| = 0.007) | **Confirmed** |
| P3 | C3 ≈ C0 on Task B LE (\|Δ mean\| ≤ 0.03 @10%, ≤ 0.01 @25%) | @10% Δ = **−0.129** (0.494 vs 0.623) — outside threshold, *negative*; @25% Δ = +0.003 ✓; @100% 0.704 vs 0.695 (single runs) | **Refuted @10% — in the opposite direction: params alone made low-label transfer worse** |
| P4 | Scaled-cell LE gains ≤ +0.02; all cells < XGB 0.711 @100% | Largest gain +0.003 (@25%, C3); @10% all cells **regressed** (0.576/0.541/0.494 vs 0.623); @100% max 0.704 < 0.711 ✓ | **Confirmed** (and understated: not just "no gains" — losses) |
| P5 | Primary: hier-full @C0 > 0.695 @100%. Secondary: 0.71–0.73, P@50 ≥ 0.80 | @C0 full **0.652** [0.536, 0.761], P@50 0.78 | **Refuted** (as frozen, P5's scope is the C0 encoder). The capstone @C3 — registered touch #4 but not a frozen prediction — landed in-window: **0.714** [0.620, 0.802], P@50 **0.82** |

## The two findings we did not predict

**1. Scale actively hurt low-label transfer.** The prereg predicted scaling
would do *nothing*; it did worse than nothing. At 10% of fraud labels, every
scaled encoder transfers worse than the v1.0 baseline (C0 0.623 → C1 0.576,
C2 0.541, C3 0.494; 3-seed means), with the params-only cell worst. A
plausible mechanism — more capacity and more repeated synthetic data means
more of the encoder's representation is spent on DE-SynPUF's synthesis noise,
which is exactly what a 378-provider fine-tune cannot afford to unlearn — but
the mechanism is post-hoc; the measurement is the claim. Either way the
practical conclusion sharpens v1.0's: on this corpus, the 17M/5-sample
configuration was not a compromise. It was the optimum.

**2. Everything that can see claim volume converges at ~0.71.** Four
independent model families now land on the same 100%-label fraud ceiling:
XGBoost on engineered features **0.711**, hybrid (features ⊕ embeddings)
**0.718**, hierarchical-full at the C3 encoder **0.714**, and — most telling —
hierarchical-*scratch* at the C0 architecture **0.714**, an untrained encoder
whose only reliable signal is chunk count and code statistics through
attention pooling. The v1.0 flat transformer's 0.695 deficit was its
truncation-induced blindness to volume, as hypothesized; but closing that gap
does not go *through* pretraining at 100% labels. Where pretraining shows up
is label efficiency, as in v1.0: at 10% labels hier-full @C0 reaches
0.631 (±0.021) while hier-scratch manages 0.414 — a 21.7-point gap from
pretrained initialization alone.

## Detail tables

### MLM quality (registered P1 protocol: v1.0 5-sample val, fixed mask stream)

| Cell | Params | Corpus | Masked top-1 | Val loss |
|---|---|---|---|---|
| C0 | 16.7M | 5s | 15.8% | 6.703 |
| C1 | 16.7M | 18s | 16.20% | 6.632 |
| C2 | 46.3M | 18s | 16.29% | 6.631 |
| C3 | 46.3M | 5s | 16.10% | 6.667 |

3.6× corpus + 2.8× params, in any combination, moved masked accuracy by less
than half a point. The corpus's information ceiling (32% `[UNK]`, synthetic
near-flat code statistics) binds every axis.

### Task A, full fine-tune, calibrated test AUROC (P2)

| Encoder | label_ip | label_cost |
|---|---|---|
| v1.0 (C0) | 0.669 | 0.715 |
| C1 | 0.662 | 0.716 |
| C2 | 0.666 | 0.713 |
| C3 | 0.663 | 0.714 |

### Task B single-sequence, pretrained-full LE (test AUPRC; 10/25% = 3-seed mean)

| Encoder | 10% | 25% | 100% |
|---|---|---|---|
| v1.0 (C0) | 0.623 | 0.679 | 0.695 |
| C1 | 0.576 | 0.637 | 0.630 |
| C2 | 0.541 | 0.671 | 0.681 |
| C3 | 0.494 | 0.682 | 0.704 |
| XGBoost (frozen baseline) | 0.594 | 0.637 | 0.711 |

### Hierarchical Task B (all claims, gated-attention MIL; test AUPRC)

| Arm | @C0 encoder | @C3 encoder (capstone) |
|---|---|---|
| full, 100% | 0.652 [0.536, 0.761] | **0.714** [0.620, 0.802] |
| scratch, 100% | 0.714 [0.625, 0.802] | 0.533 [0.427, 0.647] |
| probe, 100% | 0.599 | 0.539 |
| full, 10% (3-seed) | 0.631 (±0.021) | 0.421 |
| full, 25% (3-seed) | 0.681 (±0.017) | 0.653 |
| P@50 (full) | 0.78 | 0.82 |

Single-run 100%-label AUPRCs on an 812-provider test set carry ±0.09–0.12
bootstrap CIs; the @C0 full-vs-scratch inversion and the @C0-vs-@C3 full gap
are both inside overlapping CIs. The registered directional call (@C0 full
> 0.695) is nonetheless scored as refuted — the prereg exists precisely so
noisy misses get reported as misses.

## Test-touch accounting

Exactly the registered list, once each: Task A full ×{C1,C2,C3} (2 labels);
Task B single-seq LE evals ×{C1,C2,C3}; hier-full + hier-scratch LE +
hier-probe@100% at the C0 encoder and at the best scaled cell. Best-cell
selection followed the registered rule (highest single-seq full_1.0 **val**
AUPRC: C3 0.6777 vs C2 0.6597, C1 0.6500). Baselines never recomputed.

## Budget ledger (honest version)

Prereg point estimate $7, ceiling $10. **Actual: $13.39** (final; instance
destroyed 2026-07-25, remaining credit $7.13). Compute was *under* estimate
(~$5.5: pretrains ≈ $3.6 with C2 finishing 2× faster than projected;
fine-tune phases ≈ $1.9); the overrun is ~$7.9 of **idle-instance burn**
when the orchestrating session died overnight after the last GPU run and
teardown never fired. Lesson recorded: teardown belongs in the run chain
(`train && … && vastai destroy`), not in a supervisor that can die.

## Artifacts

- Per-cell eval reports: `task_a_transformer_{17m18s,46m18s,46m5s}.md`,
  `task_b_transformer_{17m18s,46m18s,46m5s}.md`, `task_b_transformer_hier.md`
  (@C0), `task_b_transformer_hier_46m5s.md` (capstone) + matching
  `metrics_*.json`; P1 scoring in `p1_masked_val_scoring.json`.
- Checkpoints: C1 and C3 (the capstone encoder) archived locally,
  sha-verified via split-part manifests. C2's checkpoint was the one casualty
  of the flaky instance link; it is regenerable in-distribution from the
  committed config + seed + frozen pack (~$1.5). All probability parquets,
  training metrics (`metrics.jsonl` with the new cumulative `tokens_seen`),
  and run metas are archived for every registered run.
