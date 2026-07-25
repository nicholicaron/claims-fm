# What a small pretrained claims encoder actually buys a payer

*~1,500 words. All numbers are held-out test results with member- or
provider-level bootstrap CIs; every test set was scored exactly once. Repo:
claims-fm; total compute $1.14.*

## The question

Payers sit on claims sequences, not clinical notes. The healthcare
foundation-model literature — Med-BERT, CLMBR, MOTOR — says pretraining on
those sequences yields representations that transfer across downstream
tasks. Those results come from large real-data cohorts and institutional
compute. I wanted the one-person, twenty-dollar version of a narrower,
more decision-shaped question: **for the two problems a payer actually pays
for — care-management targeting and payment integrity — when does a
pretrained claims encoder beat well-engineered gradient boosting, and when
doesn't it?**

So: pretrain a 17M-parameter BERT-style encoder on 517,390 synthetic
Medicare members (CMS DE-SynPUF, 41M coded events) with masked-code
modeling; fine-tune it twice. Task A: predict next-year inpatient admission
and top-decile cost from a two-year observation window. Task B: transfer
the encoder — unchanged vocabulary, different dataset — to flag potentially
fraudulent providers in the Kaggle provider-fraud data, which shares
DE-SynPUF's ICD-9 lineage (measured: 99.9% of Kaggle diagnosis occurrences
covered by the pretraining vocabulary, verified as a gate before any GPU
spend).

## The discipline is the credibility

Every conclusion below rests on protocol built before any transformer
existed. Logistic regression and tuned XGBoost baselines (40-draw search on
engineered features) were fit and **frozen first** — committed under a tag
before pretraining began. Leakage rules are a test suite, not a paragraph:
pretraining samples are disjoint from evaluation samples by construction;
observation windows are enforced by a builder that raises on out-of-window
events; split assignments are content-hashed and every later stage verifies
the hash. Calibration is reported everywhere and recalibration is fit on
validation only. Fraud metrics are capacity-shaped — precision at an SIU
caseload, not just AUROC. And a ledger: $0.52 pretraining, $0.32 Task A,
$0.30 Task B, $0 for the final ablation.

## Result 1: the GBM wins member risk — and we measured why

On Task A, tuned XGBoost beats every transformer variant: AUROC 0.710 vs
0.669 on admissions, 0.762 vs 0.715 on top-decile cost, non-overlapping
CIs. We would ship the GBM.

The interesting part is the diagnosis. DE-SynPUF's synthesis preserves the
aggregate signal engineered features consume — prior-year cost levels,
chronic-condition flags, utilization counts — while heavily weakening the
fine-grained sequential structure a transformer needs (visible upstream:
masked-code accuracy tops out at 15.8% against the 40–60% real-EHR models
reach). To separate "the transformer is worse" from "the sequence channel
is empty," I ran the obvious rebuttal as an experiment: re-tune XGBoost
with the frozen encoder's 320-d member embedding appended to the 159
engineered features, identical search protocol. Result: **identical to
features-alone** (0.763 vs 0.762 AUROC on cost). On this synthetic data the
encoder contains no predictive signal the features don't already carry. The
ceiling is measured, not assumed — and on real claims, where sequence
structure survives, that hybrid test is the first cheap experiment to
rerun.

Calibration earned its keep here: the tuned admission model's
class-imbalance weighting produced badly distorted probabilities (ECE
0.32). Isotonic recalibration fit on validation restored them (ECE 0.005,
reliability diagrams before/after). Payers act on probabilities — a model
that ranks well but lies about risk levels mis-sizes every outreach list.

## Result 2: pretraining transfers

The control for everything above is an identical architecture trained from
scratch at the same budget. It loses to the pretrained encoder everywhere
it was tried: on both Task A labels (AUPRC 0.184 vs 0.170 admissions,
0.246 vs 0.217 cost — the *frozen* pretrained probe matches or beats full
from-scratch training), and at every label fraction of Task B — a dataset
the encoder never saw during pretraining. Cross-dataset transfer through a
shared medical vocabulary is real, even at this scale, even on synthetic
data.

## Result 3: label efficiency is the money chart

Task B is where the thesis was staked, because it's where the operational
constraint bites: confirmed fraud labels are scarce and expensive —
SIU-investigated, adjudicated, slow. The label-efficiency grid fine-tunes
on 10%, 25%, and 100% of labeled providers (three subsample seeds each):

| Labels | XGBoost | Pretrained encoder | From scratch | Hybrid (XGB+emb) |
|---|---|---|---|---|
| 10% | 0.594 (±0.056) | 0.623 (±0.018) | 0.608 (±0.027) | 0.622 (±0.064) |
| 25% | 0.637 (±0.050) | **0.679 (±0.004)** | 0.650 (±0.019) | **0.688 (±0.019)** |
| 100% | 0.711 | 0.695 | 0.677 | **0.718** |

Below full labels the ordering is exactly the pretraining hypothesis:
pretrained > from-scratch > XGBoost. At 25% of labels the pretrained
encoder approaches what XGBoost needs the full label set to reach, and the
hybrid's best seed matches it outright. Stability is part of the result —
±0.004 vs ±0.050 across label subsamples. At 100% labels the honest
scoreboard has simple models on top (logistic regression 0.749) with the
hybrid posting the best precision@50 (82%): review the top-100 providers
and 58% are true flags at 74% recall.

The payment-integrity framing comes from my security background, and the
bridge is literal: provider billing anomalies are anomalous event streams,
the same detection posture as security telemetry — baseline the population,
rank deviations, and respect analyst capacity, which is why the operating
metric is precision at a caseload rather than a threshold-free curve.

## Equity slices

Discrimination and calibration are reported by sex, age band, and race for
both model families. Slices are broadly uniform except the smallest race
group (n=364), which shows the weakest AUROC and calibration in *both* the
GBM and the transformer — a data-representation issue, not an
architecture one. In production this argues for calibration monitoring per
subgroup, and for being explicit that calibration parity and equal
opportunity are different targets that can't generally both be satisfied;
which one governs is a policy decision, not a modeling one.

## Phase 2: does scale fix it? (pre-registered)

The natural objection to everything above is "your transformer was too small
and your corpus too thin." So we tested it, and pre-registered the test
([scaling_prereg.md](scaling_prereg.md), committed publicly before any
training): a 2×2 grid — 17M vs 46M parameters, 5 vs 18 DE-SynPUF samples
(517k → 1.86M members) — plus the one intervention aimed at *real* missing
information, a hierarchical variant that encodes all of a provider's claims
in 512-token chunks with gated-attention MIL pooling instead of truncating
at 512 tokens (v1.0 discarded 52.8% of claims).

Scored against the frozen predictions ([scaling_results.md](scaling_results.md)):

- **The MLM scaling curve is nearly flat.** 3.6× corpus and 2.8× params, in
  any combination, moved masked accuracy from 15.8% to at most 16.29%.
- **Task A did not move** — every cell within ±0.007 AUROC of v1.0, as
  predicted. The synthesis ceiling is now measured across the whole grid.
- **Scale actively hurt low-label fraud transfer** — the one outcome we
  didn't predict. At 10% labels, every scaled encoder underperforms the
  original (0.623 → 0.49–0.58 AUPRC). The 17M/5-sample configuration wasn't
  a compromise; it was the optimum.
- **Removing the truncation closed the volume gap — but so did an untrained
  encoder.** Four model families converge at the same 100%-label ceiling:
  XGBoost 0.711, hybrid 0.718, hierarchical-full 0.714, and
  hierarchical-*scratch* 0.714. At full labels the fraud ceiling on this
  dataset is claim volume plus code statistics, however you reach it.
  Pretraining's contribution stays where v1.0 found it: label efficiency
  (hier pretrained 0.631 vs hier scratch 0.414 at 10% labels).

Three of five predictions came back refuted — one on magnitude, one in the
*opposite* direction we hedged toward, and one (the hierarchical win at the
original encoder) as a plain miss, with the in-window capstone result scored
as an observation rather than a confirmation because the frozen prediction
didn't cover it. That's the point of pre-registering: the misses are
load-bearing evidence, and what they carry is the same conclusion as v1.0,
now measured rather than argued — on synthetic claims, the ceiling belongs
to the data.

## Limitations, stated plainly

**Synthetic data caps everything.** DE-SynPUF's synthesis weakens
code–outcome correlations, so absolute numbers understate real-data
performance and the Task A comparison in particular should be read as
methodology, not a verdict on transformers for risk stratification.
**ICD-9 era.** 2008–2010 data; migration is a vocabulary swap (GEMs
crosswalk to warm-start ICD-10 embeddings, or retrain — the tokenizer is
code-agnostic by design). **Constructed fraud labels.** Kaggle's provider
labels are heuristic, not SIU-adjudicated; results overstate what
confirmed-fraud labels would show, which strengthens rather than weakens
the label-scarcity argument. **Truncation.** Provider sequences cap at 512
tokens (47% of claims retained), so the transformer never sees volume — an
information asymmetry versus the baselines, stated in the report; the
pretrained-vs-scratch comparison shares the constraint and stands. *(Phase 2
removed this limitation with the hierarchical variant — and found the volume
signal it unlocked is worth ~0.02 AUPRC to any architecture that can see it,
pretrained or not; see above.)*

## With real member data, what changes

Vocabulary: ICD-10/HCPCS swap via GEMs warm-start, plus real NDC
hierarchies instead of synthesis-randomized codes. Pretraining objective:
next-visit prediction (à la MOTOR) over masked codes once sequences carry
real temporal signal. Features the synthetic data can't offer: dual
eligibility, SDOH joins, pharmacy adherence. Serving: monthly batch scoring
into care-management queues with per-cycle recalibration, drift monitors on
population mix, code mix, and calibration decay. And the first experiment
on day one: the hybrid test, because it cheaply answers whether the
sequence channel is worth a transformer at all on your data.

## Appendix: what didn't work (kept on purpose)

CMS's portal serves sample 1's 2010 beneficiary file under a sample-20
filename — recovered via Wayback Machine content digests and verified by
member-ID nesting. Kaggle encodes missing codes as the literal string
`"NA"`, which silently became 70% of "diagnosis occurrences" until the
inverted coverage statistics gave it away. Full-precision NDCs exploded the
vocabulary (278k tokens) and were truncated to 9 digits. Boolean-gather
loss tensors caused per-step graph recompilation stalls on MPS; the fix
(fixed-shape cross-entropy) also helps CUDA. A batch-cost model that
ignored pad quantization produced 8× oversized batches and an OOM on a
24GB card. `torch.compile` on a broken container toolchain stalls training
worse than eager mode ever could. An early evaluation slip compared
calibrated transformer probabilities against raw baseline ones —
caught and corrected to raw-vs-raw before freezing. Each of these is a
commit in the history rather than a footnote, because the debugging is part
of the work.
