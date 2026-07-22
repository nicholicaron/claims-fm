# Data documentation

## Sources & licensing

| Dataset | Role | Source | License / terms |
|---|---|---|---|
| CMS DE-SynPUF (2008–2010) | Pretraining (samples 3–7) + Task A (samples 1–2) | [CMS DE-SynPUF portal](https://www.cms.gov/data-research/statistics-trends-and-reports/medicare-claims-synthetic-public-use-files/cms-2008-2010-data-entrepreneurs-synthetic-public-use-file-de-synpuf) | Public, synthetic, no DUA required. Contains **no real beneficiary data**. |
| Kaggle Healthcare Provider Fraud | Task B | [Kaggle dataset](https://www.kaggle.com/datasets/rohitrox/healthcare-provider-fraud-detection-analysis) | CC0-1.0 (per Kaggle API at download); DE-SynPUF-derived synthetic data. |

Sample roles are declared in [configs/data.yaml](configs/data.yaml) and enforced by the
leakage test suite: **samples 1–2 are Task A eval-only and never enter the
pretraining corpus or vocabulary** (SPEC §3 clean-split rule). Per-file
checksums, byte sizes, row counts, and the URL each file was actually served
from are committed in [configs/data.lock.yaml](configs/data.lock.yaml).

## Provenance notes

- **Sample 1's 2010 beneficiary summary file is not correctly retrievable from
  the live CMS portal.** Every canonical URL 404s, and the portal's sample-1
  page links a file named `..._sample_20.zip`. Wayback Machine content digests
  show the archived sample-1 URL (digest `YVEQIMTO...`, stable 2013–2015)
  differs from the archived sample-20 URL (`UMPFSZFL...`), and today's live
  "sample_20" file is byte-identical to the *archived sample-1* content — CMS
  appears to serve sample 1's 2010 file under a sample-20 filename. We use the
  Wayback capture of the original sample-1 URL
  (snapshot `20141219132014`, sha256 `b28b8ac7ecc2...`). Authenticity checks:
  the zip's internal CSV is named `DE1_0_2010_Beneficiary_Summary_File_Sample_1.csv`,
  and 100% of its 112,754 beneficiary IDs appear in sample 1's 2008 and 2009
  files with strict year-over-year nesting (2010 ⊂ 2009 ⊂ 2008), consistent
  with mortality attrition and inconsistent with a different sample (samples
  are disjoint member populations). The cross-year nesting check runs for
  every sample in the test suite.
- **Carrier claims are deferred** (SPEC §7 M1 / §10): the largest files
  (~226 MB zipped per sample), 13 line items per claim, and not required for
  the v1 event vocabulary. Revisit only if ahead of schedule.
- **HCPCS codes are excluded from the event vocabulary**: ~tens of thousands
  of distinct codes concentrated in outpatient claims, absent from the Kaggle
  fraud schema, so they cannot transfer to Task B. Documented trade-off: some
  outpatient service granularity is lost for Task A.
- Sample-1 detail plots in the EDA notebook are orientation-only; no modeling
  decisions are taken from eval-sample data.

## DE-SynPUF raw data (downloaded 2026-07-21)

| Sample | Role | Beneficiary rows 08 / 09 / 10 | Inpatient claims | Outpatient claims | Drug events | Zips (MB) |
|---|---|---|---|---|---|---|
| 1 | eval_only | 116,352 / 114,538 / 112,754 | 66,773 | 790,790 | 5,552,421 | 153 |
| 2 | eval_only | 116,395 / 114,618 / 112,845 | 66,494 | 792,562 | 5,561,154 | 153 |
| 3 | pretrain | 116,390 / 114,644 / 112,812 | 66,672 | 792,415 | 5,557,147 | 153 |
| 4 | pretrain | 116,279 / 114,528 / 112,699 | 66,253 | 789,485 | 5,549,070 | 152 |
| 5 | pretrain | 116,364 / 114,539 / 112,687 | 66,414 | 790,538 | 5,549,634 | 152 |
| 6 | pretrain | 116,234 / 114,532 / 112,713 | 66,977 | 793,146 | 5,557,441 | 153 |
| 7 | pretrain | 116,352 / 114,569 / 112,747 | 66,791 | 791,916 | 5,560,085 | 153 |

Per-file sha256, byte sizes, serving URLs, and distinct-member counts:
[configs/data.lock.yaml](configs/data.lock.yaml). Cross-year beneficiary ID
nesting is 100% within every sample (2010 ⊂ 2009 ⊂ 2008 — attrition only).

Kaggle provider-fraud files: 5,410 labeled providers (train), 558k claims
across Train/Test inpatient + outpatient files, ~139k beneficiaries.

## Event sequence store (`data/processed/`)

| Store | Members | Events | Events/member mean · median · p95 | Median visits |
|---|---|---|---|---|
| `sequences_pretrain` (samples 3–7, 2008–2010) | 517,390 | 41,281,565 | 79.8 · 62 · 213 | 44 |
| `sequences_eval_only` (samples 1–2, 2008–2010) | 206,928 | 16,514,824 | 79.8 · 62 · 213 | 44 |
| `sequences_eval_only_window_2008_2009` (Task A observation) | 203,453 | 12,523,664 | 61.6 · 48 · 166 | 34 |

Pretrain token occurrences: 13.0M diagnosis (DX), 27.8M drug (RX), 0.48M
procedure (PX). Members with zero claims in 2008–2010 carry no sequence
(~11% of beneficiaries).

Event-date bounds: a ~0.03% tail of IP/OP events dates to Nov–Dec 2007
(claims that opened before the study window and ran into it); drug events are
strictly within 2008–2010. The Task A window filter (`--window 2008:2009`)
excludes pre-2008 dates by year filtering — conservative and leakage-safe.

## Vocabulary (`vocab.json`)

- **28,203 tokens** = 10,742 DX + 2,456 PX + 15,000 RX + 5 specials
  (`[PAD] [UNK] [MASK] [CLS] [VISIT]`).
- Built from **pretrain samples 3–7 only** (140,652 distinct tokens seen).
- Frequency floor **10** (sweep of full-vocab sizes at floors 5/10/25:
  131,926 / 126,407 / 118,201 — the floor barely trims because of the NDC
  profile below).
- **Decision — NDC truncated to 9-digit labeler+product** at event
  extraction: full NDC-11 yields ~278k distinct drug tokens.
- **Decision — RX capped at top-15,000 by frequency** (52.5% of RX
  occurrences; the remainder encode as `[UNK]`): DE-SynPUF synthesizes NDCs
  with a near-flat frequency profile (~121k distinct NDC-9; top 5k cover only
  32.5% of occurrences, unlike real dispensing data), and Kaggle has no drug
  data, so RX granularity cannot transfer to Task B. An uncapped RX vocab
  would put ~30M parameters into embeddings alone, breaking the 10–25M model
  budget for signal that is largely synthesis noise.
- **Decision — Kaggle literal `"NA"` strings are nulls.** The Kaggle CSVs
  encode missing codes as `"NA"`; treating them as tokens made 70% of
  apparent dx occurrences placeholder junk (first gate run measured 30%
  coverage with 90% type coverage — that inversion was the tell).

## Kaggle ↔ DE-SynPUF vocabulary overlap (M1 gate: ≥ ~60%)

**PASS — 99.9% of Kaggle dx-code occurrences** (2,085,893 real occurrences,
11,227 distinct codes) are covered by the pretraining vocabulary; 92.1% of
procedure occurrences. The top-frequency dx codes match DE-SynPUF's top codes
almost rank-for-rank (4019, 25000, 2724, V5869, …), empirically confirming
the shared ICD-9 lineage. Details + floor sensitivity:
[reports/vocab_overlap.md](reports/vocab_overlap.md).

Known Kaggle data-quality caveat: `ClmProcedureCode_*` columns were written
as floats by the dataset author, so procedure codes lost leading zeros
irrecoverably (e.g. `0066` → `66`); this affects a portion of the 7.9%
uncovered px occurrences. Diagnosis codes are strings and unaffected.
