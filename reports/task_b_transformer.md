# Task B — cross-dataset transfer & label efficiency (M5)

Provider fraud on the Kaggle dataset, same committed provider splits as the
frozen baselines. Provider = one sequence of its claims (claim = `[VISIT]`
span of DX/PX codes) encoded by the DE-SynPUF-pretrained encoder; `[CLS]`
pooled. Selection/calibration on val; test scored once, here.

**Stated information asymmetry:** sequences truncate keep-most-recent at 512
tokens — 1,018 high-volume providers truncated; 47% of claims retained overall. The XGBoost baseline sees
all-claims aggregate features (including volume), so the baseline comparison
is not information-equal; the pretrained-vs-scratch comparison is (both arms
share the constraint).

## Headline comparison (test, prevalence 9.4%)

| Model | AUROC | AUPRC | P@50 | P@100 |
|---|---|---|---|---|
| XGBoost (baseline) | 0.952 [0.935, 0.967] | 0.711 [0.617, 0.796] | 76.0% [60.0%, 90.0%] | 54.0% [43.0%, 64.0%] |
| Logistic regression (baseline) | 0.961 [0.944, 0.974] | 0.749 [0.649, 0.829] | 78.0% [64.0%, 92.0%] | 59.0% [47.0%, 71.0%] |
| Pretrained, full fine-tune | 0.944 [0.923, 0.963] | 0.695 [0.594, 0.790] | 80.0% [62.0%, 90.0%] | 56.0% [44.0%, 66.0%] |
| From scratch | 0.941 [0.920, 0.960] | 0.677 [0.572, 0.772] | 78.0% [58.0%, 90.0%] | 52.0% [41.0%, 64.0%] |
| Hybrid: XGBoost + frozen embeddings (M5.5) | 0.957 [0.938, 0.973] | 0.718 [0.615, 0.822] | 82.0% [64.0%, 92.0%] | 58.0% [46.0%, 69.0%] |

Operating point (pretrained transformer, top 100): precision 56.0%, recall 73.7%.

## Label efficiency (test AUPRC; ± is std over subsample seeds)

| Labeled providers | XGBoost | Pretrained transformer | Transformer from scratch | Hybrid (XGB+emb) |
|---|---|---|---|---|
| 10% | 0.594 (±0.056) | 0.623 (±0.018) | 0.608 (±0.027) | 0.622 (±0.064) |
| 25% | 0.637 (±0.050) | 0.679 (±0.004) | 0.650 (±0.019) | 0.688 (±0.019) |
| 100% | 0.711 | 0.695 | 0.677 | 0.718 |

![money chart](figures/task_b_money_chart.png)


## Addendum — the hybrid follow-up (M5.5)

Question tested: does the encoder add signal a GBM can't already extract?
XGBoost was re-tuned with the identical M2 protocol on
[34 provider aggregates ⊕ 320-d frozen pretrained `[CLS]` embedding]
(`scripts/run_hybrid.py`; numbers in `reports/metrics_hybrid.json`).
At 25% of labels the hybrid reaches 0.688 mean test AUPRC (best single seed
0.712 ≈ XGBoost's full-label 0.711) vs 0.637 for XGBoost alone — the
embeddings carry transferable fraud signal a GBM can use. At full labels the
hybrid is a wash on AUPRC (+0.007) but posts the best precision@50 of any
model (82%). At 10% it matches the fine-tuned transformer's mean with much
higher variance — when labels are scarcest, fine-tuning the encoder remains
the most stable option.
