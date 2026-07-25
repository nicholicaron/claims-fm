# Task B — cross-dataset transfer & label efficiency — hierarchical (Phase 2)

Provider fraud on the Kaggle dataset, same committed provider splits as the
frozen baselines. Hierarchical encoding (Phase 2): ALL of a provider's
claims in 512-token claim-aligned chunks, each encoded independently;
chunk `[CLS]` vectors pooled per provider with gated attention (MIL).
Selection on val; test scored once, here.

**No truncation:** 47% of claims retained (1,018 providers truncated). The v1.0
512-token information asymmetry vs the all-claims baselines is gone;
chunk count still implicitly encodes claim volume (as do the baselines'
volume features; the scratch arm shares the architecture).

## Headline comparison (test, prevalence 9.4%)

| Model | AUROC | AUPRC | P@50 | P@100 |
|---|---|---|---|---|
| XGBoost (baseline) | 0.952 [0.935, 0.967] | 0.711 [0.617, 0.796] | 76.0% [60.0%, 90.0%] | 54.0% [43.0%, 64.0%] |
| Logistic regression (baseline) | 0.961 [0.944, 0.974] | 0.749 [0.649, 0.829] | 78.0% [64.0%, 92.0%] | 59.0% [47.0%, 71.0%] |
| Pretrained, full fine-tune | 0.937 [0.915, 0.958] | 0.681 [0.578, 0.775] | 78.0% [64.0%, 88.0%] | 54.0% [43.0%, 64.0%] |
| Hybrid: XGBoost + frozen embeddings (M5.5) | 0.957 [0.938, 0.973] | 0.718 [0.615, 0.822] | 82.0% [64.0%, 92.0%] | 58.0% [46.0%, 69.0%] |

Operating point (pretrained transformer, top 100): precision 54.0%, recall 71.1%.

## Label efficiency (test AUPRC; ± is std over subsample seeds)

| Labeled providers | XGBoost | Pretrained transformer | Transformer from scratch | Hybrid (XGB+emb) |
|---|---|---|---|---|
| 10% | 0.594 (±0.056) | 0.541 (±0.044) | — | 0.622 (±0.064) |
| 25% | 0.637 (±0.050) | 0.671 (±0.010) | — | 0.688 (±0.019) |
| 100% | 0.711 | 0.681 | — | 0.718 |

![money chart](figures/task_b_money_chart_46m18s.png)

