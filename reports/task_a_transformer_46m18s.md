# Task A — transformer vs baselines (46m18s)

Same cohort, same committed splits, same observation-window inputs as the
frozen baselines (tag `v0.2-baselines`). Selection and calibration on val;
test scored once, here. 95% CIs: member-level bootstrap. Prediction head
reads the final-layer `[CLS]` state (design choice; no pooling search).

## any inpatient admission in 2010 (test prevalence 11.6%)

| Model | AUROC | AUPRC | Brier | ECE | Capture@1% | Capture@5% | Capture@10% |
|---|---|---|---|---|---|---|---|
| XGBoost (baseline) | 0.710 [0.699, 0.721] | 0.218 [0.205, 0.232] | 0.097 [0.094, 0.100] | 0.0053 | 3.1% [2.4%, 3.6%] | 12.8% [11.5%, 13.9%] | 22.5% [20.8%, 24.2%] |
| Logistic regression (baseline) | 0.692 [0.681, 0.703] | 0.201 [0.189, 0.213] | 0.098 [0.094, 0.101] | 0.0034 | 2.3% [2.1%, 3.3%] | 11.3% [9.9%, 12.4%] | 20.4% [18.9%, 22.0%] |
| Pretrained, full fine-tune **←** | 0.666 [0.655, 0.678] | 0.184 [0.173, 0.197] | 0.099 [0.096, 0.103] | 0.0027 | 2.2% [1.7%, 2.8%] | 11.1% [9.7%, 12.1%] | 19.5% [18.1%, 21.2%] |

Best transformer (Pretrained, full fine-tune) AUROC 0.666 vs XGBoost 0.710; calibrated with isotonic (ECE 0.0027).

### Subgroup slices (best transformer, calibrated)

| Field | Group | n | Prevalence | Mean predicted | AUROC | ECE |
|---|---|---|---|---|---|---|
| sex | 1 | 7,468 | 11.4% | 11.3% | 0.666 | 0.0058 |
| sex | 2 | 9,639 | 11.7% | 11.7% | 0.666 | 0.0005 |
| race | 1 | 14,347 | 11.9% | 11.6% | 0.661 | 0.0032 |
| race | 2 | 1,691 | 9.6% | 11.2% | 0.683 | 0.0164 |
| race | 3 | 705 | 11.3% | 10.2% | 0.736 | 0.0174 |
| race | 5 | 364 | 9.3% | 10.8% | 0.624 | 0.0180 |
| age_band | 65-69 | 3,019 | 10.8% | 10.7% | 0.691 | 0.0038 |
| age_band | 70-74 | 3,374 | 10.4% | 11.0% | 0.653 | 0.0064 |
| age_band | 75-79 | 2,915 | 11.8% | 11.5% | 0.679 | 0.0114 |
| age_band | 80-84 | 2,455 | 12.7% | 11.9% | 0.652 | 0.0078 |
| age_band | 85-199 | 2,767 | 13.4% | 12.6% | 0.663 | 0.0096 |
| age_band | <65 | 2,577 | 10.9% | 11.8% | 0.642 | 0.0105 |

![reliability](figures/task_a_tf_label_ip_reliability_46m18s.png)
![capture](figures/task_a_tf_label_ip_capture_46m18s.png)

## top-decile 2010 total cost (test prevalence 10.0%)

| Model | AUROC | AUPRC | Brier | ECE | Capture@1% | Capture@5% | Capture@10% |
|---|---|---|---|---|---|---|---|
| XGBoost (baseline) | 0.762 [0.751, 0.773] | 0.303 [0.283, 0.324] | 0.080 [0.077, 0.083] | 0.0066 | 7.7% [6.8%, 8.5%] | 20.5% [18.7%, 21.8%] | 29.9% [28.2%, 32.0%] |
| Logistic regression (baseline) | 0.743 [0.733, 0.754] | 0.269 [0.250, 0.289] | 0.082 [0.079, 0.085] | 0.0058 | 6.6% [5.8%, 7.4%] | 18.1% [16.6%, 19.5%] | 27.4% [25.9%, 29.4%] |
| Pretrained, full fine-tune **←** | 0.713 [0.700, 0.724] | 0.247 [0.230, 0.267] | 0.083 [0.080, 0.087] | 0.0026 | 6.4% [5.4%, 7.2%] | 16.9% [15.5%, 18.6%] | 27.1% [25.2%, 28.9%] |

Best transformer (Pretrained, full fine-tune) AUROC 0.713 vs XGBoost 0.762; calibrated with isotonic (ECE 0.0026).

### Subgroup slices (best transformer, calibrated)

| Field | Group | n | Prevalence | Mean predicted | AUROC | ECE |
|---|---|---|---|---|---|---|
| sex | 1 | 7,468 | 9.6% | 9.9% | 0.711 | 0.0048 |
| sex | 2 | 9,639 | 10.3% | 10.0% | 0.714 | 0.0038 |
| race | 1 | 14,347 | 10.2% | 10.0% | 0.707 | 0.0031 |
| race | 2 | 1,691 | 9.0% | 9.9% | 0.756 | 0.0103 |
| race | 3 | 705 | 9.6% | 8.6% | 0.759 | 0.0146 |
| race | 5 | 364 | 8.2% | 8.6% | 0.652 | 0.0282 |
| age_band | 65-69 | 3,019 | 8.7% | 8.9% | 0.727 | 0.0065 |
| age_band | 70-74 | 3,374 | 8.7% | 9.4% | 0.696 | 0.0069 |
| age_band | 75-79 | 2,915 | 10.8% | 10.0% | 0.721 | 0.0088 |
| age_band | 80-84 | 2,455 | 11.7% | 10.6% | 0.695 | 0.0161 |
| age_band | 85-199 | 2,767 | 11.6% | 11.3% | 0.724 | 0.0078 |
| age_band | <65 | 2,577 | 8.9% | 9.6% | 0.695 | 0.0128 |

![reliability](figures/task_a_tf_label_cost_reliability_46m18s.png)
![capture](figures/task_a_tf_label_cost_capture_46m18s.png)

