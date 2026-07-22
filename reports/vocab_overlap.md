# Kaggle ↔ DE-SynPUF vocabulary overlap

Gate: ≥ 60% of Kaggle dx-code occurrences covered by the pretraining vocab (size 28203, floor 10).

**Result: PASS — 99.9% dx occurrence coverage**

| Metric | Diagnosis | Procedure |
|---|---|---|
| Occurrence-weighted coverage | 99.9% | 92.1% |
| Unique-type coverage | 90.0% | 64.9% |
| Kaggle distinct codes | 11,227 | 1,400 |
| Kaggle code occurrences | 2,085,893 | 36,882 |

Dx occurrence coverage at alternative vocab floors:

| min_count | coverage |
|---|---|
| 1 | 100.0% |
| 5 | 100.0% |
| 10 | 99.9% |
| 25 | 99.7% |
