# Session 13 - Task 3: Random Forest Hyperparameter Grid Tuning

## Task Overview
Tune `n_estimators` $\in \{10, 50, 100\}$ and `max_depth` $\in \{2, 4, 6\}$ for `RandomForestClassifier` on the Iris dataset via a nested search loop to identify the optimal parameter configuration.

---

## Grid Tuning Search Matrix

| `n_estimators` (Trees) | `max_depth` (Depth Limit) | Test Accuracy | Performance Ranking |
|---|---|---|---|
| **10** | **2** | **91.11%** | **Optimal (Best)** |
| **10** | **4** | **91.11%** | **Optimal (Best)** |
| **10** | **6** | **91.11%** | **Optimal (Best)** |
| **50** | **2** | **91.11%** | **Optimal (Best)** |
| 50 | 4 | 88.89% | Sub-optimal |
| 50 | 6 | 88.89% | Sub-optimal |
| **100** | **2** | **91.11%** | **Optimal (Best)** |
| 100 | 4 | 88.89% | Sub-optimal |
| 100 | 6 | 88.89% | Sub-optimal |

---

## Key Insights
- **Optimal Combination**: `n_estimators = 10` (or `100`), `max_depth = 2` yields top test accuracy (**91.11%**).
- **Over-parameterization Penalty**: Restricting `max_depth=2` acts as a strong regularizer that prevents deeper trees from fitting training sample noise.
