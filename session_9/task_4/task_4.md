# Session 9 - Task 4: ElasticNet Regression & l1_ratio Hyperparameter Effect

## Task Overview
Evaluate **ElasticNet Regression**, which combines both $L_1$ (Lasso) and $L_2$ (Ridge) penalties:

$$\text{Loss}_{\text{ElasticNet}} = \frac{1}{2N} \sum (y_i - \hat{y}_i)^2 + \alpha \cdot \left[ \rho \sum |w_j| + \frac{1 - \rho}{2} \sum w_j^2 \right]$$

where $\rho = \text{\texttt{l1\_ratio}}$.

---

## Coefficient Evolution Table Across `l1_ratio` ($\rho$)

| Feature | ElasticNet ($\rho=0.1$) | ElasticNet ($\rho=0.5$) | ElasticNet ($\rho=0.9$) | Pure Lasso ($\rho=1.0$) | Pure Ridge ($\rho=0.0$) |
|---|---|---|---|---|---|
| `location_score` | +0.31983 | +0.30587 | +0.29270 | +0.30835 | +0.32499 |
| `votes` | +0.20190 | +0.18186 | +0.16111 | +0.18086 | +0.20866 |
| `book_table` | +0.12473 | +0.10733 | +0.08728 | +0.10454 | +0.12980 |
| `restaurant_age_yrs` | +0.12197 | +0.09793 | +0.07195 | +0.09336 | +0.12911 |
| `online_order` | +0.05912 | +0.03474 | +0.00940 | +0.02943 | +0.06519 |
| `approx_cost_for_two` | +0.04605 | +0.03521 | +0.02656 | +0.03573 | +0.04963 |
| `avg_dish_price` | +0.03898 | +0.02771 | +0.01269 | +0.02277 | +0.04173 |
| `delivery_time_min` | -0.02961 | -0.00981 | 0.00000 | -0.00526 | -0.03510 |
| `cuisine_count` | -0.02406 | -0.00708 | 0.00000 | -0.00336 | -0.02870 |
| `parking_available` | 0.00000 | 0.00000 | 0.00000 | 0.00000 | +0.00496 |
| `noise_feature_1` | **+0.00620** | **0.00000** | **0.00000** | **0.00000** | **+0.01009** |
| `noise_feature_2` | **+0.00091** | **0.00000** | **0.00000** | **0.00000** | **+0.00603** |
| `noise_feature_3` | **-0.00189** | **0.00000** | **0.00000** | **0.00000** | **-0.00450** |

---

## Insights
1. **As $\rho \rightarrow 0$ (Ridge Limit)**: Features are retained with smooth shrinkage; noise features retain small non-zero values.
2. **As $\rho \rightarrow 1$ (Lasso Limit)**: Sparsity increases dramatically, zeroing out uninformative features (`noise_feature_1, 2, 3` and `parking_available`).
3. **Trade-off**: ElasticNet avoids Lasso's failure mode on highly correlated feature groups (where Lasso arbitrarily picks one feature) while keeping Lasso's feature selection ability.
