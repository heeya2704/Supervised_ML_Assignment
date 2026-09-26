# Session 9 - Task 3: Ridge (L2) vs Lasso (L1) Coefficient Comparison

## Task Overview
Compare the feature coefficients produced by **Lasso Regression (L1 Regularization)** and **Ridge Regression (L2 Regularization)** when trained on the same standardized dataset.

---

## Coefficient Comparison Matrix

| Feature Name | Lasso Coefficient (L1) | Ridge Coefficient (L2) | Behavior Comparison |
|---|---|---|---|
| `location_score` | **+0.308351** | **+0.324994** | Strong active predictor in both models |
| `votes` | **+0.180861** | **+0.208657** | Active predictor in both models |
| `book_table` | **+0.104535** | **+0.129800** | Active predictor in both models |
| `restaurant_age_yrs` | **+0.093358** | **+0.129106** | Active predictor in both models |
| `online_order` | **+0.029430** | **+0.065191** | Active predictor in both models |
| `approx_cost_for_two` | **+0.035728** | **+0.049625** | Active predictor in both models |
| `avg_dish_price` | **+0.022775** | **+0.041732** | Active predictor in both models |
| `delivery_time_min` | **-0.005256** | **-0.035100** | Active negative predictor |
| `cuisine_count` | **-0.003361** | **-0.028697** | Active negative predictor |
| `parking_available` | **0.000000** | **+0.004957** | **Lasso set to 0, Ridge non-zero** |
| `noise_feature_1` | **0.000000** | **+0.010094** | **Lasso set to 0, Ridge non-zero** |
| `noise_feature_2` | **0.000000** | **+0.006030** | **Lasso set to 0, Ridge non-zero** |
| `noise_feature_3` | **0.000000** | **-0.004503** | **Lasso set to 0, Ridge non-zero** |

---

## Key Takeaways
1. **Sparsity vs Shrinkage**:
   - **Lasso ($L_1$)**: The diamond contour of the $L_1$ penalty causes weight vectors to hit corner axes at zero, generating **sparse models** by discarding noisy features.
   - **Ridge ($L_2$)**: The spherical contour of the $L_2$ penalty shrinks weights proportionally towards zero but **never forces them to exactly zero**.
2. **Feature Retention**:
   - Features like `noise_feature_1`, `noise_feature_2`, `noise_feature_3`, and `parking_available` are eliminated by Lasso (`0.000000`) but retained with tiny non-zero weights by Ridge (`0.010094`, `0.006030`, `-0.004503`, `0.004957`).
