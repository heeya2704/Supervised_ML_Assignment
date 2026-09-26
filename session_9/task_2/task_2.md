# Session 9 - Task 2: Lasso Regression (L1 Regularization)

## Task Overview
Fit a **Lasso Regression (L1 Regularization)** model on the standardized Zomato dataset and inspect feature coefficients.

$$\text{Loss}_{\text{Lasso}} = \frac{1}{2N} \sum_{i=1}^{N} (y_i - \hat{y}_i)^2 + \alpha \sum_{j=1}^{p} |w_j|$$

---

## Model Hyperparameters & Setup
- **Algorithm**: `sklearn.linear_model.Lasso`
- **Regularization Strength ($\alpha$)**: `0.03`
- **Preprocessing**: `StandardScaler()` applied to feature matrix $X$
- **Intercept ($b$)**: `4.4272`

---

## Feature Coefficients Table

| Feature Name | Scaled Lasso Coefficient | Status |
|---|---|---|
| `location_score` | **+0.308351** | Retained (Strong positive predictor) |
| `votes` | **+0.180861** | Retained (Moderate positive predictor) |
| `book_table` | **+0.104535** | Retained |
| `restaurant_age_yrs` | **+0.093358** | Retained |
| `approx_cost_for_two` | **+0.035728** | Retained |
| `online_order` | **+0.029430** | Retained |
| `avg_dish_price` | **+0.022775** | Retained |
| `delivery_time_min` | **-0.005256** | Retained |
| `cuisine_count` | **-0.003361** | Retained |
| `parking_available` | **0.000000** | **ZEROED OUT** |
| `noise_feature_1` | **0.000000** | **ZEROED OUT** |
| `noise_feature_2` | **0.000000** | **ZEROED OUT** |
| `noise_feature_3` | **0.000000** | **ZEROED OUT** |

---

## Key Observation
- The L1 penalty forces the coefficients of uninformative/noisy features (`noise_feature_1`, `noise_feature_2`, `noise_feature_3`) and weak predictors (`parking_available`) **exactly to zero**.
- Lasso inherently acts as an **automatic feature selector**.
