# Session 9 - Task 5: Zomato Restaurant Rating Feature Selector

## Task Overview
Build a feature selection module for a Zomato restaurant rating predictor using **Lasso Regression (L1 Regularization)** to identify essential predictors and eliminate irrelevant noise features.

---

## Feature Selection Results

### 1. Kept Features (Non-Zero Coefficients) - Total: 9
| Feature Name | Coefficient / Importance | Interpretation |
|---|---|---|
| `location_score` | **+0.308351** | Strongest positive driver (Prime location $\rightarrow$ higher rating) |
| `votes` | **+0.180861** | Strong positive driver (More customer engagement/reviews) |
| `book_table` | **+0.104535** | Positive driver (Table reservation feature) |
| `restaurant_age_yrs` | **+0.093358** | Positive driver (Established reputation) |
| `approx_cost_for_two` | **+0.035728** | Moderate positive driver |
| `online_order` | **+0.029430** | Moderate positive driver |
| `avg_dish_price` | **+0.022775** | Minor positive driver |
| `cuisine_count` | **-0.003361** | Minor negative driver |
| `delivery_time_min` | **-0.005256** | Minor negative driver (Longer delivery time drops rating) |

---

### 2. Dropped Features (Zero Coefficients) - Total: 4
| Feature Name | Status | Reason for Elimination |
|---|---|---|
| `parking_available` | **ELIMINATED (0.00)** | Weak predictive signal on overall restaurant rating |
| `noise_feature_1` | **ELIMINATED (0.00)** | Uninformative synthetic Gaussian noise |
| `noise_feature_2` | **ELIMINATED (0.00)** | Uninformative synthetic Uniform noise |
| `noise_feature_3` | **ELIMINATED (0.00)** | Uninformative synthetic Bernoulli noise |

---

## Conclusion & System Impact
Using Lasso as a feature selection front-end reduces dimensionality from **13 to 9 features**, stripping out 100% of synthetic noise variables and redundant metrics without loss of predictive power.
