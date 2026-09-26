# Session 8 - Task 4: L2 Regularization (Ridge Regression) on Degree-3 Polynomial Model

## Task Overview
Evaluate the effect of **L2 Regularization (Ridge Regression)** vs. an **Unregularized Degree-3 Polynomial Regression** model on synthetic Flipkart product price dataset.

---

## Experimental Setup
- **Dataset Size**: 150 product samples
- **Features**: `spec_score` (1-10 scale), `review_count` (100-5000 range), `product_rating` (3.0-5.0 scale)
- **Target**: Flipkart Product Price (Rs.) with non-linear relationships and Gaussian noise
- **Model Architecture**:
  1. `PolynomialFeatures(degree=3)` $\rightarrow$ `StandardScaler()` $\rightarrow$ `LinearRegression()`
  2. `PolynomialFeatures(degree=3)` $\rightarrow$ `StandardScaler()` $\rightarrow$ `Ridge(alpha=10.0)`

---

## Results & Performance Metrics

| Metric / Model Parameter | Unregularized Polynomial (Degree=3) | Ridge L2 Regularized ($\alpha=10.0$) |
|---|---|---|
| **Train RMSE** | Rs. 2,809.26 | Rs. 3,392.78 |
| **Test RMSE** | Rs. 3,483.03 | Rs. 3,535.80 |
| **Test $R^2$ Score** | 0.9786 | 0.9779 |
| **L2 Norm of Coefficients ($\|w\|_2$)** | **70,718.11** | **10,570.50** |

---

## Key Insights & Theoretical Rationale

1. **Coefficient Shrinkage ($\|w\|_2$)**:
   - The unregularized model exhibits an exceptionally high coefficient norm ($\|w\|_2 = 70,718.11$) due to multicollinearity between polynomial interaction terms ($x^3, x^2y, xy^2, \dots$).
   - Ridge regression successfully constrains coefficient magnitudes, reducing $\|w\|_2$ down to **10,570.50** (a **~85% reduction in parameter magnitude**).

2. **Mitigating Overfitting**:
   - By penalizing large weights $\alpha \sum w_i^2$, Ridge prevents the model from overfitting to random high-frequency variance in training samples, promoting model stability and robustness.
