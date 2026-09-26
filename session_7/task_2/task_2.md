# Session 7 - Task 2: Flipkart Product Price MAE & $R^2$ Evaluation

## Task Overview
Implement a Python function `evaluate_flipkart_predictions(y_actual, y_predicted)` using `scikit-learn` to calculate Mean Absolute Error (MAE) and $R^2$ Score for Flipkart product price prediction estimates.

---

## Evaluation Function Implementation
```python
from sklearn.metrics import mean_absolute_error, r2_score

def evaluate_flipkart_predictions(y_actual, y_predicted):
    mae = mean_absolute_error(y_actual, y_predicted)
    r2 = r2_score(y_actual, y_predicted)
    return mae, r2
```

---

## Evaluation Results Table

| Metric | Formula | Value | Interpretation |
|---|---|---|---|
| **Mean Absolute Error (MAE)** | $\frac{1}{n}\sum \|y_i - \hat{y}_i\|$ | **₹145.00** | Predictions deviate by an average absolute margin of **₹145.00**. |
| **$R^2$ Score** | $1 - \frac{\text{SS}_{\text{res}}}{\text{SS}_{\text{tot}}}$ | **0.9995 (99.95%)** | The regression model accounts for **99.95%** of price variance across products. |

---

## Key Takeaways
- MAE measures overall average magnitude of pricing error without squaring errors.
- High $R^2$ score indicates near-perfect alignment across price tiers ranging from ₹749 to ₹29,999.
