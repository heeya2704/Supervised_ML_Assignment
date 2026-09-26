# Session 4 - Task 2: Flipkart Product Feature Scaling via `MinMaxScaler`

## Task Overview
Using a dataset of Flipkart product ratings (features: `price`, `rating`, `number_of_reviews`, `discount`), apply `MinMaxScaler` so that all features are scaled between $0$ and $1$. Print the minimum and maximum value for each column before and after scaling.

---

## Code Implementation

```python
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

df = pd.read_csv('../dataset/flipkart_products.csv')
features = ['price', 'rating', 'number_of_reviews', 'discount']

# Fit & transform MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))
df_scaled = df.copy()
df_scaled[features] = scaler.fit_transform(df[features])
```

---

## Before & After Feature Min/Max Comparison

$$x_{\text{scaled}} = \frac{x - x_{\min}}{x_{\max} - x_{\min}}$$

| Feature Name | Raw Min (Before) | Raw Max (Before) | Scaled Min (After) | Scaled Max (After) |
|---|---|---|---|---|
| **`price`** | 2867.00 | 87837.00 | **0.0000** | **1.0000** |
| **`rating`** | 2.40 | 4.80 | **0.0000** | **1.0000** |
| **`number_of_reviews`** | 216.00 | 14830.00 | **0.0000** | **1.0000** |
| **`discount`** | 10.00 | 70.00 | **0.0000** | **1.0000** |

---

## Summary
`MinMaxScaler` maps all feature values bounded strictly into the range $[0, 1]$. This prevents features with large numeric scales (like product `price` in ₹ or `number_of_reviews` in thousands) from dominating algorithms sensitive to feature distance (like KNN, K-Means, or Gradient Descent).
