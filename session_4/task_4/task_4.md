# Session 4 - Task 4: Recursive Feature Elimination (RFE) on Zomato Dataset

## Task Overview
Take a Zomato restaurant dataset and use scikit-learn's `RFE` (Recursive Feature Elimination) with a `DecisionTreeRegressor` to select the 3 most important features for predicting `'average_cost_for_two'`. List the selected features and the RFE ranking.

---

## Code Implementation

```python
import pandas as pd
from sklearn.feature_selection import RFE
from sklearn.tree import DecisionTreeRegressor

df = pd.read_csv('../dataset/zomato_restaurants.csv')
X = df.drop(columns=['average_cost_for_two'])
y = df['average_cost_for_two']

# RFE with DecisionTreeRegressor
estimator = DecisionTreeRegressor(random_state=42)
rfe = RFE(estimator=estimator, n_features_to_select=3)
rfe.fit(X, y)

# Output RFE support & ranking
results = pd.DataFrame({
    'Feature': X.columns,
    'Selected': rfe.support_,
    'RFE_Rank': rfe.ranking_
}).sort_values(by='RFE_Rank')

print(results)
```

---

## Complete RFE Ranking Output Table

| Feature Name | Selected Status | RFE Rank (1 = Top Selected) |
|---|---|---|
| **`rating`** | **`True`** | **`1` (Selected)** |
| **`votes`** | **`True`** | **`1` (Selected)** |
| **`cuisine_variety`** | **`True`** | **`1` (Selected)** |
| `table_booking` | `False` | `2` |
| `location_score` | `False` | `3` |
| `online_order` | `False` | `4` |

---

## Top 3 Selected Features Summary
1. **`rating`** (Rank 1): Crucial predictor of pricing tier.
2. **`votes`** (Rank 1): Customer volume proxy heavily linked to restaurant scale and pricing.
3. **`cuisine_variety`** (Rank 1): Menu width directly drives expected meal cost for two.
