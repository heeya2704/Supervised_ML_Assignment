# Session 4 - Task 1: Standard Scaling on IPL Stats (`StandardScaler`)

## Task Overview
Download a small dataset of IPL cricket player stats (e.g., runs, wickets, matches, strike rate) and apply `StandardScaler` from scikit-learn to scale all numeric features. Save the scaled data to a new CSV file named [`ipl_scaled.csv`](ipl_scaled.csv).

---

## Code Implementation

```python
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Load dataset
df = pd.read_csv('../dataset/ipl_stats.csv')
numeric_cols = ['runs', 'wickets', 'matches', 'strike_rate']

# Apply StandardScaler (z-score standardization)
scaler = StandardScaler()
df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

# Save scaled dataset
df.to_csv('ipl_scaled.csv', index=False)
```

---

## Scaled Data Output Table (First 5 Players)

| `player` | `runs` (Scaled) | `wickets` (Scaled) | `matches` (Scaled) | `strike_rate` (Scaled) |
|---|---|---|---|---|
| Player_1 | -1.3890 | +1.5663 | -0.3595 | +1.2332 |
| Player_2 | +0.9043 | -0.8220 | +1.1801 | -0.8922 |
| Player_3 | +0.8213 | +1.0091 | -1.2604 | +0.4272 |
| Player_4 | +0.8036 | -1.1603 | +0.6069 | -0.7191 |
| Player_5 | +0.0852 | +1.6260 | -0.0975 | -0.0386 |

---

## Standardization Verification
$$z = \frac{x - \mu}{\sigma}$$
* **Transformed Feature Mean ($\mu$):** **`0.0000`**
* **Transformed Feature Standard Deviation ($\sigma$):** **`1.0000`**
* **Output File Location:** [`ipl_scaled.csv`](ipl_scaled.csv)
