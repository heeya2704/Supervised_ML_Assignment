# Session 3 - Task 2: Median Age Imputation using Pandas `fillna()`

## Task Overview
For the `'player_age'` column in your IPL dataset, fill all missing values with the median age using pandas' `fillna()` method and display the updated column.

---

## Code Implementation

```python
import pandas as pd

df = pd.read_csv('../dataset/ipl_player_stats.csv')

# Calculate median age
median_age = df['player_age'].median()
print(f"Calculated Median Age: {median_age}")

# Impute missing values with median
df['player_age'] = df['player_age'].fillna(median_age)

print(df[['player_name', 'player_age']])
```

---

## Imputation Results

* **Calculated Median Age:** **`31.0` years**

### Updated Column Table

| Index | Player Name | Original `player_age` | Imputed `player_age` | Status |
|---|---|---|---|---|
| 0 | Virat Kohli | 35.0 | 35.0 | Original |
| 1 | Rohit Sharma | 36.0 | 36.0 | Original |
| 2 | MS Dhoni | **NaN** | **31.0** | **Filled with Median** |
| 3 | KL Rahul | 31.0 | 31.0 | Original |
| 4 | Hardik Pandya | 30.0 | 30.0 | Original |
| 5 | Jasprit Bumrah | **NaN** | **31.0** | **Filled with Median** |
| 6 | Ravindra Jadeja | 35.0 | 35.0 | Original |
| 7 | Rishabh Pant | 26.0 | 26.0 | Original |
| 8 | Shubman Gill | 24.0 | 24.0 | Original |
| 9 | Suryakumar Yadav | **NaN** | **31.0** | **Filled with Median** |
| 10 | Shreyas Iyer | 29.0 | 29.0 | Original |
| 11 | Mohammed Shami | 33.0 | 33.0 | Original |
| 12 | Yuzvendra Chahal | **NaN** | **31.0** | **Filled with Median** |
| 13 | Sanju Samson | 29.0 | 29.0 | Original |
| 14 | Rashid Khan | 25.0 | 25.0 | Original |

---

## Why Median Imputation?
Median imputation is preferred over mean imputation for age data because median is robust against severe skewness and extreme age outliers (e.g. veteran players aged 42+).
