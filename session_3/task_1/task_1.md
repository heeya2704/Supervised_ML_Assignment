# Session 3 - Task 1: IPL Dataset Missing Values Identification

## Task Overview
Download a small sample of IPL match data (CSV) with some missing values in player stats. Load it using pandas and print the number of missing values in each column.

---

## Code Implementation & Output

```python
import pandas as pd

df = pd.read_csv('../dataset/ipl_player_stats.csv')
missing_per_column = df.isnull().sum()
print(missing_per_column)
```

### Missing Value Analysis Output

| Column Name | Missing Count (NaN) | Percentage Missing |
|---|---|---|
| `player_id` | 0 | 0.0% |
| `player_name` | 0 | 0.0% |
| `team` | **3** | **20.0%** |
| `player_role` | 0 | 0.0% |
| `matches` | 0 | 0.0% |
| `runs` | 0 | 0.0% |
| `wickets` | 0 | 0.0% |
| `strike_rate` | **2** | **13.3%** |
| `player_age` | **4** | **26.7%** |
| `venue` | 0 | 0.0% |
| **TOTAL** | **9** | — |

---

## Key Observation
Columns containing missing values:
1. **`player_age`** (4 missing entries) - Numerical feature
2. **`team`** (3 missing entries) - Categorical feature
3. **`strike_rate`** (2 missing entries) - Numerical feature
