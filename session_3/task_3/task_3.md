# Session 3 - Task 3: Constant Value Categorical Imputation (`fillna('Unknown')`)

## Task Overview
Suppose the `'team'` column in your IPL dataset contains missing values. Replace all missing team names with the constant value `'Unknown'` and print the first 10 rows.

---

## Code Implementation

```python
import pandas as pd

df = pd.read_csv('../dataset/ipl_player_stats.csv')

# Impute missing categorical team names with constant value 'Unknown'
df['team'] = df['team'].fillna('Unknown')

# Print first 10 rows
print(df.head(10))
```

---

## First 10 Rows Output Table

| Index | `player_id` | `player_name` | `team` | `player_role` | `venue` |
|---|---|---|---|---|---|
| 0 | IPL_101 | Virat Kohli | RCB | Batsman | Chinnaswamy |
| **1** | **IPL_102** | **Rohit Sharma** | **Unknown** *(Was NaN)* | **Batsman** | **Wankhede** |
| 2 | IPL_103 | MS Dhoni | CSK | Wicketkeeper | Chepauk |
| 3 | IPL_104 | KL Rahul | LSG | Batsman | Ekana |
| 4 | IPL_105 | Hardik Pandya | GT | Allrounder | Narendra Modi Stadium |
| 5 | IPL_106 | Jasprit Bumrah | MI | Bowler | Wankhede |
| 6 | IPL_107 | Ravindra Jadeja | CSK | Allrounder | Chepauk |
| **7** | **IPL_108** | **Rishabh Pant** | **Unknown** *(Was NaN)* | **Wicketkeeper** | **Arun Jaitley** |
| 8 | IPL_109 | Shubman Gill | GT | Batsman | Narendra Modi Stadium |
| 9 | IPL_110 | Suryakumar Yadav | MI | Batsman | Wankhede |

---

## Technical Note
For categorical nominal variables like team names, imputing missing values with a designated constant category (like `'Unknown'`) preserves all other valid rows without dropping data or inventing inaccurate team assignments.
