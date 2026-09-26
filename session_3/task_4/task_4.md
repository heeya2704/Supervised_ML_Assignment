# Session 3 - Task 4: One-Hot Encoding via Pandas `get_dummies()`

## Task Overview
Take the `'venue'` column (categorical) from your IPL dataset and apply one-hot encoding using pandas' `get_dummies()`. Show the resulting DataFrame with the new columns.

---

## Code Implementation

```python
import pandas as pd

df = pd.read_csv('../dataset/ipl_player_stats.csv')

# Apply one-hot encoding on 'venue' column
df_encoded = pd.get_dummies(df, columns=['venue'], prefix='venue', dtype=int)

# Display player_name along with newly created binary venue dummy columns
venue_cols = [col for col in df_encoded.columns if col.startswith('venue_')]
print(df_encoded[['player_name'] + venue_cols].head())
```

---

## Created Dummy Columns & DataFrame Matrix

### Unique Venues Encoded
1. `venue_Arun Jaitley`
2. `venue_Chepauk`
3. `venue_Chinnaswamy`
4. `venue_Eden Gardens`
5. `venue_Ekana`
6. `venue_Narendra Modi Stadium`
7. `venue_Rajiv Gandhi Stadium`
8. `venue_Sawai Mansingh`
9. `venue_Wankhede`

### Encoded Matrix Output (First 6 Rows)

| Index | `player_name` | `venue_Chepauk` | `venue_Chinnaswamy` | `venue_Ekana` | `venue_Narendra Modi Stadium` | `venue_Wankhede` | ... |
|---|---|---|---|---|---|---|---|
| 0 | Virat Kohli | 0 | **1** | 0 | 0 | 0 | ... |
| 1 | Rohit Sharma | 0 | 0 | 0 | 0 | **1** | ... |
| 2 | MS Dhoni | **1** | 0 | 0 | 0 | 0 | ... |
| 3 | KL Rahul | 0 | 0 | **1** | 0 | 0 | ... |
| 4 | Hardik Pandya | 0 | 0 | 0 | **1** | 0 | ... |
| 5 | Jasprit Bumrah | 0 | 0 | 0 | 0 | **1** | ... |

---

## Why One-Hot Encoding?
One-Hot Encoding converts non-ordinal categorical values (like venue location names) into $K$ binary indicator features ($0$ or $1$). This prevents machine learning models from misinterpreting arbitrary categorical text as ordinal numeric ranks.
