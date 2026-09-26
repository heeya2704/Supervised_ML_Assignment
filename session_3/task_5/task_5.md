# Session 3 - Task 5: Label Encoding `player_role` via Scikit-Learn `LabelEncoder`

## Task Overview
Use `LabelEncoder` from sklearn to encode the `'player_role'` column (e.g., Batsman, Bowler, Allrounder) in your IPL dataset, and display the mapping from original roles to encoded values.  
*(Hint: Use LabelEncoder's `classes_` attribute to see the mapping).*

---

## Code Implementation

```python
import pandas as pd
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv('../dataset/ipl_player_stats.csv')

# Initialize LabelEncoder
le = LabelEncoder()
df['player_role_encoded'] = le.fit_transform(df['player_role'])

# Display mapping using le.classes_
mapping = dict(zip(le.classes_, range(len(le.classes_))))
print("Role Mapping:", mapping)
print(df[['player_name', 'player_role', 'player_role_encoded']].head(10))
```

---

## Scikit-Learn `LabelEncoder` Class Mapping

Extracted directly using `le.classes_`:

| Original `player_role` Category | Encoded Integer Code | Array Index in `le.classes_` |
|---|---|---|
| **Allrounder** | **`0`** | `classes_[0]` |
| **Batsman** | **`1`** | `classes_[1]` |
| **Bowler** | **`2`** | `classes_[2]` |
| **Wicketkeeper** | **`3`** | `classes_[3]` |

---

## Encoded Player Output Table (First 10 Rows)

| Index | Player Name | Original Role | Encoded Integer Code |
|---|---|---|---|
| 0 | Virat Kohli | Batsman | **1** |
| 1 | Rohit Sharma | Batsman | **1** |
| 2 | MS Dhoni | Wicketkeeper | **3** |
| 3 | KL Rahul | Batsman | **1** |
| 4 | Hardik Pandya | Allrounder | **0** |
| 5 | Jasprit Bumrah | Bowler | **2** |
| 6 | Ravindra Jadeja | Allrounder | **0** |
| 7 | Rishabh Pant | Wicketkeeper | **3** |
| 8 | Shubman Gill | Batsman | **1** |
| 9 | Suryakumar Yadav | Batsman | **1** |
