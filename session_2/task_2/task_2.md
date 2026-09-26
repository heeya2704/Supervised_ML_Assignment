# Session 2 - Task 2: Spotify Dataset Train-Test Split (80/20)

## Task Overview
Split the loaded Spotify dataset into training and test sets using sklearn's `train_test_split` function, with 80% for training and 20% for testing. Print the number of rows in each set.

---

## Implementation Code & Output

```python
import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv('../dataset/spotify_top_50.csv')
X = df.drop(columns=['track_id', 'track_name', 'artist_name', 'popularity'])
y = df['popularity']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

print(f"Total Rows : {len(df)}")
print(f"Train Rows : {len(X_train)} (80%)")
print(f"Test Rows  : {len(X_test)} (20%)")
```

### Output Results
* **Total Dataset Rows:** 50
* **Training Set Rows (80%):** 40 rows
* **Testing Set Rows (20%):** 10 rows

---

## Explanation
Splitting data into an 80% training set and 20% testing set is a standard machine learning practice. The training set is used to optimize model parameters ($\beta$ coefficients), while the unseen 20% test set evaluates how well the model generalizes to new song data.
