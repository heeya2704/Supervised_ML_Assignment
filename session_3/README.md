# Session 3: Data Preprocessing & Categorical Encoding

This folder contains dataset creation scripts, code files, and markdown reports for all tasks in **Session 3**.

## Directory Structure
- [`dataset/`](dataset/): Contains `ipl_player_stats.csv`.
- [`task_1/`](task_1/): Identification & counting of missing values (`isnull().sum()`).
- [`task_2/`](task_2/): Imputing missing numerical values in `player_age` with median (`fillna()`).
- [`task_3/`](task_3/): Imputing missing categorical values in `team` with constant `'Unknown'`.
- [`task_4/`](task_4/): One-Hot Encoding categorical `venue` column using `pd.get_dummies()`.
- [`task_5/`](task_5/): Label Encoding `player_role` using `LabelEncoder` & inspecting `classes_` mapping.

## How to Run Python Scripts
```bash
python session_3/task_1/task_1.py
python session_3/task_2/task_2.py
python session_3/task_3/task_3.py
python session_3/task_4/task_4.py
python session_3/task_5/task_5.py
```
