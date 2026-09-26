# Session 13: Random Forest Ensemble Learning & Bagging

This folder contains Python scripts, datasets, and detailed markdown write-ups for all tasks in **Session 13**.

## Directory Structure
- [`dataset/`](dataset/): Contains `spotify_hits.csv`.
- [`task_1/`](task_1/): Default `RandomForestClassifier` (100 trees) trained on Iris dataset.
- [`task_2/`](task_2/): Flipkart 20-product category classifier (`Electronics`, `Fashion`, `Home`) with feature importance metrics.
- [`task_3/`](task_3/): Grid hyperparameter tuning across `n_estimators` $\in \{10, 50, 100\}$ and `max_depth` $\in \{2, 4, 6\}$.
- [`task_4/`](task_4/): Overfitting mitigation demonstration comparing a Single Decision Tree vs Random Forest on noisy label data.
- [`task_5/`](task_5/): Spotify hit predictor using Random Forest with Out-of-Bag (OOB) score evaluation and AI fix notes.

## How to Run Python Scripts
Run individual task scripts from the workspace root:
```bash
python session_13/task_1/task_1.py
python session_13/task_2/task_2.py
python session_13/task_3/task_3.py
python session_13/task_4/task_4.py
python session_13/task_5/task_5.py
```
