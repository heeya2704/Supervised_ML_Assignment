# Session 19: Hyperparameter Tuning & Handling Class Imbalance

This folder contains Python scripts and detailed markdown write-ups for all tasks in **Session 19: Hyperparameter Optimization (GridSearchCV, RandomizedSearchCV), SMOTE Oversampling, and Class Weighting**.

## Directory Structure
- [`task_1/`](task_1/): `GridSearchCV` tuning of `max_depth` and `n_estimators` for a `RandomForestClassifier` on Product Review sentiment dataset.
- [`task_2/`](task_2/): `RandomizedSearchCV` tuning of continuous `C` and `gamma` hyperparameters for `SVC` on Instagram Engagement dataset ($n\_iter=10$).
- [`task_3/`](task_3/): Minority class oversampling using `SMOTE` prior to running `GridSearchCV` hyperparameter tuning on `RandomForestClassifier`.
- [`task_4/`](task_4/): Comparative performance report (Accuracy and Minority F1-Score) before and after applying `class_weight='balanced'`.
- [`task_5/`](task_5/): Comprehensive AI-suggested parameter grid for `RandomForestClassifier` with parameter range selection rationale.

## How to Run Python Scripts
Run individual task scripts from the workspace root:
```bash
python session_19/task_1/task_1.py
python session_19/task_2/task_2.py
python session_19/task_3/task_3.py
python session_19/task_4/task_4.py
python session_19/task_5/task_5.py
```
