# Session 18: Cross-Validation Strategies

This folder contains Python scripts and detailed markdown documentation for all tasks in **Session 18: Cross-Validation Techniques (k-Fold, Stratified k-Fold, Stability Analysis)**.

## Directory Structure
- [`task_1/`](task_1/): 5-Fold cross-validation on the Iris dataset using `KNeighborsClassifier` (**97.33% mean accuracy**).
- [`task_2/`](task_2/): 10-Fold cross-validation on the Wine dataset using `DecisionTreeClassifier` (**89.86% mean accuracy**).
- [`task_3/`](task_3/): Verification of class distribution equality across folds using `StratifiedKFold`.
- [`task_4/`](task_4/): Stability and variance comparison across $cv = 3, 5, 10$ using `RandomForestClassifier` on the Wine dataset.
- [`task_5/`](task_5/): AI-assisted generic `evaluate_classifier_cv` evaluation utility for any scikit-learn model and dataset.

## How to Run Python Scripts
Run individual task scripts from the workspace root:
```bash
python session_18/task_1/task_1.py
python session_18/task_2/task_2.py
python session_18/task_3/task_3.py
python session_18/task_4/task_4.py
python session_18/task_5/task_5.py
```
