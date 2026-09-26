# Session 17: Advanced Evaluation Metrics for Imbalanced Classification

This folder contains Python scripts, visualization plots, and detailed markdown write-ups for all tasks in **Session 17**.

## Directory Structure
- [`task_1/`](task_1/): Train a `LogisticRegression` baseline model on an imbalanced Credit Card Fraud dataset (95% / 5%).
- [`task_2/`](task_2/): Plot the ROC Curve using `sklearn.metrics.roc_curve` and display the AUC score (**0.9575**).
- [`task_3/`](task_3/): Plot the Precision-Recall curve using `sklearn.metrics.precision_recall_curve` with analysis of its utility for rare positive classes.
- [`task_4/`](task_4/): Calculate Log Loss (Cross-Entropy Loss) on test set predictions (**0.0812**).
- [`task_5/`](task_5/): Evaluate high class imbalance (98% / 2%) and explain why Accuracy is highly misleading compared to ROC-AUC and Log Loss.

## How to Run Python Scripts
Run individual task scripts from the workspace root:
```bash
python session_17/task_1/task_1.py
python session_17/task_2/task_2.py
python session_17/task_3/task_3.py
python session_17/task_4/task_4.py
python session_17/task_5/task_5.py
```
