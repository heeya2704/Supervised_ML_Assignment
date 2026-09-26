# Session 8: Polynomial Regression & Regularization

This folder contains Python scripts, visualization figures, and detailed markdown write-ups for all tasks in **Session 8**.

## Directory Structure
- [`task_1/`](task_1/): Polynomial features transformation (degree=2) on mobile phone features (`ram_gb`, `storage_gb`).
- [`task_2/`](task_2/): Linear vs. Polynomial Regression comparison on non-linear synthetic data (`linear_vs_polynomial.png`).
- [`task_3/`](task_3/): Validation curve analysis over polynomial degrees $d \in \{1, 2, 3, 5, 8\}$ to observe underfitting vs. overfitting (`degree_validation_curve.png`).
- [`task_4/`](task_4/): L2 Ridge Regularization ($\alpha=10.0$) on Degree-3 polynomial model to penalize large coefficient norms on Flipkart price data.

## How to Run Python Scripts
Run individual task scripts from the workspace root:
```bash
python session_8/task_1/task_1.py
python session_8/task_2/task_2.py
python session_8/task_3/task_3.py
python session_8/task_4/task_4.py
```
