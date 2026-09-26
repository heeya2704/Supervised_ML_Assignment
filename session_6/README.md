# Session 6: Multiple Linear Regression & Multicollinearity Analysis

This folder contains dataset generator scripts, Multiple Linear Regression models, Variance Inflation Factor (VIF) analyses, and refactoring reports for all tasks in **Session 6**.

## Directory Structure
- [`dataset/`](dataset/): `mobile_prices.csv` dataset.
- [`task_1/`](task_1/): Multiple Linear Regression model predicting mobile phone price using 5 specifications.
- [`task_2/`](task_2/): Regression coefficient breakdown & real-world domain interpretations.
- [`task_3/`](task_3/): Variance Inflation Factor (VIF) calculation identifying severe multicollinearity (`ram_gb` vs `storage_gb`).
- [`task_4/`](task_4/): Standardized feature importance ranking identifying `storage_gb` as most influential.
- [`task_5/`](task_5/): Model refactoring removing collinear `storage_gb`, demonstrating VIF collapse (3064.80 $\rightarrow$ 1.03) with zero $R^2$ loss.

## How to Run Python Scripts
```bash
python session_6/create_mobile_dataset.py
python session_6/task_1/task_1.py
python session_6/task_2/task_2.py
python session_6/task_3/task_3.py
python session_6/task_4/task_4.py
python session_6/task_5/task_5.py
```
