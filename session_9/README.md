# Session 9: Regularization Techniques & Feature Selection

This folder contains Python scripts and detailed markdown write-ups for all tasks in **Session 9**.

## Directory Structure
- [`create_zomato_dataset.py`](create_zomato_dataset.py): Generator script producing realistic Zomato restaurant rating dataset with 13 features (including noise variables).
- [`dataset/`](dataset/): Contains `zomato_ratings.csv`.
- [`task_1/`](task_1/): Dataset loading, feature matrix $X$ and target $y$ splitting, and 80/20 train-test split setup.
- [`task_2/`](task_2/): Lasso Regression ($L_1$ penalty) implementation and feature coefficient inspection.
- [`task_3/`](task_3/): Comparative evaluation of Ridge ($L_2$) vs Lasso ($L_1$) feature coefficients.
- [`task_4/`](task_4/): ElasticNet regression analysis across different `l1_ratio` values ($\rho \in \{0.1, 0.5, 0.9\}$).
- [`task_5/`](task_5/): Automated Zomato rating feature selector highlighting kept vs eliminated features.

## How to Run Python Scripts
Run individual task scripts from the workspace root:
```bash
python session_9/create_zomato_dataset.py
python session_9/task_1/task_1.py
python session_9/task_2/task_2.py
python session_9/task_3/task_3.py
python session_9/task_4/task_4.py
python session_9/task_5/task_5.py
```
