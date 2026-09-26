# Session 10: Sigmoid Function & Logistic Regression Fundamentals

This folder contains Python scripts, visualization figures, and detailed markdown write-ups for all tasks in **Session 10**.

## Directory Structure
- [`task_1/`](task_1/): Sigmoid function $\sigma(x) = \frac{1}{1 + e^{-x}}$ Python implementation and testing at $x \in \{-2, 0, 3\}$.
- [`task_2/`](task_2/): Binary Logistic Regression model fitting on 10 Instagram posts predicting virality (`likes` and `has_caption`).
- [`task_3/`](task_3/): Continuous Sigmoid curve plot across domain $x \in [-10, 10]$ with $0.5$ decision threshold line (`sigmoid_plot.png`).
- [`task_4/`](task_4/): Configurable decision threshold classification function (`classify_virality`) tested at thresholds $0.5$ and $0.7$.

## How to Run Python Scripts
Run individual task scripts from the workspace root:
```bash
python session_10/task_1/task_1.py
python session_10/task_2/task_2.py
python session_10/task_3/task_3.py
python session_10/task_4/task_4.py
```
