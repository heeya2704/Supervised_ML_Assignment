# Session 22: End-to-End Zomato Restaurant Classification Pipeline

This folder contains Python scripts, visualization plots, and detailed markdown write-ups for all tasks in **Session 22: Exploratory Data Analysis, Missing Value Imputation, IQR Outlier Capping, One-Hot Encoding, StandardScaler, Baseline Classification, SMOTE Resampling, and GridSearchCV Hyperparameter Tuning**.

## Directory Structure
- [`dataset/`](dataset/): Raw Zomato restaurant dataset (`zomato_restaurants.csv`).
- [`task_1/`](task_1/): Data loading, summary statistics, `.info()`, and `.describe()` exploratory analysis.
- [`task_2/`](task_2/): Median missing value imputation & IQR outlier capping for dining cost with boxplot visualization (`outliers_boxplot.png`).
- [`task_3/`](task_3/): One-Hot Encoding for categorical features (`cuisine`, `location`) and `StandardScaler` normalization.
- [`task_4/`](task_4/): 80/20 Train-Test Split, baseline `LogisticRegression` vs `RandomForestClassifier` training, and ROC-AUC evaluation plot (`roc_curve_comparison.png`).
- [`task_5/`](task_5/): Minority class balancing using `SMOTE` and hyperparameter optimization via `GridSearchCV`.

## How to Run Python Scripts
Run individual task scripts from the workspace root:
```bash
python session_22/task_1/task_1.py
python session_22/task_2/task_2.py
python session_22/task_3/task_3.py
python session_22/task_4/task_4.py
python session_22/task_5/task_5.py
```
