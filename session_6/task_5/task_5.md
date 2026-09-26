# Session 6 - Task 5: Code Refactoring & Collinear Feature Removal

## Task Overview
Refactor the Multiple Linear Regression model by removing the highly collinear feature (`storage_gb`, which had a VIF of 3061.45). Compare model performance, coefficient stability, and VIF values before and after refactoring.

---

## Performance & Metric Comparison Table

| Metric / Parameter | Original Model (5 Features) | Refactored Model (4 Features) | Net Difference / Impact |
|---|---|---|---|
| **Features Included** | `ram`, `storage`, `battery`, `camera`, `screen` | `ram`, `battery`, `camera`, `screen` | Removed 1 redundant feature (`storage_gb`) |
| **$R^2$ Score** | **0.9907** | **0.9906** | negligible change (-0.0001) |
| **RMSE** | **₹2,549.47** | **₹2,563.60** | Virtually identical error margin |
| **`ram_gb` VIF** | **3064.80** | **1.03** | **Extreme VIF drop (3064.80 $\rightarrow$ 1.03)** |
| **`ram_gb` Coefficient** | ₹2,563.66 / GB | ₹7,206.39 / GB | Absorbed storage contribution cleanly |

---

## Detailed Coefficient & VIF Comparison

| Feature | Original Coef | Refactored Coef | Original VIF | Refactored VIF | Status |
|---|---|---|---|---|---|
| `ram_gb` | ₹2,563.66 | ₹7,206.39 | **3064.80** | **1.03** | **Stabilized ($\text{VIF} < 5$)** |
| `storage_gb` | ₹145.27 | *REMOVED* | **3061.45** | *N/A* | **Eliminated collinear predictor** |
| `battery_mah` | ₹5.09 | ₹5.11 | 50.78 | 50.77 | Preserved |
| `camera_mp` | ₹179.19 | ₹178.50 | 1.07 | 1.05 | Independent |
| `screen_size_inch` | -₹3,649.27 | -₹3,609.45 | 50.44 | 50.44 | Preserved |

---

## Key Takeaways
1. **Stabilized Coefficients:** Removing `storage_gb` eliminated multicollinearity for `ram_gb`, causing its VIF to collapse from **3064.80 down to 1.03**.
2. **Preserved Predictive Power:** The model's $R^2$ score remained virtually unchanged (**0.9906**), demonstrating that removing redundant collinear features simplifies model architecture without sacrificing accuracy.
