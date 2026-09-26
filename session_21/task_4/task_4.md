# Session 21 - Task 4: Evaluation Metrics Analysis (MAE, RMSE, MAPE)

## Task Overview
Calculate Mean Absolute Error ($\text{MAE}$), Root Mean Squared Error ($\text{RMSE}$), and Mean Absolute Percentage Error ($\text{MAPE}$) for test set predictions and identify which metric is most sensitive to large errors.

---

## Mathematical Formulas

### 1. Mean Absolute Error ($\text{MAE}$)
$$\text{MAE} = \frac{1}{n} \sum_{i=1}^{n} |y_i - \hat{y}_i|$$

### 2. Root Mean Squared Error ($\text{RMSE}$)
$$\text{RMSE} = \sqrt{\frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2}$$

### 3. Mean Absolute Percentage Error ($\text{MAPE}$)
$$\text{MAPE} = \frac{100\%}{n} \sum_{i=1}^{n} \left| \frac{y_i - \hat{y}_i}{y_i} \right|$$

---

## Metric Benchmark Results

| Metric | Computed Score |
| :--- | :---: |
| **MAE** | **1.7142°C** |
| **RMSE** | **2.1481°C** |
| **MAPE** | **5.98%** |

---

## One-Line Explanation
> **Root Mean Squared Error ($\text{RMSE}$) is the most sensitive metric to large prediction errors because squaring individual residuals heavily penalizes extreme outliers compared to linear metrics like MAE.**
