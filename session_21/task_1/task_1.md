# Session 21 - Task 1: Time Series Feature Engineering (Lag & Rolling Features)

## Task Overview
Load daily temperature data for Ahmedabad and engineer temporal features using pandas: a **3-day temperature lag** (`temp_lag3`) and a **7-day rolling average** (`temp_roll7`).

---

## Engineered Feature Formulas

### 1. 3-Day Lag Feature
$$\text{Temp}_{t-3} = \text{shift}(3)$$

### 2. 7-Day Rolling Mean Feature
$$\text{Temp}_{\text{roll7}} = \frac{1}{7} \sum_{i=0}^{6} \text{Temp}_{t-i}$$

---

## Sample Processed Output

| Date | Actual Temp (°C) | 3-Day Lag Temp (°C) | 7-Day Rolling Mean (°C) |
| :---: | :---: | :---: | :---: |
| 2024-01-07 | 17.51 | 21.05 | 18.66 |
| 2024-01-08 | 19.33 | 19.64 | 18.79 |
| 2024-01-09 | 22.84 | 19.26 | 19.45 |
| 2024-01-10 | 19.16 | 17.51 | 19.68 |
