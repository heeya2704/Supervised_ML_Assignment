# Session 5 - Task 3: Making Predictions for New Usage Inputs

## Task Overview
Use the trained Simple Linear Regression model to make battery percentage predictions for hypothetical daily Instagram usage scenarios ($3.0\text{ hours}$, $4.5\text{ hours}$, and $7.0\text{ hours}$).

---

## Prediction Results

| Instagram Usage (Hours/Day) | Model Predicted Battery Remaining (%) | Equation Verification ($y = m \cdot x + b$) |
|---|---|---|
| **3.0 hrs** | **69.31%** | $105.42 + (-12.04 \times 3.0) = 69.31\%$ |
| **4.5 hrs** | **51.25%** | $105.42 + (-12.04 \times 4.5) = 51.25\%$ |
| **7.0 hrs** | **21.16%** | $105.42 + (-12.04 \times 7.0) = 21.16\%$ |

---

## Observations
- **3.0 Hours:** Moderate daily usage leaves comfortably ~69% battery for night usage.
- **4.5 Hours:** Higher daily usage drains phone to roughly half battery (~51%).
- **7.0 Hours:** Heavy screen time leaves only ~21% battery remaining by end of day.
