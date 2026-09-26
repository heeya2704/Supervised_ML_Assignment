# Session 5 - Task 2: Fitting Simple Linear Regression Model

## Task Overview
Fit a `LinearRegression` model from `scikit-learn` using the weekly Instagram usage data to predict end-of-day phone battery percentage. Extract and interpret the model's slope ($m$) and $y$-intercept ($b$).

---

## Model Parameters & Results

- **Slope / Coefficient ($m$):** `-12.0363`
- **$Y$-Intercept ($b$):** `105.4153`

### Regression Line Equation
$$\text{Battery \%} = 105.42 - 12.04 \times (\text{Instagram Hours})$$

---

## Interpretation

1. **$Y$-Intercept ($b = 105.42\%$):**
   - Represents the theoretical baseline phone battery percentage at the end of the day when Instagram usage is **0 hours**.

2. **Slope ($m = -12.04\%/\text{hr}$):**
   - For every **1 additional hour** spent on Instagram, the end-of-day phone battery percentage drops by approximately **12.04%**.
