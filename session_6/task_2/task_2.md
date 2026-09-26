# Session 6 - Task 2: Regression Coefficients & Domain Interpretation

## Task Overview
Display regression coefficients and intercept from the fitted Multiple Linear Regression model and explain their physical domain meaning in the context of mobile phone hardware specifications.

---

## Model Coefficients & Domain Interpretations

- **Base Intercept ($b$): ₹12,922.36**
  - Theoretical baseline price of a mobile phone when all features (`RAM`, `Storage`, `Battery`, `Camera`, `Screen Size`) are set to 0.

---

## Detailed Coefficient Breakdown

| Feature | Coefficient ($w_i$) | Real-World Domain Explanation |
|---|---|---|
| **RAM (`ram_gb`)** | **+₹3,647.20** | For every additional **1 GB of RAM**, the expected mobile phone retail price increases by **₹3,647.20**. |
| **Storage (`storage_gb`)** | **+₹116.60** | For every additional **1 GB of internal flash storage**, the phone price increases by **₹116.60** (or ~₹14,924 per 128 GB upgrade). |
| **Battery (`battery_mah`)** | **+₹4.61** | For every **1 mAh increase** in battery capacity, the phone price increases by **₹4.61** (or ~₹4,610 per 1,000 mAh boost). |
| **Camera (`camera_mp`)** | **+₹173.69** | For every additional **1 Megapixel** of main camera resolution, the phone price increases by **₹173.69**. |
| **Screen Size (`screen_size_inch`)** | **+₹1,255.54** | For every **1 inch increase** in display diagonal screen size, phone price increases by **₹1,255.54**. |
