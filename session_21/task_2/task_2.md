# Session 21 - Task 2: DateTime Feature Extraction on E-Commerce Sales Data

## Task Overview
Extract calendar attributes from the `order_date` column of a Flipkart sales dataset to engineer `day_of_week`, `month`, and binary `is_weekend` indicators.

---

## Extracted Feature Attributes

1. **`day_of_week`**: Day name (`Monday` through `Sunday`).
2. **`month`**: Month name (`January` through `December`).
3. **`is_weekend`**: Binary indicator ($1$ for Saturday/Sunday, $0$ for Monday–Friday).

---

## Sample Extracted Data Table

| Order ID | Order Date | Day of Week | Month | Is Weekend | Sales Amount (₹) |
| :---: | :---: | :---: | :---: | :---: | :---: |
| OD10000 | 2024-01-01 | Monday | January | 0 | ₹1,420.50 |
| OD10001 | 2024-01-06 | Saturday | January | **1** | ₹2,850.00 |
| OD10002 | 2024-01-07 | Sunday | January | **1** | ₹3,110.25 |
