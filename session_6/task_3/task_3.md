# Session 6 - Task 3: Variance Inflation Factor (VIF) & Multicollinearity Analysis

## Task Overview
Calculate the Variance Inflation Factor (VIF) for each feature in the mobile phone price prediction dataset using `statsmodels` to detect problematic multicollinearity.

> [!NOTE]
> Standard Rule of Thumb:
> - $\text{VIF} < 5$: Low multicollinearity (acceptable).
> - $5 \le \text{VIF} \le 10$: Moderate multicollinearity.
> - $\text{VIF} > 10$: Severe multicollinearity (unstable coefficient estimates).

---

## Variance Inflation Factor (VIF) Results

| Feature Name | VIF Value | Multicollinearity Status | Diagnosis |
|---|---|---|---|
| `ram_gb` | **3064.80** | High ($\text{VIF} \gg 10$) | Severe collinearity with `storage_gb` |
| `storage_gb` | **3061.45** | High ($\text{VIF} \gg 10$) | Severe collinearity with `ram_gb` |
| `battery_mah` | **50.78** | High ($\text{VIF} > 10$) | High collinearity with `screen_size_inch` |
| `screen_size_inch` | **50.44** | High ($\text{VIF} > 10$) | High collinearity with `battery_mah` |
| `camera_mp` | **1.07** | Low ($\text{VIF} \le 5$) | Independent feature |

---

## Findings & Multicollinearity Identification
- `storage_gb` and `ram_gb` exhibit extremely high VIF values (~3060+) because storage scales linearly with RAM capacity in mobile phone product lineups.
- `battery_mah` and `screen_size_inch` also show high multicollinearity ($\text{VIF} \approx 50$) because larger screen sizes physically allow for larger battery capacity.
- These high VIF values indicate redundant information, inflating regression variance and making coefficient interpretation unreliable until one of the collinear features is removed.
