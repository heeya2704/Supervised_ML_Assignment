# Session 6 - Task 4: Feature Importance & Coefficient Interpretation

## Task Overview
Determine which feature is most influential in predicting mobile phone prices by calculating standardized regression coefficients (mean = 0, std = 1 feature scaling).

---

## Standardized Feature Importance Ranking

| Rank | Feature Name | Standardized Coef | Absolute Impact (₹) | Relative Influence |
|---|---|---|---|---|
| **1** | **`storage_gb`** | **+14,872.84** | **₹14,872.84** | **Most Influential** |
| 2 | `camera_mp` | +10,625.61 | ₹10,625.61 | High |
| 3 | `ram_gb` | +8,215.72 | ₹8,215.72 | Moderate-High |
| 4 | `battery_mah` | +3,960.51 | ₹3,960.51 | Moderate |
| 5 | `screen_size_inch` | -575.35 | ₹575.35 | Low |

---

## Influential Feature Identification & Reasoning

**Most Influential Feature:** `storage_gb` (Internal Storage in GB)

**Reasoning:**
Based on standardized linear regression coefficients, **Internal Storage (`storage_gb`)** exhibits the highest absolute standardized coefficient magnitude (+₹14,872.84 per 1 standard deviation shift). This demonstrates that internal memory tiering (e.g. 128 GB vs 512 GB) creates the largest single financial price impact across smartphone models. Manufacturers aggressively price higher storage variants, making storage capacity the primary determinant of mobile price tiers.
