# Session 6 - Task 1: Multiple Linear Regression Model for Mobile Phone Prices

## Task Overview
Fit a Multiple Linear Regression model predicting mobile phone price (`price_inr`) using five features: `ram_gb`, `storage_gb`, `battery_mah`, `camera_mp`, and `screen_size_inch`.

---

## Dataset Sample (First 5 Rows)

| RAM (GB) | Storage (GB) | Battery (mAh) | Camera (MP) | Screen Size (in) | Price (INR ₹) |
|---|---|---|---|---|---|
| 6 | 96 | 4000 | 108 | 6.3 | ₹92,500 |
| 16 | 512 | 4000 | 64 | 6.3 | ₹172,900 |
| 8 | 128 | 5000 | 48 | 6.5 | ₹98,500 |
| 8 | 128 | 4500 | 108 | 6.2 | ₹105,500 |
| 4 | 128 | 4500 | 108 | 6.5 | ₹91,200 |

---

## Multiple Linear Regression Fit Summary

- **Intercept ($b$):** ₹12,922.36
- **$R^2$ Score:** `0.9909`
- **RMSE:** ₹2,365.74

### Feature Coefficients Table

| Feature Name | Coefficient ($w_i$) | Unit Description |
|---|---|---|
| `ram_gb` | **+₹3,647.20** | Price increase per +1 GB RAM |
| `storage_gb` | **+₹116.60** | Price increase per +1 GB Storage |
| `battery_mah` | **+₹4.61** | Price increase per +1 mAh Battery capacity |
| `camera_mp` | **+₹173.69** | Price increase per +1 MP Camera resolution |
| `screen_size_inch` | **+₹1,255.54** | Price increase per +1 inch Screen display size |
