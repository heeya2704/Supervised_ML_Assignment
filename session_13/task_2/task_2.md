# Session 13 - Task 2: Flipkart Product Category Classifier & Importance

## Task Overview
Train a `RandomForestClassifier` on a Flipkart product dataset (20 products across `Electronics`, `Fashion`, and `Home`) using features `price_rs`, `rating`, and `brand_tier`. Extract feature importances.

---

## Flipkart Dataset Sample (20 Products)

| Product Name | Category | `price_rs` | `rating` | `brand_tier` |
|---|---|---|---|---|
| Laptop / TV / Console | Electronics | Rs. 25,000 – Rs. 55,000 | 4.3 – 4.7 | 3 (Premium) |
| T-Shirt / Jeans / Sneakers | Fashion | Rs. 800 – Rs. 4,500 | 3.9 – 4.4 | 1–2 |
| Sofa / Dining / Microwave | Home | Rs. 700 – Rs. 22,000 | 3.7 – 4.5 | 1–3 |

---

## Random Forest Feature Importances

| Feature Name | Gini Importance Score | Contribution Percentage | Domain Interpretation |
|---|---|---|---|
| `price_rs` | **0.551865** | **55.19%** | Primary separator distinguishing expensive tech from lower-cost apparel |
| `rating` | **0.326748** | **32.67%** | Secondary discriminator separating product tiers |
| `brand_tier` | **0.121387** | **12.14%** | Brand tiering metric |
