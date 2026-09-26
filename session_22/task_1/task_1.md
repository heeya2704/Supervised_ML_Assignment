# Session 22 - Task 1: Zomato Dataset Loading & Data Exploration

## Task Overview
Load the `zomato_restaurants.csv` dataset, display the first 5 rows, and examine basic metadata using `.info()` and `.describe()`.

---

## Dataset Summary & Metadata

- **Total Records**: 1,200 restaurants
- **Total Features**: 7 columns (`restaurant_name`, `location`, `cuisine`, `cost`, `rating`, `online_order`, `votes`)

### Feature Data Types & Null Counts

| Column Name | Non-Null Count | Data Type | Description |
| :--- | :---: | :---: | :--- |
| `restaurant_name` | 1,200 | `object` | Restaurant identifier |
| `location` | 1,200 | `object` | Neighborhood location in city |
| `cuisine` | 1,200 | `object` | Primary cuisine type |
| `cost` | 1,175 | `float64` | Average dining cost for two (25 missing values) |
| `rating` | 1,180 | `float64` | Customer rating out of 5.0 (20 missing values) |
| `online_order` | 1,200 | `object` | Online order availability (`Yes` / `No`) |
| `votes` | 1,200 | `int64` | Number of customer votes/reviews |

---

## Key Observations
1. **Missing Data**: Missing values detected in `cost` (25 rows) and `rating` (20 rows).
2. **Outliers**: High variance in `cost` (mean = ₹705.14, median = ₹590.00, max = ₹15,000.00), indicating extreme right-skewed outliers.
