# Session 9 - Task 1: Dataset Loading and Train/Test Splitting

## Task Overview
Load the Zomato restaurant rating dataset (`zomato_ratings.csv`), separate the feature matrix ($X$) from the continuous regression target ($y = \text{rating}$), and perform an 80/20 train-test split.

---

## Dataset Summary
- **Total Dataset Size**: 300 restaurant samples $\times$ 14 columns
- **Target Variable ($y$)**: `rating` (Continuous value ranging from 1.0 to 5.0 stars)
- **Features ($X$)**: 13 features combining restaurant operational metrics and noisy control variables:
  1. `online_order` (Binary)
  2. `book_table` (Binary)
  3. `votes` (Numeric)
  4. `approx_cost_for_two` (Numeric)
  5. `location_score` (Numeric)
  6. `cuisine_count` (Numeric)
  7. `avg_dish_price` (Numeric)
  8. `delivery_time_min` (Numeric)
  9. `restaurant_age_yrs` (Numeric)
  10. `parking_available` (Binary)
  11. `noise_feature_1` (Random Gaussian noise)
  12. `noise_feature_2` (Random Uniform noise)
  13. `noise_feature_3` (Random Bernoulli noise)

---

## Train-Test Split Dimensions (80/20 Ratio)
- **$X_{\text{train}}$**: 240 samples $\times$ 13 features
- **$X_{\text{test}}$**: 60 samples $\times$ 13 features
- **$y_{\text{train}}$**: 240 target ratings
- **$y_{\text{test}}$**: 60 target ratings
