# Session 11 - Task 3: IRCTC Booking Confirmation Classifier (Gaussian Naive Bayes)

## Task Overview
Predict whether an IRCTC train ticket reservation will be `'confirmed'` or `'waitlisted'` using **Gaussian Naive Bayes (`GaussianNB`)** based on booking time, train route popularity score, and weekend status.

---

## Model & Mathematical Foundation

1. **Bayes' Theorem**:
   $$P(\text{Confirmed} \mid \mathbf{X}) = \frac{P(\text{Confirmed}) \prod_{i=1}^{n} P(x_i \mid \text{Confirmed})}{P(\mathbf{X})}$$

2. **Gaussian Likelihood Assumption**:
   For continuous features $x_i$:
   $$P(x_i \mid y) = \frac{1}{\sqrt{2\pi \sigma_y^2}} \exp \left( -\frac{(x_i - \mu_y)^2}{2\sigma_y^2} \right)$$

---

## Experimental Setup & Performance

- **Dataset Size**: 120 IRCTC booking records (90 train, 30 test)
- **Features**:
  - `booking_days_before`: Days prior to train departure (Numeric, 1–90)
  - `train_popularity_score`: Route demand index (Numeric, 1.0–10.0)
  - `travel_day_weekend`: Binary flag (1 if Saturday/Sunday, 0 otherwise)
- **Classification Accuracy**: **90.00%**

| Days Before | Train Popularity Score | Weekend (0/1) | Actual Status | Predicted Status |
|---|---|---|---|---|
| 58 | 5.86 | 1 | confirmed | confirmed |
| 9 | 5.87 | 0 | confirmed | confirmed |
| 15 | 4.34 | 0 | confirmed | confirmed |
| 65 | 7.08 | 1 | confirmed | confirmed |
| 72 | 7.02 | 0 | confirmed | confirmed |
