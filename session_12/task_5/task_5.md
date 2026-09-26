# Session 12 - Task 5: Swiggy Delivery ETA Decision Tree & Split Interpretation

## Task Overview
Train a decision tree model on Swiggy food delivery metrics (`distance_km`, `prep_time_min`, `rain_intensity`, `traffic_density`) to classify delivery speed (`'Fast'`, `'Moderate'`, or `'Delayed'`). Interpret the first three splits of the tree.

---

## Decision Tree Diagram
![Swiggy Delivery Speed Decision Tree](file:///c:/Users/heeya/OneDrive/Documents/TOPS/Machine%20Learning/supervised_ML_Assignment/session_12/task_5/swiggy_tree.png)

---

## Detailed Interpretation of the Top 3 Decision Tree Splits

### 1. Root Split (Split #1 - Node 0)
- **Feature & Threshold**: `prep_time_min <= 28.5`
- **Why it helps classification**:
  - Restaurant kitchen preparation time is the single largest baseline determinant of delivery speed.
  - If prep time is $\le 28.5$ minutes, the delivery has a strong chance of being **Fast** or **Moderate**.
  - If prep time exceeds $28.5$ minutes, the delivery accumulates significant delay regardless of courier speed, routing most samples into the **Delayed** candidate branch.

### 2. Left Child Split (Split #2 - Node 1)
- **Feature & Threshold**: `distance_km <= 6.8`
- **Why it helps classification**:
  - Among orders with quick kitchen prep ($\le 28.5$ min), transit distance separates **Fast** local deliveries from longer transit times.
  - Short distances ($\le 6.8$ km) guarantee rapid arrival (**Fast** class), whereas longer distances shift the prediction towards **Moderate**.

### 3. Right Child Split (Split #3 - Node 8)
- **Feature & Threshold**: `prep_time_min <= 37.5` (or `traffic_density <= 6.4`)
- **Why it helps classification**:
  - For orders burdened by long kitchen prep times ($> 28.5$ min), this split isolates severe delays ($> 37.5$ min prep time combined with heavy traffic) into the absolute **Delayed** class.
  - It separates borderline moderate orders from severely delayed deliveries.
