# Session 22 - Task 3: Categorical Feature Encoding & Standardization

## Task Overview
Transform raw Zomato restaurant features into machine-learning-ready numerical inputs by applying **One-Hot Encoding** to categorical columns (`location`, `cuisine`, `online_order`) and **StandardScaler** normalization to numerical features (`cost`, `votes`).

---

## 1. Categorical Feature Encoding

- **Binary Feature**: `online_order` (`Yes` / `No`) $\rightarrow$ Binary Integer (`1` / `0`)
- **Multi-Class Features**: `location` (7 categories) and `cuisine` (7 categories) encoded via `OneHotEncoder(drop='first')`, generating 12 binary indicator features.

---

## 2. Numerical Feature Scaling

Numerical features standard normal transformation using **`StandardScaler`**:

$$z = \frac{x - \mu}{\sigma}$$

- **Scaled Features**: `cost_scaled`, `votes_scaled`
- **Resulting Parameters**: Mean $\approx 0.0$, Standard Deviation $\approx 1.0$

---

## 3. Encoded & Scaled Feature Matrix Summary

- **Input Dimension**: 1,200 samples $\times$ 15 processed features
- **Included Features**: `cost_scaled`, `votes_scaled`, `online_order_encoded`, 6 `location_*` features, 6 `cuisine_*` features.
