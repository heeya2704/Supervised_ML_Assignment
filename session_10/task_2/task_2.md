# Session 10 - Task 2: Logistic Regression on Instagram Virality Data

## Task Overview
Train a binary **Logistic Regression model** on a dataset of 10 Instagram posts using features `likes` and `has_caption` to predict whether a post went `viral` ($y \in \{0, 1\}$).

---

## Dataset Overview (10 Instagram Posts)

| Post # | `likes` | `has_caption` (Binary) | `viral` ($y$) |
|---|---|---|---|
| Post 1 | 150 | 0 | 0 |
| Post 2 | 1,200 | 1 | 0 |
| Post 3 | 300 | 0 | 0 |
| Post 4 | 4,500 | 1 | 1 |
| Post 5 | 80 | 0 | 0 |
| Post 6 | 2,500 | 1 | 1 |
| Post 7 | 6,000 | 1 | 1 |
| Post 8 | 400 | 0 | 0 |
| Post 9 | 3,200 | 1 | 1 |
| Post 10 | 9,500 | 1 | 1 |

---

## Model Coefficients

| Parameter / Feature | Raw Model Coefficient | Standardized Model Coefficient |
|---|---|---|
| **Intercept ($\beta_0$)** | `-32.119063` | `+0.054366` |
| **`likes` ($\beta_1$)** | `+0.017355` | `+1.353380` |
| **`has_caption` ($\beta_2$)** | `-0.000015` | `+0.728514` |

---

## Mathematical Formulation
$$\hat{p} = P(\text{viral} = 1 \mid X) = \frac{1}{1 + e^{-(\beta_0 + \beta_1 \cdot \text{likes} + \beta_2 \cdot \text{has\_caption})}}$$

- Standardized feature transformation shows that increasing `likes` by 1 standard deviation increases log-odds by **+1.3534**, while adding a caption (`has_caption=1`) increases log-odds by **+0.7285**.
