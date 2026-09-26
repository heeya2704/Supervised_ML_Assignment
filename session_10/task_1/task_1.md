# Session 10 - Task 1: Sigmoid Function Implementation

## Task Overview
Implement the **Sigmoid (Logistic) Activation Function** in Python:

$$\sigma(x) = \frac{1}{1 + e^{-x}}$$

and evaluate its output for inputs $x \in \{-2, 0, 3\}$.

---

## Evaluation Results Table

| Input ($x$) | Sigmoid Formula | Calculated Output $\sigma(x)$ | Classification Region |
|---|---|---|---|
| **$-2$** | $\frac{1}{1 + e^2} \approx \frac{1}{1 + 7.389}$ | **0.119203** | Probability $< 0.5$ (Negative Class) |
| **$0$** | $\frac{1}{1 + e^0} = \frac{1}{1 + 1}$ | **0.500000** | Decision Boundary ($p = 0.5$) |
| **$3$** | $\frac{1}{1 + e^{-3}} \approx \frac{1}{1 + 0.0498}$ | **0.952574** | Probability $> 0.5$ (Positive Class) |

---

## Mathematical Properties
1. **Domain & Range**: $\sigma(x) \in (0, 1)$ for all $x \in (-\infty, \infty)$.
2. **Symmetry**: $\sigma(-x) = 1 - \sigma(x)$.
3. **Derivative**: $\sigma'(x) = \sigma(x) \cdot (1 - \sigma(x))$.
