# Session 10 - Task 3: Sigmoid Curve Visualization & Decision Threshold

## Task Overview
Generate a continuous plot of the **Sigmoid Activation Function** $\sigma(x) = \frac{1}{1 + e^{-x}}$ over the domain $x \in [-10, 10]$ and draw the decision threshold line at $y = 0.5$ using `plt.axhline()`.

---

## Visualization Figure
![Sigmoid Curve with Decision Threshold](file:///c:/Users/heeya/OneDrive/Documents/TOPS/Machine%20Learning/supervised_ML_Assignment/session_10/task_3/sigmoid_plot.png)

---

## Key Observations
1. **Decision Threshold ($y = 0.5$)**:
   - Drawn as a dashed horizontal line (`plt.axhline(y=0.5)`).
   - Serves as the binary classification cutoff:
     - $\hat{P} \ge 0.5 \implies \text{Class 1 (Viral)}$
     - $\hat{P} < 0.5 \implies \text{Class 0 (Not Viral)}$
2. **Inflection Point / Midpoint $(0, 0.5)$**:
   - At $x = 0$ (the decision boundary where net input $z = \boldsymbol{w}^T \boldsymbol{x} + b = 0$), the model assigns an equal $50\%$ probability to both outcomes.
3. **Asymptotic Behavior**:
   - As $x \to +\infty$, $\sigma(x) \to 1.0$ (Saturates at 1).
   - As $x \to -\infty$, $\sigma(x) \to 0.0$ (Saturates at 0).
