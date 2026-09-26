# Session 14 - Task 4: AI-Assisted SVM Margin & Support Vector Visualization

## Task Overview
Visualize the **Maximum Margin Hyperplane** and **Support Vectors** for a linear SVM classifier trained on Zomato restaurant rating features (`location_score` and `votes`) predicting `Good` vs `Bad` ratings.

---

## SVM Margin & Support Vectors Diagram
![Zomato SVM Margin & Support Vectors](file:///c:/Users/heeya/OneDrive/Documents/TOPS/Machine%20Learning/supervised_ML_Assignment/session_14/task_4/zomato_svm_margin.png)

---

## Mathematical Formulation & Margin Geometry

1. **Decision Boundary**:
   $$\mathbf{w}^T \mathbf{x} + b = 0$$

2. **Upper & Lower Margins**:
   $$\mathbf{w}^T \mathbf{x} + b = +1 \quad \text{and} \quad \mathbf{w}^T \mathbf{x} + b = -1$$

3. **Margin Width Calculation**:
   $$\text{Margin Width} = \frac{2}{\|\mathbf{w}\|_2} = 1.3942$$

---

## Key Components Explained
1. **Decision Hyperplane (Solid Line)**: The central boundary line separating Good and Bad Zomato restaurant clusters.
2. **Margin Lines (Dashed Lines)**: Parallel boundary lines located at distance $\frac{1}{\|\mathbf{w}\|}$ from the decision boundary.
3. **Support Vectors (Circled Purple Points)**: The 4 critical data samples lying directly on the margin boundaries ($\mathbf{w}^T \mathbf{x}_i + b = \pm 1$). Removing any other sample from the dataset leaves the decision boundary unchanged, making SVM robust against non-boundary noise.
