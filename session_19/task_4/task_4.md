# Session 19 - Task 4: `class_weight='balanced'` Impact Evaluation

## Task Overview
Compare the classification performance of a `RandomForestClassifier` before and after setting `class_weight='balanced'` on an imbalanced dataset (92% Majority, 8% Minority).

---

## Benchmark Performance Table

| Model Setting | Overall Accuracy | Minority Class F1-Score | Macro Average F1-Score |
| :--- | :---: | :---: | :---: |
| **Without Class Weights (Default)** | **95.07% (0.9507)** | **0.5843 (58.43%)** | **0.7790** |
| **With `class_weight='balanced'`** | **94.27% (0.9427)** | **0.4941 (49.41%)** | **0.7319** |

---

## Detailed Findings

1. **Accuracy & F1-Score Metrics**:
   - Accuracy: **95.07%** (Default) vs **94.27%** (`balanced`)
   - Minority Class F1-Score: **58.43%** (Default) vs **49.41%** (`balanced`)
2. **Mechanism**: The `balanced` mode automatically assigns weights inversely proportional to class frequencies:
   $$w_j = \frac{N}{K \cdot n_j}$$
   This forces tree split criteria (Gini Impurity / Entropy) to penalize errors on minority samples far more heavily during tree construction.
