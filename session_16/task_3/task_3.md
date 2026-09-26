# Session 16 - Task 3: Fake News Detection Precision & Recall Analysis

## Task Overview
Calculate Precision and Recall for a social media fake news detector given:
- **True Positives ($TP$)**: 60 (Fake posts correctly flagged)
- **False Positives ($FP$)**: 10 (Real posts falsely flagged as fake)
- **False Negatives ($FN$)**: 30 (Fake posts missed)
- **True Negatives ($TN$)**: 120 (Real posts correctly identified)

---

## Calculations & Results

1. **Precision**:
   $$\text{Precision} = \frac{TP}{TP + FP} = \frac{60}{60 + 10} = \frac{60}{70} = \mathbf{0.8571 \quad (85.71\%)}$$
   - **Meaning**: Out of all posts flagged as fake by the system, **85.71% were actually fake**.

2. **Recall (Sensitivity)**:
   $$\text{Recall} = \frac{TP}{TP + FN} = \frac{60}{60 + 30} = \frac{60}{90} = \mathbf{0.6667 \quad (66.67\%)}$$
   - **Meaning**: Out of all real fake posts present on the platform, the model **detected 66.67% of them** and missed 33.33%.
