# Session 16 - Task 1: Manual Calculation of Confusion Matrix Metrics

## Task Overview
Calculate Accuracy, Precision, Recall, and F1-Score for a spam filter given:
- **True Positives ($TP$)**: 50 (Spam correctly flagged)
- **True Negatives ($TN$)**: 120 (Not Spam correctly identified)
- **False Positives ($FP$)**: 10 (Not Spam wrongly flagged as Spam)
- **False Negatives ($FN$)**: 20 (Spam missed)
- **Total Population ($N$)**: 200 emails

---

## Step-by-Step Mathematical Calculations

1. **Accuracy**:
   $$\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN} = \frac{50 + 120}{200} = \frac{170}{200} = \mathbf{0.8500 \quad (85.00\%)}$$

2. **Precision**:
   $$\text{Precision} = \frac{TP}{TP + FP} = \frac{50}{50 + 10} = \frac{50}{60} = \mathbf{0.8333 \quad (83.33\%)}$$

3. **Recall (Sensitivity)**:
   $$\text{Recall} = \frac{TP}{TP + FN} = \frac{50}{50 + 20} = \frac{50}{70} = \mathbf{0.7143 \quad (71.43\%)}$$

4. **F1-Score**:
   $$\text{F1-Score} = 2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}} = 2 \cdot \frac{0.8333 \cdot 0.7143}{0.8333 + 0.7143} = \frac{1.1905}{1.5476} = \mathbf{0.7692 \quad (76.92\%)}$$
