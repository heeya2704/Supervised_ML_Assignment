# Session 16 - Task 4: Food Delivery Fraud Detection System F1-Score Computation

## Task Overview
Calculate the F1-Score for a food delivery platform's automated fraud detection system given the following confusion matrix parameters:
- **True Positives ($TP$)**: 40 (Fraudulent orders correctly identified and blocked)
- **True Negatives ($TN$)**: 150 (Legitimate orders correctly processed)
- **False Positives ($FP$)**: 20 (Legitimate orders falsely flagged as fraud)
- **False Negatives ($FN$)**: 10 (Fraudulent orders missed by the system)
- **Total Population ($N$)**: 220 orders

---

## Step-by-Step Mathematical Calculation

### 1. Precision (Positive Predictive Value)
Precision measures the proportion of flagged orders that were actually fraudulent:
$$\text{Precision} = \frac{TP}{TP + FP} = \frac{40}{40 + 20} = \frac{40}{60} = \mathbf{0.6667 \quad (66.67\%)}$$

### 2. Recall (Sensitivity / True Positive Rate)
Recall measures the proportion of actual fraudulent orders detected by the model:
$$\text{Recall} = \frac{TP}{TP + FN} = \frac{40}{40 + 10} = \frac{40}{50} = \mathbf{0.8000 \quad (80.00\%)}$$

### 3. F1-Score (Harmonic Mean of Precision & Recall)
The F1-Score provides a single balanced metric evaluating precision and recall trade-offs:
$$\text{F1-Score} = 2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}} = 2 \cdot \frac{0.6667 \cdot 0.8000}{0.6667 + 0.8000} = 2 \cdot \frac{0.5333}{1.4667} = \mathbf{0.7273 \quad (72.73\%)}$$

---

## One-Line Technical Explanation
> **The F1-Score (72.73%) represents the harmonic mean of precision and recall, providing a single balanced metric that evaluates fraud detection performance without being artificially inflated by the high volume of legitimate non-fraudulent transactions ($TN=150$).**
