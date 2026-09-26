# Session 17 - Task 3: Precision-Recall Curve Analysis

## Task Overview
Plot the Precision-Recall curve using `sklearn.metrics.precision_recall_curve` for the imbalanced fraud detection model and provide a single-sentence explanation of why PR curves are superior for imbalanced datasets.

---

## Visualization
The Precision-Recall curve is saved to [`precision_recall_curve.png`](precision_recall_curve.png).

---

## One-Line Technical Explanation
> **The Precision-Recall curve is essential for imbalanced datasets because it completely ignores True Negatives ($TN$) and focuses exclusively on performance relative to the rare positive class (fraud), preventing overly optimistic evaluations caused by large majority class counts.**
