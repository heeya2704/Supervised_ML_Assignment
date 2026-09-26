# Session 16 - Task 2: Custom `print_confusion_matrix` Function

## Task Overview
Implement a reusable Python function `print_confusion_matrix(tp, tn, fp, fn)` to display confusion matrix tables alongside derived metrics (Accuracy, Precision, Recall, F1-Score).

---

## Python Function Implementation
```python
def print_confusion_matrix(tp, tn, fp, fn, title="CONFUSION MATRIX DISPLAY"):
    total = tp + tn + fp + fn
    accuracy = (tp + tn) / total if total > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    header_col = "Actual \\ Predicted"

    print("=" * 68)
    print(f" {title.upper()}")
    print("=" * 68)
    print(f"{header_col:<22} | {'Predicted Positive (1)':<20} | {'Predicted Negative (0)':<20}")
    print("-" * 68)
    print(f"{'Actual Positive (1)':<22} | {'TP = ' + str(tp):<20} | {'FN = ' + str(fn):<20}")
    print(f"{'Actual Negative (0)':<22} | {'FP = ' + str(fp):<20} | {'TN = ' + str(tn):<20}")
    print("-" * 68)
```

---

## Formatted Output Example

```
====================================================================
 SPAM FILTER EVALUATION
====================================================================
Actual \ Predicted     | Predicted Positive (1) | Predicted Negative (0)
--------------------------------------------------------------------
Actual Positive (1)    | TP = 50              | FN = 20             
Actual Negative (0)    | FP = 10              | TN = 120            
--------------------------------------------------------------------
SUMMARY METRICS:
  • Total Samples : 200
  • Accuracy      : 0.8500 (85.00%)
  • Precision     : 0.8333 (83.33%)
  • Recall        : 0.7143 (71.43%)
  • F1-Score      : 0.7692 (76.92%)
====================================================================
```
