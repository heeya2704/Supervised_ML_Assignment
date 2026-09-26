# Session 18 - Task 5: AI-Assisted Generic Cross-Validation Evaluation Function

## Task Overview
Implement a flexible Python function `evaluate_classifier_cv()` that accepts any scikit-learn classifier and dataset, executes $k$-fold cross-validation, and returns the mean accuracy, standard deviation, and raw fold scores.

---

## Code Implementation

```python
import numpy as np
from sklearn.model_selection import cross_val_score

def evaluate_classifier_cv(model, X, y, cv=5, scoring='accuracy'):
    scores = cross_val_score(model, X, y, cv=cv, scoring=scoring)
    mean_score = np.mean(scores)
    std_score = np.std(scores)
    return mean_score, std_score, scores
```

---

## AI Enhancements & Key Improvements

1. **Parameter Flexibility**: Added a dynamic `scoring` argument supporting custom metrics (`'f1'`, `'roc_auc'`, `'precision'`) beyond basic accuracy.
2. **Comprehensive Returns**: Returned structured summary statistics (`mean`, `std`) alongside full raw fold arrays for detailed variance tracking.
3. **Scikit-Learn Compatibility**: Designed to accept any estimator matching the `fit`/`predict` duck-typing interface.
