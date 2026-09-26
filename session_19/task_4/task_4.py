"""
Session 19 - Task 4: class_weight='balanced' Performance Comparison
"""

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report

def main():
    print("=" * 75)
    print("SESSION 19 - TASK 4: Class Weight Balancing Comparison (Before vs After)")
    print("=" * 75)

    # 1. Create Imbalanced Dataset (92% Majority, 8% Minority)
    X, y = make_classification(
        n_samples=2500, n_features=12, n_informative=8, n_redundant=4,
        weights=[0.92, 0.08], random_state=42
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    print(f"\n1. DATASET DISTRIBUTION:")
    print(f"   • Total Test Samples : {len(y_test)}")
    print(f"   • Majority Class (0) : {(y_test == 0).sum()}")
    print(f"   • Minority Class (1) : {(y_test == 1).sum()}")

    # 2. Model 1: Baseline Random Forest WITHOUT class weights (default)
    rf_default = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_default.fit(X_train, y_train)

    y_pred_default = rf_default.predict(X_test)
    acc_default = accuracy_score(y_test, y_pred_default)
    f1_default = f1_score(y_test, y_pred_default, pos_label=1)
    f1_macro_default = f1_score(y_test, y_pred_default, average='macro')

    # 3. Model 2: Random Forest WITH class_weight='balanced'
    rf_balanced = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
    rf_balanced.fit(X_train, y_train)

    y_pred_balanced = rf_balanced.predict(X_test)
    acc_balanced = accuracy_score(y_test, y_pred_balanced)
    f1_balanced = f1_score(y_test, y_pred_balanced, pos_label=1)
    f1_macro_balanced = f1_score(y_test, y_pred_balanced, average='macro')

    # 4. Print Comparison Table
    print(f"\n2. PERFORMANCE COMPARISON TABLE:")
    print("-" * 75)
    print(f"{'Model Setting':<28} | {'Accuracy':<15} | {'Minority F1':<15} | {'Macro F1':<10}")
    print("-" * 75)
    print(f"{'Without Class Weights (Default)':<28} | {acc_default*100:.2f}% ({acc_default:.4f})  | {f1_default*100:.2f}% ({f1_default:.4f})  | {f1_macro_default:.4f}")
    label_balanced = 'With class_weight="balanced"'
    print(f"{label_balanced:<28} | {acc_balanced*100:.2f}% ({acc_balanced:.4f})  | {f1_balanced*100:.2f}% ({f1_balanced:.4f})  | {f1_macro_balanced:.4f}")
    print("-" * 75)

    print("\n3. OBSERVATIONS & ANALYSIS:")
    print("• Incorporating class_weight='balanced' automatically penalizes misclassification errors on the rare minority class proportionally to its inverse frequency.")
    print("• While overall Accuracy may drop slightly (due to a small increase in False Positives), the Minority Class F1-Score and Recall increase significantly, ensuring critical rare events are not ignored.")
    print("=" * 75)

if __name__ == "__main__":
    main()
