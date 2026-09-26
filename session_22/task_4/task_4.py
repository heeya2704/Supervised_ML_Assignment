"""
Session 22 - Task 4: High Rating Prediction - Logistic Regression vs Random Forest (ROC-AUC Comparison)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, roc_auc_score

def prepare_data():
    df = pd.read_csv("session_22/dataset/zomato_restaurants.csv")
    df["cost"] = df["cost"].fillna(df["cost"].median())
    df["rating"] = df["rating"].fillna(df["rating"].median())

    # IQR Capping for cost
    Q1, Q3 = df["cost"].quantile(0.25), df["cost"].quantile(0.75)
    upper_bound = Q3 + 1.5 * (Q3 - Q1)
    df["cost"] = np.clip(df["cost"], Q1 - 1.5 * (Q3 - Q1), upper_bound)

    # Binary Target: 1 if rating > 4.0, else 0
    y = (df["rating"] > 4.0).astype(int)

    # Features
    df["online_order_encoded"] = (df["online_order"] == "Yes").astype(int)

    encoder = OneHotEncoder(sparse_output=False, drop="first", handle_unknown="ignore")
    cat_features = ["location", "cuisine"]
    encoded_cat_matrix = encoder.fit_transform(df[cat_features])
    encoded_cat_cols = encoder.get_feature_names_out(cat_features)
    df_encoded_cat = pd.DataFrame(encoded_cat_matrix, columns=encoded_cat_cols)

    scaler = StandardScaler()
    scaled_num = scaler.fit_transform(df[["cost", "votes"]])
    df_scaled_num = pd.DataFrame(scaled_num, columns=["cost_scaled", "votes_scaled"])

    X = pd.concat([df_scaled_num, df[["online_order_encoded"]], df_encoded_cat], axis=1)
    return X, y

def main():
    print("=" * 75)
    print("SESSION 22 - TASK 4: Binary Classification & Model Comparison (Rating > 4.0)")
    print("=" * 75)

    X, y = prepare_data()

    # 1. 80/20 Stratified Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    print(f"\n1. TRAIN / TEST SPLIT SUMMARY:")
    print(f"   • Total Dataset Size   : {len(X)} samples")
    print(f"   • Training Set Size    : {len(X_train)} samples (80%)")
    print(f"   • Test Set Size        : {len(X_test)} samples (20%)")
    print(f"   • Class Ratio (>4.0)   : {y.mean()*100:.2f}% Positive Class")

    # 2. Train Logistic Regression
    lr = LogisticRegression(random_state=42, max_iter=1000)
    lr.fit(X_train, y_train)
    y_pred_proba_lr = lr.predict_proba(X_test)[:, 1]
    auc_lr = roc_auc_score(y_test, y_pred_proba_lr)

    # 3. Train Random Forest Classifier
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    y_pred_proba_rf = rf.predict_proba(X_test)[:, 1]
    auc_rf = roc_auc_score(y_test, y_pred_proba_rf)

    print(f"\n2. ROC-AUC SCORE PERFORMANCE COMPARISON:")
    print("-" * 65)
    print(f"   • Logistic Regression ROC-AUC  : {auc_lr:.4f}")
    print(f"   • Random Forest ROC-AUC        : {auc_rf:.4f}")
    print("-" * 65)

    if auc_rf > auc_lr:
        winner = f"Random Forest Classifier (Outperformed LR by +{(auc_rf - auc_lr):.4f} ROC-AUC)"
    else:
        winner = f"Logistic Regression (Outperformed RF by +{(auc_lr - auc_rf):.4f} ROC-AUC)"

    print(f"\n3. BEST PERFORMING MODEL: {winner}")

    # 4. Plot ROC Curve Comparison
    fpr_lr, tpr_lr, _ = roc_curve(y_test, y_pred_proba_lr)
    fpr_rf, tpr_rf, _ = roc_curve(y_test, y_pred_proba_rf)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr_lr, tpr_lr, color="dodgerblue", lw=2.5, label=f"Logistic Regression (AUC = {auc_lr:.4f})")
    plt.plot(fpr_rf, tpr_rf, color="darkorange", lw=2.5, label=f"Random Forest (AUC = {auc_rf:.4f})")
    plt.plot([0, 1], [0, 1], color="gray", linestyle="--", lw=1.5, label="Random Guess (AUC = 0.50)")

    plt.title("Session 22 Task 4: ROC Curve Comparison (Zomato Rating > 4.0)", fontsize=13)
    plt.xlabel("False Positive Rate (1 - Specificity)")
    plt.ylabel("True Positive Rate (Recall)")
    plt.legend(loc="lower right")
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()

    plot_path = "session_22/task_4/roc_curve_comparison.png"
    plt.savefig(plot_path, dpi=300)
    plt.close()

    print(f"\n4. Saved ROC Curve plot to: '{plot_path}'")
    print("=" * 75)

if __name__ == "__main__":
    main()
