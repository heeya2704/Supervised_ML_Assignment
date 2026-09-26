"""
Session 22 - Task 5: Class Balancing with SMOTE & Hyperparameter Tuning via GridSearchCV
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from imblearn.over_sampling import SMOTE

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
    print("SESSION 22 - TASK 5: SMOTE Resampling & GridSearchCV Hyperparameter Tuning")
    print("=" * 75)

    X, y = prepare_data()

    # 1. Train-Test Split (80/20)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    print("\n1. INITIAL CLASS DISTRIBUTION (TRAINING SET):")
    print(f"   • Class 0 (Rating <= 4.0) : {(y_train == 0).sum()} samples")
    print(f"   • Class 1 (Rating > 4.0)  : {(y_train == 1).sum()} samples (Minority)")

    # 2. Apply SMOTE to balance minority class
    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

    print("\n2. POST-SMOTE CLASS DISTRIBUTION (BALANCED):")
    print(f"   • Class 0 (Rating <= 4.0) : {(y_train_res == 0).sum()} samples")
    print(f"   • Class 1 (Rating > 4.0)  : {(y_train_res == 1).sum()} samples (Balanced)")

    # 3. Define GridSearchCV parameter grid for RandomForestClassifier
    param_grid = {
        "n_estimators": [50, 100, 200],
        "max_depth": [3, 5, 8, 12, None],
        "min_samples_split": [2, 5, 10]
    }

    print("\n3. HYPERPARAMETER SEARCH GRID:")
    for param, values in param_grid.items():
        print(f"   • {param} : {values}")

    rf = RandomForestClassifier(random_state=42)
    grid_search = GridSearchCV(
        estimator=rf,
        param_grid=param_grid,
        cv=5,
        scoring="roc_auc",
        n_jobs=-1
    )

    grid_search.fit(X_train_res, y_train_res)

    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_
    best_cv_score = grid_search.best_score_

    # 4. Evaluate Tuned Model on Unseen Test Set
    y_pred_proba_tuned = best_model.predict_proba(X_test)[:, 1]
    test_roc_auc = roc_auc_score(y_test, y_pred_proba_tuned)

    print(f"\n4. GRIDSEARCHCV BEST RESULTS:")
    print("-" * 65)
    print(f"   • Best Hyperparameters  : {best_params}")
    print(f"   • Best 5-Fold CV ROC-AUC: {best_cv_score:.4f}")
    print(f"   • Final Test ROC-AUC    : {test_roc_auc:.4f}")
    print("-" * 65)
    print("=" * 75)

if __name__ == "__main__":
    main()
