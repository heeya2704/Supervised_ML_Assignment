"""
Session 22 - Task 3: Feature Encoding (One-Hot Encoding) & Scaling (StandardScaler)
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler

def main():
    print("=" * 75)
    print("SESSION 22 - TASK 3: Categorical Encoding & Numerical Feature Scaling")
    print("=" * 75)

    # 1. Load dataset & perform Task 2 cleaning
    df = pd.read_csv("session_22/dataset/zomato_restaurants.csv")
    df["cost"] = df["cost"].fillna(df["cost"].median())
    df["rating"] = df["rating"].fillna(df["rating"].median())

    # IQR Capping for cost
    Q1, Q3 = df["cost"].quantile(0.25), df["cost"].quantile(0.75)
    upper_bound = Q3 + 1.5 * (Q3 - Q1)
    df["cost"] = np.clip(df["cost"], Q1 - 1.5 * (Q3 - Q1), upper_bound)

    # Convert binary online_order column ('Yes'/'No') to 1/0
    df["online_order_encoded"] = (df["online_order"] == "Yes").astype(int)

    # 2. One-Hot Encoding for 'cuisine' and 'location'
    encoder = OneHotEncoder(sparse_output=False, drop="first", handle_unknown="ignore")
    cat_features = ["location", "cuisine"]
    encoded_cat_matrix = encoder.fit_transform(df[cat_features])
    encoded_cat_cols = encoder.get_feature_names_out(cat_features)
    
    df_encoded_cat = pd.DataFrame(encoded_cat_matrix, columns=encoded_cat_cols)

    print(f"\n1. CATEGORICAL ONE-HOT ENCODING SUMMARY:")
    print(f"   • Categorical Columns Encoded : {cat_features}")
    print(f"   • Generated One-Hot Features  : {len(encoded_cat_cols)} columns")
    print(f"   • Sample Encoded Columns      : {list(encoded_cat_cols[:5])}")

    # 3. Standard Scaling for numerical features ('cost', 'votes')
    num_features = ["cost", "votes"]
    scaler = StandardScaler()
    scaled_num_matrix = scaler.fit_transform(df[num_features])
    df_scaled_num = pd.DataFrame(scaled_num_matrix, columns=[f"{col}_scaled" for col in num_features])

    print(f"\n2. STANDARD SCALER SUMMARY:")
    print(f"   • Scaled Numerical Columns   : {num_features}")
    print(f"   • Scaled Cost (Mean / Std)   : {df_scaled_num['cost_scaled'].mean():.4f} / {df_scaled_num['cost_scaled'].std():.4f}")
    print(f"   • Scaled Votes (Mean / Std)  : {df_scaled_num['votes_scaled'].mean():.4f} / {df_scaled_num['votes_scaled'].std():.4f}")

    # Combine all features into final feature matrix X
    X = pd.concat([df_scaled_num, df[["online_order_encoded"]], df_encoded_cat], axis=1)

    print(f"\n3. PROCESSED FEATURE MATRIX:")
    print(f"   • Final Shape : {X.shape[0]} rows x {X.shape[1]} columns")
    print("-" * 75)
    print(X.head(5).to_string(index=False))
    print("-" * 75)
    print("=" * 75)

if __name__ == "__main__":
    main()
