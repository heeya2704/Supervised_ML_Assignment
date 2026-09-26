"""
Session 6 - Task 3
Calculate Variance Inflation Factor (VIF) for all features to identify multicollinearity.
"""

import os
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

def calculate_vif(X):
    X_const = sm.add_constant(X)
    vif_data = pd.DataFrame()
    vif_data["Feature"] = X_const.columns
    vif_data["VIF"] = [variance_inflation_factor(X_const.values, i) for i in range(X_const.shape[1])]
    # Return features excluding constant
    return vif_data[vif_data["Feature"] != "const"].reset_index(drop=True)

def main():
    print("=" * 70)
    print("SESSION 6 - TASK 3: Variance Inflation Factor (VIF) Multicollinearity Analysis")
    print("=" * 70)

    dataset_path = os.path.join(os.path.dirname(__file__), "..", "dataset", "mobile_prices.csv")
    df = pd.read_csv(dataset_path)

    feature_cols = ['ram_gb', 'storage_gb', 'battery_mah', 'camera_mp', 'screen_size_inch']
    X = df[feature_cols]

    vif_df = calculate_vif(X)

    print("\nVARIANCE INFLATION FACTOR (VIF) RESULTS:")
    print("-" * 65)
    print(f"{'Feature':<20} | {'VIF Value':<12} | {'Multicollinearity Status':<25}")
    print("-" * 65)
    for idx, row in vif_df.iterrows():
        feat = row['Feature']
        vif_val = row['VIF']
        if vif_val > 10:
            status = "HIGH (> 10) - Action Needed"
        elif vif_val > 5:
            status = "MODERATE (> 5)"
        else:
            status = "LOW (<= 5)"
        print(f"{feat:<20} | {vif_val:<12.2f} | {status:<25}")
    print("-" * 65)

    high_vif_feats = vif_df[vif_df['VIF'] > 5]['Feature'].tolist()
    print("\nMULTICOLLINEARITY DIAGNOSIS:")
    print(f"[!] Features exhibiting problematic multicollinearity (VIF > 5): {high_vif_feats}")
    print("    Note: storage_gb & ram_gb, as well as screen_size_inch & battery_mah show high correlation.")
    print("=" * 70)

if __name__ == "__main__":
    main()
