"""
Session 6 - Task 5
Refactor regression model by removing highly collinear feature (storage_gb), re-fit model, and compare metrics.
"""

import os
import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, root_mean_squared_error

def calculate_vif(X):
    X_const = sm.add_constant(X)
    vif_data = pd.DataFrame()
    vif_data["Feature"] = X_const.columns
    vif_data["VIF"] = [variance_inflation_factor(X_const.values, i) for i in range(X_const.shape[1])]
    return vif_data[vif_data["Feature"] != "const"].reset_index(drop=True)

def main():
    print("=" * 70)
    print("SESSION 6 - TASK 5: Model Refactoring & Collinear Feature Removal")
    print("=" * 70)

    dataset_path = os.path.join(os.path.dirname(__file__), "..", "dataset", "mobile_prices.csv")
    df = pd.read_csv(dataset_path)

    y = df['price_inr']

    # Original features (Full Model)
    cols_original = ['ram_gb', 'storage_gb', 'battery_mah', 'camera_mp', 'screen_size_inch']
    X_orig = df[cols_original]

    model_orig = LinearRegression()
    model_orig.fit(X_orig, y)
    y_pred_orig = model_orig.predict(X_orig)

    r2_orig = r2_score(y, y_pred_orig)
    rmse_orig = root_mean_squared_error(y, y_pred_orig)
    vif_orig = calculate_vif(X_orig)

    # Refactored features (Remove 'storage_gb' which had VIF ~3060)
    cols_refactored = ['ram_gb', 'battery_mah', 'camera_mp', 'screen_size_inch']
    X_refact = df[cols_refactored]

    model_refact = LinearRegression()
    model_refact.fit(X_refact, y)
    y_pred_refact = model_refact.predict(X_refact)

    r2_refact = r2_score(y, y_pred_refact)
    rmse_refact = root_mean_squared_error(y, y_pred_refact)
    vif_refact = calculate_vif(X_refact)

    print("\nORIGINAL VS REFACTORED MODEL COMPARISON:")
    print("-" * 65)
    print(f"{'Metric / Feature':<22} | {'Original (5 Features)':<18} | {'Refactored (4 Features)':<20}")
    print("-" * 65)
    print(f"{'R^2 Score':<22} | {r2_orig:<18.4f} | {r2_refact:<20.4f}")
    print(f"{'RMSE (Rs.)':<22} | {rmse_orig:<18.2f} | {rmse_refact:<20.2f}")
    print(f"{'Intercept (b)':<22} | {model_orig.intercept_:<18.2f} | {model_refact.intercept_:<20.2f}")
    print("-" * 65)

    print("\nCOEFFICIENTS COMPARISON:")
    for feat in cols_original:
        c_orig = dict(zip(cols_original, model_orig.coef_)).get(feat, np.nan)
        c_ref = dict(zip(cols_refactored, model_refact.coef_)).get(feat, "REMOVED")
        if isinstance(c_ref, float):
            print(f"  - {feat:<18} : Orig Coef = Rs. {c_orig:>8.2f} | Refactored Coef = Rs. {c_ref:>8.2f}")
        else:
            print(f"  - {feat:<18} : Orig Coef = Rs. {c_orig:>8.2f} | Refactored Coef = {c_ref}")

    print("\nVIF SCORES COMPARISON:")
    print("-" * 65)
    vif_merged = pd.merge(vif_orig, vif_refact, on="Feature", how="left", suffixes=('_Orig', '_Refact'))
    for idx, row in vif_merged.iterrows():
        f = row['Feature']
        vo = row['VIF_Orig']
        vr = row['VIF_Refact']
        vr_str = f"{vr:.2f}" if not np.isnan(vr) else "REMOVED"
        print(f"  - {f:<18} : Orig VIF = {vo:>8.2f} | Refactored VIF = {vr_str}")
    print("-" * 65)

    print("\nSUMMARY CONCLUSION:")
    print("1. Removing the highly collinear 'storage_gb' feature stabilized feature VIFs (ram_gb VIF dropped dramatically).")
    print("2. The model retained virtually identical R^2 and RMSE predictive performance while eliminating parameter redundancy.")
    print("=" * 70)

if __name__ == "__main__":
    main()
