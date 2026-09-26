"""
Session 8 - Task 1
PolynomialFeatures(degree=2) transformation on Mobile Phone RAM and Storage features.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures

def main():
    print("=" * 70)
    print("SESSION 8 - TASK 1: PolynomialFeatures (Degree = 2) Transformation")
    print("=" * 70)

    # Mobile phone feature dataset (RAM in GB, Storage in GB)
    mobile_data = pd.DataFrame({
        'ram_gb': [4, 6, 8, 12, 16],
        'storage_gb': [64, 128, 128, 256, 512]
    })

    print("\nOriginal Input Feature Matrix (RAM, Storage):")
    print("-" * 45)
    print(mobile_data)
    print("-" * 45)

    # PolynomialFeatures Degree 2
    poly = PolynomialFeatures(degree=2, include_bias=True)
    X_poly = poly.fit_transform(mobile_data)
    feature_names = poly.get_feature_names_out(mobile_data.columns)

    df_poly = pd.DataFrame(X_poly, columns=feature_names)

    print("\nDegree-2 Polynomial Transformed Feature Matrix:")
    print("-" * 75)
    print(df_poly)
    print("-" * 75)

    print("\nGENERATED POLYNOMIAL FEATURE TERMS:")
    for name in feature_names:
        print(f"  - {name}")
    print("=" * 70)

if __name__ == "__main__":
    main()
