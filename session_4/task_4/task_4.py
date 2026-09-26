"""
Session 4 - Task 4
RFE (Recursive Feature Elimination) with DecisionTreeRegressor on Zomato dataset to select 3 features for 'average_cost_for_two'.
"""

import os
import pandas as pd
from sklearn.feature_selection import RFE
from sklearn.tree import DecisionTreeRegressor

def main():
    print("=" * 70)
    print("SESSION 4 - TASK 4: RFE Feature Selection on Zomato Dataset")
    print("=" * 70)

    csv_path = os.path.join(os.path.dirname(__file__), "..", "dataset", "zomato_restaurants.csv")
    df = pd.read_csv(csv_path)

    target_col = 'average_cost_for_two'
    X = df.drop(columns=[target_col])
    y = df[target_col]

    # Initialize DecisionTreeRegressor and RFE with n_features_to_select=3
    estimator = DecisionTreeRegressor(random_state=42)
    rfe = RFE(estimator=estimator, n_features_to_select=3)
    rfe.fit(X, y)

    # Compile ranking results
    feature_ranking = pd.DataFrame({
        'Feature': X.columns,
        'Selected': rfe.support_,
        'RFE_Rank': rfe.ranking_
    }).sort_values(by='RFE_Rank')

    print("\nFull RFE Feature Ranking:")
    print("-" * 50)
    print(feature_ranking.to_string(index=False))
    print("-" * 50)

    selected_features = feature_ranking[feature_ranking['Selected']]['Feature'].tolist()

    print("\n" + "=" * 70)
    print("TOP 3 SELECTED FEATURES BY RFE:")
    for idx, feature in enumerate(selected_features, 1):
        print(f"  {idx}. {feature} (RFE Rank: 1)")
    print("=" * 70)

if __name__ == "__main__":
    main()
