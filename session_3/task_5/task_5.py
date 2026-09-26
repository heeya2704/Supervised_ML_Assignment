"""
Session 3 - Task 5
Label encode 'player_role' using scikit-learn's LabelEncoder and display class mapping.
"""

import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder

def main():
    print("=" * 70)
    print("SESSION 3 - TASK 5: Label Encoding for 'player_role' & Class Mapping")
    print("=" * 70)

    csv_path = os.path.join(os.path.dirname(__file__), "..", "dataset", "ipl_player_stats.csv")
    df = pd.read_csv(csv_path)

    print("\nOriginal unique roles in 'player_role':", df['player_role'].unique())

    # Initialize and fit LabelEncoder
    le = LabelEncoder()
    df['player_role_encoded'] = le.fit_transform(df['player_role'])

    # Extract mapping using classes_ attribute
    mapping = {index: label for index, label in enumerate(le.classes_)}
    reverse_mapping = {label: index for index, label in enumerate(le.classes_)}

    print("\n" + "=" * 50)
    print("LABEL ENCODER MAPPING (via le.classes_):")
    print("=" * 50)
    for role, code in reverse_mapping.items():
        print(f"  - Original Role: '{role:<12}' ---> Encoded Integer: {code}")
    print("=" * 50)

    print("\nSample DataFrame with Original and Encoded Roles:")
    print("-" * 70)
    print(df[['player_name', 'player_role', 'player_role_encoded']].head(10))
    print("=" * 70)

if __name__ == "__main__":
    main()
