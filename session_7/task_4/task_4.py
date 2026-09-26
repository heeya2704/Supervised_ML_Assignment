"""
Session 7 - Task 4
Swiggy Delivery ETA Model Selection: Comparative Metric Analysis (Model A vs Model B).
"""

def evaluate_swiggy_models():
    models = {
        'Model A': {'RMSE': 8.0, 'R2': 0.65},
        'Model B': {'RMSE': 6.0, 'R2': 0.55}
    }

    print("=" * 70)
    print("SESSION 7 - TASK 4: Swiggy Delivery Time Model Comparison & Selection")
    print("=" * 70)

    print("\nMODEL METRICS SUMMARY:")
    print("-" * 55)
    print(f"{'Model Name':<12} | {'RMSE (Minutes)':<18} | {'R^2 Score':<15}")
    print("-" * 55)
    for name, m in models.items():
        print(f"{name:<12} | {m['RMSE']:<18.1f} | {m['R2']:<15.2f}")
    print("-" * 55)

    print("\nRECOMMENDED SELECTION: MODEL B")
    print("\nEXPLANATION & REASONING:")
    print("1. Operational Customer Impact (RMSE): In food delivery services like Swiggy, minimizing the actual ETA prediction error margin is paramount.")
    print("   Model B's RMSE of 6 minutes is 2 minutes lower than Model A's 8 minutes (a 25% reduction in prediction error magnitude).")
    print("2. Trade-off Context (R^2 vs RMSE): While Model A has a higher R^2 (0.65 vs 0.55), R^2 measures proportion of variance explained.")
    print("   For customer-facing ETA promises, smaller absolute time deviations (6 mins vs 8 mins) directly improve user experience and reduce late order complaints.")
    print("=" * 70)

if __name__ == "__main__":
    evaluate_swiggy_models()
