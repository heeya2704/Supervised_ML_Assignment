"""
Session 7 - Task 3
Residual Plot for Actual vs Predicted Movie Ratings (Scale 1-5) for 20 Movies.
"""

import os
import numpy as np
import matplotlib.pyplot as plt

def main():
    print("=" * 70)
    print("SESSION 7 - TASK 3: Residual Plot Analysis for Movie Ratings")
    print("=" * 70)

    np.random.seed(42)
    # 20 movie actual ratings (scale 1.0 to 5.0)
    actual_ratings = np.array([3.5, 4.0, 2.0, 4.5, 5.0, 1.5, 3.0, 4.2, 2.8, 3.9,
                               4.8, 1.0, 3.2, 4.1, 2.5, 4.7, 3.6, 2.2, 4.4, 3.8])
    
    # Predicted ratings with small random noise
    predicted_ratings = actual_ratings + np.random.normal(0, 0.35, size=20)
    predicted_ratings = np.clip(predicted_ratings, 1.0, 5.0)

    # Compute Residuals = Actual - Predicted
    residuals = actual_ratings - predicted_ratings

    print("\nRESIDUAL COMPUTATION SAMPLES (First 5 Movies):")
    print("-" * 65)
    print(f"{'Movie #':<10} | {'Actual Rating':<16} | {'Predicted Rating':<18} | {'Residual Error':<15}")
    print("-" * 65)
    for i in range(5):
        print(f"Movie #{i+1:<3} | {actual_ratings[i]:<16.2f} | {predicted_ratings[i]:<18.2f} | {residuals[i]:<15.2f}")
    print("-" * 65)

    # Create Residual Plot
    fig, ax = plt.subplots(figsize=(8.5, 5), dpi=300)

    plt.scatter(predicted_ratings, residuals, color='#2ca02c', s=90, zorder=4, 
                edgecolors='black', label='Movie Residuals')
    
    # Reference Line at zero error
    plt.axhline(0, color='red', linestyle='--', linewidth=2, zorder=3, label='Zero Error Reference (y = 0)')

    plt.title("Residual Plot: Movie Rating Predictions (Scale 1.0 - 5.0)", fontsize=13, weight='bold', pad=15)
    plt.xlabel("Predicted Movie Rating (Scale 1.0 - 5.0)", fontsize=11)
    plt.ylabel("Residuals (Actual - Predicted Rating)", fontsize=11)
    plt.ylim(-1.5, 1.5)
    plt.xlim(0.5, 5.5)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(loc='upper right')
    plt.tight_layout()

    img_path = os.path.join(os.path.dirname(__file__), "residual_plot.png")
    plt.savefig(img_path, bbox_inches='tight')
    plt.close()

    print(f"\n[+] Saved residual plot visualization to: {img_path}")
    print("\nRESIDUAL PLOT INTERPRETATION:")
    print("1. Homoscedasticity: Residuals are randomly scattered above and below the zero line without clear patterns.")
    print("2. Unbiased Predictions: Points are evenly distributed around 0, indicating the model does not consistently over-predict or under-predict ratings.")
    print("=" * 70)

if __name__ == "__main__":
    main()
