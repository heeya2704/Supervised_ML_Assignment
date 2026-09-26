"""
Session 2 - Task 4
Plot Bias-Variance Tradeoff graph and provide Zomato restaurant rating prediction application note.
"""

import os
import numpy as np
import matplotlib.pyplot as plt

def plot_bias_variance_tradeoff(output_path):
    fig, ax = plt.subplots(figsize=(9, 5.5), dpi=300)
    
    complexity = np.linspace(1, 10, 200)
    
    # Mathematical curves representing Bias, Variance, Total Error
    bias_sq = 9.0 / (complexity**0.8)
    variance = 0.1 * (complexity**2.2)
    irreducible_error = np.full_like(complexity, 1.2)
    total_error = bias_sq + variance + irreducible_error
    
    opt_idx = np.argmin(total_error)
    opt_complexity = complexity[opt_idx]
    
    plt.plot(complexity, bias_sq, 'b--', label=r'Bias$^2$ (Underfitting)', linewidth=2.5)
    plt.plot(complexity, variance, 'g--', label='Variance (Overfitting)', linewidth=2.5)
    plt.plot(complexity, total_error, 'r-', label='Total Error', linewidth=3)
    plt.axhline(y=1.2, color='gray', linestyle=':', label='Irreducible Error')
    
    plt.axvline(x=opt_complexity, color='black', linestyle='-.', alpha=0.7, label='Optimal Complexity')
    
    plt.title("The Bias - Variance Tradeoff", fontsize=14, weight='bold', pad=15)
    plt.xlabel("Model Complexity (Degree of Polynomial / Model Depth)", fontsize=12)
    plt.ylabel("Error Rate", fontsize=12)
    
    # Annotations
    plt.text(1.5, 7.5, "High Bias\n(Underfitting)", fontsize=11, color='blue', weight='bold', bbox=dict(boxstyle="round,pad=0.3", fc="#e6f0ff", ec="blue"))
    plt.text(7.5, 7.5, "High Variance\n(Overfitting)", fontsize=11, color='green', weight='bold', bbox=dict(boxstyle="round,pad=0.3", fc="#e6ffe6", ec="green"))
    plt.text(opt_complexity - 0.7, 0.4, "Optimal Model", fontsize=10, color='black', weight='bold')
    
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(loc='upper center', frameon=True, facecolor='white', framealpha=0.9)
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    print(f"[+] Saved Bias-Variance Tradeoff graph to: {output_path}")

def main():
    print("=" * 70)
    print("SESSION 2 - TASK 4: Bias-Variance Tradeoff Graph & Zomato App Analysis")
    print("=" * 70)
    
    note = """
BIAS-VARIANCE TRADEOFF IN A ZOMATO RESTAURANT RATING PREDICTOR APP:

1. High Bias (Underfitting):
   - Occurs when using an oversimplified model (e.g., predicting restaurant rating based ONLY on average cost).
   - The model ignores key drivers like cuisine quality, location, ambiance, and review volume.
   - Impact on Zomato: Consistently makes inaccurate, generic rating predictions for both luxury fine-dining and popular street food spots.

2. High Variance (Overfitting):
   - Occurs when using an overly complex model (e.g., deep decision tree with no max-depth limit memorizing specific review text typos).
   - The model fits random noise and specific customer quirks in the training dataset.
   - Impact on Zomato: Works perfectly on historical restaurants but fails horribly when new restaurants are added, yielding erratic, unreliable rating predictions.

3. Optimal Tradeoff (Balanced Generalization):
   - Selects a model with moderate complexity that captures essential feature interactions (cost + location + cuisine type + aggregated review sentiment) while ignoring random noise.
   - Result: Yields accurate, robust rating predictions on newly launched restaurants across all cities.
"""
    print(note.strip())
    print("=" * 70)
    
    img_path = os.path.join(os.path.dirname(__file__), "bias_variance_tradeoff.png")
    plot_bias_variance_tradeoff(img_path)

if __name__ == "__main__":
    main()
