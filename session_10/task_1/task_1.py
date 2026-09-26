"""
Session 10 - Task 1: Sigmoid Function Implementation and Evaluation
"""

import numpy as np

def sigmoid(x):
    """
    Computes the sigmoid (logistic) activation function.
    
    Formula: sigma(x) = 1 / (1 + exp(-x))
    """
    return 1.0 / (1.0 + np.exp(-x))

def main():
    print("=" * 65)
    print("SESSION 10 - TASK 1: Sigmoid Function Evaluation")
    print("=" * 65)

    test_inputs = [-2, 0, 3]

    print(f"{'Input (x)':<15} | {'Formula':<25} | {'Sigmoid Output s(x)':<20}")
    print("-" * 65)

    for x in test_inputs:
        val = sigmoid(x)
        formula_str = f"1 / (1 + e^-({x}))"
        print(f"{x:<15} | {formula_str:<25} | {val:<20.6f}")

    print("-" * 65)
    print("\nPROPERTIES VERIFIED:")
    print(f"1. At x = -2 (negative input): s(-2) = {sigmoid(-2):.6f} (< 0.5)")
    print(f"2. At x =  0 (midpoint):       s(0)  = {sigmoid(0):.6f} (= 0.5 decision boundary)")
    print(f"3. At x =  3 (positive input): s(3)  = {sigmoid(3):.6f} (> 0.5)")
    print("=" * 65)

if __name__ == "__main__":
    main()
