"""
Session 10 - Task 3: Plotting the Sigmoid Curve with 0.5 Decision Threshold
"""

import os
import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def main():
    print("=" * 65)
    print("SESSION 10 - TASK 3: Plotting Sigmoid Activation Curve")
    print("=" * 65)

    x = np.linspace(-10, 10, 500)
    y = sigmoid(x)

    plt.figure(figsize=(9, 6), dpi=300)
    plt.plot(x, y, color='#2563eb', linewidth=3, label=r'$\sigma(x) = \frac{1}{1 + e^{-x}}$')

    # Threshold horizontal line at y = 0.5
    plt.axhline(y=0.5, color='#dc2626', linestyle='--', linewidth=2, label='Decision Threshold (y = 0.5)')
    
    # Midpoint vertical line at x = 0
    plt.axvline(x=0, color='#6b7280', linestyle=':', linewidth=1.5, label='Decision Boundary (x = 0)')

    # Highlight point (0, 0.5)
    plt.scatter([0], [0.5], color='#dc2626', s=100, zorder=5)
    plt.text(0.3, 0.52, 'Midpoint (0, 0.5)', fontsize=11, fontweight='bold', color='#dc2626')

    plt.title('Sigmoid Activation Function & 0.5 Decision Threshold', fontsize=14, fontweight='bold', pad=15)
    plt.xlabel('Input Value (x)', fontsize=12, labelpad=10)
    plt.ylabel('Sigmoid Probability Output s(x)', fontsize=12, labelpad=10)
    plt.ylim(-0.05, 1.05)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(fontsize=11, loc='upper left')

    output_dir = os.path.dirname(__file__)
    file_path = os.path.join(output_dir, 'sigmoid_plot.png')
    plt.savefig(file_path, bbox_inches='tight')
    plt.close()

    print(f"Sigmoid plot successfully generated and saved to: {file_path}")
    print("=" * 65)

if __name__ == "__main__":
    main()
