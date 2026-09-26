"""
Session 5 - Task 5
Visualization of Best-Fit Regression Line with Scatter Data and Predictions.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

def main():
    print("=" * 70)
    print("SESSION 5 - TASK 5: Regression Line & Scatter Visualization")
    print("=" * 70)

    # Weekly training data
    instagram_hours = np.array([1.5, 2.0, 3.5, 4.0, 2.5, 5.5, 6.0]).reshape(-1, 1)
    battery_percentage = np.array([88, 82, 65, 58, 72, 40, 32])

    # Fit model
    model = LinearRegression()
    model.fit(instagram_hours, battery_percentage)

    # Generate continuous x points for smooth line rendering
    x_line = np.linspace(0, 7, 100).reshape(-1, 1)
    y_line = model.predict(x_line)

    # Prediction points from Task 3
    test_hours = np.array([[3.0], [4.5], [7.0]])
    test_preds = model.predict(test_hours)

    # Plot graph
    fig, ax = plt.subplots(figsize=(9, 5.5), dpi=300)

    # Scatter actual data
    plt.scatter(instagram_hours, battery_percentage, color='#e1306c', s=110, zorder=4, 
                edgecolors='black', label='Actual Daily Data')

    # Regression Line
    plt.plot(x_line, y_line, color='#1f77b4', linewidth=2.5, zorder=3, 
             label=f'Fitted Regression Line (y = {model.intercept_:.2f} - {abs(model.coef_[0]):.2f}x)')

    # Test Prediction Points
    plt.scatter(test_hours, test_preds, color='#ff7f0e', marker='^', s=130, zorder=5, 
                edgecolors='black', label='Scenario Predictions (3.0, 4.5, 7.0 hrs)')

    plt.title("Simple Linear Regression: Instagram Usage vs Phone Battery %", fontsize=13, weight='bold', pad=15)
    plt.xlabel("Instagram Usage Time (Hours/Day)", fontsize=11)
    plt.ylabel("Phone Battery Remaining (%)", fontsize=11)
    plt.ylim(0, 110)
    plt.xlim(0, 7.5)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(loc='upper right', framealpha=0.95)
    plt.tight_layout()

    output_dir = os.path.dirname(__file__)
    img_path = os.path.join(output_dir, "regression_line.png")
    plt.savefig(img_path, bbox_inches='tight')
    plt.close()

    print(f"[+] Successfully generated and saved plot to: {img_path}")
    print("=" * 70)

if __name__ == "__main__":
    main()
