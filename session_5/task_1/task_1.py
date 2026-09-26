"""
Session 5 - Task 1
Generate Instagram usage vs Battery percentage data and create a scatter plot.
"""

import os
import numpy as np
import matplotlib.pyplot as plt

def generate_data():
    # 7 days of data
    days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    instagram_hours = np.array([1.5, 2.0, 3.5, 4.0, 2.5, 5.5, 6.0]) # Independent X
    battery_percentage = np.array([88, 82, 65, 58, 72, 40, 32])   # Dependent Y
    return days, instagram_hours, battery_percentage

def plot_scatter(instagram_hours, battery_percentage, output_path):
    fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
    
    plt.scatter(instagram_hours, battery_percentage, color='#e1306c', s=100, zorder=3, edgecolors='black', label='Daily Data Points')
    
    plt.title("Instagram Daily Usage vs End-of-Day Phone Battery %", fontsize=13, weight='bold', pad=15)
    plt.xlabel("Instagram Usage Time (Hours/Day)", fontsize=11)
    plt.ylabel("Phone Battery Remaining (%)", fontsize=11)
    plt.ylim(0, 100)
    plt.xlim(0, 7)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(loc='upper right')
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    print(f"[+] Saved scatter plot to: {output_path}")

def main():
    print("=" * 70)
    print("SESSION 5 - TASK 1: Daily Data Generation & Scatter Plot")
    print("=" * 70)

    days, instagram_hours, battery_percentage = generate_data()

    print("\nGenerated Daily Data:")
    print("-" * 50)
    print(f"{'Day':<6} | {'Instagram Hours':<18} | {'Battery Remaining (%)':<22}")
    print("-" * 50)
    for d, h, b in zip(days, instagram_hours, battery_percentage):
        print(f"{d:<6} | {h:<18.1f} | {b:<22}")
    print("-" * 50)

    img_path = os.path.join(os.path.dirname(__file__), "scatter_plot.png")
    plot_scatter(instagram_hours, battery_percentage, img_path)

if __name__ == "__main__":
    main()
