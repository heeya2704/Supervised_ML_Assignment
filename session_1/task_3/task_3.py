"""
Session 1 - Task 3
Draw a simple diagram showing the basic ML workflow: Data -> Preprocessing -> Modeling -> Evaluation -> Deployment.
Under each step, write a one-line example of what happens at that stage for a food delivery app like Swiggy.
"""

import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def create_ml_workflow_diagram(output_path):
    fig, ax = plt.subplots(figsize=(12, 5), dpi=300)
    ax.axis('off')
    
    steps = [
        {"title": "1. Data Collection", "desc": "Gather GPS tracks, order history,\nkitchen prep time & traffic logs."},
        {"title": "2. Preprocessing", "desc": "Clean missing timestamps, scale distances\n& encode restaurant cuisine categories."},
        {"title": "3. Modeling", "desc": "Train a Gradient Boosting Regressor\nto estimate dish prep & delivery time."},
        {"title": "4. Evaluation", "desc": "Calculate Mean Absolute Error (MAE) between\npredicted ETA and actual trip duration."},
        {"title": "5. Deployment", "desc": "Deploy trained model via microservice API\nto show live ETA on the Swiggy customer app."}
    ]
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    n = len(steps)
    box_width = 1.8
    box_height = 1.6
    spacing = 0.5
    start_x = 0.5
    y_pos = 1.5
    
    for i, step in enumerate(steps):
        x = start_x + i * (box_width + spacing)
        rect = patches.FancyBboxPatch(
            (x, y_pos), box_width, box_height,
            boxstyle="round,pad=0.2",
            fc=colors[i], ec="black", lw=1.5, alpha=0.85
        )
        ax.add_patch(rect)
        
        ax.text(
            x + box_width / 2, y_pos + box_height - 0.35, step["title"],
            color='white', weight='bold', fontsize=11, ha='center', va='center'
        )
        
        ax.text(
            x + box_width / 2, y_pos + box_height / 2 - 0.25, step["desc"],
            color='white', fontsize=8.5, ha='center', va='center', style='italic'
        )
        
        if i < n - 1:
            arrow_x = x + box_width + 0.05
            ax.annotate(
                '', xy=(arrow_x + spacing - 0.1, y_pos + box_height / 2),
                xytext=(arrow_x, y_pos + box_height / 2),
                arrowprops=dict(facecolor='#333333', shrink=0.05, width=2, headwidth=8)
            )
            
    plt.title("Machine Learning Workflow — Food Delivery App (Swiggy ETA Prediction)", fontsize=14, weight='bold', pad=20)
    plt.xlim(0, start_x + n * (box_width + spacing))
    plt.ylim(0.5, 3.8)
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    print(f"[+] Successfully saved ML Workflow diagram to: {output_path}")

def main():
    print("=" * 70)
    print("SESSION 1 - TASK 3: Basic ML Workflow & Swiggy Example")
    print("=" * 70)
    
    workflow_steps = [
        ("Step 1: Data Collection", "Swiggy collects raw telemetry data including restaurant kitchen prep times, customer locations, delivery agent GPS coordinates, live traffic data, and weather conditions."),
        ("Step 2: Preprocessing", "Swiggy handles missing values in delivery logs, encodes categorical variables (e.g., cuisine type, payment mode), and normalizes continuous features like route distance."),
        ("Step 3: Modeling", "Swiggy trains a regression model (e.g., XGBoost or Random Forest Regressor) to accurately map input telemetry features to expected delivery duration."),
        ("Step 4: Evaluation", "Swiggy measures model accuracy on historical test orders using evaluation metrics like Mean Absolute Error (MAE) and Root Mean Squared Error (RMSE)."),
        ("Step 5: Deployment", "Swiggy containerizes the trained ML model and deploys it as a real-time microservice API that updates the live ETA on the user's mobile app during checkout.")
    ]
    
    for step, example in workflow_steps:
        print(f"\n{step}:")
        print(f"  Swiggy Example: {example}")
    
    img_path = os.path.join(os.path.dirname(__file__), "ml_workflow_swiggy.png")
    create_ml_workflow_diagram(img_path)

if __name__ == "__main__":
    main()
