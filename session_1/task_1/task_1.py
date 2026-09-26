"""
Session 1 - Task 1
List three apps you use daily and describe one feature in each that likely uses machine learning.
For each, state whether it is an example of supervised, unsupervised, or reinforcement learning.
"""

def main():
    print("=" * 70)
    print("SESSION 1 - TASK 1: Daily Apps & Machine Learning Paradigm Analysis")
    print("=" * 70)
    
    apps_info = [
        {
            "app": "1. Zomato",
            "feature": "Estimated Time of Arrival (ETA) & Delivery Time Prediction",
            "description": "Predicts exact food delivery duration based on restaurant prep time, historical distance, live traffic, and weather conditions.",
            "ml_type": "Supervised Learning",
            "justification": "Trained on historical delivery records with known ground truth labels (actual trip durations)."
        },
        {
            "app": "2. Instagram",
            "feature": "Explore Feed & Content Recommendation Engine",
            "description": "Groups users with similar preferences and cluster posts/reels to recommend content aligned with user interests.",
            "ml_type": "Unsupervised Learning (with Supervised Ranking)",
            "justification": "Uses clustering/embeddings (Unsupervised) to discover implicit topic/user groups without explicit human labels, followed by supervised ranking models."
        },
        {
            "app": "3. Flipkart",
            "feature": "Dynamic Personalised Product Recommendation ('Frequently Bought Together')",
            "description": "Analyzes purchasing behavior across millions of users to recommend complementary items.",
            "ml_type": "Unsupervised Learning (Collaborative Filtering / Association Rule Mining)",
            "justification": "Discovers hidden relationships, co-occurrences, and item affinity clusters in unlabeled transactional purchase data."
        }
    ]

    for item in apps_info:
        print(f"\nApp: {item['app']}")
        print(f"Feature: {item['feature']}")
        print(f"Description: {item['description']}")
        print(f"ML Paradigm: {item['ml_type']}")
        print(f"Justification: {item['justification']}")
        print("-" * 70)

if __name__ == "__main__":
    main()
