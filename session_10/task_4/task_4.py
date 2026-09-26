"""
Session 10 - Task 4: Customizable Probability Classification Function
"""

def classify_virality(probability, threshold=0.5):
    """
    Classifies a post's virality probability into 'Viral' or 'Not Viral'
    based on a customizable decision threshold.
    
    Parameters:
        probability (float): Model predicted probability (between 0.0 and 1.0)
        threshold (float): Decision threshold cutoff (default 0.5)
        
    Returns:
        str: 'Viral' if probability >= threshold else 'Not Viral'
    """
    return "Viral" if probability >= threshold else "Not Viral"

def main():
    print("=" * 70)
    print("SESSION 10 - TASK 4: Custom Threshold Virality Classification")
    print("=" * 70)

    test_probabilities = [0.2, 0.6, 0.8]
    test_thresholds = [0.5, 0.7]

    for thresh in test_thresholds:
        print(f"\nEvaluating Decision Threshold = {thresh}:")
        print("-" * 55)
        print(f"{'Input Probability':<20} | {'Threshold':<15} | {'Classification Output':<18}")
        print("-" * 55)
        for prob in test_probabilities:
            label = classify_virality(prob, threshold=thresh)
            print(f"{prob:<20.2f} | {thresh:<15.2f} | {label:<18}")
        print("-" * 55)

    print("\nINSIGHTS ON THRESHOLD TUNING:")
    print("1. Standard Threshold (0.5): Probability 0.6 and 0.8 are classified as 'Viral'. Only 0.2 is 'Not Viral'.")
    print("2. Strict Threshold (0.7): Raises precision requirement. Probability 0.6 shifts from 'Viral' to 'Not Viral'.")
    print("3. Business Application: Raising thresholds reduces false positives (conserves advertising budget).")
    print("=" * 70)

if __name__ == "__main__":
    main()
