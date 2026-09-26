"""
Session 16 - Task 4: Food Delivery Fraud Detection System F1-Score Computation
"""

def main():
    print("=" * 75)
    print("SESSION 16 - TASK 4: Food Delivery Fraud Detection F1-Score")
    print("=" * 75)

    # Given confusion matrix:
    TP = 40
    TN = 150
    FP = 20
    FN = 10

    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    f1_score = 2 * (precision * recall) / (precision + recall)

    print(f"\n1. Inputs: TP={TP}, TN={TN}, FP={FP}, FN={FN}")
    print(f"2. Precision = {precision:.4f} ({precision*100:.2f}%)")
    print(f"3. Recall    = {recall:.4f} ({recall*100:.2f}%)")
    print(f"4. Computed F1 Score = {f1_score:.4f} ({f1_score*100:.2f}%)\n")

    print("ONE-LINE EXPLANATION OF F1-SCORE:")
    print("The F1 score (72.73%) represents the harmonic mean of precision and recall, providing a single balanced metric that evaluates fraud detection accuracy without being inflated by the large number of non-fraudulent orders (TN=150).")
    print("=" * 75)

if __name__ == "__main__":
    main()
