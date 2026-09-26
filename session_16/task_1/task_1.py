"""
Session 16 - Task 1: Manual Calculation of Confusion Matrix Metrics (Spam Filter)
"""

def main():
    print("=" * 75)
    print("SESSION 16 - TASK 1: Manual Confusion Matrix Metrics Calculation")
    print("=" * 75)

    # Confusion matrix parameters given in prompt:
    # Actual: Spam = 70, Not Spam = 130 (Total N = 200)
    # TP = 50 (Spam correctly flagged as Spam)
    # TN = 120 (Not Spam correctly identified as Not Spam)
    # FP = 10 (Not Spam wrongly flagged as Spam)
    # FN = 20 (Spam missed and marked as Not Spam)

    TP, TN, FP, FN = 50, 120, 10, 20
    N = TP + TN + FP + FN

    # Metric formulas
    accuracy = (TP + TN) / N
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    f1_score = 2 * (precision * recall) / (precision + recall)

    print("\n1. GIVEN CONFUSION MATRIX VALUES:")
    print(f"   True Positives  (TP) = {TP}")
    print(f"   True Negatives  (TN) = {TN}")
    print(f"   False Positives (FP) = {FP}")
    print(f"   False Negatives (FN) = {FN}")
    print(f"   Total Samples   (N)  = {N}")

    print("\n2. CALCULATED EVALUATION METRICS:")
    print("-" * 65)
    print(f"   Accuracy  = (TP + TN) / N             = ({TP} + {TN}) / {N} = {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"   Precision = TP / (TP + FP)            = {TP} / ({TP} + {FP})     = {precision:.4f} ({precision*100:.2f}%)")
    print(f"   Recall    = TP / (TP + FN)            = {TP} / ({TP} + {FN})     = {recall:.4f} ({recall*100:.2f}%)")
    print(f"   F1 Score  = 2*(Prec*Rec)/(Prec+Rec)   = 2*({precision:.4f}*{recall:.4f})/({precision:.4f}+{recall:.4f}) = {f1_score:.4f} ({f1_score*100:.2f}%)")
    print("-" * 65)
    print("=" * 75)

if __name__ == "__main__":
    main()
