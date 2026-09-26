"""
Session 16 - Task 3: Fake News Detection Model Precision & Recall Calculation
"""

def main():
    print("=" * 75)
    print("SESSION 16 - TASK 3: Fake News Detection Model Metrics")
    print("=" * 75)

    # Given inputs:
    # Flagged as fake: 80 posts (TP + FP = 80)
    # TP = 60 (actually fake posts flagged)
    # FN = 30 (fake posts missed)
    # TN = 120 (real posts correctly identified as not fake)
    # FP = 10 (real posts wrongly flagged as fake)

    TP = 60
    FN = 30
    TN = 120
    FP = 10

    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    f1_score = 2 * (precision * recall) / (precision + recall)
    accuracy = (TP + TN) / (TP + TN + FP + FN)

    print("\n1. FAKE NEWS MODEL CONFUSION MATRIX:")
    print(f"   True Positives  (TP) = {TP} (Correctly flagged fake posts)")
    print(f"   False Positives (FP) = {FP} (Real posts wrongly flagged as fake)")
    print(f"   False Negatives (FN) = {FN} (Fake posts missed by model)")
    print(f"   True Negatives  (TN) = {TN} (Real posts correctly passed)")

    print("\n2. CALCULATED METRICS:")
    print("-" * 65)
    print(f"   Precision = TP / (TP + FP) = {TP} / ({TP} + {FP}) = {precision:.4f} ({precision*100:.2f}%)")
    print(f"   Recall    = TP / (TP + FN) = {TP} / ({TP} + {FN}) = {recall:.4f} ({recall*100:.2f}%)")
    print(f"   F1-Score  = {f1_score:.4f} ({f1_score*100:.2f}%)")
    print(f"   Accuracy  = {accuracy:.4f} ({accuracy*100:.2f}%)")
    print("-" * 65)
    print("=" * 75)

if __name__ == "__main__":
    main()
