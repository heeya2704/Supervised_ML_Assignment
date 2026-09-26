"""
Session 16 - Task 2: Custom print_confusion_matrix Function
"""

def print_confusion_matrix(tp, tn, fp, fn, title="CONFUSION MATRIX DISPLAY"):
    """
    Prints a confusion matrix in a clean, human-readable table format.
    
    Parameters:
        tp (int): True Positives
        tn (int): True Negatives
        fp (int): False Positives
        fn (int): False Negatives
        title (str): Header title
    """
    total = tp + tn + fp + fn
    accuracy = (tp + tn) / total if total > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    header_col = "Actual \\ Predicted"

    print("=" * 68)
    print(f" {title.upper()}")
    print("=" * 68)
    print(f"{header_col:<22} | {'Predicted Positive (1)':<20} | {'Predicted Negative (0)':<20}")
    print("-" * 68)
    print(f"{'Actual Positive (1)':<22} | {'TP = ' + str(tp):<20} | {'FN = ' + str(fn):<20}")
    print(f"{'Actual Negative (0)':<22} | {'FP = ' + str(fp):<20} | {'TN = ' + str(tn):<20}")
    print("-" * 68)
    print("SUMMARY METRICS:")
    print(f"  • Total Samples : {total}")
    print(f"  • Accuracy      : {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"  • Precision     : {precision:.4f} ({precision*100:.2f}%)")
    print(f"  • Recall        : {recall:.4f} ({recall*100:.2f}%)")
    print(f"  • F1-Score      : {f1:.4f} ({f1*100:.2f}%)")
    print("=" * 68)

def main():
    # Test function with example values
    print_confusion_matrix(tp=50, tn=120, fp=10, fn=20, title="Spam Filter Evaluation")

if __name__ == "__main__":
    main()
