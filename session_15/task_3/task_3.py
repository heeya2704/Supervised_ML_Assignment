"""
Session 15 - Task 3: Weak Learner vs Strong Learner Explanation & App Examples
"""

def main():
    print("=" * 80)
    print("SESSION 15 - TASK 3: Weak Learner vs Strong Learner & Real-World Examples")
    print("=" * 80)

    explanation = """
1. WEAK LEARNER:
   - Definition: A classifier or model that performs only slightly better than random guessing (e.g. 52% - 60% accuracy). It is computationally simple, constrained, and has high bias (such as a 1-depth Decision Stump).
   - Real-World App Example (Instagram):
     A simple single-rule heuristic on Instagram that flags a comment as spam if it contains the word "crypto" or a link. On its own, it misses many spam nuances and flags valid tech discussions, making it a weak predictor.

2. STRONG LEARNER:
   - Definition: A complex model or an ensemble of sequential weak learners that achieves high predictive accuracy (e.g. 90% - 99% accuracy) and generalizes well to complex non-linear patterns.
   - Real-World App Example (Zomato / Paytm):
     Zomato's multi-layered ETA & recommendation system, or Paytm's real-time fraud detection engine. Paytm combines dozens of weak rules (transaction velocity, location mismatch, device ID change, sudden high amount) using Boosting algorithms to create a high-precision, low-error fraud classification system.

3. KEY DIFFERENCE:
   - Boosting sequentially converts a collection of high-bias WEAK LEARNERS into a low-bias, low-variance STRONG LEARNER by focusing on previously misclassified errors.
"""
    print(explanation)
    print("=" * 80)

if __name__ == "__main__":
    main()
