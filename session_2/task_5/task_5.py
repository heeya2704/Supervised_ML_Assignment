"""
Session 2 - Task 5
Real-world example of overfitting in everyday apps (Instagram/Flipkart/Swiggy) with 2-3 line explanation.
"""

def main():
    print("=" * 70)
    print("SESSION 2 - TASK 5: Real-World Overfitting Example in Daily Apps")
    print("=" * 70)

    example = """
APP: Instagram Reels / Recommendation Engine

REAL EXAMPLE OF OVERFITTING:
After a user accidentally clicks or watches a single 15-second DIY home repair video once, 
Instagram's recommendation algorithm overfits to this single transient interaction signal 
and floods the user's Explore feed and Reels feed almost exclusively with DIY repair videos 
for the next three days.

WHY IT HAPPENS (2-3 Lines):
This happens because the recommendation model assigns excessively high feature weight to recent 
micro-interactions without enough historical regularization or diversity constraints. As a result, 
the model overfits to momentary noise rather than capturing the user's long-term diverse interests.
"""
    print(example.strip())
    print("=" * 70)

if __name__ == "__main__":
    main()
