# Session 2 - Task 5: Real-World Overfitting Example (Instagram Recommendation Engine)

## Task Overview
Give a real example (outside of healthcare/hospital) of overfitting from any app you use (e.g., Instagram, Flipkart, Swiggy), and explain in 2-3 lines why it might happen in that scenario.

---

## Real-World Example: Instagram Reels Recommendation Feed

### The Scenario
After a user accidentally watches or taps on a single 15-second DIY carpentry video once, Instagram's recommendation algorithm overfits to this single transient action, immediately cluttering the user's Explore and Reels feed almost exclusively with carpentry and woodworking videos for the next few days.

---

## 2–3 Line Explanation (Why It Happens)

> *This occurs because the recommendation algorithm places excessive weight on recent micro-interactions without incorporating sufficient regularization or topic-diversity constraints.*  
> *Consequently, the model overfits to noisy, short-term user behavior (a accidental single click) rather than accurately representing the user's stable, long-term content preferences.*
