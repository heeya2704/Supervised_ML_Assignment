# Session 1 - Task 1: Everyday Apps & Machine Learning Features

## Task Overview
List three apps you use daily (such as Zomato, Instagram, or Flipkart) and describe one feature in each app that likely uses machine learning. For each, state whether it is an example of supervised, unsupervised, or reinforcement learning.

---

## Analysis & Answers

### 1. Zomato
* **Feature:** **Estimated Time of Arrival (ETA) Prediction**
* **Description:** Zomato predicts the total time required to prepare and deliver food to a user's delivery address based on factors like dish preparation complexity, current restaurant order queue, historical route distance, live traffic, and weather conditions.
* **ML Type:** **Supervised Learning**
* **Explanation:** The model is trained on millions of past orders where the ground-truth label is known: the *actual delivery duration* (continuous variable). Regression algorithms learn the mathematical relationship between input features (distance, traffic, prep time) and target delivery duration.

---

### 2. Instagram
* **Feature:** **Explore Feed & Topic Clustering**
* **Description:** Instagram automatically groups similar posts, reels, and photos, surfacing relevant content to users based on latent preferences without requiring explicit genre tags.
* **ML Type:** **Unsupervised Learning** *(with Supervised Ranking)*
* **Explanation:** Instagram uses matrix factorization, embedding vectors, and clustering algorithms (unsupervised) to discover hidden user interaction patterns and cluster content based on semantic similarity without relying on human-labeled categories.

---

### 3. Flipkart
* **Feature:** **"Frequently Bought Together" & Product Recommendations**
* **Description:** Flipkart analyzes buyer checkout history to discover co-purchasing patterns and suggest complementary products (e.g., suggesting a phone case and screen protector when buying a smartphone).
* **ML Type:** **Unsupervised Learning** *(Association Rule Mining / Collaborative Filtering)*
* **Explanation:** The algorithm uncovers hidden affinities and co-occurrence patterns across massive unlabeled transaction logs without human target labels.
