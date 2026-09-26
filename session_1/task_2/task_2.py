"""
Session 1 - Task 2
Classify given scenarios into Supervised, Unsupervised, or Reinforcement Learning with a 1-line explanation each.
"""

def main():
    print("=" * 70)
    print("SESSION 1 - TASK 2: ML Scenario Classification")
    print("=" * 70)

    scenarios = [
        {
            "id": 1,
            "scenario": "Netflix recommending movies based on your watch history.",
            "ml_type": "Supervised Learning",
            "explanation": "It predicts user preferences/ratings based on labeled historical viewing data and explicit/implicit ratings."
        },
        {
            "id": 2,
            "scenario": "Spotify grouping similar songs into playlists.",
            "ml_type": "Unsupervised Learning",
            "explanation": "It uses clustering algorithms to group songs based on inherent audio feature similarity without explicit labels."
        },
        {
            "id": 3,
            "scenario": "A self-driving car learning to park by trial and error.",
            "ml_type": "Reinforcement Learning",
            "explanation": "An autonomous agent learns optimal parking control policies by interacting with an environment through feedback rewards and penalties."
        }
    ]

    for item in scenarios:
        print(f"\nScenario {item['id']}: {item['scenario']}")
        print(f"Classification: {item['ml_type']}")
        print(f"Explanation   : {item['explanation']}")
        print("-" * 70)

if __name__ == "__main__":
    main()
