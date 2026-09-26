"""
Session 1 - Task 4
Traditional Programming vs Machine Learning for WhatsApp Spam Filtering
"""

def main():
    print("=" * 70)
    print("SESSION 1 - TASK 4: Traditional Programming vs Machine Learning (Spam Filter)")
    print("=" * 70)

    comparison = """
In Traditional Programming, developers manually write explicit, deterministic rules (if-else statements) 
to identify spam messages—for instance, flagging messages containing exact keywords like 'CONGRATS WINNER', 
'BITCOIN LOTTERY', or containing unverified URL links. While simple initially, this rule-based approach 
fails to scale because spammers continuously evade filters using obfuscations (e.g., 'C0ngr@ts W!nn3r'). 

In contrast, Machine Learning approaches the problem by learning patterns directly from data. By feeding an 
ML algorithm thousands of historical WhatsApp messages labeled as 'spam' or 'ham' (legitimate), the model 
automatically uncovers complex statistical patterns, word frequencies, n-gram combinations, and contextual 
metadata without human rule-writing. When spammers adapt their language, the ML filter seamlessly updates 
its classification boundaries by simply retraining on fresh message logs, making it far more robust and adaptive.
"""
    print(comparison.strip())
    print("=" * 70)

if __name__ == "__main__":
    main()
