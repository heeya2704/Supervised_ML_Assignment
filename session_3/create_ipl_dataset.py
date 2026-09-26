"""
Generate IPL Player Stats dataset with realistic missing values for Session 3 tasks.
"""

import os
import pandas as pd
import numpy as np

def generate_ipl_data():
    np.random.seed(42)
    
    players = [
        "Virat Kohli", "Rohit Sharma", "MS Dhoni", "KL Rahul", "Hardik Pandya",
        "Jasprit Bumrah", "Ravindra Jadeja", "Rishabh Pant", "Shubman Gill", "Suryakumar Yadav",
        "Shreyas Iyer", "Mohammed Shami", "Yuzvendra Chahal", "Sanju Samson", "Rashid Khan"
    ]
    
    teams = ["RCB", "MI", "CSK", "LSG", "GT", "MI", "CSK", "DC", "GT", "MI", "KKR", "SRH", "RR", "RR", "GT"]
    roles = ["Batsman", "Batsman", "Wicketkeeper", "Batsman", "Allrounder", "Bowler", "Allrounder", "Wicketkeeper", "Batsman", "Batsman", "Batsman", "Bowler", "Bowler", "Wicketkeeper", "Bowler"]
    venues = ["Chinnaswamy", "Wankhede", "Chepauk", "Ekana", "Narendra Modi Stadium", "Wankhede", "Chepauk", "Arun Jaitley", "Narendra Modi Stadium", "Wankhede", "Eden Gardens", "Rajiv Gandhi Stadium", "Sawai Mansingh", "Sawai Mansingh", "Narendra Modi Stadium"]
    
    matches = [237, 243, 250, 118, 123, 120, 226, 98, 91, 139, 101, 110, 145, 152, 109]
    runs = [7263, 6211, 5082, 4163, 2309, 165, 2692, 2838, 2790, 3249, 2776, 79, 45, 3888, 443]
    wickets = [4, 15, 0, 0, 53, 145, 152, 0, 0, 0, 0, 127, 187, 0, 139]
    strike_rate = [130.0, 130.05, 135.9, 134.6, 145.8, 11.2, 128.6, 147.9, 134.1, 143.3, 125.4, 75.0, 42.1, 137.5, 138.2]
    
    ages = [35.0, 36.0, 42.0, 31.0, 30.0, 30.0, 35.0, 26.0, 24.0, 33.0, 29.0, 33.0, 33.0, 29.0, 25.0]

    df = pd.DataFrame({
        'player_id': [f"IPL_{101+i}" for i in range(len(players))],
        'player_name': players,
        'team': teams,
        'player_role': roles,
        'matches': matches,
        'runs': runs,
        'wickets': wickets,
        'strike_rate': strike_rate,
        'player_age': ages,
        'venue': venues
    })

    # Intentionally inject missing values (NaN) into player_age, team, strike_rate
    df.loc[[2, 5, 9, 12], 'player_age'] = np.nan
    df.loc[[1, 7, 13], 'team'] = np.nan
    df.loc[[4, 10], 'strike_rate'] = np.nan
    
    return df

if __name__ == "__main__":
    df = generate_ipl_data()
    file_path = "session_3/dataset/ipl_player_stats.csv"
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    df.to_csv(file_path, index=False)
    print(f"[+] Saved IPL Dataset to {file_path}")
    print(df.head(10))
