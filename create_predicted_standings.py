import pandas as pd

# Load the simulated fixtures data
df = pd.read_csv('fixtures_2025_2026_simulated.csv')

# Initialize a dictionary to store team stats
teams = {}

# Process each match
for _, row in df.iterrows():
    home_team = row['home_team']
    away_team = row['away_team']
    result = row['result']
    
    # Initialize teams if not in dictionary
    if home_team not in teams:
        teams[home_team] = {'wins': 0, 'draws': 0, 'losses': 0, 'points': 0}
    if away_team not in teams:
        teams[away_team] = {'wins': 0, 'draws': 0, 'losses': 0, 'points': 0}
    
    # Update stats based on result
    if result == 'H':  # Home win
        teams[home_team]['wins'] += 1
        teams[home_team]['points'] += 3
        teams[away_team]['losses'] += 1
    elif result == 'A':  # Away win
        teams[away_team]['wins'] += 1
        teams[away_team]['points'] += 3
        teams[home_team]['losses'] += 1
    elif result == 'D':  # Draw
        teams[home_team]['draws'] += 1
        teams[home_team]['points'] += 1
        teams[away_team]['draws'] += 1
        teams[away_team]['points'] += 1

# Convert to DataFrame
standings = []
for team, stats in teams.items():
    standings.append({
        'Team': team,
        'Wins': stats['wins'],
        'Draws': stats['draws'],
        'Losses': stats['losses'],
        'Points': stats['points']
    })

standings_df = pd.DataFrame(standings)

# Sort by points (descending), then by wins (descending) as tiebreaker
standings_df = standings_df.sort_values(by=['Points', 'Wins'], ascending=[False, False])

# Add rank
standings_df.insert(0, 'Rank', range(1, len(standings_df) + 1))

# Reorder columns to match specification
standings_df = standings_df[['Rank', 'Team', 'Wins', 'Draws', 'Losses', 'Points']]

# Reset index
standings_df = standings_df.reset_index(drop=True)

# Save to CSV
standings_df.to_csv('predicted_final_standings_2025_2026.csv', index=False)

print("Predicted Final Standings for 2025-2026 Season:")
print("=" * 60)
print(standings_df.to_string(index=False))
print("\nStandings saved to: predicted_final_standings_2025_2026.csv")