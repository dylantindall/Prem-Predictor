# Create tables of the final standings for each season,
# save them as csv files into the final_standings folder.

import pandas as pd
import os

def generate_season_tables(df):
    """
    Generate final standings tables for each season and save as separate CSV files
    """
    
    # Create directory for season tables if it doesn't exist
    os.makedirs('season_tables', exist_ok=True)
    
    # Process each season
    for season in df['season'].unique():
        print(f"Generating table for season: {season}")
        
        season_df = df[df['season'] == season].copy()
        season_df = season_df.sort_values('date').reset_index(drop=True)
        
        # Get the last 10 matches of the season to determine final standings
        last_10_matches = season_df.tail(10)
        
        # Get all teams in this season
        all_teams = set(season_df['home_team'].unique()) | set(season_df['away_team'].unique())
        
        # Initialize team statistics
        team_stats = {}
        for team in all_teams:
            team_stats[team] = {
                'points': 0,
                'wins': 0,
                'draws': 0,
                'losses': 0,
                'goals_for': 0,
                'goals_against': 0,
                'goal_difference': 0,
                'played': 0
            }
        
        # Process all matches in the season to build final stats
        for _, row in season_df.iterrows():
            home_team = row['home_team']
            away_team = row['away_team']
            home_goals = row['home_goals']
            away_goals = row['away_goals']
            result = row['result']
            
            # Update games played
            team_stats[home_team]['played'] += 1
            team_stats[away_team]['played'] += 1
            
            # Update goals
            team_stats[home_team]['goals_for'] += home_goals
            team_stats[home_team]['goals_against'] += away_goals
            team_stats[away_team]['goals_for'] += away_goals
            team_stats[away_team]['goals_against'] += home_goals
            
            # Update goal difference
            team_stats[home_team]['goal_difference'] = (team_stats[home_team]['goals_for'] - 
                                                       team_stats[home_team]['goals_against'])
            team_stats[away_team]['goal_difference'] = (team_stats[away_team]['goals_for'] - 
                                                       team_stats[away_team]['goals_against'])
            
            # Update points and W/D/L
            if result == 'H':  # Home win
                team_stats[home_team]['points'] += 3
                team_stats[home_team]['wins'] += 1
                team_stats[away_team]['losses'] += 1
            elif result == 'A':  # Away win
                team_stats[away_team]['points'] += 3
                team_stats[away_team]['wins'] += 1
                team_stats[home_team]['losses'] += 1
            elif result == 'D':  # Draw
                team_stats[home_team]['points'] += 1
                team_stats[away_team]['points'] += 1
                team_stats[home_team]['draws'] += 1
                team_stats[away_team]['draws'] += 1
        
        # Convert to DataFrame
        table_data = []
        for team, stats in team_stats.items():
            table_data.append({
                'Team': team,
                'Points': stats['points'],
                'Wins': stats['wins'],
                'Draws': stats['draws'],
                'Losses': stats['losses'],
                'GF': stats['goals_for'],
                'GA': stats['goals_against'],
                'GD': stats['goal_difference']
            })
        
        table_df = pd.DataFrame(table_data)
        
        # Sort by Premier League rules: Points, GD, GF, then alphabetically
        table_df = table_df.sort_values(
            by=['Points', 'GD', 'GF', 'Team'],
            ascending=[False, False, False, True]
        ).reset_index(drop=True)
        
        # Add rank
        table_df.insert(0, 'Rank', range(1, len(table_df) + 1))
        
        # Create filename (e.g., season_1995_1996.csv)
        season_filename = season.replace('-', '_')
        filepath = f'season_tables/season_{season_filename}.csv'
        
        # Save to CSV
        table_df.to_csv(filepath, index=False)
        print(f"  Saved: {filepath}")
        print(f"  Champion: {table_df.iloc[0]['Team']} with {table_df.iloc[0]['Points']} points")
    
    print(f"\nAll season tables generated successfully in 'season_tables/' directory!")
    return

# Run the function
df = pd.read_csv("prem_results.csv")
generate_season_tables(df)

# Example: Load and display one season table
example_season = df['season'].unique()[0]
example_filename = example_season.replace('-', '_')
example_table = pd.read_csv(f'season_tables/season_{example_filename}.csv')
print(f"\nExample: {example_season} Final Standings:")
print(example_table)