"""
Calculate historical average goals for/against for each team at each matchweek.

This creates a lookup table that can be used during simulation to provide
realistic goal feature values without requiring actual future goal data.
"""

import pandas as pd
import numpy as np
import json

def calculate_team_matchweek_averages(historical_data_path='prem_results.csv',
                                     output_json='team_matchweek_goal_averages.json'):
    """
    Calculate average goals for/against/diff for each team at each matchweek
    across all their historical Premier League seasons.
    
    Returns:
    --------
    Dictionary structure:
    {
        "Liverpool": {
            "1": {"goals_for": 1.2, "goals_against": 0.8, "goal_diff": 0.4},
            "2": {"goals_for": 2.5, "goals_against": 1.6, "goal_diff": 0.9},
            ...
        },
        ...
    }
    
    These represent cumulative averages (e.g., at MW10, average total goals 
    scored in first 10 games across all Liverpool's PL seasons).
    """
    
    print("="*70)
    print("CALCULATING HISTORICAL MATCHWEEK GOAL AVERAGES")
    print("="*70)
    
    df = pd.read_csv(historical_data_path)
    print(f"\nLoaded {len(df)} historical matches")
    print(f"Seasons: {df['season'].nunique()}")
    
    # Get all unique teams
    teams = sorted(list(set(df['home_team'].unique()) | set(df['away_team'].unique())))
    print(f"Teams: {len(teams)}")
    
    # Dictionary to store averages
    team_averages = {}
    
    # Process each team
    for team in teams:
        print(f"Processing {team}...", end=' ')
        
        team_averages[team] = {}
        
        # Get all seasons this team played in
        team_seasons = df[
            (df['home_team'] == team) | (df['away_team'] == team)
        ]['season'].unique()
        
        # For each matchweek (1-38)
        for mw in range(1, 39):
            cumulative_gf = []
            cumulative_ga = []
            
            # For each season
            for season in team_seasons:
                # Get all matches up to this matchweek in this season
                season_data = df[
                    (df['season'] == season) & 
                    (df['match_week'] <= mw)
                ].copy()
                
                # Calculate cumulative goals for this season up to this MW
                gf = 0
                ga = 0
                
                for _, row in season_data.iterrows():
                    if row['home_team'] == team:
                        gf += row['home_goals']
                        ga += row['away_goals']
                    elif row['away_team'] == team:
                        gf += row['away_goals']
                        ga += row['home_goals']
                
                # Only include if team played in this matchweek
                if len(season_data[(season_data['home_team'] == team) | 
                                  (season_data['away_team'] == team)]) > 0:
                    cumulative_gf.append(gf)
                    cumulative_ga.append(ga)
            
            # Calculate average across all seasons
            if len(cumulative_gf) > 0:
                avg_gf = np.mean(cumulative_gf)
                avg_ga = np.mean(cumulative_ga)
                avg_gd = avg_gf - avg_ga
                
                team_averages[team][str(mw)] = {
                    'goals_for': round(avg_gf, 2),
                    'goals_against': round(avg_ga, 2),
                    'goal_diff': round(avg_gd, 2),
                    'sample_size': len(cumulative_gf)  # How many seasons contributed
                }
            else:
                # If no data, use zeros
                team_averages[team][str(mw)] = {
                    'goals_for': 0.0,
                    'goals_against': 0.0,
                    'goal_diff': 0.0,
                    'sample_size': 0
                }
        
        print(f"✓ ({len(team_seasons)} seasons)")
    
    # Save to JSON
    with open(output_json, 'w') as f:
        json.dump(team_averages, f, indent=2)
    
    print(f"\n✓ Saved to {output_json}")
    
    # Display sample for verification
    print("\n" + "="*70)
    print("SAMPLE: Liverpool's Historical Averages")
    print("="*70)
    
    if 'Liverpool' in team_averages:
        print(f"\n{'MW':<4} {'Avg GF':<8} {'Avg GA':<8} {'Avg GD':<8} {'Seasons':<8}")
        print("-" * 45)
        for mw in [1, 5, 10, 15, 20, 25, 30, 35, 38]:
            data = team_averages['Liverpool'][str(mw)]
            print(f"{mw:<4} {data['goals_for']:<8.2f} {data['goals_against']:<8.2f} "
                  f"{data['goal_diff']:<8.2f} {data['sample_size']:<8}")
    
    return team_averages


def get_h2h_goal_averages_from_history(df, up_to_date=None):
    """
    Calculate H2H goal averages for all team pairs from actual historical data.
    
    Parameters:
    -----------
    df : DataFrame
        Historical results data
    up_to_date : str (optional)
        Only use matches up to this date (format: 'YYYY-MM-DD')
    
    Returns:
    --------
    Dictionary with H2H goal averages:
    {
        ('Arsenal', 'Chelsea'): {
            'home_avg_goals': 1.5,  # Arsenal's avg when playing at home vs Chelsea
            'away_avg_goals': 1.2   # Arsenal's avg when playing away vs Chelsea
        },
        ...
    }
    """
    
    print("\n" + "="*70)
    print("CALCULATING H2H GOAL AVERAGES FROM HISTORY")
    print("="*70)
    
    if up_to_date:
        df = df[pd.to_datetime(df['date']) <= pd.to_datetime(up_to_date)].copy()
        print(f"Using matches up to {up_to_date}")
    
    h2h_averages = {}
    
    # Get all unique team pairs
    teams = sorted(list(set(df['home_team'].unique()) | set(df['away_team'].unique())))
    
    for i, team1 in enumerate(teams):
        for team2 in teams[i+1:]:
            # Get all matches between these teams
            h2h_matches = df[
                ((df['home_team'] == team1) & (df['away_team'] == team2)) |
                ((df['home_team'] == team2) & (df['away_team'] == team1))
            ].copy()
            
            if len(h2h_matches) == 0:
                continue
            
            # Calculate averages from team1's perspective
            team1_home_goals = []
            team1_away_goals = []
            
            for _, row in h2h_matches.iterrows():
                if row['home_team'] == team1:
                    team1_home_goals.append(row['home_goals'])
                else:
                    team1_away_goals.append(row['away_goals'])
            
            # Store both directions
            key1 = (team1, team2)
            key2 = (team2, team1)
            
            h2h_averages[key1] = {
                'home_avg_goals': np.mean(team1_home_goals) if team1_home_goals else 0.0,
                'away_avg_goals': np.mean(team1_away_goals) if team1_away_goals else 0.0
            }
            
            # For team2's perspective, swap home/away
            team2_home_goals = []
            team2_away_goals = []
            for _, row in h2h_matches.iterrows():
                if row['home_team'] == team2:
                    team2_home_goals.append(row['home_goals'])
                else:
                    team2_away_goals.append(row['away_goals'])
            
            h2h_averages[key2] = {
                'home_avg_goals': np.mean(team2_home_goals) if team2_home_goals else 0.0,
                'away_avg_goals': np.mean(team2_away_goals) if team2_away_goals else 0.0
            }
    
    print(f"✓ Calculated H2H averages for {len(h2h_averages)} team-pair combinations")
    
    return h2h_averages


if __name__ == "__main__":
    # Calculate matchweek averages
    team_averages = calculate_team_matchweek_averages(
        historical_data_path='prem_results.csv',
        output_json='team_matchweek_goal_averages.json'
    )
    
    # Calculate H2H averages
    df = pd.read_csv('prem_results.csv')
    h2h_averages = get_h2h_goal_averages_from_history(df)
    
    # Save H2H averages
    # Convert tuple keys to strings for JSON
    h2h_json = {
        f"{k[0]}__vs__{k[1]}": v for k, v in h2h_averages.items()
    }
    
    with open('h2h_goal_averages.json', 'w') as f:
        json.dump(h2h_json, f, indent=2)
    
    print("\n✓ Saved H2H averages to h2h_goal_averages.json")
    
    print("\n" + "="*70)
    print("SETUP COMPLETE!")
    print("="*70)
    print("""
Files created:
1. team_matchweek_goal_averages.json - Historical averages per team per MW
2. h2h_goal_averages.json - H2H goal averages for all team pairs

Use these files in your simulation to provide realistic goal features
without requiring actual future goal data.
    """)
