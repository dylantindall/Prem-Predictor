"""
Season simulation using HISTORICAL AVERAGE goal data.

KEY IMPROVEMENT: Instead of freezing goal features at MW7 values, this uses
each team's historical average goals for/against at each matchweek.

Example: When predicting Liverpool's MW20 match, the model uses Liverpool's
average goals scored in the first 20 games across all their PL seasons.

This provides realistic, matchweek-appropriate goal features throughout
the entire simulation without requiring actual future goal data.
"""

import pandas as pd
import numpy as np
from collections import defaultdict, deque
import json
import warnings
warnings.filterwarnings('ignore')

# Configuration
FORM_WINDOW = 5
H2H_WINDOW = 5

FEATURES = ['home_win_form', 'home_draw_form', 'home_loss_form',
            'away_win_form', 'away_draw_form', 'away_loss_form',
            'home_position', 'away_position', 'home_points', 'away_points',
            'position_gap', 'points_gap', 'home_team_label', 'away_team_label',
            'h2h_home_wins', 'h2h_draws', 'h2h_away_wins', 'home_avg_h2h_goals',
            'away_avg_h2h_goals', 'home_team_goals_for', 'home_team_goals_against',
            'home_team_goal_diff', 'away_team_goals_for', 'away_team_goals_against',
            'away_team_goal_diff']


def load_historical_averages(team_avg_path='team_matchweek_goal_averages.json',
                            h2h_avg_path='h2h_goal_averages.json'):
    """Load pre-calculated historical averages."""
    
    print("="*70)
    print("LOADING HISTORICAL AVERAGES")
    print("="*70)
    
    try:
        with open(team_avg_path, 'r') as f:
            team_averages = json.load(f)
        print(f"✓ Loaded team matchweek averages for {len(team_averages)} teams")
    except FileNotFoundError:
        print(f"❌ {team_avg_path} not found!")
        print("   Run calculate_historical_goal_averages.py first")
        return None, None
    
    try:
        with open(h2h_avg_path, 'r') as f:
            h2h_json = json.load(f)
        # Convert string keys back to tuples
        h2h_averages = {}
        for key_str, val in h2h_json.items():
            teams = key_str.split('__vs__')
            h2h_averages[(teams[0], teams[1])] = val
        print(f"✓ Loaded H2H averages for {len(h2h_averages)} team pairs")
    except FileNotFoundError:
        print(f"❌ {h2h_avg_path} not found!")
        print("   Run calculate_historical_goal_averages.py first")
        return None, None
    
    return team_averages, h2h_averages


def get_team_goal_features_for_matchweek(team, matchweek, team_averages):
    """
    Get historical average goal features for a team at a specific matchweek.
    
    Returns:
    --------
    dict with keys: goals_for, goals_against, goal_diff
    """
    if team not in team_averages:
        return {'goals_for': 0, 'goals_against': 0, 'goal_diff': 0}
    
    mw_data = team_averages[team].get(str(matchweek), {})
    return {
        'goals_for': mw_data.get('goals_for', 0),
        'goals_against': mw_data.get('goals_against', 0),
        'goal_diff': mw_data.get('goal_diff', 0)
    }


def get_h2h_goal_features(home_team, away_team, h2h_averages):
    """
    Get H2H goal averages for a specific matchup.
    
    Returns:
    --------
    dict with keys: home_avg_h2h_goals, away_avg_h2h_goals
    """
    key = (home_team, away_team)
    if key in h2h_averages:
        return {
            'home_avg_h2h_goals': h2h_averages[key]['home_avg_goals'],
            'away_avg_h2h_goals': h2h_averages[key]['away_avg_goals']
        }
    return {'home_avg_h2h_goals': 0.0, 'away_avg_h2h_goals': 0.0}


def compute_positions(team_stats, team_averages, current_matchweek):
    """
    Calculate league positions using actual points and historical goal data for tiebreakers.
    """
    rows = []
    for team, stats in team_stats.items():
        # Use historical average goals for tiebreakers
        goal_features = get_team_goal_features_for_matchweek(team, current_matchweek, team_averages)
        rows.append((
            team, 
            stats['points'], 
            goal_features['goal_diff'],  # Use historical average GD
            goal_features['goals_for']    # Use historical average GF
        ))
    rows.sort(key=lambda x: (-x[1], -x[2], -x[3], x[0]))
    return {t: i + 1 for i, (t, *_) in enumerate(rows)}


def update_fixture_features(df, idx, team_stats, h2h_results, team_averages, 
                           h2h_averages, current_matchweek):
    """
    Update all features for a fixture using:
    - Actual current form, points, positions
    - Historical average goal data for the matchweek
    - Historical H2H goal averages
    """
    row = df.loc[idx]
    home = row['home_team']
    away = row['away_team']
    key = tuple(sorted([home, away]))
    
    # Calculate positions using historical goal data for tiebreakers
    positions = compute_positions(team_stats, team_averages, current_matchweek)
    home_stats = team_stats[home]
    away_stats = team_stats[away]
    
    # Form counts (actual, from simulation)
    hw = sum(1 for r in home_stats['recent_results'] if r == 'W')
    hd = sum(1 for r in home_stats['recent_results'] if r == 'D')
    hl = sum(1 for r in home_stats['recent_results'] if r == 'L')
    aw = sum(1 for r in away_stats['recent_results'] if r == 'W')
    ad = sum(1 for r in away_stats['recent_results'] if r == 'D')
    al = sum(1 for r in away_stats['recent_results'] if r == 'L')
    
    # H2H result counts (actual, from simulation)
    h2h_home_wins = sum(1 for r in h2h_results[key] if r == 'H')
    h2h_draws = sum(1 for r in h2h_results[key] if r == 'D')
    h2h_away_wins = sum(1 for r in h2h_results[key] if r == 'A')
    
    # Get historical goal features for this matchweek
    home_goals = get_team_goal_features_for_matchweek(home, current_matchweek, team_averages)
    away_goals = get_team_goal_features_for_matchweek(away, current_matchweek, team_averages)
    
    # Get H2H goal averages (from all-time history)
    h2h_goals = get_h2h_goal_features(home, away, h2h_averages)
    
    # Update ALL features
    df.at[idx, 'home_win_form'] = hw
    df.at[idx, 'home_draw_form'] = hd
    df.at[idx, 'home_loss_form'] = hl
    df.at[idx, 'away_win_form'] = aw
    df.at[idx, 'away_draw_form'] = ad
    df.at[idx, 'away_loss_form'] = al
    
    df.at[idx, 'home_position'] = positions.get(home, 20)
    df.at[idx, 'away_position'] = positions.get(away, 20)
    df.at[idx, 'home_points'] = home_stats['points']
    df.at[idx, 'away_points'] = away_stats['points']
    df.at[idx, 'position_gap'] = abs(positions.get(home, 20) - positions.get(away, 20))
    df.at[idx, 'points_gap'] = abs(home_stats['points'] - away_stats['points'])
    
    df.at[idx, 'h2h_home_wins'] = h2h_home_wins
    df.at[idx, 'h2h_draws'] = h2h_draws
    df.at[idx, 'h2h_away_wins'] = h2h_away_wins
    
    # Use historical averages for goal features
    df.at[idx, 'home_team_goals_for'] = home_goals['goals_for']
    df.at[idx, 'home_team_goals_against'] = home_goals['goals_against']
    df.at[idx, 'home_team_goal_diff'] = home_goals['goal_diff']
    df.at[idx, 'away_team_goals_for'] = away_goals['goals_for']
    df.at[idx, 'away_team_goals_against'] = away_goals['goals_against']
    df.at[idx, 'away_team_goal_diff'] = away_goals['goal_diff']
    
    df.at[idx, 'home_avg_h2h_goals'] = h2h_goals['home_avg_h2h_goals']
    df.at[idx, 'away_avg_h2h_goals'] = h2h_goals['away_avg_h2h_goals']


def simulate_season(model, 
                   fixtures_path='fixtures_2025_2026.csv',
                   team_avg_path='team_matchweek_goal_averages.json',
                   h2h_avg_path='h2h_goal_averages.json',
                   start_simulation_from_mw=8):
    """
    Simulate season using historical average goal data.
    
    Parameters:
    -----------
    model : trained sklearn model
        Your trained RandomForestClassifier
    fixtures_path : str
        Path to fixtures CSV with MW1-7 results
    team_avg_path : str
        Path to team matchweek averages JSON
    h2h_avg_path : str
        Path to H2H averages JSON
    start_simulation_from_mw : int
        Matchweek to start predictions from
    """
    
    print("\n" + "="*70)
    print("PREMIER LEAGUE 2025-2026 SEASON SIMULATION")
    print("Using Historical Average Goal Data")
    print("="*70)
    
    # Load historical averages
    team_averages, h2h_averages = load_historical_averages(team_avg_path, h2h_avg_path)
    if team_averages is None or h2h_averages is None:
        return None, None
    
    # Load fixtures
    df = pd.read_csv(fixtures_path)
    print(f"\n✓ Loaded {len(df)} fixtures from {fixtures_path}")
    
    # Get all teams
    teams = sorted(set(df['home_team'].unique()) | set(df['away_team'].unique()))
    print(f"✓ Teams in season: {len(teams)}")
    
    # Initialize team stats (for tracking points and form only)
    team_stats = {
        t: {
            'points': 0,
            'played': 0,
            'wins': 0,
            'draws': 0,
            'losses': 0,
            'recent_results': deque(maxlen=FORM_WINDOW),
        }
        for t in teams
    }
    
    # H2H trackers (for result counts)
    h2h_results = defaultdict(lambda: deque(maxlen=H2H_WINDOW))
    
    # Process completed matchweeks
    completed_mw = start_simulation_from_mw - 1
    completed = df[df['match_week'] <= completed_mw].copy()
    completed = completed.sort_values(['match_week', 'date']).reset_index(drop=True)
    
    print(f"\n{'='*70}")
    print(f"INITIALIZING FROM COMPLETED MATCHWEEKS (1-{completed_mw})")
    print("="*70)
    
    for _, row in completed.iterrows():
        home = row['home_team']
        away = row['away_team']
        result = row['result']
        
        if pd.isna(result):
            continue
        
        team_stats[home]['played'] += 1
        team_stats[away]['played'] += 1
        
        if result == 'H':
            team_stats[home]['points'] += 3
            team_stats[home]['wins'] += 1
            team_stats[away]['losses'] += 1
            team_stats[home]['recent_results'].append('W')
            team_stats[away]['recent_results'].append('L')
        elif result == 'A':
            team_stats[away]['points'] += 3
            team_stats[away]['wins'] += 1
            team_stats[home]['losses'] += 1
            team_stats[home]['recent_results'].append('L')
            team_stats[away]['recent_results'].append('W')
        elif result == 'D':
            team_stats[home]['points'] += 1
            team_stats[away]['points'] += 1
            team_stats[home]['draws'] += 1
            team_stats[away]['draws'] += 1
            team_stats[home]['recent_results'].append('D')
            team_stats[away]['recent_results'].append('D')
        
        key = tuple(sorted([home, away]))
        h2h_results[key].append(result)
    
    # Display current standings
    positions = compute_positions(team_stats, team_averages, completed_mw)
    standings = sorted(
        [(team, stats, positions[team]) for team, stats in team_stats.items()],
        key=lambda x: x[2]
    )
    
    print(f"\nCurrent Standings after MW{completed_mw}:")
    print(f"{'Pos':<4} {'Team':<20} {'P':<3} {'W':<3} {'D':<3} {'L':<3} {'Pts':<4}")
    print("-" * 50)
    for pos, (team, stats, _) in enumerate(standings[:20], 1):
        print(f"{pos:<4} {team:<20} {stats['played']:<3} {stats['wins']:<3} "
              f"{stats['draws']:<3} {stats['losses']:<3} {stats['points']:<4}")
    
    # Simulate remaining matchweeks
    print(f"\n{'='*70}")
    print(f"SIMULATING MATCHWEEKS {start_simulation_from_mw}-38")
    print("Using historical average goal data for each matchweek")
    print("="*70)
    
    all_predictions = []
    
    for mw in range(start_simulation_from_mw, 39):
        print(f"\n🔮 Matchweek {mw}:")
        
        # Get fixtures for this matchweek
        week_mask = df['match_week'] == mw
        week_indices = df[week_mask].index
        
        if len(week_indices) == 0:
            continue
        
        # Update features using historical averages for THIS matchweek
        for idx in week_indices:
            update_fixture_features(df, idx, team_stats, h2h_results, 
                                  team_averages, h2h_averages, mw)
        
        # Prepare features for prediction
        X_week = df.loc[week_indices, FEATURES].fillna(0)
        
        # Make predictions
        predictions = model.predict(X_week)
        
        # Store predictions
        df.loc[week_indices, 'result'] = predictions
        
        # Display predictions
        for idx, pred in zip(week_indices, predictions):
            row = df.loc[idx]
            result_symbol = {'H': '🏠', 'D': '🤝', 'A': '✈️'}.get(pred, pred)
            all_predictions.append({
                'match_week': mw,
                'date': row['date'],
                'home_team': row['home_team'],
                'away_team': row['away_team'],
                'predicted_result': pred
            })
            print(f"  {row['home_team']:<20} vs {row['away_team']:<20} → {result_symbol} {pred}")
        
        # Update team stats for next matchweek
        for idx, pred in zip(week_indices, predictions):
            row = df.loc[idx]
            home = row['home_team']
            away = row['away_team']
            key = tuple(sorted([home, away]))
            
            team_stats[home]['played'] += 1
            team_stats[away]['played'] += 1
            
            if pred == 'H':
                team_stats[home]['points'] += 3
                team_stats[home]['wins'] += 1
                team_stats[away]['losses'] += 1
                team_stats[home]['recent_results'].append('W')
                team_stats[away]['recent_results'].append('L')
            elif pred == 'A':
                team_stats[away]['points'] += 3
                team_stats[away]['wins'] += 1
                team_stats[home]['losses'] += 1
                team_stats[home]['recent_results'].append('L')
                team_stats[away]['recent_results'].append('W')
            elif pred == 'D':
                team_stats[home]['points'] += 1
                team_stats[away]['points'] += 1
                team_stats[home]['draws'] += 1
                team_stats[away]['draws'] += 1
                team_stats[home]['recent_results'].append('D')
                team_stats[away]['recent_results'].append('D')
            
            h2h_results[key].append(pred)
    
    # Generate final table
    print("\n" + "="*70)
    print("PREDICTED FINAL TABLE 2025-2026")
    print("="*70)
    
    # Get final positions and historical goal data for MW38
    final_positions = compute_positions(team_stats, team_averages, 38)
    
    table_data = []
    for team, stats in team_stats.items():
        # Use historical average goals for final display
        goal_features = get_team_goal_features_for_matchweek(team, 38, team_averages)
        
        table_data.append({
            'Position': final_positions[team],
            'Team': team,
            'Played': stats['played'],
            'Won': stats['wins'],
            'Drawn': stats['draws'],
            'Lost': stats['losses'],
            'GF': round(goal_features['goals_for'], 1),
            'GA': round(goal_features['goals_against'], 1),
            'GD': round(goal_features['goal_diff'], 1),
            'Points': stats['points']
        })
    
    table_df = pd.DataFrame(table_data)
    table_df = table_df.sort_values('Position').reset_index(drop=True)
    
    print(table_df.to_string(index=False))
    
    # Highlight key positions
    print("\n🏆 Champions League Places:")
    for i in range(4):
        team = table_df.iloc[i]
        print(f"  {team['Position']}. {team['Team']:<20} - {team['Points']} points")
    
    print("\n⬇️  Relegation Zone:")
    for i in range(-3, 0):
        team = table_df.iloc[i]
        print(f"  {team['Position']}. {team['Team']:<20} - {team['Points']} points")
    
    # Save outputs
    print("\n" + "="*70)
    print("SAVING RESULTS")
    print("="*70)
    
    predictions_df = pd.DataFrame(all_predictions)
    predictions_df.to_csv('predictions/predictions.csv', index=False)
    print("✓ Predictions saved to: predictions.csv")
    
    df.to_csv('fixtures_2025_2026_simulated.csv', index=False)
    print("✓ Complete fixtures saved to: fixtures_2025_2026_simulated.csv")
    
    table_df.to_csv('predictions/predicted_final_table_2025_2026.csv', index=False)
    print("✓ Final table saved to: predicted_final_table_2025_2026.csv")
    
    print("\n" + "="*70)
    print("KEY IMPROVEMENTS IN THIS SIMULATION")
    print("="*70)
    print("""
✓ Goal features use historical averages for each matchweek
  - MW8: Uses each team's average goals through 8 games historically
  - MW20: Uses each team's average goals through 20 games historically
  - MW38: Uses each team's average goals through 38 games historically

✓ H2H goal features use all-time historical averages
  - Based on actual head-to-head records across all PL seasons

✓ Form, points, and positions update dynamically
  - Based on actual predicted results during simulation

This approach provides realistic goal feature values throughout the
entire season without requiring actual future goal data!
    """)
    
    return predictions_df, table_df


# ============================================================================
# USAGE
# ============================================================================

if __name__ == "__main__":
    print("\n" + "="*70)
    print("STEP 1: CALCULATE HISTORICAL AVERAGES (if not done)")
    print("="*70)
    
    import os
    
    # Check if averages files exist
    need_calculation = False
    if not os.path.exists('team_matchweek_goal_averages.json'):
        print("❌ team_matchweek_goal_averages.json not found")
        need_calculation = True
    else:
        print("✓ team_matchweek_goal_averages.json found")
    
    if not os.path.exists('h2h_goal_averages.json'):
        print("❌ h2h_goal_averages.json not found")
        need_calculation = True
    else:
        print("✓ h2h_goal_averages.json found")
    
    if need_calculation:
        print("\nCalculating historical averages...")
        print("This may take a few minutes...\n")
        
        # Import and run the calculation
        from calculate_historical_goal_averages import (
            calculate_team_matchweek_averages,
            get_h2h_goal_averages_from_history
        )
        
        team_averages = calculate_team_matchweek_averages(
            historical_data_path='prem_results.csv',
            output_json='team_matchweek_goal_averages.json'
        )
        
        df_hist = pd.read_csv('prem_results.csv')
        h2h_averages = get_h2h_goal_averages_from_history(df_hist)
        
        h2h_json = {
            f"{k[0]}__vs__{k[1]}": v for k, v in h2h_averages.items()
        }
        
        with open('h2h_goal_averages.json', 'w') as f:
            json.dump(h2h_json, f, indent=2)
        
        print("✓ Historical averages calculated and saved")
    
    # Step 2: Load model and simulate
    print("\n" + "="*70)
    print("STEP 2: LOADING MODEL AND SIMULATING SEASON")
    print("="*70)
    
    try:
        from model import clf
        
        if clf.model is None:
            print("❌ Model not trained yet! Run model.py first.")
        else:
            print("✓ Model loaded successfully\n")
            
            predictions, final_table = simulate_season(
                model=clf.model,
                fixtures_path='fixtures_2025_2026.csv',
                team_avg_path='team_matchweek_goal_averages.json',
                h2h_avg_path='h2h_goal_averages.json',
                start_simulation_from_mw=8
            )
            
            if predictions is not None:
                print("\n" + "="*70)
                print("SIMULATION COMPLETE!")
                print("="*70)
                print(f"\nTotal predictions: {len(predictions)}")
                print(f"Files saved:")
                print("  - predictions.csv")
                print("  - fixtures_2025_2026_simulated.csv")
                print("  - predicted_final_table_2025_2026.csv")
    
    except ImportError as e:
        print(f"❌ Could not import model: {e}")
        print("\nTo use this script:")
        print("1. First run model.py to train your classifier")
        print("2. Then run this script to simulate the season")
        print("\nExample:")
        print("  python model.py")
        print("  python simulate_with_historical_averages.py")
