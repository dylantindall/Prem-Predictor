"""
Helper script to prepare data for season simulation.
Extracts available matchweeks from the data and prepares fixtures for simulation.
"""

import pandas as pd
import numpy as np

def prepare_simulation_data(
    historical_data_path='prem_results.csv',
    current_season='2025/26',
    split_matchweek=8
):
    """
    Prepare data files for season simulation starting from matchweek 8.
    
    Parameters:
    -----------
    historical_data_path : str
        Path to your full historical results CSV
    current_season : str
        Season identifier for the season you want to simulate
    split_matchweek : int
        Matchweek to start simulating from (default: 8)
        
    Creates:
    --------
    1. prem_results_2025_2026_mw1_7.csv - Actual results for initialization
    2. fixtures_2025_2026.csv - All fixtures with features for simulation
    """
    
    print("="*70)
    print("PREPARING SIMULATION DATA")
    print("="*70)
    
    # Load full historical data
    df = pd.read_csv(historical_data_path)
    
    # Standardize column names
    df = df.rename(columns={
        'date': 'date',
        'match_week': 'match_week',
        'season': 'season',
        'result': 'result'
    })
    
    print(f"\nLoaded {len(df)} total matches")
    print(f"Seasons: {sorted(df['season'].unique())}")
    
    # Filter to current season
    current_season_data = df[df['season'] == current_season].copy()
    
    if len(current_season_data) == 0:
        raise ValueError(f"No data found for season {current_season}")
    
    print(f"\nCurrent season ({current_season}): {len(current_season_data)} matches")
    print(f"Matchweeks available: {sorted(current_season_data['match_week'].unique())}")
    
    # Split into historical (already played) and fixtures (to predict)
    historical = current_season_data[
        current_season_data['match_week'] < split_matchweek
    ].copy()
    
    all_fixtures = current_season_data.copy()
    
    print(f"\n✓ Historical matches (weeks 1-{split_matchweek-1}): {len(historical)}")
    print(f"✓ Total fixtures (all weeks): {len(all_fixtures)}")
    
    # Save historical results
    historical_output = f'prem_results_{current_season.replace("/", "_")}_mw1_{split_matchweek-1}.csv'
    historical.to_csv(historical_output, index=False)
    print(f"\n✓ Saved historical results to: {historical_output}")
    
    # Save all fixtures
    fixtures_output = f'fixtures_{current_season.replace("/", "_")}.csv'
    all_fixtures.to_csv(fixtures_output, index=False)
    print(f"✓ Saved fixtures to: {fixtures_output}")
    
    # Check for required features
    print("\n" + "="*70)
    print("FEATURE CHECK")
    print("="*70)
    
    required_frozen_features = [
        'home_avg_h2h_goals',
        'away_avg_h2h_goals',
        'home_team_goals_for',
        'home_team_goals_against',
        'home_team_goal_diff',
        'away_team_goals_for',
        'away_team_goals_against',
        'away_team_goal_diff'
    ]
    
    required_static_features = [
        'home_team_label',
        'away_team_label'
    ]
    
    missing_frozen = [f for f in required_frozen_features if f not in all_fixtures.columns]
    missing_static = [f for f in required_static_features if f not in all_fixtures.columns]
    
    if missing_frozen:
        print("\n⚠ WARNING: Missing frozen features (these are REQUIRED):")
        for f in missing_frozen:
            print(f"  - {f}")
        print("\nThese features require goal data and cannot be calculated from W/D/L alone.")
        print("You must add these to your fixtures CSV manually.")
    else:
        print("\n✓ All frozen features present")
    
    if missing_static:
        print("\n⚠ WARNING: Missing static features:")
        for f in missing_static:
            print(f"  - {f}")
        print("\nYou may need to encode team names using your classifier's label encoder.")
    else:
        print("✓ All static features present")
    
    # Show sample of what features are available
    print("\n" + "="*70)
    print("AVAILABLE FEATURES IN YOUR DATA")
    print("="*70)
    
    feature_cols = [col for col in all_fixtures.columns 
                   if col not in ['date', 'match_week', 'season', 'result', 'home_team', 'away_team']]
    
    print(f"\nTotal features: {len(feature_cols)}")
    print("\nFeature list:")
    for i, feat in enumerate(feature_cols, 1):
        print(f"  {i:2d}. {feat}")
    
    print("\n" + "="*70)
    print("NEXT STEPS")
    print("="*70)
    print(f"""
1. Verify that {fixtures_output} contains all required frozen features
2. If missing, add them from previous season's ending values
3. Ensure team labels match your trained classifier
4. Run the season simulator with:
   
   simulator.initialize_from_historical_matches(
       historical_results_df=pd.read_csv('{historical_output}'),
       fixtures_df=pd.read_csv('{fixtures_output}'),
       start_from_matchweek={split_matchweek}
   )
    """)
    
    return historical, all_fixtures


def add_missing_features_from_previous_season(
    fixtures_path,
    previous_season_data_path,
    previous_season='2024/25',
    output_path=None
):
    """
    Add missing frozen features to fixtures using previous season's final values.
    
    Parameters:
    -----------
    fixtures_path : str
        Path to fixtures CSV (missing some features)
    previous_season_data_path : str
        Path to full historical data
    previous_season : str
        Season to get final values from
    output_path : str
        Where to save updated fixtures (default: overwrites fixtures_path)
    """
    
    print("\n" + "="*70)
    print("ADDING MISSING FEATURES FROM PREVIOUS SEASON")
    print("="*70)
    
    # Load fixtures
    fixtures = pd.read_csv(fixtures_path)
    
    # Load previous season data
    prev_data = pd.read_csv(previous_season_data_path)
    prev_data = prev_data[prev_data['season'] == previous_season]
    
    # Get final matchweek values for each team
    final_matchweek = prev_data['match_week'].max()
    final_data = prev_data[prev_data['match_week'] == final_matchweek]
    
    print(f"\nUsing final values from {previous_season}, matchweek {final_matchweek}")
    
    # Create lookup dictionaries for each team's final stats
    team_stats = {}
    
    for _, row in final_data.iterrows():
        home_team = row['home_team']
        away_team = row['away_team']
        
        if home_team not in team_stats:
            team_stats[home_team] = {
                'goals_for': row.get('home_team_goals_for', 0),
                'goals_against': row.get('home_team_goals_against', 0),
                'goal_diff': row.get('home_team_goal_diff', 0)
            }
        
        if away_team not in team_stats:
            team_stats[away_team] = {
                'goals_for': row.get('away_team_goals_for', 0),
                'goals_against': row.get('away_team_goals_against', 0),
                'goal_diff': row.get('away_team_goal_diff', 0)
            }
    
    # Add features to fixtures
    frozen_features_to_add = [
        'home_team_goals_for', 'home_team_goals_against', 'home_team_goal_diff',
        'away_team_goals_for', 'away_team_goals_against', 'away_team_goal_diff'
    ]
    
    for feat in frozen_features_to_add:
        if feat not in fixtures.columns:
            fixtures[feat] = 0
    
    # Fill in values
    for idx, row in fixtures.iterrows():
        home_team = row['home_team']
        away_team = row['away_team']
        
        if home_team in team_stats:
            fixtures.at[idx, 'home_team_goals_for'] = team_stats[home_team]['goals_for']
            fixtures.at[idx, 'home_team_goals_against'] = team_stats[home_team]['goals_against']
            fixtures.at[idx, 'home_team_goal_diff'] = team_stats[home_team]['goal_diff']
        
        if away_team in team_stats:
            fixtures.at[idx, 'away_team_goals_for'] = team_stats[away_team]['goals_for']
            fixtures.at[idx, 'away_team_goals_against'] = team_stats[away_team]['goals_against']
            fixtures.at[idx, 'away_team_goal_diff'] = team_stats[away_team]['goal_diff']
    
    # Add h2h goal averages (using 0 as default since we can't calculate without goal data)
    if 'home_avg_h2h_goals' not in fixtures.columns:
        fixtures['home_avg_h2h_goals'] = 0
    if 'away_avg_h2h_goals' not in fixtures.columns:
        fixtures['away_avg_h2h_goals'] = 0
    
    # Save updated fixtures
    if output_path is None:
        output_path = fixtures_path
    
    fixtures.to_csv(output_path, index=False)
    print(f"\n✓ Updated fixtures saved to: {output_path}")
    print(f"✓ Added {len(frozen_features_to_add)} frozen features")
    
    return fixtures


def create_team_label_encoder(historical_data_path, output_path='team_labels.csv'):
    """
    Create team label encodings from historical data.
    
    Parameters:
    -----------
    historical_data_path : str
        Path to historical results CSV
    output_path : str
        Where to save team label mapping
        
    Returns:
    --------
    DataFrame with team names and their encoded labels
    """
    from sklearn.preprocessing import LabelEncoder
    
    print("\n" + "="*70)
    print("CREATING TEAM LABEL ENCODER")
    print("="*70)
    
    # Load data
    df = pd.read_csv(historical_data_path)
    
    # Get all unique teams
    teams = sorted(list(set(df['home_team'].unique()) | set(df['away_team'].unique())))
    
    print(f"\nFound {len(teams)} unique teams")
    
    # Create encoder
    le = LabelEncoder()
    labels = le.fit_transform(teams)
    
    # Create mapping
    team_mapping = pd.DataFrame({
        'team': teams,
        'label': labels
    })
    
    print("\nTeam Label Mapping:")
    print(team_mapping.to_string(index=False))
    
    # Save mapping
    team_mapping.to_csv(output_path, index=False)
    print(f"\n✓ Team labels saved to: {output_path}")
    
    return team_mapping


def add_team_labels_to_fixtures(fixtures_path, team_mapping_path, output_path=None):
    """
    Add team label encodings to fixtures dataframe.
    
    Parameters:
    -----------
    fixtures_path : str
        Path to fixtures CSV
    team_mapping_path : str
        Path to team labels CSV (from create_team_label_encoder)
    output_path : str
        Where to save updated fixtures
    """
    
    print("\n" + "="*70)
    print("ADDING TEAM LABELS TO FIXTURES")
    print("="*70)
    
    # Load data
    fixtures = pd.read_csv(fixtures_path)
    team_mapping = pd.read_csv(team_mapping_path)
    
    # Create lookup dictionary
    label_dict = dict(zip(team_mapping['team'], team_mapping['label']))
    
    # Add labels
    fixtures['home_team_label'] = fixtures['home_team'].map(label_dict)
    fixtures['away_team_label'] = fixtures['away_team'].map(label_dict)
    
    # Check for missing labels
    missing_home = fixtures[fixtures['home_team_label'].isna()]['home_team'].unique()
    missing_away = fixtures[fixtures['away_team_label'].isna()]['away_team'].unique()
    
    if len(missing_home) > 0 or len(missing_away) > 0:
        print("\n⚠ WARNING: Some teams not found in label mapping:")
        for team in set(list(missing_home) + list(missing_away)):
            print(f"  - {team}")
        print("\nThese teams need to be added to your label encoder.")
    else:
        print("\n✓ All teams successfully encoded")
    
    # Save
    if output_path is None:
        output_path = fixtures_path
    
    fixtures.to_csv(output_path, index=False)
    print(f"✓ Updated fixtures saved to: {output_path}")
    
    return fixtures


# ============================================================================
# COMPLETE WORKFLOW EXAMPLE
# ============================================================================

if __name__ == "__main__":
    print("\n" + "="*70)
    print("SEASON SIMULATION DATA PREPARATION WIZARD")
    print("="*70)
    
    print("""
This script will help you prepare the data files needed for season simulation.

You will need:
1. Your historical results CSV (prem_results.csv)
2. The season you want to simulate (e.g., '2025/26')
3. The matchweek to start simulating from (e.g., 8)
    """)
    
    # Configuration
    historical_data_path = input("Path to historical results CSV [prem_results.csv]: ").strip() or 'prem_results.csv'
    current_season = input("Season to simulate [2025/26]: ").strip() or '2025/26'
    split_matchweek = int(input("Start simulating from matchweek [8]: ").strip() or '8')
    
    # Step 1: Split data into historical and fixtures
    print("\n" + "="*70)
    print("STEP 1: Splitting Data")
    print("="*70)
    
    historical, fixtures = prepare_simulation_data(
        historical_data_path=historical_data_path,
        current_season=current_season,
        split_matchweek=split_matchweek
    )
    
    # Step 2: Check if we need to add missing features
    frozen_features = [
        'home_avg_h2h_goals', 'away_avg_h2h_goals',
        'home_team_goals_for', 'home_team_goals_against', 'home_team_goal_diff',
        'away_team_goals_for', 'away_team_goals_against', 'away_team_goal_diff'
    ]
    
    missing_frozen = [f for f in frozen_features if f not in fixtures.columns]
    
    if missing_frozen:
        print("\n" + "="*70)
        print("STEP 2: Adding Missing Frozen Features")
        print("="*70)
        
        add_features = input("\nWould you like to add missing features from previous season? (y/n) [y]: ").strip().lower()
        
        if add_features != 'n':
            previous_season = input("Previous season identifier [2024/25]: ").strip() or '2024/25'
            
            fixtures_path = f'fixtures_{current_season.replace("/", "_")}.csv'
            
            fixtures = add_missing_features_from_previous_season(
                fixtures_path=fixtures_path,
                previous_season_data_path=historical_data_path,
                previous_season=previous_season,
                output_path=fixtures_path
            )
    
    # Step 3: Check team labels
    static_features = ['home_team_label', 'away_team_label']
    missing_static = [f for f in static_features if f not in fixtures.columns]
    
    if missing_static:
        print("\n" + "="*70)
        print("STEP 3: Creating Team Label Encodings")
        print("="*70)
        
        create_labels = input("\nWould you like to create team label encodings? (y/n) [y]: ").strip().lower()
        
        if create_labels != 'n':
            team_mapping = create_team_label_encoder(
                historical_data_path=historical_data_path,
                output_path='team_labels.csv'
            )
            
            fixtures_path = f'fixtures_{current_season.replace("/", "_")}.csv'
            
            fixtures = add_team_labels_to_fixtures(
                fixtures_path=fixtures_path,
                team_mapping_path='team_labels.csv',
                output_path=fixtures_path
            )
    
    # Final summary
    print("\n" + "="*70)
    print("PREPARATION COMPLETE!")
    print("="*70)
    
    print(f"""
Your data is ready for simulation!

Files created:
1. prem_results_{current_season.replace('/', '_')}_mw1_{split_matchweek-1}.csv
   → Historical results for initialization
   
2. fixtures_{current_season.replace('/', '_')}.csv
   → All fixtures with features for simulation
   
3. team_labels.csv (if created)
   → Team name to label encoding mapping

Next steps:
1. Load your trained classifier
2. Create SeasonSimulator instance
3. Run: simulator.initialize_from_historical_matches(...)
4. Simulate: predictions = simulator.simulate_full_season(start_matchweek={split_matchweek})

Example code:
-----------
from season_simulator import SeasonSimulator
import pandas as pd

simulator = SeasonSimulator(your_trained_classifier, verbose=True)

simulator.initialize_from_historical_matches(
    historical_results_df=pd.read_csv('prem_results_{current_season.replace('/', '_')}_mw1_{split_matchweek-1}.csv'),
    fixtures_df=pd.read_csv('fixtures_{current_season.replace('/', '_')}.csv'),
    start_from_matchweek={split_matchweek}
)

predictions = simulator.simulate_full_season(start_matchweek={split_matchweek}, end_matchweek=38)
final_table = simulator.get_final_table()
    """)