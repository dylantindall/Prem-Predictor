"""
Fixture Update Manager

This script handles:
1. Adding new real results to fixtures
2. Comparing predictions vs actual results
3. Locking in past predictions for historical tracking
4. Updating the dashboard with prediction accuracy
"""

import pandas as pd
import numpy as np
from datetime import datetime
import os
import shutil

# File paths
FIXTURES_SIMULATED = 'fixtures_2025_2026_simulated.csv'
PREDICTIONS_LOCKED = 'predictions_locked.csv'
FIXTURES_WITH_ACCURACY = 'fixtures_with_accuracy.csv'


def initialize_predictions_tracking():
    """
    Initialize the predictions tracking file if it doesn't exist.
    This stores the ORIGINAL predictions before any results come in.
    """
    if not os.path.exists(PREDICTIONS_LOCKED):
        print("Initializing predictions tracking...")
        
        # Load current simulated fixtures
        df = pd.read_csv(FIXTURES_SIMULATED)
        
        # Create locked predictions file with only essential columns
        locked_df = df[['match_week', 'date', 'home_team', 'away_team', 'result']].copy()
        locked_df.rename(columns={'result': 'predicted_result'}, inplace=True)
        
        # Add tracking columns
        locked_df['actual_result'] = None
        locked_df['actual_home_goals'] = None
        locked_df['actual_away_goals'] = None
        locked_df['prediction_correct'] = None
        locked_df['prediction_status'] = 'pending'  # pending, correct, incorrect
        
        locked_df.to_csv(PREDICTIONS_LOCKED, index=False)
        print(f"✓ Created {PREDICTIONS_LOCKED}")
    else:
        print(f"✓ {PREDICTIONS_LOCKED} already exists")


def determine_result_from_goals(home_goals, away_goals):
    """Determine result (H/D/A) from goal scores."""
    if pd.isna(home_goals) or pd.isna(away_goals):
        return None
    
    if home_goals > away_goals:
        return 'H'
    elif home_goals < away_goals:
        return 'A'
    else:
        return 'D'


def update_fixtures_with_new_results(fixtures_csv='fixtures_2025_2026.csv'):
    """
    Update the tracking file with new actual results from fixtures_2025_2026.csv
    
    This compares predictions against actual results and marks them as correct/incorrect.
    """
    print("\n" + "="*70)
    print("UPDATING FIXTURES WITH NEW RESULTS")
    print("="*70)
    
    # Load files
    fixtures_df = pd.read_csv(fixtures_csv)
    locked_df = pd.read_csv(PREDICTIONS_LOCKED)
    
    updates_made = 0
    
    # Go through each fixture with actual results
    for idx, fixture in fixtures_df.iterrows():
        # Skip if no goals data (not played yet)
        if pd.isna(fixture['home_goals']) or pd.isna(fixture['away_goals']):
            continue
        
        # Find matching row in locked predictions
        match_mask = (
            (locked_df['match_week'] == fixture['match_week']) &
            (locked_df['home_team'] == fixture['home_team']) &
            (locked_df['away_team'] == fixture['away_team'])
        )
        
        if not match_mask.any():
            continue
        
        match_idx = locked_df[match_mask].index[0]
        
        # Skip if already updated
        if locked_df.at[match_idx, 'prediction_status'] != 'pending':
            continue
        
        # Get actual result
        actual_result = determine_result_from_goals(fixture['home_goals'], fixture['away_goals'])
        predicted_result = locked_df.at[match_idx, 'predicted_result']
        
        # Update the locked predictions file
        locked_df.at[match_idx, 'actual_result'] = actual_result
        locked_df.at[match_idx, 'actual_home_goals'] = int(fixture['home_goals'])
        locked_df.at[match_idx, 'actual_away_goals'] = int(fixture['away_goals'])
        locked_df.at[match_idx, 'prediction_correct'] = (predicted_result == actual_result)
        locked_df.at[match_idx, 'prediction_status'] = 'correct' if (predicted_result == actual_result) else 'incorrect'
        
        updates_made += 1
        
        # Print update
        symbol = "✓" if (predicted_result == actual_result) else "✗"
        print(f"{symbol} MW{int(fixture['match_week']):2d}: {fixture['home_team']:<20} {int(fixture['home_goals'])}-{int(fixture['away_goals'])} {fixture['away_team']:<20} (Predicted: {predicted_result}, Actual: {actual_result})")
    
    # Save updated locked predictions
    locked_df.to_csv(PREDICTIONS_LOCKED, index=False)
    
    print(f"\n✓ Updated {updates_made} match results")
    
    # Calculate and display accuracy stats
    completed = locked_df[locked_df['prediction_status'] != 'pending']
    if len(completed) > 0:
        correct = (completed['prediction_status'] == 'correct').sum()
        accuracy = (correct / len(completed)) * 100
        print(f"\nPrediction Accuracy: {correct}/{len(completed)} ({accuracy:.1f}%)")
    
    return locked_df


def create_dashboard_fixtures_file():
    """
    Create a fixtures file for the dashboard that includes:
    - All fixtures with predictions
    - Accuracy indicators for completed matches
    """
    print("\n" + "="*70)
    print("CREATING DASHBOARD FIXTURES FILE")
    print("="*70)
    
    # Load files
    simulated_df = pd.read_csv(FIXTURES_SIMULATED)
    locked_df = pd.read_csv(PREDICTIONS_LOCKED)
    
    # Merge to add accuracy information
    dashboard_df = simulated_df.copy()
    
    # Add prediction tracking columns
    dashboard_df['predicted_result'] = locked_df['predicted_result']
    dashboard_df['prediction_status'] = locked_df['prediction_status']
    dashboard_df['prediction_correct'] = locked_df['prediction_correct']
    
    # Save for dashboard use
    dashboard_df.to_csv(FIXTURES_WITH_ACCURACY, index=False)
    print(f"✓ Created {FIXTURES_WITH_ACCURACY} for dashboard")
    
    return dashboard_df


def get_prediction_summary():
    """Get summary statistics of predictions."""
    locked_df = pd.read_csv(PREDICTIONS_LOCKED)
    
    total_matches = len(locked_df)
    completed = locked_df[locked_df['prediction_status'] != 'pending']
    pending = locked_df[locked_df['prediction_status'] == 'pending']
    correct = locked_df[locked_df['prediction_status'] == 'correct']
    incorrect = locked_df[locked_df['prediction_status'] == 'incorrect']
    
    print("\n" + "="*70)
    print("PREDICTION SUMMARY")
    print("="*70)
    print(f"Total matches:      {total_matches}")
    print(f"Completed:          {len(completed)}")
    print(f"  ✓ Correct:        {len(correct)}")
    print(f"  ✗ Incorrect:      {len(incorrect)}")
    print(f"Pending:            {len(pending)}")
    
    if len(completed) > 0:
        accuracy = (len(correct) / len(completed)) * 100
        print(f"\nOverall Accuracy:   {accuracy:.1f}%")
        
        # Breakdown by result type
        print("\nAccuracy by Prediction Type:")
        for pred_type in ['H', 'D', 'A']:
            pred_matches = completed[completed['predicted_result'] == pred_type]
            if len(pred_matches) > 0:
                pred_correct = (pred_matches['prediction_correct'] == True).sum()
                pred_acc = (pred_correct / len(pred_matches)) * 100
                result_name = {'H': 'Home Win', 'D': 'Draw', 'A': 'Away Win'}[pred_type]
                print(f"  {result_name:12s}: {pred_correct}/{len(pred_matches)} ({pred_acc:.1f}%)")


def main():
    """Main workflow for updating fixtures."""
    print("\n" + "="*70)
    print("FIXTURE UPDATE WORKFLOW")
    print("="*70)
    print("""
This script:
1. Initializes prediction tracking (if first time)
2. Compares predictions against new actual results
3. Locks in past predictions with correct/incorrect markers
4. Creates fixtures file for dashboard with accuracy indicators

After running this, you can:
- Re-run run_simulation.py to update FUTURE predictions with new data
- The past predictions remain locked and won't change
    """)
    
    # Step 1: Initialize if needed
    initialize_predictions_tracking()
    
    # Step 2: Update with new results
    locked_df = update_fixtures_with_new_results()
    
    # Step 3: Create dashboard file
    create_dashboard_fixtures_file()
    
    # Step 4: Show summary
    get_prediction_summary()
    
    print("\n" + "="*70)
    print("NEXT STEPS")
    print("="*70)
    print("""
1. To update predictions for future matchweeks with new data:
   - Update start_simulation_from_mw in run_simulation.py
   - Run: python run_simulation.py
   - Run: python update_fixtures.py (this script again)

2. The dashboard will use fixtures_with_accuracy.csv to show:
   - ✓ for correct predictions
   - ✗ for incorrect predictions
   - Future predictions (no marker yet)
    """)


if __name__ == "__main__":
    main()