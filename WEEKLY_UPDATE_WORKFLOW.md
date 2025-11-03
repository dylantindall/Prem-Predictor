# Premier League Prediction System - Weekly Update Workflow

## Overview

This system tracks predictions vs actual results and allows you to update predictions as new matchweek data becomes available.

## Files Explained

1. **fixtures_2025_2026.csv** - Contains actual match results (with goals). Update this weekly with new results.

2. **fixtures_2025_2026_simulated.csv** - Contains all fixtures with predictions for future matches.

3. **predictions_locked.csv** - Stores the ORIGINAL predictions before results come in. Once a match is played, this file compares the prediction vs actual result and locks it in with a ✓ or ✗.

4. **fixtures_with_accuracy.csv** - Used by the dashboard, combines predictions with accuracy indicators.

5. **predicted_final_standings_2025_2026.csv** - Final league table based on all results (actual + predicted).

---

## Weekly Workflow

### Step 1: Add New Results
When a matchweek is complete, add the actual match results to `fixtures_2025_2026.csv`:
- Update `home_goals` and `away_goals` columns with the actual scores
- The `result` column should be: 'H' (home win), 'D' (draw), or 'A' (away win)

### Step 2: Update Prediction Tracking
Run the update script to compare predictions vs actual results:

```bash
python update_fixtures.py
```

This will:
- Compare your predictions against the new actual results
- Mark each prediction as ✓ (correct) or ✗ (incorrect)
- Lock in those predictions permanently
- Update `predictions_locked.csv` and `fixtures_with_accuracy.csv`
- Show your prediction accuracy stats

### Step 3: Re-run Predictions for Future Matches
Now that you have more actual data, update predictions for remaining matches:

1. Edit `run_simulation.py` at line 486:
   ```python
   start_simulation_from_mw=11  # Update this to the next matchweek to predict
   ```
   
   For example:
   - If you just got MW11 results, set this to `12`
   - If you just got MW15 results, set this to `16`

2. Run the simulation:
   ```bash
   python run_simulation.py
   ```
   
   This updates `fixtures_2025_2026_simulated.csv` with new predictions for future matches.

3. Update the prediction tracking again:
   ```bash
   python update_fixtures.py
   ```

### Step 4: Generate Updated Final Standings
```bash
python create_predicted_standings.py
```

This creates the predicted final league table based on:
- Actual results (matchweeks that have been played)
- Current predictions (matchweeks that haven't been played yet)

---

## Dashboard Display

The dashboard shows two pages:

### Historic Data Page
- League standings from past seasons
- Actual results from completed matches

### Model Predictions Page
- Predictions for each upcoming matchweek
- For completed matches: shows the actual result and ✓/✗ indicator
- For future matches: shows only the prediction (no indicator yet)

---

## Example Weekly Workflow

Let's say you have results through MW10 and want to update:

1. **Add MW11 results** to `fixtures_2025_2026.csv`:
   ```
   Liverpool vs Arsenal: 2-1 (H)
   Chelsea vs Man City: 1-1 (D)
   etc...
   ```

2. **Track accuracy**:
   ```bash
   python update_fixtures.py
   ```
   Output:
   ```
   ✓ MW11: Liverpool 2-1 Arsenal (Predicted: H, Actual: H)
   ✗ MW11: Chelsea 1-1 Man City (Predicted: A, Actual: D)
   ...
   Prediction Accuracy: 7/10 (70%)
   ```

3. **Update future predictions** with the new MW11 data:
   
   Edit `run_simulation.py` line 486:
   ```python
   start_simulation_from_mw=12  # Now predicting from MW12 onwards
   ```
   
   Run:
   ```bash
   python run_simulation.py
   python update_fixtures.py
   ```

4. **Generate new standings**:
   ```bash
   python create_predicted_standings.py
   ```

5. **View in dashboard** - The dashboard will now show:
   - MW11 predictions with ✓/✗ indicators showing which you got right
   - MW12-38 predictions (updated based on MW11 results)

---

## Key Points

✓ **Past predictions are locked** - Once a match is played and marked ✓ or ✗, that prediction never changes

✓ **Future predictions update** - Predictions for matches that haven't been played yet will change as you add more actual data

✓ **Track your accuracy** - See how well your model is performing over time

✓ **Continuous improvement** - As the season progresses, your model uses more real data to make better predictions

---

## File Update Summary

| When | Update | Command |
|------|--------|---------|
| New matchweek results | `fixtures_2025_2026.csv` | Manual edit |
| After adding results | Track accuracy | `python update_fixtures.py` |
| Weekly | Re-predict future matches | Edit `run_simulation.py`, then run it |
| After re-predicting | Update tracking | `python update_fixtures.py` |
| Anytime | View final standings | `python create_predicted_standings.py` |

---

## Troubleshooting

**Q: My predictions changed for a match that already happened!**
A: Make sure you ran `update_fixtures.py` after adding the actual result. This locks in the prediction.

**Q: The dashboard doesn't show ✓/✗ indicators**
A: Make sure `fixtures_with_accuracy.csv` exists. Run `update_fixtures.py` to create it.

**Q: How do I know which matchweek to predict from?**
A: Look at your `fixtures_2025_2026.csv` file - find the last matchweek that has `home_goals` and `away_goals` filled in. Predict from the next matchweek.

**Q: Can I re-run predictions multiple times?**
A: Yes! You can re-run predictions as many times as you want for future matches. Only past matches (with actual results) are locked in.
