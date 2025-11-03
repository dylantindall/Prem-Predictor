# Quick Reference: Weekly Update Steps

## Every Week After New Matchweek Results

### 1. Add Real Results
Edit `fixtures_2025_2026.csv` - add goals for completed matches

### 2. Track Prediction Accuracy  
```bash
python update_fixtures.py
```
Shows which predictions were correct (✓) or incorrect (✗)

### 3. Update Future Predictions
```bash
# Edit run_simulation.py line 486: change start_simulation_from_mw to next matchweek
python run_simulation.py
python update_fixtures.py
```

### 4. Generate Standings
```bash
python create_predicted_standings.py
```

---

## Key Files

- **fixtures_2025_2026.csv** - Add actual results here
- **predictions_locked.csv** - Past predictions (locked with ✓/✗)
- **fixtures_with_accuracy.csv** - For dashboard display
- **run_simulation.py** - Line 486: update matchweek number

---

## What Gets Locked vs Updated

✓ **Locked Forever** (once match is played):
- Original prediction for that match
- Whether it was correct/incorrect

🔄 **Updates Each Week** (for future matches):
- Predictions for matches not yet played
- Final standings prediction

---

## Dashboard Shows

**Model Predictions Page:**
- ✓ = Correctly predicted
- ✗ = Incorrectly predicted  
- (blank) = Match not played yet
