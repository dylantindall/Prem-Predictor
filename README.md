# Prem-Predictor
Premier league table/fixture prediction model, including an interactive dashboard to compare past results and model predictions.

FILE USAGE:

one time scripts:
-contains all of the single use scripts used for cleaning and processing
 data/features.

 dashboard.py:
 -all code to create and load the dashboard



 DATA:

 season_tables:
 -folder containing all league table final standings from 1995/96-2024/25.

 fixtures_2025_2026:
 -fixtures data for the 2025 season, updated up to the current matchweek,
  static columns include team labels and matchweek for each fixture, else null.

 prem_results.csv
 -contains full historic fixture data from 1995/96-2024/25.


PREDICTED DATA:

predicted_final_table_2025_2026.csv
-final standings prediction of the model.

fixtures_2025_2026_simulated:
-contains all data from the actual fixtures table, then adds the simulated
 fixtures based on the models predictions for the remaining matchweeks.

