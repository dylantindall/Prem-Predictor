import pandas as pd
import numpy as np

# # import the raw data to be read
# raw_df = pd.read_csv("raw_prem_data.csv")

# # extract necessary columns from the raw data and rename as required
# df = raw_df[["Season", "MatchWeek","Date", "HomeTeam", "AwayTeam", "FullTimeHomeTeamGoals", "FullTimeAwayTeamGoals", "FullTimeResult", "HomeTeamPoints", "AwayTeamPoints"]].rename(columns={
#              "Season":"season",
#              "MatchWeek":"match_week",
#              "Date":"date",
#              "HomeTeam":"home_team",
#              "AwayTeam":"away_team",
#              "FullTimeHomeTeamGoals":"home_goals",
#              "FullTimeAwayTeamGoals":"away_goals",
#              "FullTimeResult":"result",
#              })

# # start from the 1995-96 season where the number of teams were reduced to 22
# # from 20, for consistency in the model.
# df = df[df['season'] >= '1994-95']

# # define function to calculate a teams form across the last x games,
# # resets each season, and tallies up until there are x prior games

# def calculate_form(team, matches, x):
#     """
#     team: name of the team to calculate the form of.
#     matches: all the matches in the current season prior to the game in hand
#             (ascending oldest to newest).
#     x: the number of games to consider the form of

#     returns -> form: a vector of up to length x eg. ('W','W','L',..'D')
#                wins, draws, losses from up to the last x games
#     """

#     # find the last x matches a team has played
#     team_prior_matches = [
#     m for m in matches 
#     if m["home_team"] == team or m["away_team"] == team
# ]

    
#     # if season has just started, only count from available games.

#     available_matches = min(len(team_prior_matches), x)
#     last_available_x = team_prior_matches[-available_matches:] if team_prior_matches else []

#     form = []
#     wins = draws = losses = 0

#     for m in last_available_x:
#         if m["result"] == "D":
#             form.append("D")
#             draws += 1
#         elif (m["home_team"] == team and m["result"] == "H") or (m["away_team"] == team and m["result"] == "A"):
#             form.append("W")
#             wins += 1
#         elif (m["home_team"] == team and m["result"] == "A") or (m["away_team"] == team and m["result"] == "H"):
#             form.append("L")
#             losses += 1


#     return form, wins, draws, losses

# # add columns to the data frame for home and away teams form across the last 5 games,
# # use the number of wins draws and losses as a constant,
# # form will be used later for the dashboard

# def add_form_columns(df, x):
#     """
#     add form columns for home and away to the data frame for the last x games
#     """

#     # initialize the columns to be added
#     df['home_win_form']  = 0
#     df['home_draw_form'] = 0
#     df['home_loss_form'] = 0
#     df['away_win_form']  = 0
#     df['away_draw_form'] = 0
#     df['away_loss_form'] = 0

#     for idx in range(len(df)):
#         current_season = df.iloc[idx]['season']

#         season_mask = (df['season'] == current_season)
#         prior_matches = df[season_mask].iloc[:idx]

#         if len(prior_matches) > 0:

#             matches_list = prior_matches.to_dict('records')
            
#             # Calculate home team form
#             home_team = df.iloc[idx]['home_team']
#             _, home_wins, home_draws, home_losses = calculate_form(home_team, matches_list, x)
            
#             # Calculate away team form
#             away_team = df.iloc[idx]['away_team']
#             _, away_wins, away_draws, away_losses = calculate_form(away_team, matches_list, x)
            
#             # Update the DataFrame
#             df.iloc[idx, df.columns.get_loc('home_win_form')]  = home_wins
#             df.iloc[idx, df.columns.get_loc('home_draw_form')] = home_draws
#             df.iloc[idx, df.columns.get_loc('home_loss_form')] = home_losses
#             df.iloc[idx, df.columns.get_loc('away_win_form')]  = away_wins
#             df.iloc[idx, df.columns.get_loc('away_draw_form')] = away_draws
#             df.iloc[idx, df.columns.get_loc('away_loss_form')] = away_losses
    
#     return df

# def add_table_position_features(df):
#     """
#     Add current league position and points for home/away teams at the time of each match
#     """
#     # Ensure DataFrame is sorted properly
#     df = df.sort_values(['season', 'date']).reset_index(drop=True)
    
#     # Initialize new columns
#     df['home_position'] = 0
#     df['away_position'] = 0
#     df['home_points'] = 0
#     df['away_points'] = 0
#     df['position_gap'] = 0
#     df['points_gap'] = 0
    
#     # Process each season separately
#     for season in df['season'].unique():
#         print(f"Processing season: {season}")
#         season_mask = df['season'] == season
#         season_df = df[season_mask].copy()
        
#         # Get all teams in this season
#         all_teams = set(season_df['home_team'].unique()) | set(season_df['away_team'].unique())
        
#         # Initialize team statistics for the season
#         team_stats = {}
#         for team in all_teams:
#             team_stats[team] = {
#                 'points': 0,
#                 'played': 0,
#                 'wins': 0,
#                 'draws': 0,
#                 'losses': 0,
#                 'goals_for': 0,
#                 'goals_against': 0,
#                 'goal_difference': 0
#             }
        
#         # Process each match in chronological order
#         for idx in season_df.index:
#             row = df.loc[idx]
#             home_team = row['home_team']
#             away_team = row['away_team']
            
#             # Calculate current table position BEFORE this match
#             current_table = calculate_current_table(team_stats)
            
#             # Get positions and points for both teams
#             home_pos = current_table[home_team]['position']
#             away_pos = current_table[away_team]['position']
#             home_pts = team_stats[home_team]['points']
#             away_pts = team_stats[away_team]['points']
            
#             # Update the DataFrame
#             df.loc[idx, 'home_position'] = home_pos
#             df.loc[idx, 'away_position'] = away_pos
#             df.loc[idx, 'home_points'] = home_pts
#             df.loc[idx, 'away_points'] = away_pts
#             df.loc[idx, 'position_gap'] = abs(home_pos - away_pos)
#             df.loc[idx, 'points_gap'] = abs(home_pts - away_pts)
            
#             # Update team statistics AFTER this match
#             update_team_stats(team_stats, row)
    
#     print("Table position features added successfully!")
#     return df

# def calculate_current_table(team_stats):
#     """
#     Calculate current league table positions based on team statistics
#     """
#     # Convert to list for sorting
#     table = []
#     for team, stats in team_stats.items():
#         table.append({
#             'team': team,
#             'points': stats['points'],
#             'played': stats['played'],
#             'goal_difference': stats['goal_difference'],
#             'goals_for': stats['goals_for']
#         })
    
#     # Sort by: Points (desc), Goal Difference (desc), Goals For (desc), Team name (asc)
#     table.sort(key=lambda x: (-x['points'], -x['goal_difference'], -x['goals_for'], x['team']))
    
#     # Create position lookup
#     positions = {}
#     for i, team_data in enumerate(table):
#         positions[team_data['team']] = {
#             'position': i + 1,
#             'points': team_data['points']
#         }
    
#     return positions

# def update_team_stats(team_stats, match_row):
#     """
#     Update team statistics based on match result
#     """
#     home_team = match_row['home_team']
#     away_team = match_row['away_team']
#     result = match_row['result']
    
#     # Get goals if available (adjust column names as needed)
#     home_goals = match_row.get('home_goals', 0)  # Replace with your actual column name
#     away_goals = match_row.get('away_goals', 0)  # Replace with your actual column name
    
#     # Update games played
#     team_stats[home_team]['played'] += 1
#     team_stats[away_team]['played'] += 1
    
#     # Update goals
#     team_stats[home_team]['goals_for'] += home_goals
#     team_stats[home_team]['goals_against'] += away_goals
#     team_stats[away_team]['goals_for'] += away_goals
#     team_stats[away_team]['goals_against'] += home_goals
    
#     # Update goal difference
#     team_stats[home_team]['goal_difference'] = (team_stats[home_team]['goals_for'] - 
#                                                team_stats[home_team]['goals_against'])
#     team_stats[away_team]['goal_difference'] = (team_stats[away_team]['goals_for'] - 
#                                                team_stats[away_team]['goals_against'])
    
#     # Update points and win/draw/loss records based on result
#     if result == 'H':  # Home win
#         team_stats[home_team]['points'] += 3
#         team_stats[home_team]['wins'] += 1
#         team_stats[away_team]['losses'] += 1
#     elif result == 'A':  # Away win
#         team_stats[away_team]['points'] += 3
#         team_stats[away_team]['wins'] += 1
#         team_stats[home_team]['losses'] += 1
#     elif result == 'D':  # Draw
#         team_stats[home_team]['points'] += 1
#         team_stats[away_team]['points'] += 1
#         team_stats[home_team]['draws'] += 1
#         team_stats[away_team]['draws'] += 1

# # Apply the functions to the DataFram
# df = add_form_columns(df, x=5)
# df = add_table_position_features(df)

# print(df.columns)
# print(df.iloc[0])

# df.to_csv("prem_results.csv", index=False)



# # take the new table and add a string column for month, 
# # and cyclically encoded columns for month_sin and month_cos

df = pd.read_csv("prem_results.csv")
df["date"] = pd.to_datetime(df["date"], errors="coerce")  # convert to datetime

# # Add Month, Year column
df["month_year"] = df["date"].dt.strftime("%B, %Y")

# # Add cyclical encoding
# df["month_sin"] = np.sin(2 * np.pi * df["date"].dt.month / 12)
# df["month_cos"] = np.cos(2 * np.pi * df["date"].dt.month / 12)

# # Apply one hot encoding to the team names
# one_hot_dict = {
#     'Arsenal': 0,
#     'Aston Villa': 1,
#     'Barnsley': 2,
#     'Birmingham': 3,
#     'Blackburn': 4,
#     'Blackpool': 5,
#     'Bolton': 6,
#     'Bournemouth': 7,
#     'Bradford': 8,
#     'Brentford': 9,
#     'Brighton': 10,
#     'Burnley': 11,
#     'Cardiff': 12,
#     'Charlton': 13,
#     'Chelsea': 14,
#     'Coventry': 15,
#     'Crystal Palace': 16,
#     'Derby': 17,
#     'Everton': 18,
#     'Fulham': 19,
#     'Huddersfield': 20,
#     'Hull': 21,
#     'Ipswich': 22,
#     'Leeds': 23,
#     'Leicester': 24,
#     'Liverpool': 25,
#     'Luton': 26,
#     'Man City': 27,
#     'Man United': 28,
#     'Middlesbrough': 29,
#     'Newcastle': 30,
#     'Norwich': 31,
#     "Nott'm Forest": 32,
#     'Portsmouth': 33,
#     'QPR': 34,
#     'Reading': 35,
#     'Sheffield United': 36,
#     'Sheffield Weds': 37,
#     'Southampton': 38,
#     'Stoke': 39,
#     'Sunderland': 40,
#     'Swansea': 41,
#     'Tottenham': 42,
#     'Watford': 43,
#     'West Brom': 44,
#     'West Ham': 45,
#     'Wigan': 46,
#     'Wimbledon': 47,
#     'Wolves': 48
# }

# df["home_team_label"] = df["home_team"].map(one_hot_dict)
# df["away_team_label"] = df["away_team"].map(one_hot_dict)

# # Calculate head to head form across the last 5 meetings.
# def add_h2h_features(df, x=5):
#     """
#     Add head-to-head statistics for the last x meetings between home and away teams
    
#     Features added:
#     - h2h_home_wins: Number of wins for the home team in last x H2H matches
#     - h2h_draws: Number of draws in last x H2H matches
#     - h2h_away_wins: Number of wins for the away team in last x H2H matches
#     - home_avg_h2h_goals: Average goals scored by home team in last x H2H matches
#     - away_avg_h2h_goals: Average goals scored by away team in last x H2H matches
#     """
    
#     # Sort by date to ensure chronological order
#     df = df.sort_values(['date']).reset_index(drop=True)
    
#     # Initialize new columns
#     df['h2h_home_wins'] = 0
#     df['h2h_draws'] = 0
#     df['h2h_away_wins'] = 0
#     df['home_avg_h2h_goals'] = 0.0
#     df['away_avg_h2h_goals'] = 0.0
    
#     # Process each match
#     for idx in range(len(df)):
#         current_home = df.iloc[idx]['home_team']
#         current_away = df.iloc[idx]['away_team']
        
#         # Get all prior matches between these two teams (in any venue)
#         prior_h2h = df.iloc[:idx][
#             ((df.iloc[:idx]['home_team'] == current_home) & (df.iloc[:idx]['away_team'] == current_away)) |
#             ((df.iloc[:idx]['home_team'] == current_away) & (df.iloc[:idx]['away_team'] == current_home))
#         ]
        
#         # Get the last x meetings
#         last_x_h2h = prior_h2h.tail(x)
        
#         if len(last_x_h2h) > 0:
#             # Calculate wins/draws/losses from home team's perspective
#             home_wins = 0
#             draws = 0
#             away_wins = 0
#             home_goals_list = []
#             away_goals_list = []
            
#             for _, match in last_x_h2h.iterrows():
#                 # Check if current home team was home or away in this historical match
#                 if match['home_team'] == current_home:
#                     # Home team was playing at home in this H2H match
#                     home_goals_list.append(match['home_goals'])
#                     away_goals_list.append(match['away_goals'])
                    
#                     if match['result'] == 'H':
#                         home_wins += 1
#                     elif match['result'] == 'D':
#                         draws += 1
#                     elif match['result'] == 'A':
#                         away_wins += 1
#                 else:
#                     # Home team was playing away in this H2H match
#                     home_goals_list.append(match['away_goals'])
#                     away_goals_list.append(match['home_goals'])
                    
#                     if match['result'] == 'A':
#                         home_wins += 1
#                     elif match['result'] == 'D':
#                         draws += 1
#                     elif match['result'] == 'H':
#                         away_wins += 1
            
#             # Update the DataFrame
#             df.iloc[idx, df.columns.get_loc('h2h_home_wins')] = home_wins
#             df.iloc[idx, df.columns.get_loc('h2h_draws')] = draws
#             df.iloc[idx, df.columns.get_loc('h2h_away_wins')] = away_wins
#             df.iloc[idx, df.columns.get_loc('home_avg_h2h_goals')] = np.mean(home_goals_list)
#             df.iloc[idx, df.columns.get_loc('away_avg_h2h_goals')] = np.mean(away_goals_list)
    
#     print(f"Head-to-head features added successfully!")
#     print(f"Matches with H2H data: {(df['h2h_home_wins'] + df['h2h_draws'] + df['h2h_away_wins'] > 0).sum()}")
#     print(f"Matches without H2H data: {(df['h2h_home_wins'] + df['h2h_draws'] + df['h2h_away_wins'] == 0).sum()}")
    
#     return df

# # Apply the function
# df = add_h2h_features(df, x=5)

# # Calculate home/away goals for, against and goal difference.
# def add_team_goal_stats(df):
#     """
#     Add overall goal statistics for home and away teams before each match
    
#     Features added:
#     - home_team_goals_for: Total goals scored by home team this season (all games)
#     - home_team_goals_against: Total goals conceded by home team this season (all games)
#     - home_team_goal_diff: Goal difference for home team this season
#     - away_team_goals_for: Total goals scored by away team this season (all games)
#     - away_team_goals_against: Total goals conceded by away team this season (all games)
#     - away_team_goal_diff: Goal difference for away team this season
#     """
    
#     # Sort by season and date
#     df = df.sort_values(['season', 'date']).reset_index(drop=True)
    
#     # Initialize new columns
#     df['home_team_goals_for'] = 0
#     df['home_team_goals_against'] = 0
#     df['home_team_goal_diff'] = 0
#     df['away_team_goals_for'] = 0
#     df['away_team_goals_against'] = 0
#     df['away_team_goal_diff'] = 0
    
#     # Process each season separately
#     for season in df['season'].unique():
#         print(f"Processing season: {season}")
#         season_mask = df['season'] == season
#         season_df = df[season_mask].copy()
        
#         # Get all teams in this season
#         all_teams = set(season_df['home_team'].unique()) | set(season_df['away_team'].unique())
        
#         # Initialize team statistics for the season
#         team_stats = {}
#         for team in all_teams:
#             team_stats[team] = {
#                 'goals_for': 0,
#                 'goals_against': 0,
#                 'goal_difference': 0
#             }
        
#         # Process each match in chronological order
#         for idx in season_df.index:
#             row = df.loc[idx]
#             home_team = row['home_team']
#             away_team = row['away_team']
            
#             # Get current stats BEFORE this match
#             home_gf = team_stats[home_team]['goals_for']
#             home_ga = team_stats[home_team]['goals_against']
#             home_gd = team_stats[home_team]['goal_difference']
            
#             away_gf = team_stats[away_team]['goals_for']
#             away_ga = team_stats[away_team]['goals_against']
#             away_gd = team_stats[away_team]['goal_difference']
            
#             # Update the DataFrame with stats before this match
#             df.loc[idx, 'home_team_goals_for'] = home_gf
#             df.loc[idx, 'home_team_goals_against'] = home_ga
#             df.loc[idx, 'home_team_goal_diff'] = home_gd
#             df.loc[idx, 'away_team_goals_for'] = away_gf
#             df.loc[idx, 'away_team_goals_against'] = away_ga
#             df.loc[idx, 'away_team_goal_diff'] = away_gd
            
#             # Update team statistics AFTER this match
#             home_goals = row['home_goals']
#             away_goals = row['away_goals']
            
#             # Update home team's overall record
#             team_stats[home_team]['goals_for'] += home_goals
#             team_stats[home_team]['goals_against'] += away_goals
#             team_stats[home_team]['goal_difference'] = (team_stats[home_team]['goals_for'] - 
#                                                        team_stats[home_team]['goals_against'])
            
#             # Update away team's overall record
#             team_stats[away_team]['goals_for'] += away_goals
#             team_stats[away_team]['goals_against'] += home_goals
#             team_stats[away_team]['goal_difference'] = (team_stats[away_team]['goals_for'] - 
#                                                        team_stats[away_team]['goals_against'])
    
#     print("Team goal statistics added successfully!")
#     return df

# # Apply the function
# df = add_team_goal_stats(df)

# Save back to the same CSV (overwrite)
df.to_csv("prem_results.csv", index=False)