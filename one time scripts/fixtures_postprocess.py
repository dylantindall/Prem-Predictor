import argparse
import pandas as pd

# Mapping ESPN team names to prem_results naming conventions
ESPN_TO_PREM_RESULTS = {
    'Arsenal': 'Arsenal',
    'Aston Villa': 'Aston Villa',
    'AFC Bournemouth': 'Bournemouth',
    'Bournemouth': 'Bournemouth',
    'Brentford': 'Brentford',
    'Brighton': 'Brighton',
    'Burnley': 'Burnley',
    'Chelsea': 'Chelsea',
    'Crystal Palace': 'Crystal Palace',
    'Everton': 'Everton',
    'Fulham': 'Fulham',
    'Leeds United': 'Leeds',
    'Leeds': 'Leeds',
    'Leicester City': 'Leicester',
    'Leicester': 'Leicester',
    'Liverpool': 'Liverpool',
    'Luton Town': 'Luton',
    'Luton': 'Luton',
    'Manchester City': 'Man City',
    'Man City': 'Man City',
    'Manchester United': 'Man United',
    'Man United': 'Man United',
    'Newcastle United': 'Newcastle',
    'Newcastle': 'Newcastle',
    'Nottingham Forest': "Nott'm Forest",
    "Nott'm Forest": "Nott'm Forest",
    'Norwich City': 'Norwich',
    'Norwich': 'Norwich',
    'Sheffield United': 'Sheffield United',
    'Sheffield Wednesday': 'Sheffield Weds',
    'Sheffield Weds': 'Sheffield Weds',
    'Southampton': 'Southampton',
    'Tottenham Hotspur': 'Tottenham',
    'Tottenham': 'Tottenham',
    'Watford': 'Watford',
    'West Bromwich Albion': 'West Brom',
    'West Brom': 'West Brom',
    'West Ham United': 'West Ham',
    'West Ham': 'West Ham',
    'Wolves': 'Wolves',
    'Wolverhampton Wanderers': 'Wolves',
    'Sunderland': 'Sunderland',
    'Middlesbrough': 'Middlesbrough',
    'Derby County': 'Derby',
    'Derby': 'Derby',
    'Charlton Athletic': 'Charlton',
    'Charlton': 'Charlton',
    'Blackburn Rovers': 'Blackburn',
    'Blackburn': 'Blackburn',
    'QPR': 'QPR',
}

# One-hot dictionary from data_processing.py (labels)
ONE_HOT_DICT = {
    'Arsenal': 0,
    'Aston Villa': 1,
    'Barnsley': 2,
    'Birmingham': 3,
    'Blackburn': 4,
    'Blackpool': 5,
    'Bolton': 6,
    'Bournemouth': 7,
    'Bradford': 8,
    'Brentford': 9,
    'Brighton': 10,
    'Burnley': 11,
    'Cardiff': 12,
    'Charlton': 13,
    'Chelsea': 14,
    'Coventry': 15,
    'Crystal Palace': 16,
    'Derby': 17,
    'Everton': 18,
    'Fulham': 19,
    'Huddersfield': 20,
    'Hull': 21,
    'Ipswich': 22,
    'Leeds': 23,
    'Leicester': 24,
    'Liverpool': 25,
    'Luton': 26,
    'Man City': 27,
    'Man United': 28,
    'Middlesbrough': 29,
    'Newcastle': 30,
    'Norwich': 31,
    "Nott'm Forest": 32,
    'Portsmouth': 33,
    'QPR': 34,
    'Reading': 35,
    'Sheffield United': 36,
    'Sheffield Weds': 37,
    'Southampton': 38,
    'Stoke': 39,
    'Sunderland': 40,
    'Swansea': 41,
    'Tottenham': 42,
    'Watford': 43,
    'West Brom': 44,
    'West Ham': 45,
    'Wigan': 46,
    'Wimbledon': 47,
    'Wolves': 48
}

# Features list from model.py
FEATURES = ['home_win_form', 'home_draw_form', 'home_loss_form',
            'away_win_form', 'away_draw_form', 'away_loss_form',
            'home_position', 'away_position', 'home_points', 'away_points',
            'position_gap', 'points_gap', 'home_team_label', 'away_team_label',
            'h2h_home_wins', 'h2h_draws', 'h2h_away_wins', 'home_avg_h2h_goals',
            'away_avg_h2h_goals', 'home_team_goals_for', 'home_team_goals_against',
            'home_team_goal_diff', 'away_team_goals_for', 'away_team_goals_against',
            'away_team_goal_diff']


def main(input_csv: str, output_csv: str):
    df = pd.read_csv(input_csv)

    # Map team names to prem_results conventions
    df['home_team'] = df['home_team'].map(lambda x: ESPN_TO_PREM_RESULTS.get(x, x))
    df['away_team'] = df['away_team'].map(lambda x: ESPN_TO_PREM_RESULTS.get(x, x))

    # Normalize date to YYYY/MM/DD (handles DD/MM/YYYY input)
    if 'date' in df.columns:
        parsed_dates = pd.to_datetime(df['date'], errors='coerce', dayfirst=True)
        df['date'] = parsed_dates.dt.strftime('%Y/%m/%d')

    # Add team label columns
    df['home_team_label'] = df['home_team'].map(ONE_HOT_DICT)
    df['away_team_label'] = df['away_team'].map(ONE_HOT_DICT)

    # Ensure all feature columns exist; initialize as empty (NaN)
    for col in FEATURES:
        if col not in df.columns:
            df[col] = pd.NA

    # Write output
    df.to_csv(output_csv, index=False)
    print(f"Wrote postprocessed fixtures to {output_csv} ({len(df)} rows)")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Postprocess ESPN fixtures to match prem_results conventions and features')
    parser.add_argument('--input', default='fixtures_2025_2026.csv', help='Input fixtures CSV path')
    parser.add_argument('--output', default='fixtures_2025_2026.csv', help='Output CSV path')
    args = parser.parse_args()

    main(args.input, args.output)


