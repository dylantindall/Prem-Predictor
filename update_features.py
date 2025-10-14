import argparse
from collections import defaultdict, deque
from typing import Dict, Deque, Tuple

import pandas as pd

"""
Update the fixtures 2025-2026 CSV file so that the features are updated
based on the results of the previous matchweek.
"""


FORM_WINDOW = 5
H2H_WINDOW = 5


def ensure_columns(df: pd.DataFrame) -> pd.DataFrame:
    required = [
        'match_week', 'date', 'home_team', 'away_team',
        'home_team_label', 'away_team_label',
        'home_goals', 'away_goals', 'result',
        'home_win_form', 'home_draw_form', 'home_loss_form',
        'away_win_form', 'away_draw_form', 'away_loss_form',
        'home_position', 'away_position', 'home_points', 'away_points',
        'position_gap', 'points_gap',
        'h2h_home_wins', 'h2h_draws', 'h2h_away_wins',
        'home_avg_h2h_goals', 'away_avg_h2h_goals',
        'home_team_goals_for', 'home_team_goals_against', 'home_team_goal_diff',
        'away_team_goals_for', 'away_team_goals_against', 'away_team_goal_diff',
    ]
    for col in required:
        if col not in df.columns:
            df[col] = pd.NA
    return df


def compute_positions(team_table: Dict[str, dict]) -> Dict[str, int]:
    rows = []
    for team, stats in team_table.items():
        rows.append((team, stats['points'], stats['goal_diff'], stats['goals_for']))
    rows.sort(key=lambda x: (-x[1], -x[2], -x[3], x[0]))
    return {t: i + 1 for i, (t, *_rest) in enumerate(rows)}


def main(input_csv: str, output_csv: str):
    df = pd.read_csv(input_csv)
    df = ensure_columns(df)

    # Normalize types where possible
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'], errors='coerce')

    # Identify all teams in this fixtures file
    teams = sorted(set(df['home_team']).union(set(df['away_team'])))

    # Initialize per-team cumulative state
    team_table = {
        t: {
            'points': 0,
            'played': 0,
            'wins': 0,
            'draws': 0,
            'losses': 0,
            'goals_for': 0,
            'goals_against': 0,
            'goal_diff': 0,
            'recent_results': deque(maxlen=FORM_WINDOW),  # 'W','D','L'
        }
        for t in teams
    }

    # H2H tracker: for each unordered pair store last outcomes and goals from home perspective
    h2h_results: Dict[Tuple[str, str], Deque[str]] = defaultdict(lambda: deque(maxlen=H2H_WINDOW))
    h2h_home_goals: Dict[Tuple[str, str], Deque[int]] = defaultdict(lambda: deque(maxlen=H2H_WINDOW))
    h2h_away_goals: Dict[Tuple[str, str], Deque[int]] = defaultdict(lambda: deque(maxlen=H2H_WINDOW))

    # Sort by match week, keep internal order stable
    df = df.sort_values(['match_week', 'date'], kind='mergesort').reset_index(drop=True)

    # Process week by week, so features for week N depend only on results up to week N-1
    for mw in sorted(df['match_week'].dropna().unique()):
        week_mask = df['match_week'] == mw
        week_slice = df[week_mask].copy()

        # 1) Compute pre-match features for the entire week from current cumulative state
        for idx, row in week_slice.iterrows():
            home = row['home_team']
            away = row['away_team']
            key = tuple(sorted([home, away]))

            positions = compute_positions(team_table)
            home_stats = team_table[home]
            away_stats = team_table[away]

            # Form counts
            hw = sum(1 for r in home_stats['recent_results'] if r == 'W')
            hd = sum(1 for r in home_stats['recent_results'] if r == 'D')
            hl = sum(1 for r in home_stats['recent_results'] if r == 'L')
            aw = sum(1 for r in away_stats['recent_results'] if r == 'W')
            ad = sum(1 for r in away_stats['recent_results'] if r == 'D')
            al = sum(1 for r in away_stats['recent_results'] if r == 'L')

            # H2H recent
            h2h_home_wins = sum(1 for r in h2h_results[key] if r == 'H')
            h2h_draws = sum(1 for r in h2h_results[key] if r == 'D')
            h2h_away_wins = sum(1 for r in h2h_results[key] if r == 'A')
            home_avg_h2h_goals = (
                float(pd.Series(h2h_home_goals[key]).mean()) if len(h2h_home_goals[key]) > 0 else 0.0
            )
            away_avg_h2h_goals = (
                float(pd.Series(h2h_away_goals[key]).mean()) if len(h2h_away_goals[key]) > 0 else 0.0
            )

            # Write features to main df
            orig_idx = row.name
            df.at[orig_idx, 'home_win_form'] = hw
            df.at[orig_idx, 'home_draw_form'] = hd
            df.at[orig_idx, 'home_loss_form'] = hl
            df.at[orig_idx, 'away_win_form'] = aw
            df.at[orig_idx, 'away_draw_form'] = ad
            df.at[orig_idx, 'away_loss_form'] = al

            df.at[orig_idx, 'home_position'] = positions.get(home, 20)
            df.at[orig_idx, 'away_position'] = positions.get(away, 20)
            df.at[orig_idx, 'home_points'] = home_stats['points']
            df.at[orig_idx, 'away_points'] = away_stats['points']
            df.at[orig_idx, 'position_gap'] = abs(positions.get(home, 20) - positions.get(away, 20))
            df.at[orig_idx, 'points_gap'] = abs(home_stats['points'] - away_stats['points'])

            df.at[orig_idx, 'h2h_home_wins'] = h2h_home_wins
            df.at[orig_idx, 'h2h_draws'] = h2h_draws
            df.at[orig_idx, 'h2h_away_wins'] = h2h_away_wins
            df.at[orig_idx, 'home_avg_h2h_goals'] = home_avg_h2h_goals
            df.at[orig_idx, 'away_avg_h2h_goals'] = away_avg_h2h_goals

            df.at[orig_idx, 'home_team_goals_for'] = home_stats['goals_for']
            df.at[orig_idx, 'home_team_goals_against'] = home_stats['goals_against']
            df.at[orig_idx, 'home_team_goal_diff'] = home_stats['goal_diff']
            df.at[orig_idx, 'away_team_goals_for'] = away_stats['goals_for']
            df.at[orig_idx, 'away_team_goals_against'] = away_stats['goals_against']
            df.at[orig_idx, 'away_team_goal_diff'] = away_stats['goal_diff']

        # 2) After the week is fully feature-filled, apply results to update cumulative state
        for idx, row in week_slice.iterrows():
            home = row['home_team']
            away = row['away_team']
            key = tuple(sorted([home, away]))
            home_goals = row['home_goals'] if pd.notna(row['home_goals']) else None
            away_goals = row['away_goals'] if pd.notna(row['away_goals']) else None

            if home_goals is None or away_goals is None:
                continue

            try:
                hg = int(home_goals)
                ag = int(away_goals)
            except Exception:
                continue

            if hg > ag:
                res = 'H'
            elif ag > hg:
                res = 'A'
            else:
                res = 'D'

            # Persist result in main df
            df.at[row.name, 'result'] = res

            home_stats = team_table[home]
            away_stats = team_table[away]

            if res == 'H':
                home_stats['points'] += 3
                home_stats['wins'] += 1
                away_stats['losses'] += 1
                home_stats['recent_results'].append('W')
                away_stats['recent_results'].append('L')
            elif res == 'A':
                away_stats['points'] += 3
                away_stats['wins'] += 1
                home_stats['losses'] += 1
                home_stats['recent_results'].append('L')
                away_stats['recent_results'].append('W')
            else:
                home_stats['points'] += 1
                away_stats['points'] += 1
                home_stats['draws'] += 1
                away_stats['draws'] += 1
                home_stats['recent_results'].append('D')
                away_stats['recent_results'].append('D')

            home_stats['played'] += 1
            away_stats['played'] += 1

            # Goals aggregation
            home_stats['goals_for'] += hg
            home_stats['goals_against'] += ag
            away_stats['goals_for'] += ag
            away_stats['goals_against'] += hg
            home_stats['goal_diff'] = home_stats['goals_for'] - home_stats['goals_against']
            away_stats['goal_diff'] = away_stats['goals_for'] - away_stats['goals_against']

            # Update H2H trackers from the home perspective for this fixture
            h2h_results[key].append(res)
            h2h_home_goals[key].append(hg)
            h2h_away_goals[key].append(ag)

    # Write back
    # Restore date formatting if needed
    if 'date' in df.columns:
        # Keep original date strings where possible; otherwise ISO
        try:
            df['date'] = df['date'].dt.strftime('%Y/%m/%d')
        except Exception:
            pass

    df.to_csv(output_csv, index=False)
    print(f"Updated fixtures written to {output_csv} ({len(df)} rows)")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Update fixtures file features based on entered goals')
    parser.add_argument('--input', default='fixtures_2025_2026.csv', help='Input fixtures CSV path')
    parser.add_argument('--output', default='fixtures_2025_2026.csv', help='Output fixtures CSV path')
    args = parser.parse_args()

    main(args.input, args.output)


