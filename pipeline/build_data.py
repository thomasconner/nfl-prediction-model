"""Build compact play and game tables from nflverse files (2021-2026).

Usage: python3 -I build_data.py <data_dir>
Writes plays.parquet and games.parquet into <data_dir>.
"""
import sys, os
import pandas as pd
import pyarrow.parquet as pq

D = sys.argv[1]
COLS = [
    'game_id', 'season', 'season_type', 'week', 'posteam', 'defteam', 'home_team', 'away_team',
    'play_type', 'qb_dropback', 'rush', 'pass', 'epa', 'success', 'wp', 'down', 'ydstogo', 'yardline_100',
    'qb_kneel', 'qb_spike', 'qb_scramble', 'sack', 'fumble', 'fumble_lost', 'interception',
    'special_teams_play', 'special', 'half_seconds_remaining', 'game_seconds_remaining', 'qtr',
    'score_differential', 'passer_player_id', 'passer_player_name', 'rusher_player_id', 'cpoe',
    'yards_gained', 'penalty', 'penalty_team', 'xpass', 'pass_oe', 'air_yards', 'field_goal_result',
    'kick_distance', 'third_down_converted', 'third_down_failed', 'fourth_down_converted',
    'fourth_down_failed', 'goal_to_go', 'fixed_drive', 'fixed_drive_result', 'touchdown',
    'return_touchdown', 'td_team', 'qb_hit', 'pass_attempt', 'rush_attempt', 'complete_pass',
    'extra_point_result', 'two_point_conv_result', 'play_id',
]
frames = []
YEARS = [int(y) for y in os.environ.get('NFL_YEARS', '2021,2022,2023,2024,2025,2026').split(',')]
for y in YEARS:
    path = f'{D}/pbp_{y}.parquet'
    have = set(pq.read_schema(path).names)
    use = [c for c in COLS if c in have]
    df = pd.read_parquet(path, columns=use)
    if 'season' not in df:
        df['season'] = y
    frames.append(df)
    print(y, len(df), 'missing:', [c for c in COLS if c not in have])
plays = pd.concat(frames, ignore_index=True)
plays.to_parquet(f'{D}/plays.parquet', index=False)

g = pd.read_csv(f'{D}/games.csv')
g = g[(g.season >= min(YEARS)) & (g.season <= max(YEARS))].copy()
keep = ['game_id', 'season', 'game_type', 'week', 'gameday', 'away_team', 'away_score', 'home_team',
        'home_score', 'location', 'result', 'spread_line', 'total_line', 'div_game', 'away_rest',
        'home_rest', 'away_qb_id', 'home_qb_id', 'away_qb_name', 'home_qb_name', 'away_coach',
        'home_coach', 'overtime']
g = g[keep]
g.to_parquet(f'{D}/games.parquet', index=False)
print('games', len(g), g.groupby(['season', 'game_type']).size().to_dict())
print('2026 played', g[(g.season == 2026) & g.result.notna()].week.max())
