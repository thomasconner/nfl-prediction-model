"""Health check: did every game of a finished week get its picks logged before kickoff?

Usage: python3 -I check_log.py <pipeline_dir> <season> <log_dir> [week]
Week defaults to the latest finished week in out/dash_<season>/meta.json. Prints JSON with the games
that have no Tuesday pick and the games that have no pre-kickoff pick (a pre-kickoff run failed or
didn't happen).
"""
import sys, os, json, glob

work, season, log_dir = os.path.abspath(sys.argv[1]), int(sys.argv[2]), os.path.abspath(sys.argv[3])
meta = json.load(open(f'{work}/out/dash_{season}/meta.json'))[0]
week = int(sys.argv[4]) if len(sys.argv) > 4 else meta['through_week']
games = [g for g in json.load(open(f'{work}/out/dash_{season}/games.json')) if g['week'] == week]
tuesday, pregame = set(), set()
for f in glob.glob(f'{log_dir}/{season}/week_{week:02d}_picks.json'):
    tuesday |= {e['game_id'] for e in json.load(open(f)) if e.get('logged_live')}
for f in glob.glob(f'{log_dir}/{season}/week_{week:02d}_pregame_*.json'):
    pregame |= {e['game_id'] for e in json.load(open(f))}
print(json.dumps(dict(
    season=season, week=week, games=len(games),
    with_tuesday_pick=sum(g['game_id'] in tuesday for g in games),
    with_pregame_pick=sum(g['game_id'] in pregame for g in games),
    missing_tuesday=[g['matchup'] for g in games if g['game_id'] not in tuesday],
    missing_pregame=[f"{g['matchup']} ({g['gameday']})" for g in games if g['game_id'] not in pregame]), indent=1))
