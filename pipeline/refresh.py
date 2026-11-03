"""Weekly refresh for the NFL Report Cards dashboard.

Usage: python3 -I refresh.py <work_dir> <season>
  <work_dir> holds this bundle's files (model.py, weekly.py, out/*.json ...).
Downloads nflverse play-by-play (season-5 .. season) and the schedule into <work_dir>/data,
rebuilds the compact tables, runs weekly.py, sanity-checks the output and prints a JSON summary.
"""
import sys, os, json, subprocess, urllib.request
import pandas as pd

work, season = os.path.abspath(sys.argv[1]), int(sys.argv[2])
data = os.path.join(work, 'data')
os.makedirs(data, exist_ok=True)
PBP = 'https://github.com/nflverse/nflverse-data/releases/download/pbp/play_by_play_{y}.parquet'
GAMES = 'https://raw.githubusercontent.com/nflverse/nfldata/master/data/games.csv'


def fetch(url, dest):
    for attempt in range(3):
        try:
            urllib.request.urlretrieve(url, dest)
            return
        except Exception as e:  # network hiccup: retry, then give up loudly
            err = e
    raise SystemExit(f'download failed: {url}: {err}')


years = list(range(season - 5, season + 1))
for y in years:
    dest = f'{data}/pbp_{y}.parquet'
    if y == season or not os.path.exists(dest):
        fetch(PBP.format(y=y), dest)
fetch(GAMES, f'{data}/games.csv')
env = dict(os.environ, NFL_DATA=data, NFL_YEARS=','.join(map(str, years)))
subprocess.run([sys.executable, '-I', f'{work}/build_data.py', data], check=True, env=env, cwd=work, stdout=subprocess.DEVNULL)
subprocess.run([sys.executable, '-I', f'{work}/weekly.py', f'{work}/out', str(season)], check=True, env=env, cwd=work, stdout=subprocess.DEVNULL)

od = f'{work}/out/dash_{season}'
load = lambda n: json.load(open(f'{od}/{n}.json'))
meta, weekly, games, tg, cards = load('meta')[0], load('weekly'), load('games'), load('team_games'), load('cards')
W = meta['through_week']
problems = []
if len(cards) != 32: problems.append(f'cards has {len(cards)} rows')
if len(weekly) != 32 * (W + 1): problems.append(f'weekly has {len(weekly)} rows, expected {32 * (W + 1)}')
if len(tg) != 2 * len(games): problems.append('team_games is not two rows per game')
for c in cards:
    if c.get('projected_wins') is None or c.get('verdict') is None: problems.append(f"card {c['team']} is missing fields")
fin = [g for g in games if g['status'] == 'final']
mr = sum(g['model_correct'] for g in fin if g['model_correct'] is not None)
vg = [g for g in fin if g['vegas_correct'] is not None]
vr = sum(g['vegas_correct'] for g in vg)
cd = pd.DataFrame(cards)
movers = cd.reindex(cd.proj_change_week.abs().sort_values(ascending=False).index).head(3)
summary = dict(
    season=season, through_week=W, next_week=meta['next_week'], problems=problems,
    files={n: f'{od}/{n}.json' for n in ('meta', 'weekly', 'games', 'team_games', 'cards')},
    model_record=f'{mr:g} of {len(fin)}', vegas_record=f'{vr:g} of {len(vg)}',
    top3=[f"{r.team} ({r.verdict})" for r in cd.sort_values('strength_rank').head(3).itertuples()],
    biggest_moves=[f"{r.team} {r.proj_change_week:+.1f} projected wins (now {r.projected_wins:.1f})" for r in movers.itertuples()],
    fluff=[r.team for r in cd[cd.verdict == 'Fluff'].itertuples()],
    jax=[f"{r.record}, {r.verdict}, strength #{r.strength_rank}, projected {r.projected_wins:.1f} wins" for r in cd[cd.team == 'JAX'].itertuples()][0],
)
print(json.dumps(summary, indent=1))
