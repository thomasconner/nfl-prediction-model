"""Pre-kickoff update: re-pick the games starting soon with the latest starters and injury report.

Usage: python3 -I pregame.py <pipeline_dir> <season> <log_dir> [--window MINUTES] [--now 2026-11-08T12:10]
Run right after `refresh.py <pipeline_dir> <season> --extra`, which downloads the latest schedule,
expected starters and injury reports and rebuilds out/dash_<season>/.
Finds games kicking off within the window (default 150 minutes, Eastern time). If there are none it
prints {"games": 0} and changes nothing. Otherwise it adds log/<season>/week_NN_pregame_<time>.json
(append-only), reruns weekly.py so those picks become the official ones, and rebuilds the site.
"""
import sys, os, json, subprocess
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from log_week import Context, write_once

ET = ZoneInfo('America/New_York')
args = [a for a in sys.argv[1:]]
work, season, log_dir = os.path.abspath(args[0]), int(args[1]), os.path.abspath(args[2])
window = int(args[args.index('--window') + 1]) if '--window' in args else 150
now = datetime.fromisoformat(args[args.index('--now') + 1]).replace(tzinfo=ET) if '--now' in args else datetime.now(ET)

ctx = Context(work, season)
soon = []
for g in ctx.games:
    if g['status'] != 'upcoming':
        continue
    t = ctx.sched.loc[g['game_id']].gametime
    if not isinstance(t, str):
        continue
    kick = datetime.fromisoformat(f"{g['gameday']}T{t}").replace(tzinfo=ET)
    if now <= kick <= now + timedelta(minutes=window):
        soon.append(g)
if not soon:
    print(json.dumps(dict(games=0, checked_at=now.strftime('%Y-%m-%d %H:%M ET'), window_minutes=window)))
    sys.exit(0)

week = soon[0]['week']
stamp = now.strftime('%Y%m%dT%H%M')
path = f'{log_dir}/{season}/week_{week:02d}_pregame_{stamp}.json'
rows = ctx.pick_rows({g['game_id'] for g in soon}, True, 'pregame')
write_once(path, rows)
env = dict(os.environ, NFL_LOG=log_dir, NFL_DATA=os.path.abspath(os.environ.get('NFL_DATA', f'{work}/data')))
subprocess.run([sys.executable, '-I', f'{work}/weekly.py', f'{work}/out', str(season)], check=True, env=env, cwd=work, stdout=subprocess.DEVNULL)
subprocess.run([sys.executable, '-I', f'{work}/build_site.py', work, str(season), os.path.dirname(work)], check=True, cwd=work, stdout=subprocess.DEVNULL)
print(json.dumps(dict(
    games=len(rows), log_file=os.path.relpath(path, os.path.dirname(log_dir)),
    picks=[dict(game=f"{r['away']} at {r['home']}", pick=f"{r['model_pick']} by {abs(r['model_margin_home']):.1f}",
                vegas=r['vegas_spread_home'], notes=r['notes']) for r in rows]), indent=1))
