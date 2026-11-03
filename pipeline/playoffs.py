"""Pre-kickoff picks for playoff games, logged like the regular season's.

Usage: python3 -I playoffs.py <pipeline_dir> <season> <log_dir> [--window MINUTES] [--now 2027-01-16T15:10]
Run right after `refresh.py <pipeline_dir> <season> --extra`. Finds playoff games (wild card through
Super Bowl) with both teams set that kick off within the window (default 150 minutes, Eastern), picks
them with the live model (full regular-season ratings plus the quarterback-change, injury and rest
add-ons; the Super Bowl is a neutral site) and adds log/<season>/playoffs_<time>.json (append-only).
Prints {"games": 0} and changes nothing when no playoff game is in the window.
"""
import sys, os, json
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pandas as pd
from scipy.stats import norm
import model as M
from live_addons import Addons
from log_week import model_version, write_once

ET = ZoneInfo('America/New_York')
args = sys.argv[1:]
work, season, log_dir = os.path.abspath(args[0]), int(args[1]), os.path.abspath(args[2])
window = int(args[args.index('--window') + 1]) if '--window' in args else 150
now = datetime.fromisoformat(args[args.index('--now') + 1]).replace(tzinfo=ET) if '--now' in args else datetime.now(ET)
data = os.environ.get('NFL_DATA', f'{work}/data')

sched = pd.read_csv(f'{data}/games.csv')
sched = sched[sched.season == season]
post = sched[(sched.game_type != 'REG') & (sched.result.isna() | ('--test-played' in args)) & sched.gametime.notna()]
soon = []
for _, g in post.iterrows():
    if not (isinstance(g.home_team, str) and isinstance(g.away_team, str)) or 'TBD' in (g.home_team, g.away_team):
        continue
    kick = datetime.fromisoformat(f'{g.gameday}T{g.gametime}').replace(tzinfo=ET)
    if now <= kick <= now + timedelta(minutes=window):
        soon.append(g)
if not soon:
    print(json.dumps(dict(games=0, checked_at=now.strftime('%Y-%m-%d %H:%M ET'), window_minutes=window)))
    sys.exit(0)

cfg = json.load(open(f'{work}/out/method.json'))
r = M.rating(cfg, season, 18)
AD = Addons(work, M.load()[0], data, season)
names = {}
for side in ('home', 'away'):
    names.update(dict(zip(sched[f'{side}_qb_id'], sched[f'{side}_qb_name'])))
num = lambda v: None if pd.isna(v) else float(v)
stamp_utc = datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')
version = model_version(work)
rows = []
for g in soon:
    w = int(g.week)
    neutral = g.location == 'Neutral'
    qh, uh = AD.qb_delta(sched, g.home_team, g.home_qb_id, w)
    qa, ua = AD.qb_delta(sched, g.away_team, g.away_qb_id, w)
    oh, nh = AD.players_out(g.home_team, w)
    oa, na = AD.players_out(g.away_team, w)
    rest = float(g.home_rest - g.away_rest) if pd.notna(g.home_rest) and pd.notna(g.away_rest) else 0.0
    m = AD.margin(r[g.home_team] - r[g.away_team], 17, neutral, qh - qa, (oa or 0.0) - (oh or 0.0), rest)
    p = float(norm.cdf(m / AD.sd))
    notes = [f"{t}: {names.get(q, q)} starting instead of {names.get(u, u)}" for t, q, u, d in
             ((g.home_team, g.home_qb_id, uh, qh), (g.away_team, g.away_qb_id, ua, qa)) if d]
    notes += [f"{t} without " + ', '.join(n) for t, n in ((g.home_team, nh), (g.away_team, na)) if n]
    rows.append(dict(
        season=season, week=w, game_type=g.game_type, game_id=g.game_id, gameday=g.gameday, gametime=g.gametime,
        home=g.home_team, away=g.away_team, neutral=bool(neutral), home_rest=num(g.home_rest), away_rest=num(g.away_rest),
        ratings_through_week=18, home_rating=round(float(r[g.home_team]), 2), away_rating=round(float(r[g.away_team]), 2),
        model_margin_home=round(m, 1), model_home_win_prob=round(p, 3), model_pick=g.home_team if m >= 0 else g.away_team,
        qb_change=round(qh - qa, 2), players_out=round((oa or 0.0) - (oh or 0.0), 2), rest_diff=rest,
        injury_report=bool(oh is not None or oa is not None), notes=' | '.join(notes) or None,
        vegas_spread_home=num(g.spread_line), vegas_total=num(g.total_line),
        vegas_home_moneyline=num(g.home_moneyline), vegas_away_moneyline=num(g.away_moneyline), line_kind='at_log',
        home_qb=None if pd.isna(g.home_qb_name) else g.home_qb_name, away_qb=None if pd.isna(g.away_qb_name) else g.away_qb_name,
        qb_kind='expected', home_usual_qb=names.get(uh), away_usual_qb=names.get(ua),
        home_qb_change=bool(qh), away_qb_change=bool(qa),
        stage='pregame', logged_live=True, logged_at=stamp_utc, model_version=version))
path = f"{log_dir}/{season}/playoffs_{now.strftime('%Y%m%dT%H%M')}.json"
write_once(path, rows)
print(json.dumps(dict(games=len(rows), log_file=os.path.relpath(path, os.path.dirname(log_dir)),
                      picks=[dict(game=f"{x['away']} at {x['home']} ({x['game_type']})",
                                  pick=f"{x['model_pick']} by {abs(x['model_margin_home']):.1f}",
                                  vegas=x['vegas_spread_home'], notes=x['notes']) for x in rows]), indent=1))
