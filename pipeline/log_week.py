"""Permanent weekly log: what the model said before each game, and what it published.

Usage: python3 -I log_week.py <pipeline_dir> <season> <log_dir> [--backfill]
Run after weekly.py. Writes, only if the file does not exist yet (the log is append-only):
  <log_dir>/<season>/week_NN_picks.json        picks for the coming week (NN = next_week), made
                                               before kickoff, with the line as of logging time
  <log_dir>/<season>/after_week_NN_cards.json  the report cards as published after week NN
--backfill also writes reconstructed picks for weeks already played that have no log file:
  same pre-game ratings (data through the week before), but the line is the closing line and the
  quarterbacks are the actual starters, and each row says logged_live = false.
"""
import sys, os, json, hashlib
from datetime import datetime, timezone
import pandas as pd

work, season, log_dir = os.path.abspath(sys.argv[1]), int(sys.argv[2]), os.path.abspath(sys.argv[3])
backfill = '--backfill' in sys.argv
data = os.environ.get('NFL_DATA', f'{work}/data')
od = f'{work}/out/dash_{season}'
dest = f'{log_dir}/{season}'
os.makedirs(dest, exist_ok=True)
now = datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')

h = hashlib.sha256()
for f in ('model.py', 'analysis.py', 'reportcard.py', 'weekly.py', 'flags.py', 'out/method.json',
          'out/ledger_params.json', 'out/prior_params.json'):
    h.update(open(f'{work}/{f}', 'rb').read())
VERSION = h.hexdigest()[:12]

load = lambda n: json.load(open(f'{od}/{n}.json'))
meta, games, weekly, cards = load('meta')[0], load('games'), load('weekly'), load('cards')
W, NEXT = meta['through_week'], meta['next_week']
sched = pd.read_csv(f'{data}/games.csv')
sched = sched[(sched.season == season) & (sched.game_type == 'REG')].set_index('game_id')
strength = {(r['week'], r['team']): r['strength'] for r in weekly}

# each team's usual starter so far (most starts in finished games), to flag quarterback changes
played = sched[sched.result.notna()]
starts = pd.concat([played[['week', 'home_team', 'home_qb_name']].set_axis(['week', 'team', 'qb'], axis=1),
                    played[['week', 'away_team', 'away_qb_name']].set_axis(['week', 'team', 'qb'], axis=1)])


def usual_qb(team, before_week):
    s = starts[(starts.team == team) & (starts.week < before_week)].qb.dropna()
    return s.value_counts().index[0] if len(s) else None


def num(v):
    return None if pd.isna(v) else float(v)


def pick_rows(week, live):
    rows = []
    for g in [x for x in games if x['week'] == week]:
        s = sched.loc[g['game_id']]
        rw = week - 1 if not live else W
        hq, aq = (None if pd.isna(s.home_qb_name) else s.home_qb_name), (None if pd.isna(s.away_qb_name) else s.away_qb_name)
        hu, au = usual_qb(g['home'], week), usual_qb(g['away'], week)
        rows.append(dict(
            season=season, week=week, game_id=g['game_id'], gameday=g['gameday'], gametime=None if pd.isna(s.gametime) else s.gametime,
            home=g['home'], away=g['away'], neutral=bool(s.location == 'Neutral'), div_game=bool(s.div_game),
            roof=None if pd.isna(s.roof) else s.roof, home_rest=num(s.home_rest), away_rest=num(s.away_rest),
            ratings_through_week=rw, home_rating=strength.get((rw, g['home'])), away_rating=strength.get((rw, g['away'])),
            model_margin_home=g['model_margin_home'],
            model_home_win_prob=round(g['model_pick_prob'] if g['model_pick'] == g['home'] else 1 - g['model_pick_prob'], 3),
            model_pick=g['model_pick'],
            vegas_spread_home=num(s.spread_line), vegas_total=num(s.total_line),
            vegas_home_moneyline=num(s.home_moneyline), vegas_away_moneyline=num(s.away_moneyline),
            line_kind='at_log' if live else 'closing',
            home_qb=hq, away_qb=aq, qb_kind='expected' if live else 'actual',
            home_usual_qb=hu, away_usual_qb=au,
            home_qb_change=bool(hq and hu and hq != hu), away_qb_change=bool(aq and au and aq != au),
            logged_live=live, logged_at=now, model_version=VERSION))
    return rows


def write_once(path, payload):
    if os.path.exists(path):
        return False
    json.dump(payload, open(path, 'w'), indent=1)
    return True


written = []
if NEXT:
    p = f'{dest}/week_{NEXT:02d}_picks.json'
    if write_once(p, pick_rows(NEXT, True)):
        written.append(p)
p = f'{dest}/after_week_{W:02d}_cards.json'
if write_once(p, dict(season=season, through_week=W, logged_at=now, model_version=VERSION, cards=cards)):
    written.append(p)
if backfill:
    for w in range(1, W + 1):
        p = f'{dest}/week_{w:02d}_picks.json'
        if write_once(p, pick_rows(w, False)):
            written.append(p)
print(json.dumps(dict(model_version=VERSION, through_week=W, next_week=NEXT, written=[os.path.relpath(x, log_dir) for x in written]), indent=1))
