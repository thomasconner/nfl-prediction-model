"""Permanent weekly log: what the model said before each game, and what it published.

Usage: python3 -I log_week.py <pipeline_dir> <season> <log_dir> [--backfill]
Run after weekly.py. Writes, only if the file does not exist yet (the log is append-only):
  <log_dir>/<season>/week_NN_picks.json        picks for the coming week (NN = next_week), made
                                               on Tuesday, with the line as of logging time
  <log_dir>/<season>/after_week_NN_cards.json  the report cards as published after week NN
--backfill also writes reconstructed picks for weeks already played that have no log file:
  same pre-game ratings (data through the week before), but the line is the closing line and the
  quarterbacks are the actual starters, and each row says logged_live = false.
pregame.py uses pick_rows() to add a week_NN_pregame_<time>.json file before each kickoff slot.
"""
import sys, os, json, hashlib
from datetime import datetime, timezone
import pandas as pd

VERSION_FILES = ('model.py', 'analysis.py', 'reportcard.py', 'weekly.py', 'flags.py', 'live_addons.py',
                 'out/method.json', 'out/ledger_params.json', 'out/prior_params.json', 'out/addons.json')


def model_version(work):
    h = hashlib.sha256()
    for f in VERSION_FILES:
        if os.path.exists(f'{work}/{f}'):
            h.update(open(f'{work}/{f}', 'rb').read())
    return h.hexdigest()[:12]


def num(v):
    return None if pd.isna(v) else float(v)


class Context:
    def __init__(self, work, season):
        self.work, self.season = work, season
        data = os.environ.get('NFL_DATA', f'{work}/data')
        od = f'{work}/out/dash_{season}'
        load = lambda n: json.load(open(f'{od}/{n}.json'))
        self.meta, self.games, self.weekly, self.cards = load('meta')[0], load('games'), load('weekly'), load('cards')
        sched = pd.read_csv(f'{data}/games.csv')
        self.sched = sched[(sched.season == season) & (sched.game_type == 'REG')].set_index('game_id')
        self.strength = {(r['week'], r['team']): r['strength'] for r in self.weekly}
        played = self.sched[self.sched.result.notna()]
        self.starts = pd.concat([played[['week', 'home_team', 'home_qb_name']].set_axis(['week', 'team', 'qb'], axis=1),
                                 played[['week', 'away_team', 'away_qb_name']].set_axis(['week', 'team', 'qb'], axis=1)])
        self.version = model_version(work)
        self.now = datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')

    def usual_qb(self, team, before_week):
        s = self.starts[(self.starts.team == team) & (self.starts.week < before_week)].qb.dropna()
        return s.value_counts().index[0] if len(s) else None

    def pick_rows(self, game_ids, live, stage):
        W = self.meta['through_week']
        rows = []
        for g in [x for x in self.games if x['game_id'] in game_ids]:
            s = self.sched.loc[g['game_id']]
            week = g['week']
            rw = W if live else week - 1
            hq = None if pd.isna(s.home_qb_name) else s.home_qb_name
            aq = None if pd.isna(s.away_qb_name) else s.away_qb_name
            hu, au = self.usual_qb(g['home'], week), self.usual_qb(g['away'], week)
            rows.append(dict(
                season=self.season, week=week, game_id=g['game_id'], gameday=g['gameday'],
                gametime=None if pd.isna(s.gametime) else s.gametime,
                home=g['home'], away=g['away'], neutral=bool(s.location == 'Neutral'), div_game=bool(s.div_game),
                roof=None if pd.isna(s.roof) else s.roof, home_rest=num(s.home_rest), away_rest=num(s.away_rest),
                ratings_through_week=rw, home_rating=self.strength.get((rw, g['home'])), away_rating=self.strength.get((rw, g['away'])),
                model_margin_home=g['model_margin_home'],
                model_home_win_prob=round(g['model_pick_prob'] if g['model_pick'] == g['home'] else 1 - g['model_pick_prob'], 3),
                model_pick=g['model_pick'],
                qb_change=g.get('qb_change'), players_out=g.get('players_out'), rest_diff=g.get('rest_diff'),
                injury_report=g.get('injury_report'), notes=g.get('notes'),
                vegas_spread_home=num(s.spread_line), vegas_total=num(s.total_line),
                vegas_home_moneyline=num(s.home_moneyline), vegas_away_moneyline=num(s.away_moneyline),
                line_kind='at_log' if live else 'closing',
                home_qb=hq, away_qb=aq, qb_kind='expected' if live else 'actual',
                home_usual_qb=hu, away_usual_qb=au,
                home_qb_change=bool(hq and hu and hq != hu), away_qb_change=bool(aq and au and aq != au),
                stage=stage, logged_live=live, logged_at=self.now, model_version=self.version))
        return rows


def write_once(path, payload):
    if os.path.exists(path):
        return False
    os.makedirs(os.path.dirname(path), exist_ok=True)
    json.dump(payload, open(path, 'w'), indent=1)
    return True


if __name__ == '__main__':
    work, season, log_dir = os.path.abspath(sys.argv[1]), int(sys.argv[2]), os.path.abspath(sys.argv[3])
    ctx = Context(work, season)
    dest = f'{log_dir}/{season}'
    W, NEXT = ctx.meta['through_week'], ctx.meta['next_week']
    written = []
    if NEXT:
        ids = {g['game_id'] for g in ctx.games if g['week'] == NEXT}
        p = f'{dest}/week_{NEXT:02d}_picks.json'
        if write_once(p, ctx.pick_rows(ids, True, 'tuesday')):
            written.append(p)
    p = f'{dest}/after_week_{W:02d}_cards.json'
    if write_once(p, dict(season=season, through_week=W, logged_at=ctx.now, model_version=ctx.version, cards=ctx.cards)):
        written.append(p)
    if '--backfill' in sys.argv:
        for w in range(1, W + 1):
            ids = {g['game_id'] for g in ctx.games if g['week'] == w}
            p = f'{dest}/week_{w:02d}_picks.json'
            if write_once(p, ctx.pick_rows(ids, False, 'rebuilt')):
                written.append(p)
    print(json.dumps(dict(model_version=ctx.version, through_week=W, next_week=NEXT,
                          written=[os.path.relpath(x, log_dir) for x in written]), indent=1))
