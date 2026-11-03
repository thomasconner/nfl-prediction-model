"""Weekly dashboard data: season-to-date time series, game picks, and current report cards.

Usage: python3 -I weekly.py <out_dir> <season> [through_week]
  through_week defaults to the last regular-season week whose games all have results.
Writes <out_dir>/dash_<season>/{meta,weekly,games,team_games,cards,accuracy}.json

Projection rating at week w = (n*r_w + k_w*rho*r_prev)/(n + k_w), k_w = k0*max(0, 1 - w/8)
(out/prior_params.json, fit by fit_prior.py). From Week 8 on it equals the report-card rating.
Each played game is predicted from ratings frozen after the week before it; unplayed games from
the latest ratings.
"""
import sys, os, json, subprocess
from datetime import datetime, timezone
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd
from scipy.stats import norm
import model as M

out_dir, SEASON = sys.argv[1], int(sys.argv[2])
HERE = os.path.dirname(os.path.abspath(__file__))
cfg = json.load(open(f'{out_dir}/method.json'))
K = json.load(open(f'{out_dir}/ledger_params.json'))['logistic_k']
PP = json.load(open(f'{out_dir}/prior_params.json'))
BT = pd.read_csv(f'{out_dir}/proj_backtest.csv').set_index('week')
RHO, K0, TAPER = PP['rho'], PP['k0'], PP['taper_weeks']
NICK = {'ARI': 'Cardinals', 'ATL': 'Falcons', 'BAL': 'Ravens', 'BUF': 'Bills', 'CAR': 'Panthers', 'CHI': 'Bears',
        'CIN': 'Bengals', 'CLE': 'Browns', 'DAL': 'Cowboys', 'DEN': 'Broncos', 'DET': 'Lions', 'GB': 'Packers',
        'HOU': 'Texans', 'IND': 'Colts', 'JAX': 'Jaguars', 'KC': 'Chiefs', 'LA': 'Rams', 'LAC': 'Chargers',
        'LV': 'Raiders', 'MIA': 'Dolphins', 'MIN': 'Vikings', 'NE': 'Patriots', 'NO': 'Saints', 'NYG': 'Giants',
        'NYJ': 'Jets', 'PHI': 'Eagles', 'PIT': 'Steelers', 'SEA': 'Seahawks', 'SF': '49ers', 'TB': 'Buccaneers',
        'TEN': 'Titans', 'WAS': 'Commanders'}

_, g_all = M.load()
gs = pd.read_parquet(f'{M.DATA}/games.parquet')
gs = gs[(gs.season == SEASON) & (gs.game_type == 'REG')].copy()
done_weeks = gs.groupby('week').result.apply(lambda x: x.notna().all())
W = int(sys.argv[3]) if len(sys.argv) > 3 else int(max([w for w, ok in done_weeks.items() if ok] or [0]))
NEXT = W + 1 if W < gs.week.max() else None
tg = M.team_games(cfg)
tg = tg[tg.season == SEASON]
prev = M.rating(cfg, SEASON - 1, 18)
teams = sorted(NICK)


def params(w):
    p = PP['by_week'][str(min(w, 17))]
    return p['b'], p['h'], p['sd']


def proj_rating(w):
    pr = (RHO * prev).reindex(teams).fillna(0)
    if w == 0:
        return pr, None
    raw = M.rating(cfg, SEASON, w).reindex(teams)
    n = tg[tg.week <= w].groupby('team').size().reindex(teams).fillna(0)
    k = K0 * max(0.0, 1 - w / TAPER)
    r = (n * raw.fillna(0) + k * pr) / (n + k) if k > 0 else raw
    return r, M.rating(cfg, SEASON, w, with_parts=True).reindex(teams)


def verdict(sr, wpct, luck):
    if sr <= 6: return 'Contender'
    if wpct >= 0.6 and (sr >= 17 or (sr >= 13 and luck >= 1.0)): return 'Fluff'
    if luck >= 1.0 and sr <= 16: return 'Good but lucky'
    if luck <= -1.0 and sr <= 16: return 'Better than their record'
    if sr <= 12: return 'Solid'
    if sr <= 22: return 'Middle of the pack'
    return 'Bad'


def win_prob(r, home, away, neutral, w):
    b, h, sd = params(w)
    m = b * (r[home] - r[away]) + (0.0 if neutral else h)
    return m, norm.cdf(m / sd)


ratings, weekly = {}, []
for w in range(0, W + 1):
    r, parts = proj_rating(w)
    ratings[w] = r
    d = tg[tg.week <= w]
    wins = d.groupby('team').win.apply(lambda x: (x == 1).sum()).reindex(teams).fillna(0).astype(int)
    losses = d.groupby('team').win.apply(lambda x: (x == 0).sum()).reindex(teams).fillna(0).astype(int)
    ties = d.groupby('team').win.apply(lambda x: (x == 0.5).sum()).reindex(teams).fillna(0).astype(int)
    games = wins + losses + ties
    wpct = ((wins + 0.5 * ties) / games.replace(0, np.nan))
    srank = r.rank(ascending=False, method='min').astype(int)
    rrank = wpct.rank(ascending=False, method='min')
    rem = gs[gs.week > w]
    exp_rem = pd.Series(0.0, index=teams)
    for _, gm in rem.iterrows():
        _, p = win_prob(r, gm.home_team, gm.away_team, gm.location == 'Neutral', w)
        exp_rem[gm.home_team] += p
        exp_rem[gm.away_team] += 1 - p
    proj = wins + 0.5 * ties + exp_rem
    for t in teams:
        n = int(games[t])
        exp_w = n / (1 + np.exp(-K * r[t])) if n else 0.0
        luck = round(float(wins[t] + 0.5 * ties[t] - exp_w), 1) if n else 0.0
        if w == 0:
            v = 'Preseason'
        elif w < 4:
            v = 'Too early'
        else:
            v = verdict(int(srank[t]), float(wpct[t]), luck)
        weekly.append(dict(
            team=t, week=w, wins=int(wins[t]), losses=int(losses[t]), ties=int(ties[t]),
            record=f'{wins[t]}-{losses[t]}' + (f'-{ties[t]}' if ties[t] else ''),
            strength=round(float(r[t]), 2), strength_rank=int(srank[t]),
            record_rank=None if n == 0 else int(rrank[t]),
            luck_wins=luck, verdict=v, proj_wins=round(float(proj[t]), 2),
            typical_miss=round(float(BT.loc[w, 'mae_model']), 2)))
WK = pd.DataFrame(weekly)

# ---------- games ----------
grows, trows = [], []
for _, gm in gs.sort_values(['week', 'gameday', 'game_id']).iterrows():
    played = pd.notna(gm.result)
    w_r = int(gm.week) - 1 if played else W
    w_r = min(w_r, W)
    m, p = win_prob(ratings[w_r], gm.home_team, gm.away_team, gm.location == 'Neutral', w_r)
    pick = gm.home_team if m >= 0 else gm.away_team
    spread = None if pd.isna(gm.spread_line) else float(gm.spread_line)
    vpick = None if spread is None or spread == 0 else (gm.home_team if spread > 0 else gm.away_team)
    res = float(gm.result) if played else None
    winner = None if not played else ('TIE' if res == 0 else (gm.home_team if res > 0 else gm.away_team))
    mc = None if not played else (0.5 if winner == 'TIE' else float(pick == winner))
    vc = None if (not played or vpick is None) else (0.5 if winner == 'TIE' else float(vpick == winner))
    grows.append(dict(
        game_id=gm.game_id, week=int(gm.week), gameday=gm.gameday, away=gm.away_team, home=gm.home_team,
        matchup=f'{gm.away_team} at {gm.home_team}' if gm.location != 'Neutral' else f'{gm.away_team} vs {gm.home_team}',
        status='final' if played else 'upcoming',
        model_margin_home=round(float(m), 1), model_pick=pick, model_pick_prob=round(float(max(p, 1 - p)), 3),
        model_says=f'{pick} by {abs(m):.1f}',
        vegas_spread_home=spread, vegas_pick=vpick,
        vegas_says=None if spread is None else ('Pick\'em' if spread == 0 else f'{vpick} by {abs(spread):g}'),
        away_score=None if not played else int(gm.away_score), home_score=None if not played else int(gm.home_score),
        result_home=res, winner=winner,
        final=None if not played else f'{gm.away_team} {int(gm.away_score)}, {gm.home_team} {int(gm.home_score)}',
        model_correct=mc, vegas_correct=vc,
        model_miss=None if not played else round(abs(res - m), 1),
        vegas_miss=None if (not played or spread is None) else round(abs(res - spread), 1)))
    for t, o, sgn, ha in ((gm.home_team, gm.away_team, 1, 'vs'), (gm.away_team, gm.home_team, -1, 'at')):
        if gm.location == 'Neutral':
            ha = 'vs'
        tm = sgn * m
        ts = None if not played else int(gm.home_score if sgn == 1 else gm.away_score)
        os_ = None if not played else int(gm.away_score if sgn == 1 else gm.home_score)
        if played:
            wl = 'T' if ts == os_ else ('W' if ts > os_ else 'L')
            result = f'{wl} {ts}-{os_}'
        else:
            result = 'Upcoming'
        trows.append(dict(
            team=t, game_id=gm.game_id, week=int(gm.week), opp=f'{ha} {o}', status='final' if played else 'upcoming',
            model_says=f'{pick} by {abs(m):.1f}', model_margin=round(float(tm), 1),
            model_win_prob=round(float(p if sgn == 1 else 1 - p), 3),
            vegas_says=None if spread is None else ('Pick\'em' if spread == 0 else f'{vpick} by {abs(spread):g}'),
            result=result, margin=None if not played else ts - os_,
            model_right=None if mc is None else ('Yes' if mc == 1 else ('Push' if mc == 0.5 else 'No')),
            vegas_right=None if vc is None else ('Yes' if vc == 1 else ('Push' if vc == 0.5 else 'No'))))
G, TG = pd.DataFrame(grows), pd.DataFrame(trows)

# ---------- current cards (report card logic), aligned to this file's rating ----------
subprocess.run([sys.executable, '-I', f'{HERE}/reportcard.py', out_dir, str(SEASON), str(W)], check=True,
               stdout=subprocess.DEVNULL, cwd=HERE)
cards = pd.DataFrame(json.load(open(f'{out_dir}/cards_{SEASON}_wk{W}.json')))
cur = WK[WK.week == W].set_index('team')
last = WK[WK.week == max(W - 1, 0)].set_index('team')
pre = WK[WK.week == 0].set_index('team')
cards = cards.set_index('team')
if W < TAPER:
    # early season: card strength/verdict come from the projection rating (includes the fading preseason prior)
    cards['strength'] = cur.strength.round(1)
    cards['strength_rank'] = cur.strength_rank
    cards['verdict'] = cur.verdict
    cards['luck_wins'] = cur.luck_wins
mismatch = (cards.verdict != cur.verdict).sum()
cards['projected_wins'] = cur.proj_wins.round(1)
cards['nickname'] = [NICK[t] for t in cards.index]
cards['proj_change_week'] = (cur.proj_wins - last.proj_wins).round(1)
cards['proj_change_preseason'] = (cur.proj_wins - pre.proj_wins).round(1)
cards['preseason_proj'] = pre.proj_wins.round(1)
cards['preseason_rank'] = pre.strength_rank
cards['rank_change_week'] = (last.strength_rank - cur.strength_rank)
cards['rank_change_preseason'] = (pre.strength_rank - cur.strength_rank)
cards['typical_miss'] = float(BT.loc[W, 'mae_model'])
cards['flags'] = [' | '.join(f) for f in cards['flags']]
cards['next_game'] = None
if NEXT:
    nx = TG[TG.week == NEXT].set_index('team')
    for t in cards.index:
        if t in nx.index:
            cards.loc[t, 'next_game'] = f"Week {NEXT}: {nx.loc[t, 'opp']} · model: {nx.loc[t, 'model_says']}"
        else:
            cards.loc[t, 'next_game'] = f'Week {NEXT}: bye'
cards = cards.reset_index()

model_n = G.model_correct.notna().sum()
meta = [dict(season=SEASON, through_week=W, next_week=NEXT, games_played=int(model_n),
             updated=datetime.now(timezone.utc).strftime('%Y-%m-%d'),
             data_through=str(gs[gs.week == W].gameday.max()))]
acc = BT.reset_index()[['week', 'mae_model', 'mae_pace']]

od = f'{out_dir}/dash_{SEASON}'
os.makedirs(od, exist_ok=True)


def dump(name, df):
    rows = json.loads(df.to_json(orient='records'))
    json.dump(rows, open(f'{od}/{name}.json', 'w'), separators=(',', ':'))
    return len(rows)


for name, df in (('meta', pd.DataFrame(meta)), ('weekly', WK), ('games', G), ('team_games', TG), ('cards', cards), ('accuracy', acc)):
    print(name, dump(name, df), 'rows', os.path.getsize(f'{od}/{name}.json'), 'bytes')
print('through week', W, 'next', NEXT, 'card/weekly verdict mismatches', int(mismatch))
