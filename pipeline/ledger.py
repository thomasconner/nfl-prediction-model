"""Luck ledger back-test: at Week 9, compare actual wins to wins expected from efficiency,
then check what happened in Weeks 10-18. Also where Super Bowl teams ranked."""
import sys, os, json
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import pandas as pd
import model as M

out_dir = sys.argv[1]
cfg = json.load(open(f'{out_dir}/{sys.argv[2] if len(sys.argv) > 2 else "method.json"}'))
tg = M.team_games(cfg)
_, g = M.load()

# map rating (pts/game) -> expected win% with a logistic fit on game results, 2022-2025
rows = []
for s in range(2022, 2026):
    for w in M.CHECKPOINTS:
        r = M.rating(cfg, s, w)
        d = tg[(tg.season == s) & (tg.week <= w)]
        rec = d.groupby('team').agg(wins=('win', 'sum'), games=('win', 'size'))
        fut = tg[(tg.season == s) & (tg.week > w)].groupby('team').agg(fwins=('win', 'sum'), fgames=('win', 'size'),
                                                                     fpd=('pf', 'sum'), fpa=('pa', 'sum'))
        for t in rec.index:
            rows.append(dict(season=s, cp=w, team=t, rating=r[t], wins=rec.wins[t], games=rec.games[t],
                             fwins=fut.fwins.get(t, np.nan), fgames=fut.fgames.get(t, np.nan),
                             fpd=(fut.fpd.get(t, np.nan) - fut.fpa.get(t, np.nan)) / fut.fgames.get(t, np.nan)))
L = pd.DataFrame(rows)
L['winp'] = L.wins / L.games
L['fwinp'] = L.fwins / L.fgames
# expected win% from rating: logistic win prob per game vs average opponent; scale fit on all team-checkpoints
from math import exp
best = None
for k in np.linspace(0.04, 0.2, 33):
    p = 1 / (1 + np.exp(-k * L.rating))
    ll = -(L.wins * np.log(p) + (L.games - L.wins) * np.log(1 - p)).sum()
    if best is None or ll < best[0]:
        best = (ll, k)
K = best[1]
L['exp_winp'] = 1 / (1 + np.exp(-K * L.rating))
L['luck_wins'] = L.wins - L.exp_winp * L.games
print('logistic scale', round(K, 3))

wk9 = L[L.cp == 9].copy()
X = np.c_[np.ones(len(wk9)), wk9.winp, wk9.exp_winp]
b = np.linalg.lstsq(X, wk9.fwinp, rcond=None)[0]
print('future win%% = %.3f + %.3f*record + %.3f*expected' % tuple(b))
for lab, col in (('record', 'winp'), ('expected (efficiency)', 'exp_winp')):
    print(lab, 'r with future win% =', round(np.corrcoef(wk9[col], wk9.fwinp)[0, 1], 3))

# flags
wk9['flag'] = np.select([wk9.luck_wins >= 1.5, wk9.luck_wins <= -1.5], ['Lucky (record > strength)', 'Unlucky (record < strength)'], 'In line')
summ = wk9.groupby('flag').agg(teams=('team', 'size'), winp_wk1_9=('winp', 'mean'), winp_wk10_18=('fwinp', 'mean'),
                               declined=('fwinp', lambda x: np.nan))
for f in summ.index:
    d = wk9[wk9.flag == f]
    summ.loc[f, 'declined'] = (d.fwinp < d.winp).mean()
print(summ.round(3))
# among good-record teams only (win% >= .625 at week 9)
gd = wk9[wk9.winp >= 0.625]
print('Good-record teams at Wk 9:', gd.groupby(gd.luck_wins >= 1.0).agg(n=('team', 'size'), before=('winp', 'mean'), after=('fwinp', 'mean'), after_pd=('fpd', 'mean')).round(3))
bd = wk9[wk9.winp <= 0.375]
print('Bad-record teams at Wk 9:', bd.groupby(bd.luck_wins <= -1.0).agg(n=('team', 'size'), before=('winp', 'mean'), after=('fwinp', 'mean'), after_pd=('fpd', 'mean')).round(3))
wk9.to_csv(f'{out_dir}/ledger_wk9.csv', index=False)
L.to_csv(f'{out_dir}/ledger_all.csv', index=False)
print(wk9.sort_values('luck_wins', ascending=False).head(12)[['season', 'team', 'wins', 'games', 'exp_winp', 'luck_wins', 'fwins', 'fgames']].round(2).to_string())
print(wk9.sort_values('luck_wins').head(12)[['season', 'team', 'wins', 'games', 'exp_winp', 'luck_wins', 'fwins', 'fgames']].round(2).to_string())

# Super Bowl teams: season-end rank in rating vs record
gp = pd.read_parquet(f'{M.DATA}/games.parquet')
sb = gp[gp.game_type == 'SB']
for _, row in sb[sb.season.between(2022, 2025)].iterrows():
    s = row.season
    r = M.rating(cfg, s, 18).rank(ascending=False)
    wp = tg[tg.season == s].groupby('team').win.mean().rank(ascending=False, method='min')
    winner = row.home_team if row.result > 0 else row.away_team
    for t in (row.home_team, row.away_team):
        print(s, 'SB', t, 'WON' if t == winner else 'lost', 'rating rank', int(r[t]), 'record rank', int(wp[t]))
json.dump({'logistic_k': K}, open(f'{out_dir}/ledger_params.json', 'w'))
