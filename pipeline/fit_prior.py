"""Fit the preseason starting point for in-season projections.

Rating used for projections at week w = (n*r_w + k*rho*r_prev) / (n + k), where r_w is the champion
rating through week w, n the team's games played, r_prev last season's full rating. Week 0 = rho*r_prev.
Grid-search rho, k on 2022-2025: RMSE of predicting every later regular-season game margin from
checkpoints w = 0..8 (pooled), with slope and HFA fit per checkpoint. Also back-tests projected final
wins by week against record pace.
Writes out/prior_params.json and out/proj_backtest.csv.
"""
import sys, os, json
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np, pandas as pd
from scipy.stats import norm
import model as M

HERE = os.path.dirname(os.path.abspath(__file__))
cfg = json.load(open(f'{HERE}/out/method.json'))
_, g = M.load()
SEASONS = tuple(int(x) for x in os.environ.get('NFL_TUNE', '2022,2023,2024,2025').split(','))
WEEKS = range(0, 18)
tg = M.team_games(cfg)
raw, ngames, prev = {}, {}, {}
for s in SEASONS:
    prev[s] = M.rating(cfg, s - 1, 18)
    for w in WEEKS:
        if w == 0:
            continue
        raw[(s, w)] = M.rating(cfg, s, w)
        d = tg[(tg.season == s) & (tg.week <= w)]
        ngames[(s, w)] = d.groupby('team').size()


def k_for(w, k0):
    """Prior weight tapers from k0 games at week 0 to zero at week 8 (per-week grid: best k ~4-6 in
    weeks 1-3, ~2 in weeks 4-7, 0 from week 8; the A/B at weeks 6/9/12 rejected a fixed prior)."""
    return k0 * max(0.0, 1 - w / 8)


def blend(s, w, rho, k0):
    k = k_for(w, k0)
    pr = rho * prev[s]
    if w == 0:
        return pr
    r, n = raw[(s, w)], ngames[(s, w)].reindex(raw[(s, w)].index).fillna(0)
    pr = pr.reindex(r.index).fillna(0)
    if k == 0:
        return r
    return (n * r + k * pr) / (n + k)


def rows_for(rho, k, weeks):
    out = []
    for w in weeks:
        for s in SEASONS:
            r = blend(s, w, rho, k)
            gs = g[(g.season == s) & (g.week > w) & g.result.notna()]
            out.append(pd.DataFrame(dict(w=w, s=s, x=(gs.home_team.map(r) - gs.away_team.map(r)).values,
                                         hf=(gs.location != 'Neutral').astype(float).values, y=gs.result.values,
                                         home=gs.home_team.values, away=gs.away_team.values, week=gs.week.values)))
    return pd.concat(out)


def fit_rmse(df):
    se, n = 0.0, 0
    params = {}
    for w, d in df.groupby('w'):
        X = np.c_[d.x, d.hf]
        b, h = np.linalg.lstsq(X, d.y, rcond=None)[0]
        res = d.y - X @ [b, h]
        params[w] = (b, h, float(res.std()))
        se += (res ** 2).sum(); n += len(d)
    return np.sqrt(se / n), params


res = []
for rho in (0.3, 0.4, 0.5, 0.6):
    for k0 in (0, 3, 4, 5, 6):
        rm, _ = fit_rmse(rows_for(rho, k0, range(1, 13)))
        res.append((rho, k0, rm))
R = pd.DataFrame(res, columns=['rho', 'k0', 'rmse']).sort_values('rmse')
print(R.head(8).to_string(index=False))
print('no prior:', R[R.k0 == 0].rmse.iloc[0])
rho, k = 0.5, 6.0  # best of the grid (ridge is flat: 0.4-0.6, 4-6 within 0.01 RMSE)
print('using rho', rho, 'k0', k, 'rmse', R[(R.rho == rho) & (R.k0 == k)].rmse.iloc[0])
# week 0 alone: does rho matter? (slope absorbs scale; check fit)
rm0, p0 = fit_rmse(rows_for(rho, k, [0]))
print('week0 rmse', round(rm0, 3), 'b,h,sd', [round(v, 3) for v in p0[0]])
# per-checkpoint params for weeks 0..17
allrows = rows_for(rho, k, WEEKS)
_, params = fit_rmse(allrows)
json.dump(dict(rho=rho, k0=k, taper_weeks=8, by_week={int(w): dict(b=float(b), h=float(h), sd=float(sd)) for w, (b, h, sd) in params.items()}),
          open(f'{HERE}/out/prior_params.json', 'w'), indent=1)

# back-test projected final wins at each checkpoint vs record pace
fin = {}
for s in SEASONS:
    d = tg[tg.season == s]
    fin[s] = d.groupby('team').win.sum()
bt = []
for w in range(0, 18):
    b, h, sd = params[w]
    for s in SEASONS:
        r = blend(s, w, rho, k)
        d = tg[(tg.season == s) & (tg.week <= w)]
        won = d.groupby('team').win.sum().reindex(r.index).fillna(0)
        played = d.groupby('team').size().reindex(r.index).fillna(0)
        gs = g[(g.season == s) & (g.week > w)]
        exp = pd.Series(0.0, index=r.index)
        for _, gm in gs.iterrows():
            m = b * (r[gm.home_team] - r[gm.away_team]) + (h if gm.location != 'Neutral' else 0)
            p = norm.cdf(m / sd)
            exp[gm.home_team] += p; exp[gm.away_team] += 1 - p
        proj = won + exp
        total = played + gs.groupby('home_team').size().reindex(r.index).fillna(0) + gs.groupby('away_team').size().reindex(r.index).fillna(0)
        pace = np.where(played > 0, won / played.replace(0, np.nan) * total, total / 2)
        f = fin[s].reindex(r.index)
        bt.append(dict(week=w, season=s, mae_model=float((proj - f).abs().mean()), mae_pace=float(np.abs(pace - f).mean()),
                       mae_eight5=float((total / 2 - f).abs().mean())))
B = pd.DataFrame(bt).groupby('week')[['mae_model', 'mae_pace', 'mae_eight5']].mean().round(2)
print(B.to_string())
B.reset_index().to_csv(f'{HERE}/out/proj_backtest.csv', index=False)
