"""Grade the champion once on the 2025 holdout, and run the one-week-ahead test vs. Vegas."""
import sys, os, json
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import pandas as pd
import model as M

out_dir = sys.argv[1]
champ = json.load(open(f'{out_dir}/champion.json'))
methods = {
    'Champion (raw net EPA/play)': M.method(champ),
    'Original plan (A0)': M.method(M.A0),
    'Point differential': lambda s, w: M.baseline_rating('pd', s, w),
    'Win-loss record': lambda s, w: M.baseline_rating('record', s, w),
}
rows = []
scored = {}
for name, fn in methods.items():
    for mode in ('tune', 'holdout'):
        o = M.score(fn, mode)
        scored[(name, mode)] = o
        rows.append(dict(method=name, sample='2022-24 (tuning)' if mode == 'tune' else '2025 (holdout)', **M.summarize(o)))
for mode in ('tune', 'holdout'):
    o = scored[('Champion (raw net EPA/play)', mode)]
    v = o.assign(yhat=o.vegas, err=o.y - o.vegas)
    rows.append(dict(method='Vegas closing spread (set at game time)', sample='2022-24 (tuning)' if mode == 'tune' else '2025 (holdout)', **M.summarize(v)))
res = pd.DataFrame(rows)
print(res.to_string())
res.to_csv(f'{out_dir}/holdout_table.csv', index=False)

ho = {k[0]: v for k, v in scored.items() if k[1] == 'holdout'}
c = ho['Champion (raw net EPA/play)']
for other in ('Original plan (A0)', 'Point differential', 'Win-loss record'):
    print('2025: champion vs', other, M.compare(ho[other], c))

# per-checkpoint holdout RMSE
print(c.groupby('cp').err.apply(lambda e: round(float(np.sqrt((e ** 2).mean())), 3)).to_dict())

# ---------- one-week-ahead vs Vegas ----------
_, g = M.load()
recs = []
for s in (2022, 2023, 2024, 2025):
    gs = g[(g.season == s) & g.result.notna()]
    for w in range(4, 19):
        r = M.rating(champ, s, w - 1)
        pdg = M.baseline_rating('pd', s, w - 1)
        wk = gs[gs.week == w]
        recs.append(pd.DataFrame({'season': s, 'week': w, 'game_id': wk.game_id.values,
                                  'x': wk.home_team.map(r).values - wk.away_team.map(r).values,
                                  'xpd': wk.home_team.map(pdg).values - wk.away_team.map(pdg).values,
                                  'home': (wk.location != 'Neutral').astype(float).values,
                                  'y': wk.result.values, 'vegas': wk.spread_line.values}))
wa = pd.concat(recs, ignore_index=True)
wa['bucket'] = pd.cut(wa.week, [3, 6, 10, 14, 18], labels=['4-6', '7-10', '11-14', '15-18'])
tr, te = wa[wa.season <= 2024], wa[wa.season == 2025].copy()
for b in wa.bucket.cat.categories:
    t = tr[tr.bucket == b]
    coef = np.linalg.lstsq(np.c_[t.x, t.home], t.y, rcond=None)[0]
    m = te.bucket == b
    te.loc[m, 'yhat'] = coef[0] * te.loc[m, 'x'] + coef[1] * te.loc[m, 'home']
    # does the rating add anything to Vegas? fit y = a*vegas + c*x
    coef2 = np.linalg.lstsq(np.c_[t.vegas, t.x], t.y, rcond=None)[0]
    te.loc[m, 'yhat_blend'] = coef2[0] * te.loc[m, 'vegas'] + coef2[1] * te.loc[m, 'x']
rm = lambda e: float(np.sqrt((e ** 2).mean()))
nz = te.y != 0
print('ONE-WEEK-AHEAD 2025 holdout: champion RMSE', round(rm(te.y - te.yhat), 3), 'SU', round(float((np.sign(te.yhat[nz]) == np.sign(te.y[nz])).mean()), 3))
print('  Vegas RMSE', round(rm(te.y - te.vegas), 3), 'SU', round(float((np.sign(te.vegas[nz]) == np.sign(te.y[nz])).mean()), 3))
print('  Vegas+rating blend RMSE', round(rm(te.y - te.yhat_blend), 3))
a = te.assign(err=te.y - te.vegas, cp=0); b = te.assign(err=te.y - te.yhat, cp=0)
print('  P(champion beats Vegas)', M.compare(a, b))
b2 = te.assign(err=te.y - te.yhat_blend, cp=0)
print('  P(blend beats Vegas)', M.compare(a, b2))
te.to_csv(f'{out_dir}/one_week_ahead_2025.csv', index=False)
