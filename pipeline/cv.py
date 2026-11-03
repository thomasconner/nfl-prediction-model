"""Forward selection of add-ons with leave-one-season-out cross-validation.

Base model: margin = sum_b slope_b * rating_diff * [week bucket b] + home_field * h  (refit in every fold).
Each candidate add-on enters as one more term. Every season is predicted by a model fit on the other
seasons, so all predictions are out of sample. A candidate is added when it lowers the pooled
out-of-sample RMSE and wins at least MIN_P of paired bootstrap resamples (by game) against the
current set. Vegas is reported for comparison only and never used as an input.

Usage: python3 -I cv.py <pipeline_dir>   reads out/addon_games.csv, writes out/addon_selection.json
"""
import sys, os, json
import numpy as np
import pandas as pd

HERE = os.path.abspath(sys.argv[1]) if len(sys.argv) > 1 else os.path.dirname(os.path.abspath(__file__))
MIN_P = float(os.environ.get('MIN_P', 0.8))
R = pd.read_csv(f'{HERE}/out/addon_games.csv')
R['inj_all'] = R.inj_ol + R.inj_skill + R.inj_defense
BUCKETS = [(0, 0), (1, 1), (2, 2), (3, 3), (4, 5), (6, 7), (8, 10), (11, 17)]
for i, (lo, hi) in enumerate(BUCKETS):
    R[f'd{i}'] = R['diff'] * R.wk.between(lo, hi)
BASE = [f'd{i}' for i in range(len(BUCKETS))] + ['home_field']
CANDS = ['qb_change', 'inj_all', 'inj_ol', 'inj_skill', 'inj_defense', 'home_div', 'home_trend', 'rest_diff', 'bye',
         'travel', 'early_body', 'ftn_int', 'ftn_drop', 'cpoe']
seasons = sorted(R.season.unique())


def oos(cols):
    pred = pd.Series(0.0, index=R.index)
    coefs = {}
    for s in seasons:
        tr, te = R[R.season != s], R[R.season == s]
        b = np.linalg.lstsq(tr[cols].values, tr.y.values, rcond=None)[0]
        pred[te.index] = te[cols].values @ b
        coefs[int(s)] = dict(zip(cols, map(float, b)))
    return pred, coefs


def rmse(p, idx=None):
    e = (R.y - p) if idx is None else (R.y[idx] - p[idx])
    return float(np.sqrt((e ** 2).mean()))


def su(p, idx=None):
    d = R if idx is None else R.loc[idx]
    pp = p if idx is None else p[idx]
    nz = d.y != 0
    return float((np.sign(pp[nz]) == np.sign(d.y[nz])).mean())


def p_better(pa, pb, n=4000, seed=11):
    rng = np.random.default_rng(seed)
    ea, eb = ((R.y - pa) ** 2).values, ((R.y - pb) ** 2).values
    idx = rng.integers(0, len(R), (n, len(R)))
    return float((eb[idx].mean(1) < ea[idx].mean(1)).mean())


cur = list(BASE)
p_cur, _ = oos(cur)
steps = [dict(step='base (live model)', rmse=rmse(p_cur), su=su(p_cur))]
print(f"base: RMSE {rmse(p_cur):.3f}  SU {su(p_cur):.1%}  (n={len(R)})")
remaining = list(CANDS)
while remaining:
    trials = []
    for c in remaining:
        if c.startswith('inj_') and c != 'inj_all' and 'inj_all' in cur:
            continue
        if c == 'inj_all' and any(x in cur for x in ('inj_ol', 'inj_skill', 'inj_defense')):
            continue
        p, _ = oos(cur + [c])
        trials.append((rmse(p), c, p))
    if not trials:
        break
    trials.sort(key=lambda t: t[0])
    print('  candidates:', ', '.join(f'{c} {r:.3f}' for r, c, _ in trials))
    r_best, c_best, p_best = trials[0]
    pb = p_better(p_cur, p_best)
    if r_best < rmse(p_cur) and pb >= MIN_P:
        cur.append(c_best)
        remaining.remove(c_best)
        steps.append(dict(step=f'+ {c_best}', rmse=r_best, su=su(p_best), p_better=pb))
        print(f'ADD {c_best}: RMSE {rmse(p_cur):.3f} -> {r_best:.3f}, SU {su(p_cur):.1%} -> {su(p_best):.1%}, P(better) {pb:.2f}')
        p_cur = p_best
    else:
        print(f'stop: best remaining {c_best} RMSE {r_best:.3f} (P(better) {pb:.2f})')
        break

p_base, _ = oos(BASE)
_, coefs_all = oos(cur)
full = np.linalg.lstsq(R[cur].values, R.y.values, rcond=None)[0]
v = R.spread.notna()
print('\nOut-of-sample by season (base -> selected | Vegas closing line):')
for s in seasons:
    i = R.index[R.season == s]
    iv = R.index[(R.season == s) & v]
    print(f"  {s}: n={len(i)}  RMSE {rmse(p_base, i):.2f} -> {rmse(p_cur, i):.2f} | {float(np.sqrt(((R.y[iv] - R.spread[iv]) ** 2).mean())):.2f}   "
          f"SU {su(p_base, i):.1%} -> {su(p_cur, i):.1%} | {su(R.spread.fillna(0), iv):.1%}")
print(f"  all: RMSE {rmse(p_base):.3f} -> {rmse(p_cur):.3f} | {float(np.sqrt(((R.y[v] - R.spread[v]) ** 2).mean())):.3f}   "
      f"SU {su(p_base):.1%} -> {su(p_cur):.1%} | {su(R.spread.fillna(0), R.index[v]):.1%}   P(selected beats base) {p_better(p_base, p_cur):.2f}")
print('\nCoefficients fit on all seasons:', {c: round(float(b), 3) for c, b in zip(cur, full)})
json.dump(dict(selected=[c for c in cur if c not in BASE], steps=steps, buckets=BUCKETS,
               coef_all=dict(zip(cur, map(float, full))), coef_by_fold=coefs_all, min_p=MIN_P),
          open(f'{HERE}/out/addon_selection.json', 'w'), indent=1)
