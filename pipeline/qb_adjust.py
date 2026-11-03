"""Quarterback adjustment test.

For each game, the base prediction is the live model's (ratings frozen after the week before,
preseason prior fading by Week 8, per-week slope and home field). The add-on:

    delta_team = (q_expected_qb - q_team) * DROPBACKS_PER_GAME
    margin_B   = margin_live + c * (delta_home - delta_away)

q_expected_qb: the starter's EPA per dropback over the previous 3 seasons plus this season so far,
               shrunk toward replacement level with M_QB dropbacks of prior.
q_team:        the team's EPA per dropback this season so far (what the rating already contains),
               shrunk toward last season's team value with M_TEAM dropbacks.
Variant "change": the add-on only when the starter is not the team's usual starter so far.

c is fit on 2022-2024 and graded on 2025 and 2026 (never used for fitting).
Starters for past games are the actual starters (nflverse schedule); for upcoming games nflverse
lists the expected starter, which is what the live model would use.
"""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd
import model as M

HERE = os.path.dirname(os.path.abspath(__file__))
cfg = json.load(open(f'{HERE}/out/champion_4yr.json' if os.path.exists(f'{HERE}/out/champion_4yr.json') else f'{HERE}/out/method.json'))
PP = json.load(open(f'{HERE}/out/prior_params.json'))
RHO, K0, TAPER = PP['rho'], PP['k0'], PP['taper_weeks']
TRAIN, TEST = (2022, 2023, 2024), (2025, 2026)
M_QB, M_TEAM = float(os.environ.get('M_QB', 300)), 150.0

pl, g = M.load()
db = pl[(pl.qb_dropback == 1) & pl.epa.notna()][['season', 'week', 'posteam', 'passer_player_id', 'epa']]
db = db[db.passer_player_id.notna()]
DPG = db.groupby(['season', 'week', 'posteam']).size().mean()
# replacement level: QBs with under 300 dropbacks in a season (backups, spot starters)
qs = db.groupby(['season', 'passer_player_id']).epa.agg(['sum', 'size'])
Q_REP = float(qs[qs['size'] < 300]['sum'].sum() / qs[qs['size'] < 300]['size'].sum())
print(f'dropbacks per team-game {DPG:.1f}, replacement EPA/dropback {Q_REP:+.3f}')
team_season = db.groupby(['season', 'posteam']).epa.agg(['sum', 'size'])
qb_week = db.groupby(['season', 'week', 'passer_player_id']).epa.agg(['sum', 'size'])
team_week = db.groupby(['season', 'week', 'posteam']).epa.agg(['sum', 'size'])
tg = M.team_games(cfg)
raw, prev = {}, {}


def rating(s, w):
    if s not in prev:
        prev[s] = M.rating(cfg, s - 1, 18)
    pr = RHO * prev[s]
    if w == 0:
        return pr
    if (s, w) not in raw:
        raw[(s, w)] = M.rating(cfg, s, w)
    r = raw[(s, w)]
    n = tg[(tg.season == s) & (tg.week <= w)].groupby('team').size().reindex(r.index).fillna(0)
    k = K0 * max(0.0, 1 - w / TAPER)
    pr = pr.reindex(r.index).fillna(0)
    return (n * r + k * pr) / (n + k) if k > 0 else r


def q_qb(pid, s, w):
    a = qs.loc[[(y, pid) for y in range(s - 3, s) if (y, pid) in qs.index]]
    tot, n = a['sum'].sum(), a['size'].sum()
    cur = qb_week.loc[s] if s in qb_week.index.get_level_values(0) else None
    if cur is not None:
        c = cur[(cur.index.get_level_values(0) < w) & (cur.index.get_level_values(1) == pid)]
        tot += c['sum'].sum(); n += c['size'].sum()
    return (tot + M_QB * Q_REP) / (n + M_QB), n


def q_team(team, s, w):
    t = team_week.loc[s]
    c = t[(t.index.get_level_values(0) < w) & (t.index.get_level_values(1) == team)]
    last = team_season.loc[(s - 1, team)] if (s - 1, team) in team_season.index else None
    base = last['sum'] / last['size'] if last is not None else 0.0
    return (c['sum'].sum() + M_TEAM * base) / (c['size'].sum() + M_TEAM)


# each team's main starter (most starts) by season, for offseason changes
_st = pd.concat([g[['season', 'home_team', 'home_qb_id']].set_axis(['season', 'team', 'qb'], axis=1),
                 g[['season', 'away_team', 'away_qb_id']].set_axis(['season', 'team', 'qb'], axis=1)])
last_main = _st.dropna().groupby(['season', 'team']).qb.agg(lambda x: x.value_counts().index[0]).to_dict()
rows = []
for s in TRAIN + TEST:
    gs = g[(g.season == s) & g.result.notna()].sort_values('week')
    starters = []
    for _, x in gs.iterrows():
        starters += [(x.week, x.home_team, x.home_qb_id), (x.week, x.away_team, x.away_qb_id)]
    st = pd.DataFrame(starters, columns=['week', 'team', 'qb'])
    for w in sorted(gs.week.unique()):
        r = rating(s, w - 1)
        b, h = PP['by_week'][str(min(w - 1, 17))]['b'], PP['by_week'][str(min(w - 1, 17))]['h']
        for _, x in gs[gs.week == w].iterrows():
            out = dict(season=s, week=w, game_id=x.game_id, y=x.result, spread=x.spread_line,
                       m0=b * (r[x.home_team] - r[x.away_team]) + (0 if x.location == 'Neutral' else h))
            for side in ('home', 'away'):
                t, q = x[f'{side}_team'], x[f'{side}_qb_id']
                prior_st = st[(st.team == t) & (st.week < w)].qb
                usual = prior_st.value_counts().index[0] if len(prior_st) else None
                if usual is None and os.environ.get('OFFSEASON') == '1':
                    usual = last_main.get((s - 1, t))
                qq, nq = q_qb(q, s, w)
                out[f'd_{side}'] = (qq - q_team(t, s, w)) * DPG
                out[f'chg_{side}'] = bool(usual is not None and q != usual)
                out[f'nq_{side}'] = nq
            rows.append(out)
R = pd.DataFrame(rows)
R['dq'] = R.d_home - R.d_away
R['dq_chg'] = np.where(R.chg_home, R.d_home, 0) - np.where(R.chg_away, R.d_away, 0)
R['any_chg'] = R.chg_home | R.chg_away
tr, te = R[R.season.isin(TRAIN)], R[R.season.isin(TEST)]


def fit_c(col):
    x, e = tr[col], tr.y - tr.m0
    return float((x * e).sum() / (x * x).sum())


def score(d, pred):
    err = d.y - pred
    nz = d.y != 0
    return dict(rmse=float(np.sqrt((err ** 2).mean())), mae=float(err.abs().mean()),
                su=float((np.sign(pred[nz]) == np.sign(d.y[nz])).mean()), n=len(d))


def boot(d, pa, pb, n=4000, seed=7):
    rng = np.random.default_rng(seed)
    ea, eb = (d.y - pa).values ** 2, (d.y - pb).values ** 2
    idx = rng.integers(0, len(d), (n, len(d)))
    return float((eb[idx].mean(1) < ea[idx].mean(1)).mean())


res = {}
for name, col in (('always', 'dq'), ('change', 'dq_chg')):
    c = fit_c(col)
    res[name] = c
    print(f'\n== variant {name}: c = {c:.3f} (fit on 2022-2024)')
    for label, d in (('test 2025+2026', te), ('test 2025', te[te.season == 2025]), ('test 2026', te[te.season == 2026]),
                     ('test QB-change games', te[te.any_chg]), ('test other games', te[~te.any_chg]), ('train', tr)):
        pa, pb = d.m0, d.m0 + c * d[col]
        sa, sb = score(d, pa), score(d, pb)
        line = f"{label:22s} n={sa['n']:4d}  RMSE {sa['rmse']:.2f} -> {sb['rmse']:.2f}  MAE {sa['mae']:.2f} -> {sb['mae']:.2f}  SU {sa['su']:.1%} -> {sb['su']:.1%}"
        if not label.startswith('train'):
            line += f'  P(better) {boot(d, pa, pb):.2f}'
        dv = d[d.spread.notna()]
        if len(dv):
            line += f"  | Vegas RMSE {score(dv, dv.spread)['rmse']:.2f} SU {score(dv, dv.spread)['su']:.1%}"
        print(line)
R.to_csv(f'{HERE}/out/qb_adjust_games.csv', index=False)
json.dump(dict(c_always=res['always'], c_change=res['change'], m_qb=M_QB, m_team=M_TEAM, q_rep=Q_REP, dpg=DPG),
          open(f'{HERE}/out/qb_adjust_params.json', 'w'), indent=1)
