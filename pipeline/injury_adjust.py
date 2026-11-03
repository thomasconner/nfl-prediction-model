"""Injury-report test (non-quarterbacks), on top of the quarterback-change adjustment.

For each team and game: players listed Out or Doubtful on that week's final injury report, weighted
by how much of the team's snaps they played in its earlier games this season (missed games count
as zero, so a player already out for weeks adds little). Three groups:
  OL (T, G, C, OT, OG, OL), skill (WR, TE, RB, FB), defense (all defensive positions).
Coefficients are fit on 2022-2024 residuals of the live model + QB adjustment, graded on 2025-2026.
"""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd
import model as M

HERE = os.path.dirname(os.path.abspath(__file__))
D = M.DATA
G = pd.read_csv(f'{HERE}/out/qb_adjust_games.csv')
QB = json.load(open(f'{HERE}/out/qb_adjust_params.json'))
G['m1'] = G.m0 + QB['c_change'] * G.dq_chg
players = pd.read_csv(f'{D}/players.csv', usecols=['gsis_id', 'pfr_id'], low_memory=False).dropna()
pfr2gsis = dict(zip(players.pfr_id, players.gsis_id))
OL, SKILL = {'T', 'G', 'C', 'OT', 'OG', 'OL'}, {'WR', 'TE', 'RB', 'FB'}
DEF = {'DE', 'DT', 'NT', 'DL', 'LB', 'ILB', 'OLB', 'MLB', 'CB', 'S', 'SS', 'FS', 'DB', 'EDGE'}
feats = []
for s in sorted(G.season.unique()):
    sn = pd.read_parquet(f'{D}/snap_{s}.parquet')
    sn = sn[sn.game_type == 'REG']
    sn['gsis'] = sn.pfr_player_id.map(pfr2gsis)
    inj = pd.read_parquet(f'{D}/inj_{s}.parquet')
    inj = inj[(inj.game_type == 'REG') & inj.report_status.isin(['Out', 'Doubtful'])]
    team_games = sn.groupby(['team', 'week']).game_id.first().reset_index()
    for (t, w), d in inj.groupby(['team', 'week']):
        prior_games = team_games[(team_games.team == t) & (team_games.week < w)]
        ng = len(prior_games)
        if ng == 0:
            continue
        ps = sn[(sn.team == t) & (sn.week < w)]
        row = dict(season=s, week=w, team=t, ol=0.0, skill=0.0, defense=0.0)
        for _, p in d.iterrows():
            if p.position == 'QB':
                continue
            mine = ps[ps.gsis == p.gsis_id]
            off = mine.offense_pct.sum() / ng
            dfn = mine.defense_pct.sum() / ng
            if p.position in OL:
                row['ol'] += off
            elif p.position in SKILL:
                row['skill'] += off
            elif p.position in DEF:
                row['defense'] += dfn
        feats.append(row)
F = pd.DataFrame(feats)
F[['ol', 'skill', 'defense']] = F[['ol', 'skill', 'defense']].astype(float)
team_of = G.game_id.str.split('_', expand=True)
G['away'], G['home'] = team_of[2], team_of[3]
for side in ('home', 'away'):
    G = G.merge(F.rename(columns={'team': side, 'ol': f'ol_{side}', 'skill': f'skill_{side}', 'defense': f'def_{side}'}),
                on=['season', 'week', side], how='left')
for c in ('ol', 'skill', 'def'):
    G[f'{c}_h'], G[f'{c}_a'] = G[f'{c}_home'].fillna(0), G[f'{c}_away'].fillna(0)
    G[f'x_{c}'] = G[f'{c}_a'] - G[f'{c}_h']   # more missing at away helps home
X = ['x_ol', 'x_skill', 'x_def']
tr, te = G[G.season <= 2024], G[G.season >= 2025]
coef = np.linalg.lstsq(tr[X].values, (tr.y - tr.m1).values, rcond=None)[0]
print('starter-equivalents out per team-game (mean):', {c: round(float(pd.concat([G[f'{c}_h'], G[f'{c}_a']]).mean()), 2) for c in ('ol', 'skill', 'def')})
print('points per starter-equivalent out (fit 2022-2024):', dict(zip(X, np.round(coef, 2))))
c_one = float((tr[X].sum(axis=1) * (tr.y - tr.m1)).sum() / (tr[X].sum(axis=1) ** 2).sum())
print('single coefficient for all non-QB starters out:', round(c_one, 2))


def rmse(e): return float(np.sqrt((e ** 2).mean()))


def boot(d, pa, pb, n=4000, seed=7):
    rng = np.random.default_rng(seed)
    ea, eb = (d.y - pa).values ** 2, (d.y - pb).values ** 2
    idx = rng.integers(0, len(d), (n, len(d)))
    return float((eb[idx].mean(1) < ea[idx].mean(1)).mean())


for label, d in (('test 2025+2026', te), ('test 2025', te[te.season == 2025]), ('test 2026', te[te.season == 2026]), ('train', tr)):
    p3 = d.m1 + d[X].values @ coef
    p1 = d.m1 + c_one * d[X].sum(axis=1)
    nz = d.y != 0
    su = lambda p: float((np.sign(p[nz]) == np.sign(d.y[nz])).mean())
    print(f"{label:16s} n={len(d):4d} live {rmse(d.y - d.m0):.2f}/{su(d.m0):.1%}  +QB {rmse(d.y - d.m1):.2f}/{su(d.m1):.1%}  "
          f"+inj(3) {rmse(d.y - p3):.2f}/{su(p3):.1%}  +inj(1) {rmse(d.y - p1):.2f}/{su(p1):.1%}"
          + (f"  P(+QB+inj1 beats live) {boot(d, d.m0, p1):.2f}  P(+inj1 beats +QB) {boot(d, d.m1, p1):.2f}" if label != 'train' else '')
          + (f"  | Vegas {rmse((d.y - d.spread).dropna()):.2f}" if d.spread.notna().any() else ''))
json.dump(dict(coef=dict(zip(X, map(float, coef))), c_one=c_one), open(f'{HERE}/out/injury_adjust_params.json', 'w'), indent=1)
