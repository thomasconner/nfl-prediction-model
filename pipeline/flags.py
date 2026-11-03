"""Test sustainability flags at Week 9 (2022-2025): do they predict Weeks 10-18 beyond the rating?"""
import sys, os, json
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import pandas as pd
import model as M
import analysis as A

out_dir = sys.argv[1]
cfg = json.load(open(f'{out_dir}/champion_4yr.json'))
pl = A._all_plays()
s = M._scrimmage(pl[pl.season_type == 'REG'])
db = s[s.qb_dropback == 1]
qbs = db.groupby(['season', 'passer_player_id']).epa.agg(['sum', 'size'])


def qb_baseline(pid, season):
    """EPA/dropback over the previous three seasons (needs 200+ dropbacks)."""
    rows = [qbs.loc[(y, pid)] for y in range(season - 3, season) if (y, pid) in qbs.index]
    if not rows:
        return np.nan
    tot = sum(r['size'] for r in rows)
    return sum(r['sum'] for r in rows) / tot if tot >= 200 else np.nan


def team_flags(season, week):
    d = s[(s.season == season) & (s.week <= week)]
    off = d.groupby('posteam')
    out = pd.DataFrame({'rush_epa': d[d.qb_dropback != 1].groupby('posteam').epa.mean(),
                        'pass_epa': d[d.qb_dropback == 1].groupby('posteam').epa.mean(),
                        'off_epa': off.epa.mean(), 'def_epa': d.groupby('defteam').epa.mean()})
    out['rush_pct'] = out.rush_epa.rank(pct=True)
    out['pass_pct'] = out.pass_epa.rank(pct=True)
    dd = d[d.qb_dropback == 1]
    out['def_int_rate'] = dd.groupby('defteam').interception.mean()
    out['def_sack_rate'] = dd.groupby('defteam').sack.mean()
    t3 = d[d.down == 3]
    out['def_3rd'] = t3.groupby('defteam').third_down_converted.mean()
    # current QB = most dropbacks in the team's latest game
    last_wk = d.groupby('posteam').week.max()
    cur = {}
    for t, w in last_wk.items():
        x = dd[(dd.posteam == t) & (dd.week == w)].passer_player_id.value_counts()
        cur[t] = x.index[0] if len(x) else None
    out['qb'] = pd.Series(cur)
    out['qb_epa'] = [dd[(dd.posteam == t) & (dd.passer_player_id == q)].epa.mean() if q else np.nan for t, q in cur.items()]
    out['qb_base'] = [qb_baseline(q, season) if q else np.nan for q in cur.values()]
    out['qb_excess'] = out.qb_epa - out.qb_base
    return out


if __name__ == '__main__':
    L = pd.read_csv(f'{out_dir}/ledger_all.csv')
    L = L[L.cp == 9]
    rows = []
    for y in range(2022, 2026):
        f = team_flags(y, 9)
        f['season'] = y
        rows.append(f.reset_index(names='team'))
    F = pd.concat(rows).merge(L, on=['season', 'team'])
    # future offense EPA (weeks 10-18)
    fut = s[(s.week >= 10) & s.season.between(2022, 2025)]
    fo = fut.groupby(['season', 'posteam']).epa.mean().rename('fut_off').reset_index().rename(columns={'posteam': 'team'})
    fd = fut.groupby(['season', 'defteam']).epa.mean().rename('fut_def').reset_index().rename(columns={'defteam': 'team'})
    F = F.merge(fo, on=['season', 'team']).merge(fd, on=['season', 'team'])

    def ols(y, cols, data):
        X = np.c_[np.ones(len(data)), data[cols].values]
        b, *_ = np.linalg.lstsq(X, data[y].values, rcond=None)
        resid = data[y].values - X @ b
        se = np.sqrt(np.diag(np.linalg.inv(X.T @ X)) * resid.var(ddof=X.shape[1]))
        return pd.DataFrame({'coef': b, 'se': se, 't': b / se}, index=['const'] + cols).round(4)

    print('Future offense EPA ~ first-half pass vs rush EPA')
    print(ols('fut_off', ['pass_epa', 'rush_epa'], F))
    q = F.dropna(subset=['qb_excess'])
    print('Future offense EPA ~ first-half offense + QB excess over his 3-yr baseline (n=%d)' % len(q))
    print(ols('fut_off', ['off_epa', 'qb_excess'], q))
    print('Future defense EPA ~ first-half defense + INT rate + sack rate + 3rd-down allowed')
    print(ols('fut_def', ['def_epa', 'def_int_rate', 'def_sack_rate', 'def_3rd'], F))
    print('Future PD/g ~ rating + luck wins')
    print(ols('fpd', ['rating', 'luck_wins'], F))
    F.to_csv(f'{out_dir}/flags_wk9.csv', index=False)
    print(F[(F.team == 'IND') & (F.season == 2025)].T)
