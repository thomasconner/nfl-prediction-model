"""Split-half stability: weeks 1-9 vs weeks 10-18, team-seasons 2022-2025.

For each stat: how well the first half predicts (a) the same stat later and (b) later point differential.
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import pandas as pd
import model as M
import analysis as A

pl = A._all_plays()
pl = pl[pl.season_type == 'REG']
s = M._scrimmage(pl)
g = pd.read_parquet(f'{M.DATA}/games.parquet')
g = g[(g.game_type == 'REG') & g.result.notna()]
res = pd.concat([
    g[['season', 'week', 'game_id', 'home_team', 'home_score', 'away_score']].rename(columns={'home_team': 'team', 'home_score': 'pf', 'away_score': 'pa'}),
    g[['season', 'week', 'game_id', 'away_team', 'away_score', 'home_score']].rename(columns={'away_team': 'team', 'away_score': 'pf', 'home_score': 'pa'})])
res['margin'] = res.pf - res.pa
res['win'] = np.sign(res.margin) * 0.5 + 0.5
res['close'] = res.margin.abs() <= 8


def half_stats(lo, hi):
    out = {}
    d = s[(s.week >= lo) & (s.week <= hi)]
    r = res[(res.week >= lo) & (res.week <= hi)]
    grp = ['season', 'posteam']
    gd = ['season', 'defteam']
    out['Net EPA/play'] = d.groupby(grp).epa.mean() - d.groupby(gd).epa.mean().rename_axis(grp)
    out['Offense EPA/play'] = d.groupby(grp).epa.mean()
    out['Defense EPA/play (allowed)'] = d.groupby(gd).epa.mean().rename_axis(grp)
    out['Pass offense EPA/dropback'] = d[d.qb_dropback == 1].groupby(grp).epa.mean()
    out['Rush offense EPA/play'] = d[d.qb_dropback != 1].groupby(grp).epa.mean()
    out['Pass defense EPA/dropback'] = d[d.qb_dropback == 1].groupby(gd).epa.mean().rename_axis(grp)
    out['Rush defense EPA/play'] = d[d.qb_dropback != 1].groupby(gd).epa.mean().rename_axis(grp)
    out['Net success rate'] = d.groupby(grp).success.mean() - d.groupby(gd).success.mean().rename_axis(grp)
    ed = d[d.down.isin([1, 2])]
    out['Net early-down EPA'] = ed.groupby(grp).epa.mean() - ed.groupby(gd).epa.mean().rename_axis(grp)
    t3 = d[d.down == 3]
    out['3rd-down conversion (off)'] = t3.groupby(grp).third_down_converted.mean()
    out['3rd-down conversion (def)'] = t3.groupby(gd).third_down_converted.mean().rename_axis(grp)
    rz = d[d.yardline_100 <= 20].groupby(['season', 'posteam', 'game_id', 'fixed_drive']).fixed_drive_result.first().reset_index()
    rz['td'] = rz.fixed_drive_result == 'Touchdown'
    out['Red-zone TD rate (off)'] = rz.groupby(grp).td.mean()
    out['INTs thrown per dropback'] = d[d.qb_dropback == 1].groupby(grp).interception.mean()
    out['INTs made per dropback (def)'] = d[d.qb_dropback == 1].groupby(gd).interception.mean().rename_axis(grp)
    p = pl[(pl.week >= lo) & (pl.week <= hi)]
    f = p[p.fumble == 1]
    out['Share of own fumbles lost'] = f.groupby(grp).fumble_lost.mean()
    out['Share of opp fumbles recovered'] = f.groupby(gd).fumble_lost.mean().rename_axis(grp)
    take = (d.groupby(gd).interception.sum().rename_axis(grp) + p[p.fumble_lost == 1].groupby(gd).size().rename_axis(grp))
    give = (d.groupby(grp).interception.sum() + p[p.fumble_lost == 1].groupby(grp).size())
    ng = r.groupby(['season', 'team']).size().rename_axis(grp)
    out['Turnover margin per game'] = (take.reindex(ng.index).fillna(0) - give.reindex(ng.index).fillna(0)) / ng
    st = p[p.play_type.isin(['kickoff', 'punt', 'field_goal', 'extra_point']) & p.epa.notna()]
    out['Special teams EPA/game'] = (st.groupby(grp).epa.sum() - st.groupby(gd).epa.sum().rename_axis(grp)) / ng
    fg = p[p.field_goal_result.notna()]
    out['Opponent FG make rate'] = (fg.field_goal_result == 'made').groupby([fg.season, fg.defteam]).mean().rename_axis(grp)
    out['Sacks taken per dropback'] = d[d.qb_dropback == 1].groupby(grp).sack.mean()
    out['Sacks made per dropback (def)'] = d[d.qb_dropback == 1].groupby(gd).sack.mean().rename_axis(grp)
    rr = r.groupby(['season', 'team'])
    out['Win %'] = rr.win.mean().rename_axis(grp)
    out['Point differential/game'] = rr.margin.mean().rename_axis(grp)
    cl = r[r.close].groupby(['season', 'team']).win.mean().rename_axis(grp)
    out['Win % in one-score games'] = cl
    out['avg_wp'] = None
    out.pop('avg_wp')
    return pd.DataFrame(out)


H1, H2 = half_stats(1, 9), half_stats(10, 18)
H1 = H1[H1.index.get_level_values(0).isin([2022, 2023, 2024, 2025])]
H2 = H2.reindex(H1.index)
rows = []
for c in H1.columns:
    m = H1[c].notna() & H2[c].notna()
    same = np.corrcoef(H1.loc[m, c], H2.loc[m, c])[0, 1]
    m2 = H1[c].notna() & H2['Point differential/game'].notna()
    fut = np.corrcoef(H1.loc[m2, c], H2.loc[m2, 'Point differential/game'])[0, 1]
    # sign so that "good" = positive for future PD
    rows.append(dict(stat=c, repeats_r=same, predicts_future_pd_r=fut, n=int(m.sum())))
out = pd.DataFrame(rows).sort_values('repeats_r', ascending=False)
print(out.round(3).to_string())
out.to_csv(os.path.join(sys.argv[1], 'stability.csv'), index=False)
# by season for the key ones
for c in ['Net EPA/play', 'Win %', 'Point differential/game', 'Win % in one-score games', 'Rush offense EPA/play', 'Pass offense EPA/dropback']:
    print(c, {y: round(float(np.corrcoef(H1.xs(y)[c].dropna(), H2.xs(y)['Point differential/game'].reindex(H1.xs(y)[c].dropna().index))[0, 1]), 2) for y in (2022, 2023, 2024, 2025)})
