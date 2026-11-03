"""Descriptive splits used for the luck ledger, case studies and report cards."""
import os, sys
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import pandas as pd
import model as M

PYTH_EXP = 2.37


def _all_plays():
    if 'allplays' not in M._CACHE:
        pl = pd.read_parquet(f'{M.DATA}/plays.parquet')
        g = pd.read_parquet(f'{M.DATA}/games.parquet')
        valid = set(g.loc[g.result.notna(), 'game_id'])
        M._CACHE['allplays'] = pl[pl.game_id.isin(valid)]
    return M._CACHE['allplays']


def games_for(team, spans, include_post=False):
    """spans: list of (season, wk_lo, wk_hi). Returns that team's games."""
    g = pd.read_parquet(f'{M.DATA}/games.parquet')
    g = g[g.result.notna()]
    if not include_post:
        g = g[g.game_type == 'REG']
    m = np.zeros(len(g), bool)
    for s, lo, hi in spans:
        m |= ((g.season == s) & (g.week >= lo) & (g.week <= hi)).values
    g = g[m & ((g.home_team == team) | (g.away_team == team)).values].copy()
    g['opp'] = np.where(g.home_team == team, g.away_team, g.home_team)
    g['pf'] = np.where(g.home_team == team, g.home_score, g.away_score)
    g['pa'] = np.where(g.home_team == team, g.away_score, g.home_score)
    return g.sort_values(['season', 'week'])


def split_stats(team, spans, include_post=False):
    g = games_for(team, spans, include_post)
    ids = set(g.game_id)
    pl = _all_plays()
    pl = pl[pl.game_id.isin(ids)]
    s = M._scrimmage(pl)
    off, dfn = s[s.posteam == team], s[s.defteam == team]
    n = len(g)
    out = dict(games=n, wins=int((g.pf > g.pa).sum()), losses=int((g.pf < g.pa).sum()),
               pd_pg=float((g.pf - g.pa).mean()), pf_pg=float(g.pf.mean()), pa_pg=float(g.pa.mean()))
    close = g[(g.pf - g.pa).abs() <= 8]
    out['one_score'] = f"{int((close.pf > close.pa).sum())}-{int((close.pf < close.pa).sum())}"
    out['pyth_wins'] = float(n * (g.pf.sum() ** PYTH_EXP) / (g.pf.sum() ** PYTH_EXP + g.pa.sum() ** PYTH_EXP))
    for side, d in (('off', off), ('def', dfn)):
        out[f'{side}_epa'] = float(d.epa.mean())
        out[f'{side}_sr'] = float(d.success.mean())
        out[f'{side}_pass_epa'] = float(d[d.qb_dropback == 1].epa.mean())
        out[f'{side}_rush_epa'] = float(d[d.qb_dropback != 1].epa.mean())
        ed = d[d.down.isin([1, 2])]
        out[f'{side}_early_epa'] = float(ed.epa.mean())
        out[f'{side}_early_sr'] = float(ed.success.mean())
        t3 = d[d.down == 3]
        out[f'{side}_3rd_conv'] = float(t3.third_down_converted.sum() / max(1, len(t3)))
        out[f'{side}_3rd_ydstogo'] = float(t3.ydstogo.mean())
        out[f'{side}_sack_rate'] = float(d[d.qb_dropback == 1].sack.mean())
        out[f'{side}_int_rate'] = float(d[d.qb_dropback == 1].interception.mean())
        out[f'{side}_explosive'] = float(((d.yards_gained >= 20) & (d.qb_dropback == 1) | (d.yards_gained >= 10) & (d.qb_dropback != 1)).mean())
        # red zone: drives with a scrimmage play inside the 20, share ending in a TD
        rz = d[d.yardline_100 <= 20].groupby(['game_id', 'fixed_drive']).fixed_drive_result.first()
        out[f'{side}_rz_td'] = float((rz == 'Touchdown').mean()) if len(rz) else np.nan
    # turnovers and fumble luck
    allp = pl
    out['giveaways_pg'] = float((off.interception.sum() + off.fumble_lost.sum()) / n)
    out['takeaways_pg'] = float((dfn.interception.sum() + dfn.fumble_lost.sum()) / n)
    own_f = allp[(allp.fumble == 1) & (allp.posteam == team)]
    opp_f = allp[(allp.fumble == 1) & (allp.defteam == team)]
    out['own_fumbles_lost_pct'] = float(own_f.fumble_lost.mean()) if len(own_f) else np.nan
    out['opp_fumbles_recovered_pct'] = float(opp_f.fumble_lost.mean()) if len(opp_f) else np.nan
    # special teams + kicking
    st = allp[allp.play_type.isin(['kickoff', 'punt', 'field_goal', 'extra_point']) & allp.epa.notna()]
    out['st_epa_pg'] = float((st[st.posteam == team].epa.sum() - st[st.defteam == team].epa.sum()) / n)
    ofg = allp[(allp.field_goal_result.notna()) & (allp.defteam == team)]
    out['opp_fg_pct'] = float((ofg.field_goal_result == 'made').mean()) if len(ofg) else np.nan
    # game control: average win probability across the team's offensive and defensive snaps
    wp_off = off.wp.dropna()
    wp_def = 1 - dfn.wp.dropna()
    out['avg_wp'] = float(pd.concat([wp_off, wp_def]).mean())
    # primary QB
    db = off[off.qb_dropback == 1]
    qb = db.passer_player_name.value_counts()
    out['qbs'] = ', '.join(f'{k} ({v})' for k, v in qb.head(3).items())
    return out


def qb_season_table(min_db=200):
    """EPA/dropback and CPOE per passer per season (all teams), regular season."""
    pl = _all_plays()
    s = M._scrimmage(pl[pl.season_type == 'REG'])
    db = s[s.qb_dropback == 1]
    t = db.groupby(['season', 'passer_player_id', 'passer_player_name']).agg(
        dropbacks=('epa', 'size'), epa=('epa', 'mean'), cpoe=('cpoe', 'mean'), int_rate=('interception', 'mean'),
        sack_rate=('sack', 'mean')).reset_index()
    return t[t.dropbacks >= min_db]
