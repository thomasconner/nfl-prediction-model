"""Team-strength ratings and the A/B scoreboard.

A method = a config dict. Each method turns plays into a team rating (points per game
vs. an average team) at a checkpoint week. The scoreboard freezes ratings at Weeks 6, 9, 12
and scores how well they predict the margin of every later regular-season game.
"""
import os
import numpy as np
import pandas as pd

DATA = os.environ.get('NFL_DATA', '/tmp/claude-0/-home-claude/78131004-1175-5698-a8cc-38bdbc13664e/scratchpad/data')
P = 62.0  # scrimmage plays per team-game, converts EPA/play to points per game
CHECKPOINTS = (6, 9, 12)
TUNE = tuple(int(x) for x in os.environ.get('NFL_TUNE', '2022,2023,2024').split(','))
HOLDOUT = (2025,)

A0 = dict(
    opp_adj=1.0,        # strength of opponent adjustment (0 = none)
    wp_lo=0.10, wp_hi=0.90,  # competitive-situation filter on offense win probability
    excl_2min=False,    # drop last 2 minutes of each half
    fumble_neutral=True,  # charge every fumble the league-average fumble outcome
    int_def=1.0,        # share of defensive interception value kept (rest regressed to league rate)
    int_off=1.0,        # same for interceptions thrown
    def_w=1.0,          # weight on defense relative to offense
    rush_w=1.0,         # weight on rushing-play EPA (rest shrunk to league rush mean)
    sr_blend=0.0,       # share of rating from success rate instead of EPA
    down34_w=1.0,       # weight of 3rd/4th-down plays
    clip=None,          # cap on EPA per play (explosive/turnover damping)
    st_w=0.0,           # weight on special-teams EPA
    half_life=None,     # recency weighting half-life in weeks (None = equal)
    prior_k=0.0,        # weight of last season's rating, in games
    prior_rho=0.6,      # how much of last season's rating carries over
    pd_blend=0.0,       # share of rating from actual point differential per game
)

_CACHE = {}


def load():
    if 'plays' not in _CACHE:
        pl = pd.read_parquet(f'{DATA}/plays.parquet')
        g = pd.read_parquet(f'{DATA}/games.parquet')
        pl = pl[pl.season_type == 'REG'].copy()
        g = g[g.game_type == 'REG'].copy()
        valid = set(g.loc[g.result.notna(), 'game_id'])
        pl = pl[pl.game_id.isin(valid)]
        _CACHE['plays'] = pl
        _CACHE['games'] = g
    return _CACHE['plays'], _CACHE['games']


def _scrimmage(pl):
    m = pl.play_type.isin(['pass', 'run']) & pl.epa.notna() & (pl.qb_kneel != 1) & (pl.qb_spike != 1)
    return pl[m].copy()


def team_games(cfg):
    """Per team-game sums for offense and defense under a config (all seasons)."""
    key = tuple(sorted((k, cfg[k]) for k in ('wp_lo', 'wp_hi', 'excl_2min', 'fumble_neutral', 'int_def',
                                               'int_off', 'rush_w', 'down34_w', 'clip')) + [('pass_only', cfg.get('pass_only', False))])
    if key in _CACHE:
        return _CACHE[key]
    pl, g = load()
    s = _scrimmage(pl)
    s['is_pass'] = (s.qb_dropback == 1)
    epa = s.epa.copy()
    if cfg['fumble_neutral']:
        fmean = s[s.fumble == 1].groupby('season').epa.mean()
        fm = s.fumble == 1
        epa[fm] = s.loc[fm, 'season'].map(fmean)
    if cfg['rush_w'] != 1.0:
        rmean = s[~s.is_pass].groupby('season').epa.mean()
        rm = ~s.is_pass
        epa[rm] = cfg['rush_w'] * epa[rm] + (1 - cfg['rush_w']) * s.loc[rm, 'season'].map(rmean)
    if cfg['clip'] is not None:
        epa = epa.clip(-cfg['clip'], cfg['clip'])
    # interceptions: offense and defense can keep different shares of the realized value
    intm = (s.interception == 1)
    int_rate = s[s.is_pass].groupby('season').interception.mean()
    int_epa = s[intm].groupby('season').epa.mean()
    exp_int = s.season.map(int_rate * int_epa) * s.is_pass
    def _int_regress(e, beta):
        # keep `beta` of each realized INT's value; charge every dropback (1-beta) of the league-average INT cost
        if beta == 1.0:
            return e.copy()
        out = e.copy()
        out[intm] = beta * out[intm]
        return out + (1 - beta) * exp_int
    epa_off = _int_regress(epa, cfg['int_off'])
    epa_def = _int_regress(epa, cfg['int_def'])
    w = pd.Series(1.0, index=s.index)
    keep = s.wp.between(cfg['wp_lo'], cfg['wp_hi'])
    if cfg['excl_2min']:
        keep &= s.half_seconds_remaining > 120
    if cfg['down34_w'] != 1.0:
        w[s.down.isin([3, 4])] = cfg['down34_w']
    if cfg.get('pass_only', False):
        keep &= s.is_pass
    s = s.assign(eo=epa_off * w, ed=epa_def * w, w=w, sr=s.success.fillna(0) * w)[keep]
    off = s.groupby(['season', 'week', 'game_id', 'posteam', 'defteam']).agg(
        off_epa=('eo', 'sum'), off_n=('w', 'sum'), off_sr=('sr', 'sum')).reset_index()
    dfn = s.groupby(['season', 'week', 'game_id', 'defteam', 'posteam']).agg(
        def_epa=('ed', 'sum'), def_n=('w', 'sum'), def_sr=('sr', 'sum')).reset_index()
    off = off.rename(columns={'posteam': 'team', 'defteam': 'opp'})
    dfn = dfn.rename(columns={'defteam': 'team', 'posteam': 'opp'})
    tg = off.merge(dfn, on=['season', 'week', 'game_id', 'team', 'opp'], how='outer').fillna(0)
    # primary passer for each team-game (most dropbacks, all situations)
    sc = _scrimmage(pl)
    qbg = (sc[sc.qb_dropback == 1].groupby(['game_id', 'posteam', 'passer_player_id']).size()
           .reset_index(name='n').sort_values('n').drop_duplicates(['game_id', 'posteam'], keep='last'))
    tg = tg.merge(qbg.rename(columns={'posteam': 'team', 'passer_player_id': 'qb'})[['game_id', 'team', 'qb']],
                  on=['game_id', 'team'], how='left')
    # special teams and results
    st = pl[pl.play_type.isin(['kickoff', 'punt', 'field_goal', 'extra_point']) & pl.epa.notna()]
    sto = st.groupby(['game_id', 'posteam']).epa.sum()
    std = st.groupby(['game_id', 'defteam']).epa.sum()
    tg['st_net'] = [sto.get((gid, t), 0.0) - std.get((gid, t), 0.0) for gid, t in zip(tg.game_id, tg.team)]
    res = pd.concat([
        g[['game_id', 'home_team', 'home_score', 'away_score']].rename(columns={'home_team': 'team', 'home_score': 'pf', 'away_score': 'pa'}),
        g[['game_id', 'away_team', 'away_score', 'home_score']].rename(columns={'away_team': 'team', 'away_score': 'pf', 'home_score': 'pa'}),
    ])
    tg = tg.merge(res, on=['game_id', 'team'], how='left')
    tg['win'] = np.where(tg.pf > tg.pa, 1.0, np.where(tg.pf < tg.pa, 0.0, 0.5))
    _CACHE[key] = tg
    return tg


def _opp_adjust(d, val_off, val_def, n_off, n_def, lam, iters=40):
    teams = sorted(set(d.team))
    mu = d[val_off].sum() / d[n_off].sum()
    O = pd.Series(0.0, index=teams)
    D = pd.Series(0.0, index=teams)
    sum_no = d.groupby('team')[n_off].sum()
    sum_nd = d.groupby('team')[n_def].sum()
    for _ in range(iters if lam else 1):
        oadj = d[val_off] - d[n_off] * (mu + lam * d.opp.map(D).fillna(0))
        dadj = d[val_def] - d[n_def] * (mu + lam * d.opp.map(O).fillna(0))
        O = oadj.groupby(d.team).sum() / sum_no
        D = dadj.groupby(d.team).sum() / sum_nd
        O -= O.mean()
        D -= D.mean()
    return O, D


def rating(cfg, season, week, with_parts=False):
    """Team rating in points per game vs. average, using regular-season games through `week`."""
    tg = team_games(cfg)
    d = tg[(tg.season == season) & (tg.week <= week)].copy()
    if d.empty:
        return None
    gw = np.ones(len(d))
    if cfg['half_life']:
        gw = 0.5 ** ((d.week.max() - d.week) / cfg['half_life'])
    for c in ('off_epa', 'off_n', 'def_epa', 'def_n', 'off_sr', 'def_sr'):
        d[c] = d[c] * gw
    d['gw'] = gw
    if cfg.get('qb_filter', False):
        # offense measured only in games started by the team's current QB (if he has 3+ starts)
        last = d.sort_values('week').groupby('team').qb.last()
        cur = d.team.map(last)
        nstart = (d.qb == cur).groupby(d.team).transform('sum')
        drop = (d.qb != cur) & (nstart >= 3)
        for c in ('off_epa', 'off_n', 'off_sr'):
            d.loc[drop, c] = 0.0
    O, D = _opp_adjust(d, 'off_epa', 'def_epa', 'off_n', 'def_n', cfg['opp_adj'])
    epa_r = P * (O - cfg['def_w'] * D)
    r = epa_r.copy()
    if cfg['sr_blend']:
        So, Sd = _opp_adjust(d, 'off_sr', 'def_sr', 'off_n', 'def_n', cfg['opp_adj'])
        sr_r = P * SR_TO_EPA * (So - cfg['def_w'] * Sd)
        r = (1 - cfg['sr_blend']) * r + cfg['sr_blend'] * sr_r
    gws = d.groupby('team').gw.sum()
    if cfg['st_w']:
        r = r + cfg['st_w'] * (d.st_net * d.gw).groupby(d.team).sum() / gws
    pdg = ((d.pf - d.pa) * d.gw).groupby(d.team).sum() / gws
    if cfg['pd_blend']:
        r = (1 - cfg['pd_blend']) * r + cfg['pd_blend'] * pdg
    if cfg.get('rec_blend', 0.0):
        winp = (d.win * d.gw).groupby(d.team).sum() / gws
        r = (1 - cfg['rec_blend']) * r + cfg['rec_blend'] * WIN_TO_PTS * (winp - 0.5)
    if cfg['prior_k']:
        prev = rating(dict(cfg, prior_k=0.0), season - 1, 18)
        n = d.groupby('team').size()
        pr = cfg['prior_rho'] * prev.reindex(r.index).fillna(0) if prev is not None else 0 * r
        r = (n * r + cfg['prior_k'] * pr) / (n + cfg['prior_k'])
    if with_parts:
        return pd.DataFrame({'rating': r, 'off': P * O, 'def': -P * D, 'epa_rating': epa_r, 'pd_pg': pdg,
                             'games': d.groupby('team').size()})
    return r


WIN_TO_PTS = 26.9  # point differential per game per unit of win% (team-seasons 2021-2026)
SR_TO_EPA = 2.36  # set by calibrate_sr() on tuning seasons (2.364)


def calibrate_sr():
    """Slope of team-season net EPA/play on net success rate (2022-2024)."""
    tg = team_games(A0)
    d = tg[tg.season.isin(TUNE)]
    a = d.groupby(['season', 'team']).agg(oe=('off_epa', 'sum'), on=('off_n', 'sum'), de=('def_epa', 'sum'),
                                          dn=('def_n', 'sum'), os_=('off_sr', 'sum'), ds=('def_sr', 'sum'))
    ne = a.oe / a.on - a.de / a.dn
    ns = a.os_ / a.on - a.ds / a.dn
    return float(np.polyfit(ns, ne, 1)[0])


# ---------- baselines ----------
def baseline_rating(kind, season, week):
    tg = team_games(A0)
    d = tg[(tg.season == season) & (tg.week <= week)]
    if kind == 'record':
        return d.groupby('team').win.mean() - 0.5
    if kind == 'pd':
        return (d.pf - d.pa).groupby(d.team).mean()
    raise ValueError(kind)


# ---------- scoreboard ----------
def predictions(rate_fn, seasons):
    """rate_fn(season, week) -> Series. Returns one row per (checkpoint, later game)."""
    _, g = load()
    rows = []
    for s in seasons:
        gs = g[(g.season == s) & g.result.notna()]
        for w in CHECKPOINTS:
            r = rate_fn(s, w)
            later = gs[gs.week > w]
            x = later.home_team.map(r).values - later.away_team.map(r).values
            rows.append(pd.DataFrame({'season': s, 'cp': w, 'game_id': later.game_id.values, 'week': later.week.values,
                                      'x': x, 'home': (later.location != 'Neutral').astype(float).values,
                                      'y': later.result.values, 'vegas': later.spread_line.values}))
    return pd.concat(rows, ignore_index=True)


def fit_predict(pr, train_seasons, test_seasons):
    """Per checkpoint, fit y = b*x + h*home on train seasons; predict test seasons."""
    out = []
    for w in CHECKPOINTS:
        tr = pr[(pr.cp == w) & pr.season.isin(train_seasons)]
        te = pr[(pr.cp == w) & pr.season.isin(test_seasons)].copy()
        X = np.c_[tr.x, tr.home]
        b, h = np.linalg.lstsq(X, tr.y, rcond=None)[0]
        te['yhat'] = b * te.x + h * te.home
        te['b'] = b
        out.append(te)
    return pd.concat(out)


def score(rate_fn, mode='tune'):
    """mode 'tune': leave-one-season-out over 2022-2024. mode 'holdout': fit 2022-2024, test 2025."""
    if mode == 'tune':
        pr = predictions(rate_fn, TUNE)
        out = pd.concat([fit_predict(pr, [t for t in TUNE if t != s], [s]) for s in TUNE])
    else:
        pr = predictions(rate_fn, TUNE + HOLDOUT)
        out = fit_predict(pr, TUNE, HOLDOUT)
    out['err'] = out.y - out.yhat
    return out.sort_values(['season', 'cp', 'game_id']).reset_index(drop=True)


def summarize(out):
    nz = out.y != 0
    return dict(rmse=float(np.sqrt((out.err ** 2).mean())), mae=float(out.err.abs().mean()),
                su=float((np.sign(out.yhat[nz]) == np.sign(out.y[nz])).mean()), n=len(out))


def compare(a, b, n_boot=4000, seed=7):
    """Paired cluster bootstrap by game. Returns RMSE gain of B over A and P(B better)."""
    assert (a.game_id.values == b.game_id.values).all() and (a.cp.values == b.cp.values).all()
    d = pd.DataFrame({'g': a.game_id.values, 'da': a.err.values ** 2, 'db': b.err.values ** 2})
    per = d.groupby('g')[['da', 'db']].sum()
    cnt = d.groupby('g').size()
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(per), size=(n_boot, len(per)))
    A = per.da.values[idx].sum(1) / cnt.values[idx].sum(1)
    B = per.db.values[idx].sum(1) / cnt.values[idx].sum(1)
    gain = np.sqrt(d.da.mean()) - np.sqrt(d.db.mean())
    return dict(rmse_gain=float(gain), p_b_better=float((B < A).mean()))


def method(cfg):
    return lambda s, w: rating(cfg, s, w)
