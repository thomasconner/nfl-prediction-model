"""Build the game table for testing add-ons to the live model (2022 through the latest finished week).

Every row is a regular-season game with the live model's pre-game inputs (ratings frozen after the
week before, preseason prior fading by Week 8) and candidate features known before kickoff.
No Vegas information is used as a feature; the closing spread is kept only as the benchmark.

Usage: python3 -I addons.py <pipeline_dir>   (NFL_DATA points at the data folder)
Writes <pipeline_dir>/out/addon_games.csv
"""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd
import model as M

HERE = os.path.dirname(os.path.abspath(__file__))
D = M.DATA
cfg = json.load(open(f'{HERE}/out/method.json' if os.path.exists(f'{HERE}/out/method.json') else f'{HERE}/out/champion_4yr.json'))
PP = json.load(open(f'{HERE}/out/prior_params.json'))
RHO, K0, TAPER = PP['rho'], PP['k0'], PP['taper_weeks']
SEASONS = tuple(int(x) for x in os.environ.get('NFL_SEASONS', '2022,2023,2024,2025,2026').split(','))
M_QB, M_TEAM = 300.0, 150.0
TZ = {'ARI': -7, 'ATL': -5, 'BAL': -5, 'BUF': -5, 'CAR': -5, 'CHI': -6, 'CIN': -5, 'CLE': -5, 'DAL': -6, 'DEN': -7,
      'DET': -5, 'GB': -6, 'HOU': -6, 'IND': -5, 'JAX': -5, 'KC': -6, 'LA': -8, 'LAC': -8, 'LV': -8, 'MIA': -5,
      'MIN': -6, 'NE': -5, 'NO': -6, 'NYG': -5, 'NYJ': -5, 'PHI': -5, 'PIT': -5, 'SEA': -8, 'SF': -8, 'TB': -5,
      'TEN': -6, 'WAS': -5}  # standard-time offsets; only differences matter

pl, g = M.load()
g = g.copy()
sched = pd.read_csv(f'{D}/games.csv')[['game_id', 'gametime']]
g = g.merge(sched, on='game_id', how='left')
tg = M.team_games(cfg)

# ---------- ratings as the live model had them ----------
raw, prev = {}, {}


def k_at(w):
    return K0 * max(0.0, 1 - w / TAPER)


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
    k = k_at(w)
    pr = pr.reindex(r.index).fillna(0)
    return (n * r + k * pr) / (n + k) if k > 0 else r


# ---------- FTN charting: luck in interceptions and drops ----------
sc = M._scrimmage(pl)
sc = sc[sc.season >= min(SEASONS)]
ftn = pd.concat([pd.read_parquet(f'{D}/ftn_{y}.parquet') for y in SEASONS if os.path.exists(f'{D}/ftn_{y}.parquet')])
ftn = ftn[['nflverse_game_id', 'nflverse_play_id', 'is_interception_worthy', 'is_drop']].rename(columns={'nflverse_game_id': 'game_id', 'nflverse_play_id': 'play_id'})
sc = sc.merge(ftn, on=['game_id', 'play_id'], how='left')
iw = sc.is_interception_worthy.fillna(False).astype(bool)
intc = sc.interception.fillna(0) == 1
E_IW = float(sc.loc[iw, 'epa'].mean())                       # expected EPA of an interception-worthy throw
E_INC = float(sc.loc[(sc['pass'] == 1) & (sc.complete_pass == 0) & ~intc, 'epa'].mean())
E_CATCH = float(sc.loc[(sc['pass'] == 1) & (sc.complete_pass == 1), 'epa'].mean())
adj_int = np.where(iw, E_IW - sc.epa, np.where(intc, E_INC - sc.epa, 0.0))
drop = sc.is_drop.fillna(False).astype(bool)
adj_drop = np.where(drop, 0.5 * (E_CATCH - sc.epa), 0.0)
sc['adj_int'], sc['adj_drop'] = adj_int, adj_drop
print(f'FTN: interception-worthy throw {E_IW:+.2f} EPA, incompletion {E_INC:+.2f}, completion {E_CATCH:+.2f}')
cp = sc[sc.cpoe.notna()]


def cum(df, col, team_col, s, w):
    d = df[(df.season == s) & (df.week < w)]
    return d.groupby(team_col)[col].sum(), d.groupby(team_col).size()


# ---------- quarterbacks ----------
db = pl[(pl.qb_dropback == 1) & pl.epa.notna() & pl.passer_player_id.notna()][['season', 'week', 'posteam', 'passer_player_id', 'epa']]
DPG = db.groupby(['season', 'week', 'posteam']).size().mean()
qs = db.groupby(['season', 'passer_player_id']).epa.agg(['sum', 'size'])
Q_REP = float(qs[qs['size'] < 300]['sum'].sum() / qs[qs['size'] < 300]['size'].sum())
team_season = db.groupby(['season', 'posteam']).epa.agg(['sum', 'size'])


def q_qb(pid, s, w):
    tot = n = 0.0
    for y in range(s - 3, s):
        if (y, pid) in qs.index:
            tot += qs.loc[(y, pid), 'sum']; n += qs.loc[(y, pid), 'size']
    c = db[(db.season == s) & (db.week < w) & (db.passer_player_id == pid)].epa
    return (tot + c.sum() + M_QB * Q_REP) / (n + len(c) + M_QB)


def q_team(t, s, w):
    c = db[(db.season == s) & (db.week < w) & (db.posteam == t)].epa
    base = team_season.loc[(s - 1, t), 'sum'] / team_season.loc[(s - 1, t), 'size'] if (s - 1, t) in team_season.index else 0.0
    return (c.sum() + M_TEAM * base) / (len(c) + M_TEAM)


# ---------- injuries (non-QB), weighted by earlier snap share ----------
players = pd.read_csv(f'{D}/players.csv', usecols=['gsis_id', 'pfr_id'], low_memory=False).dropna()
pfr2gsis = dict(zip(players.pfr_id, players.gsis_id))
OL, SKILL = {'T', 'G', 'C', 'OT', 'OG', 'OL'}, {'WR', 'TE', 'RB', 'FB'}
inj_feat = {}
for s in SEASONS:
    if not os.path.exists(f'{D}/inj_{s}.parquet'):
        continue
    sn = pd.read_parquet(f'{D}/snap_{s}.parquet')
    sn = sn[sn.game_type == 'REG'].copy()
    sn['gsis'] = sn.pfr_player_id.map(pfr2gsis)
    inj = pd.read_parquet(f'{D}/inj_{s}.parquet')
    inj = inj[(inj.game_type == 'REG') & inj.report_status.isin(['Out', 'Doubtful']) & (inj.position != 'QB')]
    ngames = sn.groupby(['team', 'week']).game_id.first().reset_index()
    for (t, w), d in inj.groupby(['team', 'week']):
        ng = int(((ngames.team == t) & (ngames.week < w)).sum())
        if ng == 0:
            continue
        ps = sn[(sn.team == t) & (sn.week < w) & sn.gsis.isin(d.gsis_id)]
        share_off = ps.groupby('gsis').offense_pct.sum() / ng
        share_def = ps.groupby('gsis').defense_pct.sum() / ng
        pos = d.set_index('gsis_id').position
        inj_feat[(s, w, t)] = dict(
            ol=float(sum(share_off.get(p, 0) for p in pos.index if pos[p] in OL)),
            skill=float(sum(share_off.get(p, 0) for p in pos.index if pos[p] in SKILL)),
            defense=float(sum(share_def.get(p, 0) for p in pos.index if pos[p] not in OL | SKILL)))

# ---------- build ----------
rows = []
main_qb = pd.concat([g[['season', 'home_team', 'home_qb_id']].set_axis(['season', 'team', 'qb'], axis=1),
                     g[['season', 'away_team', 'away_qb_id']].set_axis(['season', 'team', 'qb'], axis=1)])
for s in SEASONS:
    gs = g[(g.season == s) & g.result.notna()].sort_values('week')
    if gs.empty:
        continue
    st = main_qb[main_qb.season == s]
    st_wk = pd.concat([gs[['week', 'home_team', 'home_qb_id']].set_axis(['week', 'team', 'qb'], axis=1),
                       gs[['week', 'away_team', 'away_qb_id']].set_axis(['week', 'team', 'qb'], axis=1)])
    for w in sorted(gs.week.unique()):
        r = rating(s, w - 1)
        n_games = tg[(tg.season == s) & (tg.week < w)].groupby('team').size()
        shrink = lambda t: (n_games.get(t, 0) / (n_games.get(t, 0) + k_at(w - 1))) if k_at(w - 1) > 0 else 1.0
        sums = {c: cum(sc, c, 'posteam', s, w) for c in ('adj_int', 'adj_drop')}
        sums_d = {c: cum(sc, c, 'defteam', s, w) for c in ('adj_int', 'adj_drop')}
        co, cn = cum(cp, 'cpoe', 'posteam', s, w)
        cd, cdn = cum(cp, 'cpoe', 'defteam', s, w)
        for _, x in gs[gs.week == w].iterrows():
            neutral = x.location == 'Neutral'
            h, a = x.home_team, x.away_team
            row = dict(season=s, week=w, game_id=x.game_id, home=h, away=a, y=float(x.result), spread=x.spread_line,
                       diff=float(r[h] - r[a]), home_field=0.0 if neutral else 1.0, wk=min(w - 1, 17))
            # quarterback change
            for side, t, q in (('h', h, x.home_qb_id), ('a', a, x.away_qb_id)):
                prior = st_wk[(st_wk.team == t) & (st_wk.week < w)].qb
                usual = prior.value_counts().index[0] if len(prior) else None
                row[f'qbd_{side}'] = (q_qb(q, s, w) - q_team(t, s, w)) * DPG if (usual is not None and q != usual) else 0.0
            row['qb_change'] = row['qbd_h'] - row['qbd_a']
            # injuries
            ih, ia = inj_feat.get((s, w, h), {}), inj_feat.get((s, w, a), {})
            for c in ('ol', 'skill', 'defense'):
                row[f'inj_{c}'] = ia.get(c, 0.0) - ih.get(c, 0.0)
            # home field variants
            row['home_div'] = row['home_field'] * float(x.div_game)
            row['home_trend'] = row['home_field'] * (s - 2024)
            # rest
            hr, ar = float(x.home_rest), float(x.away_rest)
            row['rest_diff'] = float(np.clip(hr - ar, -7, 7))
            row['bye'] = float(hr >= 13) - float(ar >= 13)
            row['short'] = float(hr <= 5) - float(ar <= 5)
            # travel and body clock (home stadium only; neutral sites skipped)
            if not neutral:
                tzd = TZ[h] - TZ[a]
                row['travel'] = float(abs(tzd))
                hh = int(str(x.gametime).split(':')[0]) if isinstance(x.gametime, str) else 13
                body_away = hh + (TZ[a] - (-5))  # away team's home clock at kickoff (gametime is Eastern)
                row['early_body'] = float(body_away <= 11)
                row['late_body'] = float(body_away >= 23)
            else:
                row['travel'] = row['early_body'] = row['late_body'] = 0.0
            # FTN luck: rating change if interceptions and drops were luck-adjusted (offense minus defense, per play)
            for c in ('adj_int', 'adj_drop'):
                so, no = sums[c]
                sd, nd = sums_d[c]
                val = {}
                for t in (h, a):
                    off = so.get(t, 0.0) / no.get(t, 1) if no.get(t, 0) else 0.0
                    dfn = sd.get(t, 0.0) / nd.get(t, 1) if nd.get(t, 0) else 0.0
                    val[t] = M.P * (off - dfn) * shrink(t)
                row[f'ftn_{c[4:]}'] = val[h] - val[a]
            # completion % over expected (Next Gen-style, from play-by-play), offense minus defense, shrunk
            cpo = {t: (co.get(t, 0.0) / (cn.get(t, 0) + 150)) - (cd.get(t, 0.0) / (cdn.get(t, 0) + 150)) for t in (h, a)}
            row['cpoe'] = cpo[h] - cpo[a]
            rows.append(row)
R = pd.DataFrame(rows)
R.to_csv(f'{HERE}/out/addon_games.csv', index=False)
print('games', len(R), 'by season', R.groupby('season').size().to_dict())
print(R.drop(columns=['game_id', 'home', 'away']).describe().T[['mean', 'std']].round(3).to_string())
