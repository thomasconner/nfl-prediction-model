"""Report cards for all 32 teams at (season, week).

Usage: python3 -I reportcard.py <out_dir> <season> <week>
Writes <out_dir>/cards_<season>_wk<week>.json and .csv
"""
import sys, os, json
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import pandas as pd
from scipy.stats import norm
import model as M
import analysis as A
from flags import qb_baseline

out_dir, SEASON, WEEK = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
cfg = json.load(open(f'{out_dir}/champion_4yr.json'))
K = json.load(open(f'{out_dir}/ledger_params.json'))['logistic_k']
NAMES = {'ARI': 'Arizona Cardinals', 'ATL': 'Atlanta Falcons', 'BAL': 'Baltimore Ravens', 'BUF': 'Buffalo Bills',
         'CAR': 'Carolina Panthers', 'CHI': 'Chicago Bears', 'CIN': 'Cincinnati Bengals', 'CLE': 'Cleveland Browns',
         'DAL': 'Dallas Cowboys', 'DEN': 'Denver Broncos', 'DET': 'Detroit Lions', 'GB': 'Green Bay Packers',
         'HOU': 'Houston Texans', 'IND': 'Indianapolis Colts', 'JAX': 'Jacksonville Jaguars', 'KC': 'Kansas City Chiefs',
         'LA': 'Los Angeles Rams', 'LAC': 'Los Angeles Chargers', 'LV': 'Las Vegas Raiders', 'MIA': 'Miami Dolphins',
         'MIN': 'Minnesota Vikings', 'NE': 'New England Patriots', 'NO': 'New Orleans Saints', 'NYG': 'New York Giants',
         'NYJ': 'New York Jets', 'PHI': 'Philadelphia Eagles', 'PIT': 'Pittsburgh Steelers', 'SEA': 'Seattle Seahawks',
         'SF': 'San Francisco 49ers', 'TB': 'Tampa Bay Buccaneers', 'TEN': 'Tennessee Titans', 'WAS': 'Washington Commanders'}
GRADES = ['A+', 'A+', 'A', 'A', 'A', 'A-', 'A-', 'A-', 'B+', 'B+', 'B+', 'B', 'B', 'B', 'B-', 'B-', 'B-',
          'C+', 'C+', 'C+', 'C', 'C', 'C', 'C-', 'C-', 'C-', 'D', 'D', 'D', 'F', 'F', 'F']


def grade(rank):
    return GRADES[int(rank) - 1]


# ---------- slope for projecting margins from this week ----------
def slope_for_week(w):
    _, g = M.load()
    xs, ys, hs = [], [], []
    for s in (2022, 2023, 2024, 2025):
        r = M.rating(cfg, s, w)
        gs = g[(g.season == s) & (g.week > w) & g.result.notna()]
        xs += list(gs.home_team.map(r) - gs.away_team.map(r))
        hs += list((gs.location != 'Neutral').astype(float))
        ys += list(gs.result)
    b, h = np.linalg.lstsq(np.c_[xs, hs], ys, rcond=None)[0]
    resid = np.array(ys) - (b * np.array(xs) + h * np.array(hs))
    return b, h, resid.std()


parts = M.rating(cfg, SEASON, WEEK, with_parts=True)
tg = M.team_games(cfg)
d = tg[(tg.season == SEASON) & (tg.week <= WEEK)]
pl = A._all_plays()
pl = pl[(pl.season == SEASON) & (pl.season_type == 'REG') & (pl.week <= WEEK)]
s = M._scrimmage(pl)
gall = pd.read_parquet(f'{M.DATA}/games.parquet')
gall = gall[(gall.season == SEASON) & (gall.game_type == 'REG')]
b, h, sd = slope_for_week(min(max(WEEK, 6), 12))

cards = []
teams = sorted(parts.index)
rk = parts.rating.rank(ascending=False, method='min')
ork = parts.off.rank(ascending=False, method='min')
drk = parts['def'].rank(ascending=False, method='min')
winp = d.groupby('team').win.mean()
wrk = winp.rank(ascending=False, method='min')
dd = s[s.qb_dropback == 1]
rush = s[s.qb_dropback != 1].groupby('posteam').epa.mean()
pas = dd.groupby('posteam').epa.mean()
rush_pct = rush.rank(pct=True)
def_int = dd.groupby('defteam').interception.mean()
def_int_rk = def_int.rank(ascending=False, method='min')
league_sched = []
for t in teams:
    my = d[d.team == t].sort_values('week')
    played = list(my.opp)
    rem = gall[((gall.home_team == t) | (gall.away_team == t)) & (gall.week > WEEK)]
    league_sched.append((t, parts.rating.reindex(played).mean(),
                         parts.rating.reindex(np.where(rem.home_team == t, rem.away_team, rem.home_team)).mean()))
sched = pd.DataFrame(league_sched, columns=['team', 'sos_played', 'sos_left']).set_index('team')
sp_rk = sched.sos_played.rank(ascending=False, method='min')
sl_rk = sched.sos_left.rank(ascending=False, method='min')

for t in teams:
    my = d[d.team == t]
    g_ = len(my)
    w_ = int((my.win == 1).sum()); l_ = int((my.win == 0).sum()); ti = int((my.win == 0.5).sum())
    r = parts.loc[t]
    exp_w = g_ / (1 + np.exp(-K * r.rating))
    luck = round((w_ + 0.5 * ti) - exp_w, 1)
    close = my[(my.pf - my.pa).abs() <= 8]
    cw, cl = int((close.pf > close.pa).sum()), int((close.pf < close.pa).sum())
    off_p, def_p = s[s.posteam == t], s[s.defteam == t]
    allp = pl
    take = int(def_p.interception.sum() + allp[(allp.defteam == t) & (allp.fumble_lost == 1)].shape[0])
    give = int(off_p.interception.sum() + allp[(allp.posteam == t) & (allp.fumble_lost == 1)].shape[0])
    own_f = allp[(allp.fumble == 1) & (allp.posteam == t)]
    opp_f = allp[(allp.fumble == 1) & (allp.defteam == t)]
    ofg = allp[(allp.field_goal_result.notna()) & (allp.defteam == t)]
    st = allp[allp.play_type.isin(['kickoff', 'punt', 'field_goal', 'extra_point']) & allp.epa.notna()]
    st_pg = (st[st.posteam == t].epa.sum() - st[st.defteam == t].epa.sum()) / g_
    # QB
    last_w = my.week.max()
    x = dd[(dd.posteam == t) & (dd.week == last_w)].passer_player_id.value_counts()
    qid = x.index[0] if len(x) else None
    qname = dd[dd.passer_player_id == qid].passer_player_name.iloc[0] if qid else None
    starts = int((my.qb == qid).sum())
    first_qb = my.sort_values('week').qb.iloc[0]
    qb_epa = float(dd[(dd.posteam == t) & (dd.passer_player_id == qid)].epa.mean()) if qid else np.nan
    qb_base = qb_baseline(qid, SEASON) if qid else np.nan
    # remaining schedule projection
    rem = gall[((gall.home_team == t) | (gall.away_team == t)) & (gall.week > WEEK)]
    exp_rem = 0.0
    for _, gm in rem.iterrows():
        home = gm.home_team == t
        opp = gm.away_team if home else gm.home_team
        hf = 0.0 if gm.location == 'Neutral' else (h if home else -h)
        m = b * (r.rating - parts.rating[opp]) + hf
        exp_rem += norm.cdf(m / sd)
    proj_w = w_ + 0.5 * ti + exp_rem
    # flags
    flags = []
    if rush_pct[t] >= 0.85 and rush[t] > pas[t]:
        flags.append('Offense leans on rushing efficiency, which repeats about half as well as passing (0.29 vs. 0.55)')
    if drk[t] <= 8 and ork[t] >= 17:
        flags.append('Strength comes mostly from defense, which repeats about half as well as offense (0.33 vs. 0.59)')
    if not np.isnan(qb_base) and qb_epa - qb_base >= 0.15 and starts >= 3:
        flags.append(f'QB is playing {qb_epa - qb_base:+.2f} EPA/dropback above his 3-year norm (watch item; not proven to predict a drop)')
    if qid is not None and first_qb != qid:
        flags.append(f'Starting QB has changed this season ({qname} has {starts} of {g_} starts)')
    if luck >= 1.5:
        flags.append(f'Record is {luck:.1f} wins better than its play; every such team since 2022 got worse')
    elif luck >= 1.0:
        flags.append(f'Record is {luck:.1f} wins better than its play')
    if luck <= -1.5:
        flags.append(f'Record is {-luck:.1f} wins worse than its play; 6 of 8 such teams since 2022 improved')
    elif luck <= -1.0:
        flags.append(f'Record is {-luck:.1f} wins worse than its play')
    # verdict
    sr = int(rk[t])
    wpct = (w_ + 0.5 * ti) / g_
    if sr <= 6:
        verdict = 'Contender'
    elif wpct >= 0.6 and (sr >= 17 or (sr >= 13 and luck >= 1.0)):
        verdict = 'Fluff'
    elif luck >= 1.0 and sr <= 16:
        verdict = 'Good but lucky'
    elif luck <= -1.0 and sr <= 16:
        verdict = 'Better than their record'
    elif sr <= 12:
        verdict = 'Solid'
    elif sr <= 22:
        verdict = 'Middle of the pack'
    else:
        verdict = 'Bad'
    luck_grade = 'Lucky' if luck >= 1.0 else ('Unlucky' if luck <= -1.0 else 'Neutral')
    cards.append(dict(
        team=t, name=NAMES[t], season=SEASON, week=WEEK, games=g_, record=f'{w_}-{l_}' + (f'-{ti}' if ti else ''), win_pct=round(wpct, 3),
        record_rank=int(wrk[t]), verdict=verdict,
        strength=round(float(r.rating), 1), strength_rank=sr, strength_grade=grade(sr),
        offense=round(float(r.off), 1), offense_rank=int(ork[t]), offense_grade=grade(ork[t]),
        defense=round(float(r['def']), 1), defense_rank=int(drk[t]), defense_grade=grade(drk[t]),
        expected_wins=round(float(exp_w), 1), luck_wins=round(float(luck), 1), luck=luck_grade,
        one_score=f'{cw}-{cl}', turnover_margin=take - give,
        own_fumbles_lost=f'{int(own_f.fumble_lost.sum())}/{len(own_f)}', opp_fumbles_recovered=f'{int(opp_f.fumble_lost.sum())}/{len(opp_f)}',
        opp_fg_pct=round(float((ofg.field_goal_result == 'made').mean()), 3) if len(ofg) else None,
        special_teams_pg=round(float(st_pg), 1),
        pass_epa=round(float(pas[t]), 3), rush_epa=round(float(rush[t]), 3),
        sos_played=round(float(sched.sos_played[t]), 1), sos_played_rank=int(sp_rk[t]), sos_played_grade=grade(sp_rk[t]),
        sos_left=round(float(sched.sos_left[t]), 1) if not np.isnan(sched.sos_left[t]) else None,
        sos_left_rank=int(sl_rk[t]) if not np.isnan(sched.sos_left[t]) else None,
        projected_wins=round(float(proj_w), 1),
        qb=qname, qb_starts=starts, qb_epa=round(qb_epa, 3) if not np.isnan(qb_epa) else None,
        qb_baseline=round(float(qb_base), 3) if not np.isnan(qb_base) else None,
        flags=flags))

out = pd.DataFrame(cards).sort_values('strength_rank')
out.to_csv(f'{out_dir}/cards_{SEASON}_wk{WEEK}.csv', index=False)
json.dump(cards, open(f'{out_dir}/cards_{SEASON}_wk{WEEK}.json', 'w'), indent=1)
print('slope', round(b, 3), 'hfa', round(h, 2), 'sd', round(sd, 2))
cols = ['team', 'record', 'verdict', 'strength', 'strength_grade', 'offense_grade', 'defense_grade', 'luck_wins',
        'one_score', 'turnover_margin', 'sos_played_grade', 'projected_wins', 'qb']
print(out[cols].to_string(index=False))
