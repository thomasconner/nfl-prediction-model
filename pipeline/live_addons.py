"""Live add-ons for next-game picks: quarterback change, players ruled out, rest days.

Chosen by leave-one-season-out cross-validation over 2022-2026 (addons.py builds the test table,
cv.py selects; out/addons.json holds the live coefficients). Vegas information is never an input.

    margin = slope[week bucket] * rating_diff + home_field * h
             + c_qb * qb_change + c_inj * players_out + c_rest * rest_diff

qb_change   (expected starter's EPA/dropback - team's EPA/dropback so far) * dropbacks per game,
            home minus away, only for a team whose expected starter is not its usual starter.
players_out starters listed Out or Doubtful (not QBs), weighted by their share of the team's
            earlier snaps this season; away minus home.
rest_diff   home rest days minus away rest days, capped at +/-7.
"""
import os, json
import numpy as np
import pandas as pd

M_QB, M_TEAM = 300.0, 150.0
OL, SKILL = {'T', 'G', 'C', 'OT', 'OG', 'OL'}, {'WR', 'TE', 'RB', 'FB'}


class Addons:
    def __init__(self, pipeline_dir, plays, data_dir, season):
        self.cfg = json.load(open(f'{pipeline_dir}/out/addons.json'))
        self.coef = self.cfg['coef']
        self.buckets = [tuple(b) for b in self.cfg['buckets']]
        self.sd = self.cfg['sd']
        self.season = season
        db = plays[(plays.qb_dropback == 1) & plays.epa.notna() & plays.passer_player_id.notna()]
        self.db = db[['season', 'week', 'posteam', 'passer_player_id', 'epa']]
        qs = self.db.groupby(['season', 'passer_player_id']).epa.agg(['sum', 'size'])
        self.qs = qs
        self.q_rep = self.cfg['q_rep']
        self.dpg = self.cfg['dpg']
        self.team_season = self.db.groupby(['season', 'posteam']).epa.agg(['sum', 'size'])
        # injuries and snaps for this season, if downloaded
        self.inj = self.snaps = None
        p_inj, p_snap, p_pl = f'{data_dir}/inj_{season}.parquet', f'{data_dir}/snap_{season}.parquet', f'{data_dir}/players.csv'
        if os.path.exists(p_inj) and os.path.exists(p_snap) and os.path.exists(p_pl):
            players = pd.read_csv(p_pl, usecols=['gsis_id', 'pfr_id'], low_memory=False).dropna()
            m = dict(zip(players.pfr_id, players.gsis_id))
            sn = pd.read_parquet(p_snap)
            sn = sn[sn.game_type == 'REG'].copy()
            sn['gsis'] = sn.pfr_player_id.map(m)
            self.snaps = sn
            inj = pd.read_parquet(p_inj)
            self.inj = inj[(inj.game_type == 'REG') & (inj.position != 'QB')]

    # ---- features ----
    def q_qb(self, pid, s, w):
        tot = n = 0.0
        for y in range(s - 3, s):
            if (y, pid) in self.qs.index:
                tot += self.qs.loc[(y, pid), 'sum']; n += self.qs.loc[(y, pid), 'size']
        c = self.db[(self.db.season == s) & (self.db.week < w) & (self.db.passer_player_id == pid)].epa
        return (tot + c.sum() + M_QB * self.q_rep) / (n + len(c) + M_QB)

    def q_team(self, t, s, w):
        c = self.db[(self.db.season == s) & (self.db.week < w) & (self.db.posteam == t)].epa
        ts = self.team_season
        base = ts.loc[(s - 1, t), 'sum'] / ts.loc[(s - 1, t), 'size'] if (s - 1, t) in ts.index else 0.0
        return (c.sum() + M_TEAM * base) / (len(c) + M_TEAM)

    def usual_qb(self, sched, t, w):
        played = sched[(sched.season == self.season) & (sched.week < w) & sched.result.notna()]
        q = pd.concat([played.loc[played.home_team == t, 'home_qb_id'], played.loc[played.away_team == t, 'away_qb_id']]).dropna()
        return q.value_counts().index[0] if len(q) else None

    def qb_delta(self, sched, t, qid, w):
        if not isinstance(qid, str):
            return 0.0, None
        usual = self.usual_qb(sched, t, w)
        if usual is None or qid == usual:
            return 0.0, usual
        return (self.q_qb(qid, self.season, w) - self.q_team(t, self.season, w)) * self.dpg, usual

    def players_out(self, t, w):
        """Starter-equivalents ruled out (Out/Doubtful) on this week's report; None if no report yet."""
        if self.inj is None:
            return None, []
        rep = self.inj[(self.inj.week == w) & (self.inj.team == t)]
        if rep.empty:
            return None, []
        out = rep[rep.report_status.isin(['Out', 'Doubtful'])]
        sn = self.snaps
        ng = sn[(sn.team == t) & (sn.week < w)].game_id.nunique()
        if ng == 0 or out.empty:
            return 0.0, []
        ps = sn[(sn.team == t) & (sn.week < w) & sn.gsis.isin(out.gsis_id)]
        off = ps.groupby('gsis').offense_pct.sum() / ng
        dfn = ps.groupby('gsis').defense_pct.sum() / ng
        total, names = 0.0, []
        for _, p in out.iterrows():
            v = off.get(p.gsis_id, 0.0) if p.position in OL | SKILL else dfn.get(p.gsis_id, 0.0)
            total += v
            if v >= 0.4:
                names.append(f'{p.full_name} ({p.position}, {p.report_status})')
        return float(total), names

    # ---- prediction ----
    def margin(self, rating_diff, wk, neutral, qb_change, players_out, rest_diff):
        slope = next(self.coef[f'd{i}'] for i, (lo, hi) in enumerate(self.buckets) if lo <= wk <= hi)
        m = slope * rating_diff + (0.0 if neutral else self.coef['home_field'])
        m += self.coef.get('qb_change', 0.0) * qb_change
        m += self.coef.get('inj_all', 0.0) * players_out
        m += self.coef.get('rest_diff', 0.0) * float(np.clip(rest_diff, -7, 7))
        return float(m)
