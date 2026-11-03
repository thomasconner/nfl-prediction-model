"""Scorecard from the permanent log: how the model's pre-game picks did against Vegas and reality.

Usage: python3 -I evaluate.py <pipeline_dir> <log_dir> [season ...] [--live-only] [--out report.md]
Joins every log/<season>/week_NN_picks.json with final scores from <pipeline_dir>/data/games.csv
(NFL_DATA overrides the data folder) and prints a markdown report: winners picked, average miss,
the Tuesday line vs. the closing line, calibration, home-field bias, and games with a quarterback change.
"""
import sys, os, json, glob
import numpy as np
import pandas as pd
from scipy.stats import norm

args = [a for a in sys.argv[1:] if not a.startswith('--')]
work, log_dir = os.path.abspath(args[0]), os.path.abspath(args[1])
seasons = [int(s) for s in args[2:] if s.isdigit()]
live_only = '--live-only' in sys.argv
out_path = sys.argv[sys.argv.index('--out') + 1] if '--out' in sys.argv else None
data = os.environ.get('NFL_DATA', f'{work}/data')
VEGAS_SD = 13.5  # spread to win chance (normal), the usual rule of thumb for NFL margins

rows = []
for f in sorted(glob.glob(f'{log_dir}/*/week_*_picks.json') + glob.glob(f'{log_dir}/*/week_*_pregame_*.json')):
    rows += json.load(open(f))
ALL = pd.DataFrame(rows)
ALL['stage'] = ALL.get('stage', pd.Series(index=ALL.index, dtype=object))
ALL['stage'] = ALL.stage.fillna(ALL.logged_live.map({True: 'tuesday', False: 'rebuilt'}))
# the official pick is the newest one logged before kickoff
L = ALL.sort_values('logged_at').groupby('game_id').tail(1).copy()
if seasons:
    L = L[L.season.isin(seasons)]
if live_only:
    L = L[L.logged_live]
g = pd.read_csv(f'{data}/games.csv').set_index('game_id')
L = L.join(g[['result', 'spread_line']].rename(columns={'spread_line': 'closing_spread_home'}), on='game_id')
L = L[L.result.notna()].copy()
lines = []
w = lines.append
if L.empty:
    w('No finished games in the log yet.')
else:
    def right(pred, res):
        return np.where(res == 0, 0.5, np.where(np.sign(pred) == np.sign(res), 1.0, 0.0))
    L['model_right'] = right(L.model_margin_home, L.result)
    L['vegas_right'] = np.where(L.vegas_spread_home.fillna(0) == 0, np.nan, right(L.vegas_spread_home, L.result))
    L['close_right'] = np.where(L.closing_spread_home.fillna(0) == 0, np.nan, right(L.closing_spread_home, L.result))
    L['model_err'] = L.result - L.model_margin_home
    L['vegas_err'] = L.result - L.vegas_spread_home
    L['close_err'] = L.result - L.closing_spread_home
    L['home_won'] = np.where(L.result == 0, 0.5, (L.result > 0).astype(float))
    w(f'# Model scorecard ({len(L)} finished games; {int(L.logged_live.sum())} logged live before kickoff, '
      f"{int((L.stage == 'pregame').sum())} of them re-picked within hours of kickoff)\n')
    w('| | Winners picked | Average miss (pts) | Bias, actual minus predicted home margin |')
    w('|---|---|---|---|')
    w(f"| Model | {L.model_right.mean():.1%} | {L.model_err.abs().mean():.2f} | {L.model_err.mean():+.2f} |")
    w(f"| Vegas line when logged | {np.nanmean(L.vegas_right):.1%} | {L.vegas_err.abs().mean():.2f} | {L.vegas_err.mean():+.2f} |")
    w(f"| Vegas closing line | {np.nanmean(L.close_right):.1%} | {L.close_err.abs().mean():.2f} | {L.close_err.mean():+.2f} |")
    w('\nA positive bias means home teams did better than predicted (home-field advantage set too low).\n')

    live = L[L.logged_live & L.vegas_spread_home.notna() & L.closing_spread_home.notna()]
    if len(live):
        mv = live.closing_spread_home - live.vegas_spread_home
        toward = np.sign(mv) == np.sign(live.model_margin_home - live.vegas_spread_home)
        moved = mv != 0
        w(f'## Line movement (live picks only, {len(live)} games)\n')
        w(f'The line moved in {int(moved.sum())} games; it moved toward the model in {int((toward & moved).sum())} of them.\n')

    w('## Calibration: when the model said X%, how often did its pick win?\n')
    w('| Model win chance | Games | Said | Won |')
    w('|---|---|---|---|')
    p = np.maximum(L.model_home_win_prob, 1 - L.model_home_win_prob)
    bins = [0.5, 0.55, 0.6, 0.65, 0.7, 0.8, 1.0001]
    lab = ['50-55%', '55-60%', '60-65%', '65-70%', '70-80%', '80%+']
    L['bucket'] = pd.cut(p, bins, labels=lab, right=False)
    for b, d in L.groupby('bucket', observed=True):
        pp = np.maximum(d.model_home_win_prob, 1 - d.model_home_win_prob)
        w(f'| {b} | {len(d)} | {pp.mean():.0%} | {d.model_right.mean():.0%} |')
    brier_m = ((L.model_home_win_prob - L.home_won) ** 2).mean()
    vp = norm.cdf(L.vegas_spread_home / VEGAS_SD)
    brier_v = np.nanmean((vp - L.home_won) ** 2)
    w(f'\nBrier score (lower is better): model {brier_m:.4f}, Vegas line converted at sd {VEGAS_SD} {brier_v:.4f}.\n')

    w('## Where the model and Vegas disagreed\n')
    w('| Gap between model and line | Games | Model closer | Vegas closer |')
    w('|---|---|---|---|')
    gap = (L.model_margin_home - L.vegas_spread_home).abs()
    L['gap_b'] = pd.cut(gap, [0, 2, 4, 7, 100], labels=['0-2', '2-4', '4-7', '7+'], right=False)
    for b, d in L.groupby('gap_b', observed=True):
        mc = (d.model_err.abs() < d.vegas_err.abs()).sum()
        vc = (d.model_err.abs() > d.vegas_err.abs()).sum()
        w(f'| {b} pts | {len(d)} | {mc} | {vc} |')

    qc = L[L.home_qb_change | L.away_qb_change]
    w(f'\n## Games with a quarterback change ({len(qc)} games)\n')
    if len(qc):
        w(f'Model picked {qc.model_right.mean():.0%} with average miss {qc.model_err.abs().mean():.1f}; '
          f'Vegas {np.nanmean(qc.vegas_right):.0%} and {qc.vegas_err.abs().mean():.1f}. '
          f'Other games: model {L.drop(qc.index).model_err.abs().mean():.1f}, Vegas {L.drop(qc.index).vegas_err.abs().mean():.1f}.')
        sgn = np.where(qc.home_qb_change & ~qc.away_qb_change, 1, np.where(qc.away_qb_change & ~qc.home_qb_change, -1, 0))
        one = sgn != 0
        if one.any():
            w(f' In games where only one team changed QB, that team did {(-(qc.model_err * sgn)[one]).mean():+.1f} points worse than the model predicted.')

    both = ALL[ALL.game_id.isin(L.game_id)].pivot_table(index='game_id', columns='stage', values='model_margin_home', aggfunc='last')
    if {'tuesday', 'pregame'} <= set(both.columns):
        b = both.dropna(subset=['tuesday', 'pregame']).join(L.set_index('game_id').result)
        if len(b):
            w(f"\n## Tuesday pick vs. pre-kickoff pick ({len(b)} games)\n")
            w(f"Average miss: Tuesday {(b.result - b.tuesday).abs().mean():.2f}, pre-kickoff {(b.result - b.pregame).abs().mean():.2f}; "
              f"picks changed sides in {int((np.sign(b.tuesday) != np.sign(b.pregame)).sum())} games.")
    if 'model_version' in L:
        w('\n## By model version\n')
        w('| Version | Games | Model right | Model miss |')
        w('|---|---|---|---|')
        for v_, d in L.groupby('model_version'):
            w(f'| {v_} | {len(d)} | {d.model_right.mean():.0%} | {d.model_err.abs().mean():.1f} |')
    w('\n## By week\n')
    w('| Week | Games | Model right | Vegas right | Model miss | Vegas miss |')
    w('|---|---|---|---|---|---|')
    for (s, wk), d in L.groupby(['season', 'week']):
        w(f'| {s} W{wk} | {len(d)} | {d.model_right.mean():.0%} | {np.nanmean(d.vegas_right):.0%} | {d.model_err.abs().mean():.1f} | {d.vegas_err.abs().mean():.1f} |')
report = '\n'.join(lines)
print(report)
if out_path:
    open(out_path, 'w').write(report + '\n')
