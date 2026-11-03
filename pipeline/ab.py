"""Iterative A/B testing of rating methods on 2022-2024 (leave-one-season-out).

Rule: one change per round. B replaces A if B wins >=90% of bootstrap resamples.
If B is simpler, B replaces A unless A wins >=90% (ties go to the simpler method).
Sweeps repeat until a full sweep accepts no change.
"""
import sys, json, os
sys.path.insert(0, os.path.dirname(__file__))
import pandas as pd
import model as M

OFF = dict(opp_adj=0.0, wp=(0.0, 1.0), excl_2min=False, fumble_neutral=False, int_def=1.0, int_off=1.0,
           def_w=1.0, rush_w=1.0, sr_blend=0.0, down34_w=1.0, clip=None, st_w=0.0, half_life=None,
           prior_k=0.0, pd_blend=0.0, qb_filter=False, pass_only=False, rec_blend=0.0)
CANDIDATES = [
    ('opp_adj', [0.0, 0.5, 1.0], 'opponent adjustment'),
    ('wp', [(0.0, 1.0), (0.02, 0.98), (0.05, 0.95), (0.10, 0.90)], 'garbage-time filter (win prob range)'),
    ('excl_2min', [False, True], 'drop last 2 minutes of each half'),
    ('fumble_neutral', [False, True], 'fumble-recovery luck removed'),
    ('int_def', [1.0, 0.5, 0.0], 'defensive interceptions regressed to league rate'),
    ('int_off', [1.0, 0.5], 'interceptions thrown regressed to league rate'),
    ('def_w', [1.0, 0.8, 0.6], 'defense weighted less than offense'),
    ('rush_w', [1.0, 0.6, 0.3], 'rushing results shrunk toward league average'),
    ('sr_blend', [0.0, 0.25, 0.5], 'success rate blended in'),
    ('down34_w', [1.0, 0.6], '3rd/4th-down plays down-weighted'),
    ('clip', [None, 5.0, 3.0], 'big plays capped (EPA per play)'),
    ('st_w', [0.0, 0.5, 1.0], 'special teams added'),
    ('half_life', [None, 12.0, 6.0], 'recent games weighted more (half-life, weeks)'),
    ('prior_k', [0.0, 3.0, 6.0, 10.0], "last season's rating as a prior (weight in games)"),
    ('prior_rho', [0.6, 0.4, 0.8], 'share of last season carried over'),
    ('pd_blend', [0.0, 0.25, 0.5], 'actual point differential blended in'),
    ('qb_filter', [False, True], "offense judged only on the current starting QB's games"),
    ('pass_only', [False, True], 'passing plays only (offense and defense)'),
    ('rec_blend', [0.0, 0.25, 0.5, 0.75], 'win-loss record blended in'),
]


def get(cfg, p):
    return (cfg['wp_lo'], cfg['wp_hi']) if p == 'wp' else cfg.get(p, OFF.get(p))


def put(cfg, p, v):
    c = dict(cfg)
    if p == 'wp':
        c['wp_lo'], c['wp_hi'] = v
    else:
        c[p] = v
    return c


def complexity(cfg):
    n = sum(get(cfg, p) != v for p, v in OFF.items())
    return n


_scores = {}


def run(cfg):
    k = json.dumps(cfg, sort_keys=True)
    if k not in _scores:
        _scores[k] = M.score(M.method(cfg))
    return _scores[k]


def main(out_dir):
    champ = dict(M.A0)
    log = []
    rnd = 0
    sweep = 0
    while True:
        sweep += 1
        changed = False
        for p, vals, label in CANDIDATES:
            if p == 'prior_rho' and champ['prior_k'] == 0:
                continue
            for v in vals:
                if v == get(champ, p):
                    continue
                chal = put(champ, p, v)
                rnd += 1
                a, b = run(champ), run(chal)
                c = M.compare(a, b)
                ca, cb = complexity(champ), complexity(chal)
                kind = 'simpler' if cb < ca else ('more complex' if cb > ca else 'same complexity')
                if kind == 'simpler':
                    accept = c['p_b_better'] > 0.10
                else:
                    accept = c['p_b_better'] >= 0.90
                sa, sb = M.summarize(a), M.summarize(b)
                log.append(dict(sweep=sweep, round=rnd, param=p, change=label, from_val=str(get(champ, p)),
                                to_val=str(v), kind=kind, rmse_a=sa['rmse'], rmse_b=sb['rmse'],
                                rmse_gain=c['rmse_gain'], p_b_better=c['p_b_better'],
                                su_a=sa['su'], su_b=sb['su'], decision='B wins' if accept else 'A stays'))
                print(f"s{sweep} r{rnd:>3} {p:>14} {str(get(champ, p)):>12} -> {str(v):<12} {kind:<15} "
                      f"A {sa['rmse']:.3f} B {sb['rmse']:.3f} gain {c['rmse_gain']:+.3f} P(B) {c['p_b_better']:.2f} "
                      f"{'B WINS' if accept else ''}", flush=True)
                if accept:
                    champ = chal
                    changed = True
                    break  # move to next parameter with the new champion
        if not changed or sweep >= 6:
            break
    tag = os.environ.get('NFL_TAG', '')
    pd.DataFrame(log).to_csv(f'{out_dir}/ab_log{tag}.csv', index=False)
    with open(f'{out_dir}/champion{tag}.json', 'w') as f:
        json.dump(champ, f, indent=1)
    print('CHAMPION', champ, M.summarize(run(champ)))


if __name__ == '__main__':
    os.makedirs(sys.argv[1], exist_ok=True)
    main(sys.argv[1])
