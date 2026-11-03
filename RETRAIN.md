# Retraining playbook

Run this after each regular season ends, once the last Tuesday update has gone out. The goal is to use the season's log to decide whether the method should change. Nothing goes live without Thomas's OK: do all the work on a branch and report back.

## What the log holds

`log/<season>/` is append-only. Each Tuesday's update adds files and never overwrites one.

- **`week_NN_picks.json`**: every game of week NN as the model saw it before kickoff.
  - The ratings used, the predicted margin and win chance.
  - The Vegas spread, total and moneylines at logging time.
  - The expected starting quarterbacks and each team's usual starter.
  - Rest days, roof, division game, and the model version (a hash of the code and parameters).
  - `logged_live: false` marks a row rebuilt afterwards rather than logged before the game. All of 2026 Weeks 1–8 were rebuilt: same pre-game ratings, but closing lines and actual starters.
- **`after_week_NN_cards.json`**: the report cards exactly as published that week, verdicts included. Use these to check how each verdict held up in real time, not just in back-tests.

Final scores, closing lines and play-by-play come from nflverse at evaluation time.

## Steps

All commands run from the repo root unless noted.

1. **Data.** `python3 -I pipeline/refresh.py pipeline <season>` downloads play-by-play for the six seasons ending with `<season>`, plus the schedule.
2. **Scorecard.**
   ```
   NFL_DATA=pipeline/data python3 -I pipeline/evaluate.py pipeline log <season> --out research/<season>_scorecard.md
   ```
   Run it again with `--live-only` for games that were logged before kickoff.
3. **Read the leads.** The scorecard points at what to test:
   - **Home-field bias:** a non-zero "actual minus predicted home margin".
   - **Calibration:** dots off the line means the conversion from rating to win chance needs a new spread.
   - **Quarterback-change games:** misses far above other games means the model needs to know who is starting.
   - **Line movement:** did lines move toward the model? If so, its disagreements carry information.
   - **Early-season weeks:** whether the preseason starting point carries too much or too little weight.
4. **A/B sweep with the new season added.** From `pipeline/`:
   ```
   NFL_DATA=data NFL_TUNE=2022,...,<season> NFL_TAG=_<season> python3 -I ab.py out
   ```
   This writes `out/champion_<season>.json` and `out/ab_log_<season>.csv`. The acceptance rule is unchanged: a change must win at least 90% of bootstrap resamples, and simpler wins ties.
   - Add new candidates to `CANDIDATES` in `ab.py` when the log supports them. Examples: a smaller home-field value, or a quarterback-change adjustment (`qb_filter` is an existing starting point).
   - Keep the season you are judging out of anything you tune on when you report its accuracy.
5. **Refit what depends on the method.**
   - Luck curve: `python3 -I ledger.py out champion_<season>.json`. This writes `ledger_params.json`.
   - Preseason starting point: `NFL_DATA=data NFL_TUNE=2022,...,<season> python3 -I fit_prior.py`. This writes `prior_params.json` and `proj_backtest.csv`.
   - Split-half stability, for the warning flags: `python3 -I stability.py out`.
6. **Verdict check.** For each verdict in `after_week_08_cards.json` / `after_week_09_cards.json`, compare those teams' win percentage before and after, and their playoff results.
7. **Report.** Work on branch `retrain-<next season>`. Write `research/retrain_<season>.md`: what was tested, what won and by how much, what changed in the verdict rules, and how the new method would have done on the season just finished. Push the branch and summarize for Thomas. **Do not merge to main.**
8. **When Thomas approves.**
   - Replace `pipeline/out/method.json` (and `ledger_params.json`, `prior_params.json` and `proj_backtest.csv` if refit). The model version hash changes automatically.
   - Change the season in the weekly scheduled task to the new one.
