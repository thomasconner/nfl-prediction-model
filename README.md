# NFL Report Cards

**Site:** https://thomasconner.github.io/nfl-report-cards/

Is an NFL team actually good, or is its record lying? Each team is graded on every play, not just its win-loss record.

- **Strength** is net EPA per play: what each snap adds on offense, minus what the defense allows. It is adjusted for opponents and shown as points per game above or below an average team.
- **Luck** is a team's wins minus the wins a team of its strength usually gets.
- **Projections** combine the record so far with each remaining game's win chance. A preseason starting point counts early in the season and fades out by Week 8.

Built with Claude from [nflverse](https://github.com/nflverse) data. For fun, not betting advice.

## Layout

- `/` is the static site: `index.html`, the `data/*.json` files it loads, and a vendored `d3.min.js` (v7.9.0, ISC license).
- `log/<season>/` is the permanent record. It is added to every Tuesday and never rewritten. Each week it stores the picks as made before kickoff, with that day's Vegas line and the expected starting quarterbacks, plus the report cards as published. The model is graded on this log and retrained from it; see `RETRAIN.md`.
- `pipeline/` holds the code that builds the data and the page.
  - `refresh.py <pipeline_dir> <season>` downloads nflverse play-by-play and schedules, rebuilds the ratings, and writes `out/dash_<season>/`.
  - `build_site.py <pipeline_dir> <season> <site_dir>` writes the site.
  - `log_week.py <pipeline_dir> <season> <log_dir>` adds this week's entries to the log.
  - `evaluate.py <pipeline_dir> <log_dir> [season]` prints the scorecard: winners picked vs. Vegas, calibration, home-field bias, quarterback-change games.
  - `ab.py`, `ledger.py`, `fit_prior.py`, `stability.py`, `holdout.py` are the research tools used to choose and tune the method (`out/method.json`).

## Weekly update

A scheduled job runs every Tuesday afternoon (Eastern) during the regular season:

```
python3 -I pipeline/refresh.py pipeline 2026
NFL_DATA=pipeline/data python3 -I pipeline/log_week.py pipeline 2026 log
python3 -I pipeline/build_site.py pipeline 2026 .
```

It then commits the new `data/` and `log/` files and pushes.
