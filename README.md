# NFL Report Cards

**Site:** https://thomasconner.github.io/nfl-report-cards/

Is an NFL team actually good, or is its record lying? Each team is graded on every play, not just its win-loss record.

- **Strength** is net EPA per play: what each snap adds on offense, minus what the defense allows. It is adjusted for opponents and shown as points per game above or below an average team.
- **Luck** is a team's wins minus the wins a team of its strength usually gets.
- **Projections** combine the record so far with each remaining game's win chance. A preseason starting point counts early in the season and fades out by Week 8.

Built with Claude from [nflverse](https://github.com/nflverse) data. For fun, not betting advice.

## Layout

- `/` is the static site: `index.html`, the `data/*.json` files it loads, and a vendored `d3.min.js` (v7.9.0, ISC license).
- `pipeline/` holds the code that builds the data and the page.
  - `refresh.py <pipeline_dir> <season>` downloads nflverse play-by-play and schedules, rebuilds the ratings, and writes `out/dash_<season>/`.
  - `build_site.py <pipeline_dir> <season> <site_dir>` writes the site.

## Weekly update

A scheduled job runs every Tuesday afternoon (Eastern) during the regular season:

```
python3 -I pipeline/refresh.py pipeline 2026
python3 -I pipeline/build_site.py pipeline 2026 .
```

It then commits the new `data/` files and pushes.
