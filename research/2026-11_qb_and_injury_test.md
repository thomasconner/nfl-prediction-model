# Quarterback-change and injury-report test (November 3, 2026)

**Question:** the model misses most when a team starts a different quarterback or is missing starters. Would adding that information help?

**Setup:**
- The base is the live model's pre-game prediction for every regular-season game, weeks 2–18.
- Coefficients were fit on 2022–2024 and graded on 2025 and 2026 Weeks 1–8. Neither of those was used for fitting.
- Scripts: `pipeline/qb_adjust.py` and `pipeline/injury_adjust.py`. Data comes from `refresh.py --extra`.

## Quarterback change

The add-on:

```
adjustment = c × (expected starter's EPA/dropback − team's EPA/dropback so far) × 35.7 dropbacks per game
```

It applies only when the starter isn't the team's usual starter so far this season. The starter's value is his last 3 seasons plus this one, shrunk toward replacement level (−0.079 EPA/dropback) with 300 dropbacks of prior. c = 0.33.

Applying it to every game, as "regress each QB to his track record", did nothing: c ≈ 0. This season's play already captures that.

| Test games | Live RMSE | With QB change | Chance it's better (bootstrap) | Vegas |
|---|---|---|---|---|
| 2025 + 2026 (393) | 12.98 | 12.89 | 93% | 12.24 |
| QB-change games only (123) | 14.78 | 14.55 | 93% | 13.62 |
| Other games (270) | 12.06 | 12.06 | – | 11.56 |

Variants changed little:
- 150 or 600 dropbacks of prior instead of 300: almost no difference.
- Counting offseason changes (a team's first start against last season's main starter): about the same, 96–98%.

## Injury reports (non-quarterbacks)

Players listed Out or Doubtful are weighted by their share of the team's earlier snaps this season. Games they missed count as zero. The fit gives about 1 point per starter out: OL 0.7, WR/TE/RB 1.6, defense 1.0.

| Test games (393) | Live | + QB change | + QB change + injuries | Vegas |
|---|---|---|---|---|
| RMSE | 12.98 | 12.89 | 12.87 | 12.24 |
| Winners picked | 63.4% | 63.2% | 64.7% | 65.5% |

The injuries add-on beats QB change alone in only 65% of bootstrap resamples, which is not enough evidence yet. Together the two beat the live model in 88%.

## Takeaways

- **The quarterback-change adjustment clears the 90% bar on held-out seasons.** The gain is in games where the starter changes.
- **Injuries help pick winners, but the accuracy gain is small and not yet significant.** Keep logging and retest in January.
- **Together they close about 15% of the gap to Vegas.** The market still knows more: practice reports, severity, matchups and late news.
- **Timing:** final injury designations come out later in the week. Using them live needs a second weekly run before Sunday, not just Tuesday.
