# Model scorecard (150 finished games; 29 logged live before kickoff, 1 of them re-picked within hours of kickoff)

| | Winners picked | Average miss (pts) | Bias, actual minus predicted home margin |
|---|---|---|---|
| Model | 65.0% | 10.03 | -2.52 |
| Vegas line when logged | 68.3% | 9.56 | -1.89 |
| Vegas closing line | 68.3% | 9.57 | -1.89 |

A positive bias means home teams did better than predicted (home-field advantage set too low).

## Line movement (live picks only, 29 games)

The line moved in 21 games; it moved toward the model in 7 of them.

## Calibration: when the model said X%, how often did its pick win?

| Model win chance | Games | Said | Won |
|---|---|---|---|
| 50-55% | 39 | 52% | 65% |
| 55-60% | 44 | 58% | 66% |
| 60-65% | 31 | 62% | 48% |
| 65-70% | 21 | 68% | 71% |
| 70-80% | 12 | 75% | 83% |
| 80%+ | 3 | 82% | 100% |

Brier score (lower is better): model 0.2291, Vegas line converted at sd 13.5 0.2142.

## Where the model and Vegas disagreed

| Gap between model and line | Games | Model closer | Vegas closer |
|---|---|---|---|
| 0-2 pts | 63 | 22 | 41 |
| 2-4 pts | 54 | 25 | 29 |
| 4-7 pts | 30 | 10 | 20 |
| 7+ pts | 3 | 2 | 1 |

## Games with a quarterback change (31 games)

Model picked 47% with average miss 13.1; Vegas 53% and 12.6. Other games: model 9.2, Vegas 8.8.
 In games where only one team changed QB, that team did +1.7 points worse than the model predicted.

## Tuesday pick vs. pre-kickoff pick (1 games)

Average miss: Tuesday 18.00, pre-kickoff 13.30; picks changed sides in 0 games.

## By model version

| Version | Games | Model right | Model miss |
|---|---|---|---|
| 47e8113beede | 135 | 65% | 9.9 |
| 5cbc78af498d | 15 | 67% | 11.4 |

## By week

| Week | Games | Model right | Vegas right | Model miss | Vegas miss |
|---|---|---|---|---|---|
| 2026 W1 | 16 | 69% | 75% | 12.4 | 11.3 |
| 2026 W2 | 16 | 56% | 69% | 12.9 | 11.8 |
| 2026 W3 | 16 | 69% | 56% | 7.8 | 8.1 |
| 2026 W4 | 16 | 75% | 56% | 6.0 | 7.3 |
| 2026 W5 | 15 | 63% | 63% | 10.9 | 10.4 |
| 2026 W6 | 14 | 50% | 57% | 6.8 | 6.6 |
| 2026 W7 | 14 | 50% | 71% | 12.0 | 10.1 |
| 2026 W8 | 14 | 71% | 79% | 10.7 | 10.8 |
| 2026 W9 | 15 | 80% | 93% | 9.7 | 8.3 |
| 2026 W10 | 14 | 64% | 64% | 11.3 | 10.8 |
