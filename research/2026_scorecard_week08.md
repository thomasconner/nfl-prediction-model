# Model scorecard (121 finished games; 0 logged live before kickoff)

| | Winners picked | Average miss (pts) | Bias, actual minus predicted home margin |
|---|---|---|---|
| Model | 63.2% | 9.93 | -2.04 |
| Vegas line when logged | 65.7% | 9.57 | -1.33 |
| Vegas closing line | 65.7% | 9.57 | -1.33 |

A positive bias means home teams did better than predicted (home-field advantage set too low).

## Calibration: when the model said X%, how often did its pick win?

| Model win chance | Games | Said | Won |
|---|---|---|---|
| 50-55% | 32 | 52% | 70% |
| 55-60% | 37 | 58% | 65% |
| 60-65% | 25 | 62% | 36% |
| 65-70% | 16 | 68% | 69% |
| 70-80% | 8 | 74% | 88% |
| 80%+ | 3 | 82% | 100% |

Brier score (lower is better): model 0.2361, Vegas line converted at sd 13.5 0.2242.

## Where the model and Vegas disagreed

| Gap between model and line | Games | Model closer | Vegas closer |
|---|---|---|---|
| 0-2 pts | 50 | 15 | 35 |
| 2-4 pts | 44 | 20 | 24 |
| 4-7 pts | 24 | 10 | 14 |
| 7+ pts | 3 | 2 | 1 |

## Games with a quarterback change (24 games)

Model picked 40% with average miss 14.1; Vegas 48% and 13.9. Other games: model 8.9, Vegas 8.5.
 In games where only one team changed QB, that team did +0.2 points worse than the model predicted.

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
