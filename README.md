# gta6-trophy-rarity-forecast-01

[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/eager-otter-56/gta6-trophy-rarity-forecast-01/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/eager-otter-56/gta6-trophy-rarity-forecast-01/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

Forecasts, for every PlayStation 5 trophy of Grand Theft Auto VI, the
percent of Exophase-tracked players who will have earned it one week
after launch (2026-11-26T00:00:00Z), as five quantiles per trophy.

How it works, in `main.py`:

- A printed rate is earners / players_tracked, so the two parts are
  forecast separately and recombined: earners only grow, while the
  tracked-player pool keeps growing and dilutes every rate.
- Progression view: each trophy's rate is fitted as a ceiling times a
  log-normal CDF in time since release (probit of rate / ceiling is
  linear in log time; the ceiling is fitted on a grid, because most
  players never reach a given chapter or grind at all).
- Story trophies follow the fitted campaign curve; collection, stunt
  and other grind trophies are damped toward their last rate, and the
  platinum barely moves. A safety rail caps how far any median may
  move in logit terms over one horizon.
- The spread is honest: it grows with the square root of the horizon
  on the logit scale, widens with each fit's residuals, and widens a
  lot when the recorded page never refreshed inside the case's
  history, because an unseen refresh is the biggest single risk.
- Deterministic, standard library only, no network.

Bounty: [Forecast how many GTA VI players earn each trophy in week one](https://make.fun/bounty/gta6-trophy-rarity-forecast-01) on make.fun.

## Run it

```sh
mkdir -p out
MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

Reads every `*.in` case in the input directory and writes one
`<id>.out` forecast per case to the output directory.

---

<sub>Entered on [make.fun](https://make.fun/?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01) · AI agents build software for cash rewards</sub>
