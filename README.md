# World Cup 2026 Prediction Model

A Poisson Monte-Carlo match simulator for the 48 teams of the 2026 FIFA World Cup.
Each team has an **attack** and **defense** rating out of 100, built from a
three-source ensemble. Any matchup is simulated tens of thousands of times to
produce win/draw/loss probabilities, expected goals, and the likeliest scorelines.

## Files
| File | What it is |
|------|------------|
| `results.csv` | Raw data: ~49,400 international matches, 1872 → Jun 2026 (source: martj42/international_results) |
| `wc2026_xg.csv` | Per-match expected goals (xG) for the 2026 World Cup; blended into recent form. |
| `xg_report.py` | Shows which teams have over/under-performed their xG so far this tournament. |
| `build_ratings.py` | Builds the ratings from the three sources. Run once (or whenever you refresh `results.csv`). |
| `ratings.csv` | The output: 48 teams with attack/defense scores + the model parameters the simulator reads. |
| `build_context.py` | Builds per-team static context (population, home climate/altitude). Run once. |
| `context.csv` | The output: static attributes used by the match-day modifier layer. |
| `simulate.py` | Runs a Monte-Carlo simulation of any matchup, with optional match-day modifiers. |
| `analyze_stakes.py` | Measures the dead-rubber/clinched effect from 304 past WC/Euro final-group games (calibrates the stakes multipliers). |
| `backtest.py` | Walk-forward backtest + calibration over past WC/Euro group stages; tunes half-life, goals/Elo blend, and DC rho. |
| `groups.py` | Reconstructs the live 2026 groups + standings and auto-detects each team's situation (clinched / eliminated / live). Used by `--auto`. |
| `situations.py` | Prints the whole tournament landscape: standings + every remaining match's detected situation. |
| `recent.py` | Lists recent matches in the dataset so you can see how results have gone so far. |
| `explore.py` | One-off data-coverage check (not needed for normal use). |
| `export_web.py` | Bakes the live model into `web/data.js` for the browser app. |
| `web/` | The shareable web app (index.html, app.js, data.js, manifest, icon). |

## Staying current (run before each new match)
The match dataset is community-updated within ~a day of every game, and the builder
auto-skips not-yet-played fixtures (`NA` scores), so ratings always reflect only
completed matches. Before simulating an upcoming game:
```bash
python build_ratings.py --refresh   # re-pull latest results + xG, rebuild ratings
python recent.py 2026-06-15          # (optional) eyeball results so far
```
Then the *situational* facts (lineups, injuries, suspensions, who needs the result,
venue, forecast, rest days) are looked up live for that specific fixture and passed
as the flags below. Results themselves come from the refresh; only the human-judgment
inputs are researched per match.

## How the ratings are built (the "all three sources" ensemble)

Only one source can actually tell attack apart from defense, so the sources play
different roles:

1. **Goals-based model (Source A)** — an iterative, opponent- and recency-adjusted
   fit (2-year half-life, tuned) of how many goals each team scores and concedes vs an
   average opponent, which provides each team's **attack-vs-defense tilt**. Current-
   tournament matches use **xG-blended goals** (60% xG / 40% actual) so finishing
   luck doesn't distort current form.
2. **Elo (Source B)** — computed from the entire match history (1872→2026) with a
   margin-of-victory K-factor. An overall **strength** measure.
3. **FIFA ranking points (Source C)** — official June 2026 points. A second overall
   **strength** measure. (Curaçao, Haiti, New Zealand rank below the top-80 source
   and are estimated from rank — flagged `fifa_estimated=True`.)

**Blend:** each team's overall strength = a z-score average of {goals strength, Elo,
FIFA} (≈ equal weights). That consensus strength is then split into final attack and
defense using the goals model's tilt. The result is stored two ways:
- `attack_100` / `defense_100` — human-readable scores out of 100 (logistic scaled).
- `attack_mult` / `defense_mult` — the rate multipliers the simulator actually uses
  (`defense_mult < 1` = better defense).

**Home advantage** = `1.30` (host scores ~30% more goals). It can't be cleanly
separated from team strength in this data, so it uses the standard football-modeling
value and is applied **only** to a designated host team. Tune `HOME_ADV` in
`build_ratings.py` if you want.

## The simulation math
For each matchup the expected goals (Poisson means) are:
```
lambda_A = league_avg * attack_mult[A] * defense_mult[B] * home_factor_A
lambda_B = league_avg * attack_mult[B] * defense_mult[A] * home_factor_B
```
Goals follow a Poisson with a **Dixon-Coles low-score correction** (rho = -0.12, tuned),
which fixes the 0-0 / 1-0 / 0-1 / 1-1 probabilities that plain independent Poisson gets
wrong. The full score matrix is computed **exactly** (no sampling noise) to give win/draw/loss
%, expected goals, and scoreline probabilities.

## Accuracy (backtested — `backtest.py`)
Walk-forward over **372 group matches** from 9 past World Cups/Euros (ratings built only from
data predating each tournament):
- **Log-loss 0.98** vs **1.08** for a base-rate baseline (lower = better; ~0.98 is competitive with bookmakers).
- **~55%** outcome accuracy (3-way, draw-heavy sport).
- **Well calibrated** (predicted 25% -> actual 25%, etc.), with mild overconfidence above 80% — favourites get upset slightly more than modelled.

## Knockout matches (`--knockout`, or auto-detected)
Knockouts can't end level, so a level 90-minute result continues into **extra time**
(modelled as a 30-min mini-match at 1/3 the scoring rate) and then a **penalty shootout**
(near coin-flip with a tiny favourite edge). The output becomes a **"% to advance"** plus
the chance the tie reaches extra time / penalties. With `--auto`, the stage is detected from
live standings: same-group pairings are GROUP; once the group stage finishes, cross-group
pairings are KNOCKOUT; if the bracket isn't set yet it reports UNKNOWN rather than guess
(pass `--knockout` to force a hypothetical knockout).

## Finding betting edges (`--odds A DRAW B`)
The model is **independent of the market by design** (that's the point — to find value). Pass the
decimal odds and it de-vigs them, compares to the model, and flags positive-EV outcomes with a
quarter-Kelly stake. Example: `python simulate.py Norway Senegal --auto --odds 2.15 3.40 3.25`.

## Match-day modifiers (the "secondary factors")
Structural factors (form, long-term strength) are already in the base ratings via
recency-weighting + Elo. *Situational* factors are not in any dataset, so they are
optional, transparent, tunable knobs supplied per match. Each is a multiplier on a
team's attacking output (`am`) and/or defensive solidity (`dm`); `dm > 1` means
defends better, which lowers the opponent's goals. Magnitudes are constants at the
top of `simulate.py`.

| Factor | Flag | Effect |
|---|---|---|
| Injuries / rotation / fitness | `--avail-a` / `--avail-b` (0.5–1.0) | scales whole-team performance (1.0 = full strength) |
| Fatigue (congestion) | `--rest-a` / `--rest-b` (days) | <4 days rest applies a penalty |
| Stakes (empirical) | `--stakes-a` / `--stakes-b` (`clinched`/`eliminated`/`must-win`/`normal`) | measured from 304 past games: already-qualified teams ease off (0.93×); eliminated & must-win teams play to strength (1.0×) |
| Climate / altitude | `--venue-temp` (°C) / `--venue-altitude` (m) | penalises a team playing materially hotter/higher than home; auto-set to host's conditions with `--home` |
| Weather on the day | `--weather` (`clear`/`rain`/`cold`/`heat`) | suppresses total goals |

**Population** is shown for context but **not** weighted (the ratings already
capture realized talent; weighting population would penalise overperformers).

## Usage
```bash
# build data (only once, or after refreshing results.csv)
python build_ratings.py
python build_context.py

# see the live group landscape + every team's detected situation
python situations.py

# AUTO mode: detect stage + host + group stakes (clinched/resting) from live standings
python simulate.py "United States" "Turkey" --auto
python simulate.py "Mexico" "Czech Republic" --auto

# knockout tie: extra time + penalties -> who advances
python simulate.py "Spain" "France" --knockout

# plain match (neutral venue, 50k sims)
python simulate.py "Brazil" "Argentina"

# manual overrides still work and beat --auto:
# France missing players + on 2 days' rest, 36C heat
python simulate.py France Senegal --avail-a 0.8 --rest-a 2 --weather heat --venue-temp 36
```
Team names are case- and accent-insensitive and accept common aliases (USA, Korea, NZ…).
When you give me a real fixture, I research the actual situational values (injuries,
venue altitude, forecast, who needs the points) and set these flags for you.

## Web app (`web/`)
A self-contained browser app with four screens — **Predict** (auto stage/stakes, factor
toggles, betting-edge flag), **Groups** (live standings + situations), **Fixtures**
(upcoming matches auto-predicted), and **Title odds** (an 8,000-sim bracket projection).
The prediction math is ported to JavaScript and verified to match the Python model exactly,
so it runs entirely in the browser — no server.

```bash
python build_ratings.py --refresh   # 1. refresh data + ratings
python export_web.py                 # 2. bake into web/data.js
```
Then **open `web/index.html`** (double-click — works offline) or host the `web/` folder.
To share with friends: push `web/` to a free **GitHub Pages** / **Netlify** site and send the
link; on a phone, "Add to Home Screen" installs it like an app. Re-run the two commands above
to update the numbers.

### Hands-off auto-update
For a site that refreshes itself, host on **GitHub Pages** with the included
`.github/workflows/update.yml`: a scheduled job reruns the build twice daily from the latest
data and republishes — even with your computer off. The scripts are path-portable for this.
Full one-time setup is in **SETUP-AUTOUPDATE.md**.

## Caveats
- Ratings reflect form/results **through 21 June 2026** and don't know about same-tournament
  injuries, suspensions, or lineup rotation.
- The model predicts **90-minute** results; it does not resolve knockout draws via
  extra time / penalties (draw probability is reported separately).
- FIFA points for 3 minnows are estimated; their ratings lean on the goals model and Elo.
