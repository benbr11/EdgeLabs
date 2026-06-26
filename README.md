# EdgeLabs — multi-sport prediction & betting-edge models

A suite of **independent, walk-forward prediction models** across six sports. Each one
builds team/fighter ratings from raw historical data, predicts win/loss (and goals,
scorelines, methods, or margins where the sport allows), and flags **betting value** by
comparing its own probabilities against de-vigged market odds.

> **The one rule everywhere: no look-ahead.** Every accuracy number below is *out-of-sample* —
> each historical game/fight is predicted using **only data that existed before it happened**.
> In-sample "accuracy" is worthless (it just memorizes the past); the backtest scripts here go
> out of their way to prevent leakage, and report numbers honestly even when they're modest.

The six models:

| Sport | Engine | Build → Predict → Backtest |
|---|---|---|
| ⚽ **World Cup 2026** | Poisson Monte-Carlo + Dixon-Coles, 3-source ensemble | `build_ratings.py` → `simulate.py` → `backtest.py` |
| 🥊 **UFC** | Opponent-quality Elo + stat profile + grappler premium; method/round model | `ufc_model.py` → `export_ufc.py` / `show_rankings.py` → `ufc_backtest.py` |
| 🏀 **NBA** | Opponent-adjusted off/def, Gaussian point-margin | `build_nba.py` → `blend_nba.py` → `backtest_nba.py` |
| 🏈 **NFL** | EPA→points, Gaussian point-margin (+ players, kickers) | `build_nfl.py` → `blend_nfl.py` → `backtest_nfl.py` |
| 🏒 **NHL** | Recency Poisson/Dixon-Coles att/def + MOV-Elo z-blend | `build_nhl.py` → `nhl_predict.py` → `backtest_nhl.py` |
| ⚾ **MLB** | Iterative run-rate att/def + starting-pitcher RA9 factor, Poisson | `build_mlb.py` → `blend_mlb.py` → `backtest_mlb.py` |
| ⚽ **Club leagues** | EPL/La Liga/Serie A/Bundesliga/Ligue 1; Poisson + MOV-Elo, Dixon-Coles 1X2 | `build_leagues.py` → `web/leagues_app.js` → `backtest_leagues.py` |

Plus a player/individual-stats layer (`build_player_*.py`, `backtest_players.py`,
`backtest_player_yoy.py`) and self-contained browser apps in `web/`.

---

## Accuracy scoreboard (out-of-sample, walk-forward)

These are the latest backtest figures. Each is **reproducible** by running the named script
(they re-fetch source data and re-derive ratings as-of each game date), and will shift slightly
as new results come in. The **market benchmark** is the closing-line implied accuracy — i.e. the
aggregate of every sharp bettor's opinion — so "at or above market" means *matching or beating
expert consensus*.

| Sport | OOS winner accuracy | Market benchmark | High-confidence tier | Reproduce |
|---|---|---|---|---|
| 🥊 UFC | **69.8%** | ~64% | **82.5%** on "Best Bets" (model ≥75%, ~¼ of fights) | `python ufc_backtest.py` |
| 🏀 NBA | **67.4%** | ~69% | 83.7% on ≥80%-confidence picks | `python backtest_nba.py` |
| 🏈 NFL | **65.9%** | ~66% | ~75% on ≥75%-confidence picks | `python backtest_nfl.py` |
| ⚾ MLB | **56.4%** | ~58–60% | 75% on ≥80%-confidence picks | `python backtest_mlb.py` |
| 🏒 NHL | **55.5%** | ~57–59% | 68.5% on ≥80%-confidence picks | `python backtest_nhl.py` |
| ⚽ Club leagues (1X2) | **52.1%** | ~50–55% (3-way) | 87.5% on ≥80%-confidence picks | `python backtest_leagues.py` |
| ⚽ World Cup (1X2) | **~55%** · log-loss **0.98** vs 1.08 baseline | bookmaker-competitive | — | `python backtest.py` |

**How to read this:**
- **No model "predicts almost all games."** The sharpest sportsbooks on earth hit ~64% in UFC,
  ~66% in the NFL, etc. A model claiming 85%+ on *all* games is leaking or overfit and loses live.
  The honest ceiling is *market level*, and several of these sit at or above it.
- **The 80%+ accuracy lives in the high-confidence tier**, not on every game — exactly like a sharp
  bettor only fires on their best spots. Those tiers are calibrated (when the model says 80%, that
  bucket really wins ~80%).
- **Calibration > raw accuracy.** Every model is checked bucket-by-bucket (predicted 25% → actual
  ~25%). A 56% MLB model that's perfectly calibrated is more useful than a flashier miscalibrated one.
- **It does not reliably beat point spreads** (the single hardest thing in betting). The edge is in
  straight-up picks, the high-confidence tiers, and value flags vs. the moneyline — not ATS.

---

## Shared design philosophy

1. **Walk-forward, no look-ahead** — ratings are rebuilt as-of each game's date; nothing from the
   game itself or any later game leaks in. The backtest scripts re-implement each model's exact
   rating math with a date cutoff and never touch the live ratings files.
2. **Market-independent by design** — the models never read the odds before predicting. That's the
   whole point: to find value, you have to disagree with the market for a reason.
3. **Decaying consensus priors** — the team-sport models (NBA/NFL/NHL/MLB) blend a preseason
   expert-consensus ranking that **fades to nothing** as real games accrue
   (`W = min(1, BASE + gamesPlayed/season_length)`), so early-season predictions aren't naked.
4. **Honest reporting** — modest numbers are reported as modest; leakage-flagged "optimistic"
   variants are labeled as such and never used as the headline.

---

## Sport-by-sport

### ⚽ World Cup 2026 (the original model)
Poisson Monte-Carlo simulator for the 48-team tournament. Each team gets attack/defense ratings
from a **three-source ensemble**:
- **Goals model** — iterative, opponent- and recency-adjusted (2-yr half-life); current-tournament
  matches use **xG-blended goals** (60% xG / 40% actual). Provides the attack-vs-defense *tilt*.
- **Elo** — full 1872→2026 match history, margin-of-victory K-factor. Overall strength.
- **FIFA ranking points** — official June 2026. Second strength measure.

Blended by z-score average, split into attack/defense by the goals tilt. Matches are simulated with
a **Dixon-Coles** low-score correction (ρ = −0.12); the full score matrix is computed exactly.
Supports knockout resolution (extra time + penalty shootout), live group/stakes auto-detection
(`--auto`), and per-match situational modifiers (injuries, rest, stakes, climate/altitude, weather).

```bash
python build_ratings.py --refresh          # pull latest results + xG, rebuild ratings
python situations.py                        # live group landscape + each team's situation
python simulate.py "Spain" "France" --knockout
python simulate.py "Brazil" "Argentina"     # plain neutral-venue match
python simulate.py Norway Senegal --auto --odds 2.15 3.40 3.25   # flag betting value
```
Situational flags: `--avail-a/b` (injuries 0.5–1.0), `--rest-a/b` (days), `--stakes-a/b`
(clinched/eliminated/must-win/normal), `--venue-temp`/`--venue-altitude`, `--weather`. Team names
are case- and accent-insensitive with aliases (USA, Korea, NZ…).

### 🥊 UFC
Performance-adjusted, dominance-scaled, **opponent-quality Elo** that is *robbery-aware* (a dominant
fighter who eats a bad decision keeps a high rating), combined with a striking/grappling stat profile
and a **grappler premium** gated to genuinely elite grapplers. A separate `method_round` model
produces P(goes the distance), the KO/Sub/Decision split, and a round distribution. Rankings are
**not** fed into the win probability (they'd be circular).
```bash
python ufc_model.py            # (re)build ufc_fighters.csv, ufc_fight_log.csv, ufc_ratings.csv
python show_rankings.py        # current pound-for-pound / per-division ratings
python ufc_backtest.py         # honest OOS backtest + vs-market + ROI on the odds overlap
python check_acceptance.py     # sanity checks (e.g. elites rank #1, protected robberies hold)
python export_ufc.py           # bake into the web app
```

### 🏀 NBA / 🏈 NFL
Opponent-adjusted **off/def ratings**, recency-weighted (NBA half-life ≈160d, NFL ≈230d), mapped to a
win probability through a **Gaussian point-margin** model (`P(home) = Φ(proj_margin / SD)`). NFL ratings
are built from play-by-play **EPA→points** and include per-player and kicker layers. Both blend the
decaying consensus prior (`blend_nba.py` / `blend_nfl.py`).
```bash
python build_nba.py && python blend_nba.py && python backtest_nba.py
python build_nfl.py && python blend_nfl.py && python backtest_nfl.py
python export_pro.py            # bake NFL/NBA/MLB into web/{nfl,nba,mlb}_data.js
```

### 🏒 NHL
Recency-weighted (half-life 70d) Poisson attack/defense + MOV-weighted Elo, z-blended, with goal-level
calibration and OT/SO resolution (a slight favourite edge on ties). Goalie/skater layers included.
```bash
python build_nhl.py && python build_nhl_xg.py && python blend_nhl.py
python nhl_predict.py            # CLI game prediction
python backtest_nhl.py           # honest OOS backtest (model-only headline; consensus variant flagged)
python nhl_export.py             # bake into the web app
```

### ⚾ MLB
Iterative run-rate attack/defense ratings + a **starting-pitcher RA9 factor** (regressed to league
average), combined in a Poisson run model (`SP ≈ 60% of run prevention`). Decaying consensus prior over
a 162-game season.
```bash
python build_mlb.py && python blend_mlb.py && python backtest_mlb.py
```

### ⚽ Club leagues (EPL / La Liga / Serie A / Bundesliga / Ligue 1)
Same Dixon-Coles 1X2 engine as the World Cup model (half-life 400d, ρ = −0.12), built from
football-data.co.uk results; consumed by `web/leagues_app.js`.
```bash
python build_leagues.py && python backtest_leagues.py && python validate_leagues.py
```

### 👤 Player / individual stats
`build_player_intl.py`, `build_player_xg.py`, `build_nfl_players.py` build per-player layers; backtested
out-of-sample by `backtest_players.py` and `backtest_player_yoy.py` (year-over-year stability).

---

## Web apps (`web/`)
Self-contained browser apps — the prediction math is ported to JavaScript and verified to match the
Python models exactly, so they run entirely in the browser (no server). Build → export → open:
```bash
python build_ratings.py --refresh && python export_web.py   # World Cup app
python export_pro.py                                         # NFL / NBA / MLB app
python nhl_export.py                                         # NHL
python export_ufc.py                                         # UFC
```
Then open the relevant `web/*.html` (works offline by double-click), or host the `web/` folder. To
share: push `web/` to **GitHub Pages**/Netlify and send the link; on a phone, "Add to Home Screen"
installs it like an app.

**Live site:** https://benbr11.github.io/EdgeLabs/

### Hands-off auto-update
`.github/workflows/update.yml` reruns the builds twice daily from the latest data and republishes the
GitHub Pages site — even with your computer off. Full one-time setup is in **SETUP-AUTOUPDATE.md**.

---

## Caveats / honest limitations
- **Variance is the sport, especially in UFC/NHL/MLB.** Even a perfect model loses a large minority of
  games; that's why the confidence tiers exist — they tell you *when* to trust a pick.
- **No model beats the closing point spread reliably.** The value is in moneyline/straight-up picks and
  the high-confidence tiers, not ATS.
- **Consensus priors for past seasons** use the *current* preseason consensus as an imperfect stand-in
  (no point-in-time historical consensus exists); the team-sport backtests therefore report the
  **model-only** number as the headline and flag the consensus-blended variant separately.
- **Situational data isn't in the ratings** — injuries, starting goalies, weather, short-notice fights,
  lineup rotation are looked up live per game and passed as flags (where supported), not baked in.
- **UFC method/round accuracy isn't formally backtested yet** — the model produces method (KO/Sub/Dec)
  and round predictions, but their standalone hit-rate is an open measurement.
- **World Cup ratings** reflect results through ~21 June 2026; FIFA points for 3 minnows are estimated.

---

## File map (quick reference)
- **World Cup:** `build_ratings.py`, `build_context.py`, `simulate.py`, `groups.py`, `situations.py`,
  `recent.py`, `analyze_stakes.py`, `analyze_shootouts.py`, `backtest.py`, `export_web.py`,
  `results.csv`, `ratings.csv`, `context.csv`, `wc2026_xg.csv`
- **UFC:** `ufc_model.py`, `ufc_positional.py`, `ufc_backtest.py`, `build_ufc_rankings.py`,
  `show_rankings.py`, `check_acceptance.py`, `export_ufc.py`, `ufc_*.csv`, `raw_*card*.{csv,json}`
- **NBA:** `build_nba.py`, `blend_nba.py`, `backtest_nba.py`, `nba_ratings.csv`, `nba_players.csv`, `consensus_nba.csv`
- **NFL:** `build_nfl.py`, `build_nfl_players.py`, `build_nfl_roster.py`, `blend_nfl.py`, `backtest_nfl.py`, `nfl_*.csv`, `consensus_nfl.csv`
- **NHL:** `build_nhl.py`, `build_nhl_xg.py`, `blend_nhl.py`, `nhl_predict.py`, `nhl_export.py`, `backtest_nhl.py`, `nhl_*.csv`, `consensus_nhl.csv`
- **MLB:** `build_mlb.py`, `blend_mlb.py`, `backtest_mlb.py`, `mlb_*.csv`, `consensus_mlb.csv`
- **Club leagues:** `build_leagues.py`, `backtest_leagues.py`, `validate_leagues.py`
- **Players:** `build_player_intl.py`, `build_player_xg.py`, `build_statsbomb_xg.py`, `backtest_players.py`, `backtest_player_yoy.py`, `player_*.csv`, `squads.csv`
- **Shared / web:** `export_pro.py`, `web/`, `.github/workflows/`, `SETUP-AUTOUPDATE.md`
