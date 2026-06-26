# -*- coding: utf-8 -*-
"""
mlb_signals.py -- POINT-IN-TIME day-of run-rate adjustments shared byte-identically by
build_mlb.py (live) and backtest_mlb.py (OOS measurement). Every function here uses ONLY
information available before first pitch.

The model's base prediction produces two Poisson run rates:
    lh = home expected runs,  la = away expected runs.
Each signal returns a multiplicative adjustment to (lh, la). Compose them, then run the
Poisson grid (unchanged). Strength knobs are env-vars so each signal can be A/B'd and a
non-improving signal reverted to OFF (strength 0) without code churn.

Signals
-------
1. park_factor(venue_runmult)            -> scales BOTH rates (run environment).
2. weather_factor(temp, wind_mph, dir,   -> scales BOTH rates (outdoor only; roof -> 1.0).
                  roof)
3. platoon_factor(home_split, away_split,-> ASYMMETRIC: away SP hand vs home lineup scales lh;
                  away_sp_hand, home_sp_hand,  home SP hand vs away lineup scales la.
                  lg_vs_l, lg_vs_r)
4. bullpen_factor(home_pen_fatigue,      -> ASYMMETRIC: home pen fatigue scales la up;
                  away_pen_fatigue)           away pen fatigue scales lh up.

All strengths default to values chosen so the signal is testable; the backtest decides
KEEP/REVERT and the kept strength is hard-set here.
"""
import os, math

_E = lambda k, d: float(os.environ.get(k, d))

# ---- strength knobs (0 => signal OFF / identity). Set after OOS measurement (mlb_sweep.py). ----
# OOS verdict on 2024-25 walk-forward (N=4857), vs baseline SU 56.50% / Brier 0.2433 / LL 0.6794:
#   PARK    -> REVERTED (W=0): symmetric run-env scaler; flat SU, LL slightly worse. Moves
#              TOTALS not the moneyline. (PF table is still computed for Goal-2 segmentation.)
#   WEATHER -> REVERTED (W=0): same -- symmetric scaler, no moneyline signal. (Kept for segments.)
#   PLATOON -> REVERTED (W=0): prior-season team-vs-hand OPS HURT monotonically (too noisy /
#              already in the team rating). (Handedness still surfaced for Goal-2 segmentation.)
#   BULLPEN -> KEPT at 0.3, 3-day lookback: SU 56.66%, Brier 0.2432, LL 0.6793 -- improves all
#              three metrics and BOTH seasons independently (2024 56.55%, 2025 56.77%). Small
#              (+0.16 SU pt) but season-consistent and directionally sound (tired pen -> more runs).
PARK_W    = _E("MLB_PARK_W", 0.0)     # REVERTED
WEATHER_W = _E("MLB_WEATHER_W", 0.0)  # REVERTED
PLATOON_W = _E("MLB_PLATOON_W", 0.0)  # REVERTED
PEN_W     = _E("MLB_PEN_W", 0.3)      # KEPT (OOS winner); bullpen-fatigue run penalty
PEN_LOOKBACK_DAYS = int(_E("MLB_PEN_LOOKBACK", 3))   # KEPT 3-day window (beat 1/2/4-day)

# clamp helpers so a single signal can't explode a run rate
def _clamp(x, lo, hi): return max(lo, min(hi, x))


def park_factor(venue_runmult):
    """venue_runmult ~ (runs at venue) / (league runs), from PRIOR seasons. Scales total runs.
    Returned multiplier applied to BOTH lh and la. PARK_W=1 -> full effect, 0 -> none."""
    if not venue_runmult or PARK_W == 0.0:
        return 1.0, 1.0
    m = 1.0 + PARK_W * (venue_runmult - 1.0)
    m = _clamp(m, 0.80, 1.25)
    return m, m


def weather_factor(temp, wind_mph, wind_dir, roof):
    """Outdoor weather effect on total runs. roof True (dome/closed) -> identity.
    Wind 'Out To ...' boosts, 'In From ...' suppresses, crosswinds ~neutral; hotter air
    carries -> small boost. Scales BOTH rates. WEATHER_W scales the whole delta."""
    if roof or WEATHER_W == 0.0:
        return 1.0, 1.0
    delta = 0.0
    d = (wind_dir or "").lower()
    if wind_mph:
        # wind component along the hit trajectory: out=+, in=-, cross/var/calm=0
        sign = 0.0
        if d.startswith("out"):
            sign = 1.0
        elif d.startswith("in"):
            sign = -1.0
        # ~0.7% run change per mph of straight out/in wind (capped)
        delta += sign * _clamp(wind_mph, 0, 20) * 0.007
    if temp is not None and temp > 0:           # temp>0 guards dome sentinel (0)
        # ~+0.4% per deg above 70F, -0.4% below; capped at +/-30F
        delta += _clamp(temp - 70, -30, 30) * 0.004
    m = 1.0 + WEATHER_W * delta
    m = _clamp(m, 0.85, 1.20)
    return m, m


def _platoon_edge(team_split, sp_hand, lg_vs_l, lg_vs_r):
    """How much better/worse than league-average this lineup hits the opposing starter's
    hand, as a fractional OPS delta. team_split={'vl':ops_vs_LHP,'vr':ops_vs_RHP} from PRIOR
    season. sp_hand 'L'/'R'. Returns (lineup_ops_vs_hand - lg_ops_vs_hand)/lg_ops_vs_hand."""
    if not team_split or sp_hand not in ("L", "R"):
        return 0.0
    if sp_hand == "L":
        ops = team_split.get("vl"); lg = lg_vs_l
    else:
        ops = team_split.get("vr"); lg = lg_vs_r
    if not ops or not lg:
        return 0.0
    return (ops - lg) / lg


def platoon_factor(home_split, away_split, away_sp_hand, home_sp_hand, lg_vs_l, lg_vs_r):
    """ASYMMETRIC. The AWAY starter (hand=away_sp_hand) faces the HOME lineup -> adjusts home
    runs (lh). The HOME starter faces the AWAY lineup -> adjusts away runs (la). PLATOON_W
    scales the OPS delta into a run multiplier (OPS delta ~ run delta, slightly damped)."""
    if PLATOON_W == 0.0:
        return 1.0, 1.0
    eh = _platoon_edge(home_split, away_sp_hand, lg_vs_l, lg_vs_r)   # home lineup vs away SP hand
    ea = _platoon_edge(away_split, home_sp_hand, lg_vs_l, lg_vs_r)   # away lineup vs home SP hand
    mh = _clamp(1.0 + PLATOON_W * eh, 0.85, 1.18)
    ma = _clamp(1.0 + PLATOON_W * ea, 0.85, 1.18)
    return mh, ma


def bullpen_factor(home_pen_fatigue, away_pen_fatigue):
    """ASYMMETRIC. pen_fatigue is recent reliever IP load (last 1-3 days) normalized so 0 =
    rested, ~1 = heavily worked. A fatigued bullpen concedes MORE runs to the opponent:
    home fatigue -> away runs (la) up; away fatigue -> home runs (lh) up. PEN_W scales it."""
    if PEN_W == 0.0:
        return 1.0, 1.0
    mh = _clamp(1.0 + PEN_W * (away_pen_fatigue or 0.0), 1.0, 1.15)  # away pen tired -> lh up
    ma = _clamp(1.0 + PEN_W * (home_pen_fatigue or 0.0), 1.0, 1.15)  # home pen tired -> la up
    return mh, ma


def apply_all(lh, la, *, park=None, weather=None, platoon=None, bullpen=None):
    """Compose the enabled signals onto (lh, la). Each arg is the tuple returned above
    (or None to skip). Returns adjusted (lh, la)."""
    for mult in (park, weather, platoon, bullpen):
        if mult is not None:
            lh *= mult[0]; la *= mult[1]
    return lh, la
