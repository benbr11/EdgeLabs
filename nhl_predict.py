# -*- coding: utf-8 -*-
"""
NHL game prediction with the situational factors that actually matter (the hockey
analogue of simulate.py). Loads nhl_ratings.csv + nhl_goalies.csv and applies:
  - STARTING GOALIE (biggest single-game factor) via that goalie's GSAx vs the team baseline
  - REST / back-to-back (2nd night of a B2B = tired, worse both ways)
  - INJURIES / availability (scoring knob)
  - HOME / away ice
  - recency is already in the ratings (time-weighted) so "form" is baked in
Outputs win % (incl. OT/SO), regulation split, expected goals, total over/under, puck line.
"""
import csv, os, math, sys
PROJ = os.path.dirname(os.path.abspath(__file__))
R = {}; AVG = HOME = None
for r in csv.DictReader(open(PROJ + r"\nhl_ratings.csv", encoding="utf-8")):
    R[r["team"]] = {"att": float(r["attack_mult"]), "dfn": float(r["defense_mult"]),
                    "gsax": float(r["gsax_per_shot"])}
    AVG = float(r["avg_goals"]); HOME = float(r["home_adv"])
GOALIE = {}
try:
    for r in csv.DictReader(open(PROJ + r"\nhl_goalies.csv", encoding="utf-8")):
        GOALIE[r["goalie"]] = {"team": r["team"], "gsax_ps": float(r["gsax_per_shot"]),
                               "games": int(r["games"]), "sv": float(r["sv_pct"])}
except FileNotFoundError:
    pass
SOG = 29.0   # ~shots on goal a goalie faces per game (scales the goalie's per-shot GSAx to goals)

# Probability recalibration. The raw Poisson favourite win-probabilities were found to be
# systematically OVERCONFIDENT at every tier in the honest walk-forward backtest
# (backtest_nhl.py, n=2792 OOS games): e.g. the 80-100% bucket won only ~68%, the 70-80%
# bucket ~64%. Hockey is the highest-variance major sport, so the true favourite edge is
# smaller than a deterministic Poisson implies. We soften winH with a LOGIT TEMPERATURE
# (monotonic, centred on 0.5: a coin-flip stays a coin-flip, ordering/SU hit-rate is
# preserved, only the SCALE shrinks toward 0.5). T=2.0 minimises both Brier and log-loss
# on the OOS backtest while leaving the straight-up hit-rate unchanged (~55.5%).
WINPROB_TEMP = 2.0

def recalibrate(p, T=WINPROB_TEMP):
    """Soften an over-confident win prob toward 0.5 via a logit temperature (T>1)."""
    if T == 1.0:
        return p
    p = min(max(p, 1e-9), 1 - 1e-9)
    z = math.log(p / (1 - p)) / T
    return 1.0 / (1.0 + math.exp(-z))

# Near-pick'em ABSTAIN band (validated OOS in backtest_nhl.py). Projected goal margin below
# ABSTAIN_MARGIN -> shrink the win prob toward 0.50 by ABSTAIN_SHRINK and flag graded=False.
ABSTAIN_MARGIN = 0.20
ABSTAIN_SHRINK = 1.0

def rest_factor(days):
    # 2nd night of a back-to-back (0-1 days) hurts; well-rested (3+) is neutral-to-fresh
    if days is None: return 1.0
    if days <= 1: return 0.95          # tired: ~5% fewer goals for, defence a touch worse
    if days == 2: return 0.99
    return 1.0

def predict(home, away, goalieH=None, goalieA=None, restH=2, restA=2,
            availH=1.0, availA=1.0, neutral=False, sims_lines=(5.5, 6.5)):
    a, b = R[home], R[away]
    hf = 1.0 if neutral else HOME
    # base expected goals (recency already in ratings)
    lh = AVG * a["att"] * b["dfn"] * hf
    la = AVG * b["att"] * a["dfn"]
    # availability (missing scorers) + rest (tired teams score less, leak a bit more)
    lh *= availH * rest_factor(restH); la *= availA * rest_factor(restA)
    lh *= (2 - rest_factor(restA)) ** 0.5    # away tired -> home scores a bit more (and vice-versa)
    la *= (2 - rest_factor(restH)) ** 0.5
    # STARTING GOALIE: adjust vs the team's baseline goaltending already in the rating.
    # a better-than-team goalie suppresses the opponent's goals; a backup leaks more.
    def gadj(goalie, team):
        if not goalie or goalie not in GOALIE: return 0.0
        return (R[team]["gsax"] - GOALIE[goalie]["gsax_ps"]) * SOG   # +goals to opp if goalie worse than team avg
    la += gadj(goalieH, home)        # home goalie affects away goals
    lh += gadj(goalieA, away)
    lh = max(0.5, lh); la = max(0.5, la)
    # score matrix
    P = lambda k, l: math.exp(-l) * l**k / math.factorial(k)
    pH = pT = pA = 0.0; over = {ln: 0.0 for ln in sims_lines}; pl_home = 0.0
    for i in range(14):
        for j in range(14):
            m = P(i, lh) * P(j, la)
            if i > j: pH += m
            elif i == j: pT += m
            else: pA += m
            for ln in sims_lines:
                if i + j > ln: over[ln] += m
            if i - j >= 2: pl_home += m      # home covers -1.5 puck line
    fav = pH / (pH + pA) if pH + pA else 0.5
    winH = pH + pT * (0.5 + (fav - 0.5) * 0.35)    # ties -> OT/SO, slight favourite edge
    winH = recalibrate(winH)                       # soften OOS-overconfident scale toward 0.5
    # ABSTAIN / regress the near-pick'em band (validated OOS fix, backtest_nhl.py change #3).
    # Games whose projected goal margin |lh-la| is below ABSTAIN_MARGIN are a coin flip the
    # model genuinely cannot call: in the n=2792 walk-forward backtest that band hit only
    # ~42-46% (worse than 50%). We shrink those toward 0.50 (graded=False) and surface the
    # flag so callers can ABSTAIN. Excluding the band lifts the graded straight-up hit-rate
    # to ~57% (from 55.5% over all games). Byte-identical to backtest_nhl.py.
    graded = True
    if ABSTAIN_MARGIN > 0.0 and abs(lh - la) < ABSTAIN_MARGIN:
        winH = 0.5 + (winH - 0.5) * (1.0 - ABSTAIN_SHRINK)
        graded = False
    return {"lh": lh, "la": la, "winH": winH, "winA": 1 - winH, "regH": pH, "regT": pT, "regA": pA,
            "over": over, "pl_home": pl_home, "graded": graded}

def show(home, away, **kw):
    r = predict(home, away, **kw)
    tags = []
    if kw.get("goalieH"): tags.append(f"{home} G:{kw['goalieH']}")
    if kw.get("goalieA"): tags.append(f"{away} G:{kw['goalieA']}")
    if kw.get("restH") is not None and kw["restH"] <= 1: tags.append(f"{home} B2B")
    if kw.get("restA") is not None and kw["restA"] <= 1: tags.append(f"{away} B2B")
    if kw.get("availH", 1) < 1: tags.append(f"{home} avail {kw['availH']}")
    if not r.get("graded", True): tags.append("PICK'EM->ABSTAIN")
    print(f"{home} vs {away}" + (f"  [{', '.join(tags)}]" if tags else ""))
    print(f"  exp goals {r['lh']:.2f}-{r['la']:.2f} | {home} win {r['winH']*100:.0f}% / {away} {r['winA']*100:.0f}% "
          f"| over 5.5 {r['over'][5.5]*100:.0f}% over 6.5 {r['over'][6.5]*100:.0f}% | {home} -1.5 {r['pl_home']*100:.0f}%")

if __name__ == "__main__":
    if len(sys.argv) >= 3:
        show(sys.argv[1], sys.argv[2])
    else:
        print("=== situational factors demo (COL vs DAL) ===")
        show("COL", "DAL")
        show("COL", "DAL", restA=1)                       # Dallas on a back-to-back
        # swap in a backup vs an elite goalie for Colorado
        elite = max(GOALIE, key=lambda g: GOALIE[g]["gsax_ps"] if GOALIE[g]["games"] >= 80 else -9)
        backup = min(GOALIE, key=lambda g: GOALIE[g]["gsax_ps"] if GOALIE[g]["games"] >= 30 else 9)
        show("COL", "DAL", goalieH=elite)
        show("COL", "DAL", goalieH=backup)
        show("COL", "DAL", availH=0.90)                   # Colorado missing a top scorer
        print(f"\n(elite goalie used: {elite} {GOALIE[elite]['gsax_ps']:+.4f}/shot; backup: {backup} {GOALIE[backup]['gsax_ps']:+.4f}/shot)")
