# -*- coding: utf-8 -*-
"""
Build attack/defense ratings for the 48 teams of the 2026 FIFA World Cup
using a 3-source ensemble:

  SOURCE A  Goals-based attack/defense   (iterative, opponent- & recency-adjusted)
  SOURCE B  Elo rating                   (computed from full match history 1872->2026)
  SOURCE C  FIFA ranking points          (official, June 2026)

The goals model is the ONLY source that distinguishes attack from defense, so it
provides each team's attack/defense *tilt*. Elo + FIFA + goals together set each
team's overall *strength level*. Final per-team attack & defense rate multipliers
are reconstructed from (consensus strength, goals tilt) and the engine stores both
those multipliers (used by simulate.py) and human-readable 0-100 scores.

Output: ratings.csv
"""
import csv, math, datetime, argparse, urllib.request

import os
PROJ = os.path.dirname(os.path.abspath(__file__))
RESULTS_URL = "https://raw.githubusercontent.com/martj42/international_results/master/results.csv"
GOALSCORERS_URL = "https://raw.githubusercontent.com/martj42/international_results/master/goalscorers.csv"
SHOOTOUTS_URL = "https://raw.githubusercontent.com/martj42/international_results/master/shootouts.csv"
XG_URL = "https://raw.githubusercontent.com/mominullptr/FIFA-World-Cup-2026-Dataset/main/matches_detailed.csv"
SQUADS_URL = "https://raw.githubusercontent.com/mominullptr/FIFA-World-Cup-2026-Dataset/main/squads_and_players.csv"
PLAYERS_URL = "https://raw.githubusercontent.com/mominullptr/FIFA-World-Cup-2026-Dataset/main/parsed_players.json"

# `python build_ratings.py --refresh` re-pulls the latest match results first, so
# ratings reflect every game played up to now. Run this before simulating a new match.
_ap = argparse.ArgumentParser()
_ap.add_argument("--refresh", action="store_true", help="re-download latest results.csv before building")
_args, _ = _ap.parse_known_args()
if _args.refresh:
    print("Refreshing results.csv from source ...")
    try:
        urllib.request.urlretrieve(RESULTS_URL, PROJ + r"\results.csv")
        print("  results done.")
    except Exception as e:
        print(f"  results refresh FAILED ({e}); using existing file.")
    for url, fn in [(GOALSCORERS_URL, "goalscorers.csv"), (SHOOTOUTS_URL, "shootouts.csv")]:
        try:
            urllib.request.urlretrieve(url, PROJ + "\\" + fn); print(f"  {fn} done.")
        except Exception as e:
            print(f"  {fn} refresh failed ({e}); using existing file.")
    try:
        urllib.request.urlretrieve(XG_URL, PROJ + r"\wc2026_xg.csv")
        print("  xG done.")
    except Exception as e:
        print(f"  xG refresh failed ({e}); using existing file.")
    for url, fn in [(SQUADS_URL, "squads.csv"), (PLAYERS_URL, "players_raw.json")]:
        try:
            urllib.request.urlretrieve(url, PROJ + "\\" + fn); print(f"  {fn} done.")
        except Exception as e:
            print(f"  {fn} refresh failed ({e}); using existing file.")

# ---- the 48 qualified teams (names as they appear in results.csv) -------------
TEAMS = [
    "Canada","Mexico","United States","Australia","Iran","Iraq","Japan","Jordan",
    "Qatar","Saudi Arabia","South Korea","Uzbekistan","Algeria","Cape Verde",
    "DR Congo","Egypt","Ghana","Ivory Coast","Morocco","Senegal","South Africa",
    "Tunisia","Curaçao","Haiti","Panama","Argentina","Brazil","Colombia","Ecuador",
    "Paraguay","Uruguay","New Zealand","Austria","Belgium","Bosnia and Herzegovina",
    "Croatia","Czech Republic","England","France","Germany","Netherlands","Norway",
    "Portugal","Scotland","Spain","Sweden","Switzerland","Turkey",
]

# ---- SOURCE C: FIFA points (June 11 2026). 3 teams below top-80 are estimated. -
FIFA = {
    "Canada":1559.48,"Mexico":1687.48,"United States":1671.23,"Australia":1579.34,
    "Iran":1619.58,"Iraq":1446.28,"Japan":1661.58,"Jordan":1387.74,"Qatar":1450.31,
    "Saudi Arabia":1423.88,"South Korea":1591.63,"Uzbekistan":1458.73,"Algeria":1571.03,
    "Cape Verde":1371.11,"DR Congo":1474.43,"Egypt":1562.37,"Ghana":1346.88,
    "Ivory Coast":1540.87,"Morocco":1755.10,"Senegal":1684.07,"South Africa":1428.38,
    "Tunisia":1476.41,"Curaçao":1295.0,"Haiti":1285.0,"Panama":1539.16,
    "Argentina":1877.27,"Brazil":1765.86,"Colombia":1698.35,"Ecuador":1598.52,
    "Paraguay":1505.35,"Uruguay":1673.07,"New Zealand":1280.0,"Austria":1597.40,
    "Belgium":1742.24,"Bosnia and Herzegovina":1387.22,"Croatia":1714.87,
    "Czech Republic":1505.74,"England":1828.02,"France":1870.70,"Germany":1735.77,
    "Netherlands":1753.57,"Norway":1557.44,"Portugal":1767.85,"Scotland":1503.34,
    "Spain":1874.71,"Sweden":1509.79,"Switzerland":1650.06,"Turkey":1605.73,
}
FIFA_ESTIMATED = {"Curaçao","Haiti","New Zealand"}

# ---- xG overlay -------------------------------------------------------------
# Blend current-tournament expected goals (chance quality -- less noisy than actual
# goals) into recent results. Only the goals model uses xG; Elo keeps real results.
W_XG = 0.6
XG_NAME = {"Cabo Verde":"Cape Verde","Congo DR":"DR Congo","Czechia":"Czech Republic",
           "Côte d'Ivoire":"Ivory Coast","IR Iran":"Iran","Türkiye":"Turkey","USA":"United States"}
m_xg = {}
try:
    with open(PROJ + r"\wc2026_xg.csv", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r.get("status") != "Completed" or not r.get("home_xg"):
                continue
            h = XG_NAME.get(r["home_team_name"], r["home_team_name"])
            a = XG_NAME.get(r["away_team_name"], r["away_team_name"])
            m_xg[(r["date"], frozenset((h, a)))] = {h: float(r["home_xg"]), a: float(r["away_xg"])}
    print(f"Loaded xG for {len(m_xg)} completed matches")
except FileNotFoundError:
    print("wc2026_xg.csv not found -- using actual goals only")

# =============================================================================
# Load matches
# =============================================================================
rows = []
with open(PROJ + r"\results.csv", encoding="utf-8") as f:
    for r in csv.DictReader(f):
        try:
            d = datetime.date.fromisoformat(r["date"])
            hs = int(r["home_score"]); as_ = int(r["away_score"])
        except (ValueError, KeyError):
            continue
        rows.append((d, r["home_team"], r["away_team"], hs, as_,
                     r["neutral"].strip().upper() == "TRUE"))
rows.sort(key=lambda x: x[0])
MAXDATE = rows[-1][0]
print(f"Loaded {len(rows):,} matches through {MAXDATE}")

# =============================================================================
# SOURCE B: Elo from full history (World-Football-Elo style)
# =============================================================================
elo = {}
BASE_ELO = 1500.0
HFA_ELO = 65.0          # home-field advantage in Elo points
def k_factor(tournament_weight, gd):
    # margin-of-victory multiplier (standard WFE)
    if gd <= 1: g = 1.0
    elif gd == 2: g = 1.5
    else: g = (11 + gd) / 8.0
    return tournament_weight * g

for d, h, a, hs, as_, neutral in rows:
    eh = elo.get(h, BASE_ELO); ea = elo.get(a, BASE_ELO)
    adj = 0.0 if neutral else HFA_ELO
    exp_h = 1.0 / (1.0 + 10 ** ((ea - (eh + adj)) / 400.0))
    res_h = 1.0 if hs > as_ else (0.5 if hs == as_ else 0.0)
    K = k_factor(30.0, abs(hs - as_))
    delta = K * (res_h - exp_h)
    elo[h] = eh + delta
    elo[a] = ea - delta

# =============================================================================
# SOURCE A: iterative goals-based attack/defense (opponent- & recency-adjusted)
# =============================================================================
CUTOFF = datetime.date(2015, 1, 1)
HALFLIFE_DAYS = 730.0           # 2-year half-life recency weighting (tuned via backtest.py)
WC_BOOST = 1.5                  # current World Cup games weighted extra (most current) but not so much an outlier distorts
def weight(d):
    age = (MAXDATE - d).days
    return 0.5 ** (age / HALFLIFE_DAYS)

# Build recent matches with xG-blended "effective goals" (eh, ea): where current-
# tournament xG exists, mix it with the actual score; otherwise use actual goals.
recent = []
xg_used = 0
for (d, h, a, hs, as_, neutral) in (m for m in rows if m[0] >= CUTOFF):
    key = (d.isoformat(), frozenset((h, a)))
    is_wc = key in m_xg and h in m_xg[key] and a in m_xg[key]   # a current-WC match (has xG)
    if is_wc:
        eh = W_XG * m_xg[key][h] + (1 - W_XG) * hs
        ea = W_XG * m_xg[key][a] + (1 - W_XG) * as_
        xg_used += 1
    else:
        eh, ea = float(hs), float(as_)
    wb = WC_BOOST if is_wc else 1.0
    recent.append((d, h, a, hs, as_, neutral, eh, ea, wb))
print(f"xG-blended {xg_used} current-WC matches (xG weight {W_XG}, WC weight boost x{WC_BOOST})")

# weighted league average goals per team per match
tot_g = tot_w = 0.0
for d, h, a, hs, as_, neutral, eh, ea, wb in recent:
    w = weight(d) * wb; tot_g += w * (eh + ea); tot_w += w * 2
AVG = tot_g / tot_w
print(f"Weighted avg goals/team/match since {CUTOFF}: {AVG:.3f}")

# Fit attack/defense VENUE-BLIND (every match treated as neutral). This yields
# clean neutral-venue strengths -- the right baseline for a World Cup, which is
# almost entirely neutral-site -- and avoids the att/def-vs-home-advantage
# confounding that collapses a joint fit. Home advantage is then measured
# separately below, controlled for team strength.
att = {}; dfn = {}                         # multiplicative; 1.0 = average
for d, h, a, hs, as_, neutral, eh, ea, wb in recent:
    for t in (h, a):
        att.setdefault(t, 1.0); dfn.setdefault(t, 1.0)

for iteration in range(60):
    na = {t: 0.0 for t in att}; da = {t: 0.0 for t in att}
    nd = {t: 0.0 for t in att}; dd = {t: 0.0 for t in att}
    for d, h, a, hs, as_, neutral, eh, ea, wb in recent:
        w = weight(d) * wb
        na[h] += w * eh;  da[h] += w * AVG * dfn[a]
        nd[a] += w * eh;  dd[a] += w * AVG * att[h]
        na[a] += w * ea;  da[a] += w * AVG * dfn[h]
        nd[h] += w * ea;  dd[h] += w * AVG * att[a]
    for t in att:
        if da[t] > 0: att[t] = na[t] / da[t]
        if dd[t] > 0: dfn[t] = nd[t] / dd[t]
    # normalise to geometric mean 1 (identifiability)
    for dct in (att, dfn):
        gm = math.exp(sum(math.log(max(v,1e-6)) for v in dct.values()) / len(dct))
        for t in dct: dct[t] /= gm

# Home advantage. It cannot be cleanly separated from attack/defense in this
# dataset (every match's home side is a different team, so a fit just absorbs the
# effect into the ratings). We therefore use the well-established football-modeling
# value: a host scores ~30% more goals (~+0.4 on a ~1.4 base). Applied ONLY to a
# designated host team in simulate.py; tune via this constant. The raw observed
# home/away goal ratio is printed for reference (it is inflated by scheduling --
# strong teams host weak ones in qualifiers -- so we do not use it directly).
HOME_ADV = 1.30
hg = ag = 0.0
for d, h, a, hs, as_, neutral, eh, ea, wb in recent:
    if neutral: continue
    w = weight(d); hg += w * hs; ag += w * as_
print(f"Raw home/away goal ratio (reference): {hg/ag:.3f}  ->  using HOME_ADV = {HOME_ADV}")
home_adv = HOME_ADV

# Squad market value (EUR millions, Transfermarkt 2026) -- "a team is only as strong
# as the players on the field." A heavily-weighted talent signal; this is what puts
# France (the most valuable squad in the world) at the top, as the eye-test expects.
SQUAD_VALUE = {
    "France":1520,"England":1360,"Spain":1220,"Portugal":1010,"Germany":947,
    "Brazil":928,"Argentina":808,"Netherlands":754,"Norway":590,"Belgium":548,
    "Ivory Coast":522,"Senegal":478,"Turkey":474,"Morocco":448,"Sweden":406,
    "Croatia":387,"United States":386,"Ecuador":369,"Uruguay":359,"Switzerland":333,
    "Colombia":302,"Japan":271,"Algeria":257,"Austria":245,"Ghana":235,"Canada":199,
    "Mexico":192,"Czech Republic":188,"Scotland":170,"Paraguay":154,
    "Bosnia and Herzegovina":146,"DR Congo":144,"South Korea":139,"Egypt":116,
    "Uzbekistan":85,"Australia":77,"Tunisia":70,"Haiti":56,"Cape Verde":49,
    "South Africa":49,"Saudi Arabia":41,"Panama":35,"New Zealand":34,"Iran":32,
    "Curaçao":26,"Iraq":21,"Jordan":20,"Qatar":20,
}

# StatsBomb shot-level xG (per game), from build_statsbomb_xg.py over recent major
# tournaments. Real chance quality -- the backbone of the TikTok-style model.
SB_XG = {}
SB_MAP = {"Cape Verde Islands": "Cape Verde"}
try:
    for r in csv.DictReader(open(PROJ + r"\statsbomb_xg.csv", encoding="utf-8")):
        t = SB_MAP.get(r["team"], r["team"])
        if t in TEAMS:
            SB_XG[t] = (float(r["xgf_pg"]), float(r["xga_pg"]), int(r["games"]))
    print(f"Loaded StatsBomb shot-xG for {len(SB_XG)}/48 teams")
except FileNotFoundError:
    print("statsbomb_xg.csv not found -- skipping shot-xG source")

# =============================================================================
# BLEND -> consensus strength, keep goals-based attack/defense tilt
# =============================================================================
def zscores(d):
    vals = list(d.values()); m = sum(vals)/len(vals)
    sd = (sum((v-m)**2 for v in vals)/len(vals)) ** 0.5 or 1.0
    return {k:(v-m)/sd for k,v in d.items()}

A_log = {t: math.log(att[t]) for t in TEAMS}      # attack log-strength
D_log = {t: -math.log(dfn[t]) for t in TEAMS}      # defense log-strength (higher=better)
g_str = {t: A_log[t] + D_log[t] for t in TEAMS}    # goals-implied overall
tilt  = {t: A_log[t] - D_log[t] for t in TEAMS}    # +ve = attack-leaning

zg = zscores(g_str)
ze = zscores({t: elo[t] for t in TEAMS})
zf = zscores({t: FIFA[t] for t in TEAMS})
zv = zscores({t: math.log(SQUAD_VALUE[t]) for t in TEAMS})   # squad value, log scale

# StatsBomb shot-xG strength & attack/defense tilt. Shrink small samples toward the
# field average (K games of prior); impute the goals-model value where no SB data.
if SB_XG:
    K = 5.0
    lf = sum(v[0] for v in SB_XG.values()) / len(SB_XG)
    la = sum(v[1] for v in SB_XG.values()) / len(SB_XG)
    sx_att = {}; sx_def = {}
    for t, (xf, xa, g) in SB_XG.items():
        xf = (g*xf + K*lf) / (g+K); xa = (g*xa + K*la) / (g+K)
        sx_att[t] = math.log(xf/lf); sx_def[t] = -math.log(xa/la)
    sb_mean = sum(sx_att[t]+sx_def[t] for t in sx_att) / len(sx_att)
    sx_str = {t: (sx_att[t]+sx_def[t]) if t in sx_att else sb_mean for t in TEAMS}  # neutral if no data
    zx = zscores(sx_str)
    tilt = {t: (0.5*tilt[t] + 0.5*(sx_att[t]-sx_def[t])) if t in sx_att else tilt[t] for t in TEAMS}
else:
    zx = zscores(g_str)

# weights: shot-xG (chance quality), goals/current form, squad value (talent), Elo, FIFA
W_SBXG, W_GOALS, W_VALUE, W_ELO, W_FIFA = 0.20, 0.22, 0.26, 0.16, 0.16   # current form (goals incl. WC) now > historical xG
cons_z = {t: W_SBXG*zx[t] + W_GOALS*zg[t] + W_VALUE*zv[t] + W_ELO*ze[t] + W_FIFA*zf[t] for t in TEAMS}

# put consensus strength back on the goals log-scale, then split by tilt
gm = sum(g_str.values())/len(TEAMS)
gs = (sum((v-gm)**2 for v in g_str.values())/len(TEAMS)) ** 0.5
G_star = {t: gm + gs*cons_z[t] for t in TEAMS}
A_star = {t: (G_star[t] + tilt[t]) / 2 for t in TEAMS}   # final attack log-strength
D_star = {t: (G_star[t] - tilt[t]) / 2 for t in TEAMS}   # final defense log-strength
att_mult = {t: math.exp(A_star[t]) for t in TEAMS}        # used by the simulator
dfn_mult = {t: math.exp(-D_star[t]) for t in TEAMS}       # <1 = good defense

# --- ABSOLUTE GOAL-LEVEL calibration ----------------------------------------
# Geometric-mean-normalised multipliers leave the ARITHMETIC mean of att*dfn above 1
# (Jensen), so the average predicted total sits ~20% above reality -> inflated
# totals/BTTS/over-under (W/D/L is unaffected, it depends on the att/dfn RATIO).
# Rescale BOTH multipliers equally so the average lambda over all pairings == AVG.
# This preserves every matchup's ratio (W/D/L, player-odds scale) and only fixes the level.
import itertools as _it
_mean_adp = sum(att_mult[a]*dfn_mult[b] for a, b in _it.permutations(TEAMS, 2)) / (len(TEAMS)*(len(TEAMS)-1))
_k = _mean_adp ** 0.5
for t in TEAMS:
    att_mult[t] /= _k; dfn_mult[t] /= _k
print(f"Goal-level calibration: mean(att*dfn) {_mean_adp:.3f} -> 1.000 (rescaled /{_k:.4f}); "
      f"avg total now ~{2*AVG*sum(att_mult[a]*dfn_mult[b] for a,b in _it.permutations(TEAMS,2))/(len(TEAMS)*(len(TEAMS)-1)):.2f}")

# human-readable 0-100 (logistic on within-field z of the log-strengths)
zA = zscores(A_star); zD = zscores(D_star)
def to100(z): return round(100.0 / (1.0 + math.exp(-1.15 * z)), 2)
attack100 = {t: to100(zA[t]) for t in TEAMS}
defense100 = {t: to100(zD[t]) for t in TEAMS}

# =============================================================================
# Write ratings.csv  (+ store global params on every row for the simulator)
# =============================================================================
out = sorted(TEAMS, key=lambda t: -(attack100[t]+defense100[t]))
with open(PROJ + r"\ratings.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["team","attack_100","defense_100","attack_mult","defense_mult",
                "elo","fifa_points","fifa_estimated","matches_since_2015",
                "league_avg_goals","home_adv_mult"])
    mcount = {t:0 for t in TEAMS}
    for d,h,a,hs,as_,n,eh,ea,wb in recent:
        if h in mcount: mcount[h]+=1
        if a in mcount: mcount[a]+=1
    for t in out:
        w.writerow([t, attack100[t], defense100[t],
                    round(att_mult[t],4), round(dfn_mult[t],4),
                    round(elo[t],1), FIFA[t], t in FIFA_ESTIMATED,
                    mcount[t], round(AVG,4), round(home_adv,4)])

print(f"\nWrote ratings.csv  (league_avg={AVG:.3f}, home_adv={home_adv:.3f})\n")
print(f"{'TEAM':<24}{'ATK':>7}{'DEF':>7}{'Elo':>8}{'FIFA':>9}")
for t in out:
    flag = "*" if t in FIFA_ESTIMATED else " "
    print(f"{t:<24}{attack100[t]:>7.2f}{defense100[t]:>7.2f}{elo[t]:>8.0f}{FIFA[t]:>8.0f}{flag}")
print("\n* FIFA points estimated (team ranked outside top 80).")
