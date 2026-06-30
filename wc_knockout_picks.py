# -*- coding: utf-8 -*-
"""
WC 2026 knockout betting board: predict each upcoming match and tier it
LOCK (obvious, safe) / LEAN / STAY AWAY (upset-prone, don't bet).

Knockout-aware: a draw in 90' -> extra time -> penalties (~coin flip), so the
ADVANCE probability (not the 90-min win prob) is what matters, and a strong team
in a low-scoring tie can still be a coin flip. Mirrors simulate.py's compressed
Dixon-Coles + extra-time/penalty math, on current (refreshed) ratings.

Upset-risk flags:
  * coin-flip     : favorite advances < 60%
  * one-goal game : low total expected goals (<2.2) -> a single moment/penalties decides it
  * goes-the-distance : high 90' draw prob with a non-dominant favorite -> likely penalties
"""
import csv, math, os, sys, unicodedata
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass
BASE = os.path.dirname(os.path.abspath(__file__))
COMPRESS = 0.60; RHO = -0.12; VAR_BASE, VAR_SLOPE = 6.0, 0.34
HOSTS = {"United States", "Canada", "Mexico"}
XG_NAME = {"Cabo Verde": "Cape Verde", "Congo DR": "DR Congo", "Czechia": "Czech Republic",
           "Côte d'Ivoire": "Ivory Coast", "IR Iran": "Iran", "Türkiye": "Turkey", "USA": "United States"}

R = {r["team"]: r for r in csv.DictReader(open(os.path.join(BASE, "ratings.csv"), encoding="utf-8"))}
for t in R:
    R[t]["a"] = float(R[t]["attack_100"]); R[t]["d"] = float(R[t]["defense_100"])
    R[t]["am"] = float(R[t]["attack_mult"]); R[t]["dm"] = float(R[t]["defense_mult"])
avg = float(next(iter(R.values()))["league_avg_goals"])
HADV = float(next(iter(R.values()))["home_adv_mult"])
TEAMS = list(R.keys())
C2 = 0.0; n2 = 0
for a in TEAMS:
    apa = R[a]["am"] ** COMPRESS
    for b in TEAMS:
        if a == b: continue
        C2 += apa * (R[b]["dm"] ** COMPRESS); n2 += 1
C2 /= n2

def fold(s):
    return "".join(c for c in unicodedata.normalize("NFKD", str(s).lower()) if not unicodedata.combining(c))
_FOLD = {fold(t): t for t in TEAMS}
def resolve(name):
    if name in R: return name
    if name in XG_NAME and XG_NAME[name] in R: return XG_NAME[name]
    f = fold(name)
    if f in _FOLD: return _FOLD[f]
    return None

# in-tournament availability multipliers (key absences not already in group ratings)
AVAIL = {}
try:
    for r in csv.DictReader(open(os.path.join(BASE, "wc_availability.csv"), encoding="utf-8")):
        AVAIL[r["team"]] = float(r["avail"])
except FileNotFoundError:
    pass

# venue conditions (altitude/heat) + each team's home baseline -> relative fitness penalty.
# A team is only penalised at a venue HOTTER or HIGHER than what it's acclimatised to.
CTX = {}
try:
    for r in csv.DictReader(open(os.path.join(BASE, "context.csv"), encoding="utf-8")):
        CTX[r["team"]] = (float(r["home_alt_m"]), float(r["home_temp_c"]))
except (FileNotFoundError, KeyError):
    pass
VENUES = {}
try:
    for r in csv.DictReader(open(os.path.join(BASE, "wc_venues.csv"), encoding="utf-8")):
        VENUES[r["city"]] = (float(r["altitude_m"]), float(r["temp_c"]), r["open_air"].strip() == "1")
except FileNotFoundError:
    pass
ALT_PEN_PER_KM = 0.05; ALT_BUFFER = 500; HEAT_PEN_PER_C = 0.005; HEAT_BUFFER = 8
def venue_factor(team, city):
    """Fitness multiplier (<=1.0) for a team at a venue, vs its home altitude/heat."""
    if city not in VENUES or team not in CTX:
        return 1.0
    valt, vtemp, openair = VENUES[city]; halt, htemp = CTX[team]
    f = 1.0
    if valt > halt + ALT_BUFFER:
        f *= max(0.0, 1 - ALT_PEN_PER_KM * (valt - halt - ALT_BUFFER) / 1000.0)
    if openair and vtemp > htemp + HEAT_BUFFER:
        f *= max(0.0, 1 - HEAT_PEN_PER_C * (vtemp - htemp - HEAT_BUFFER))
    return f

def nb_pmf(mu, r, mg):
    return [math.exp(math.lgamma(k+r)-math.lgamma(r)-math.lgamma(k+1)+r*math.log(r/(r+mu))+k*math.log(mu/(r+mu))) for k in range(mg+1)]
def matrix(lamA, lamB, dA, dB):
    mg = max(12, int(lamA+lamB)+8)
    ph = nb_pmf(lamA, dA, mg); pa = nb_pmf(lamB, dB, mg)
    M = [[ph[i]*pa[j] for j in range(mg+1)] for i in range(mg+1)]
    M[0][0]*=max(0.,1-lamA*lamB*RHO); M[0][1]*=max(0.,1+lamA*RHO); M[1][0]*=max(0.,1+lamB*RHO); M[1][1]*=max(0.,1-RHO)
    s=sum(sum(r) for r in M); M=[[v/s for v in r] for r in M]; rng=range(mg+1)
    pA=sum(M[i][j] for i in rng for j in rng if i>j); pB=sum(M[i][j] for i in rng for j in rng if j>i); pD=sum(M[i][i] for i in rng)
    exA=sum(i*sum(M[i]) for i in rng); exB=sum(j*sum(M[i][j] for i in rng) for j in rng)
    return pA, pD, pB, exA, exB

def predict_ko(A, B, neutral, home_is_A, city=None):
    # injury/availability AND venue (altitude/heat) both weaken a team: lower attack, worse defense
    avA = AVAIL.get(A, 1.0) * venue_factor(A, city); avB = AVAIL.get(B, 1.0) * venue_factor(B, city)
    amA, dmA = R[A]["am"]*avA, R[A]["dm"]/avA
    amB, dmB = R[B]["am"]*avB, R[B]["dm"]/avB
    lamA = avg*(amA**COMPRESS)*(dmB**COMPRESS)/C2
    lamB = avg*(amB**COMPRESS)*(dmA**COMPRESS)/C2
    if not neutral:
        if home_is_A: lamA *= HADV
        else: lamB *= HADV
    dA = VAR_BASE+VAR_SLOPE*((R[A]["a"]+R[A]["d"])/2); dB = VAR_BASE+VAR_SLOPE*((R[B]["a"]+R[B]["d"])/2)
    pA, pD, pB, exA, exB = matrix(lamA, lamB, dA, dB)
    # extra time = 1/3 of 90', then penalties ~ coin flip (tiny favorite edge)
    petA, petD, petB, _, _ = matrix(lamA/3.0, lamB/3.0, dA, dB)
    share = pA/(pA+pB) if (pA+pB) > 0 else 0.5
    psA = min(0.55, max(0.45, 0.5 + (share-0.5)*0.2))
    advA = pA + pD*(petA + petD*psA)
    advB = pB + pD*(petB + petD*(1-psA))
    return pA, pD, pB, advA, advB, exA, exB

# ---- load upcoming fixtures ----
games = []
for r in csv.DictReader(open(os.path.join(BASE, "wc2026_xg.csv"), encoding="utf-8")):
    if r.get("status") == "Completed": continue
    hA = resolve(r.get("home_team_name", "")); aB = resolve(r.get("away_team_name", ""))
    if hA is None or aB is None:
        print(f"  (skip: {r.get('home_team_name')} vs {r.get('away_team_name')} on {r.get('date')} -- team not resolved)")
        continue
    neutral = hA not in HOSTS          # host plays at home; otherwise neutral knockout
    games.append((r.get("date", ""), hA, aB, neutral, r.get("city", "")))

rows = []
for date, A, B, neutral, city in games:
    pA, pD, pB, advA, advB, exA, exB = predict_ko(A, B, neutral, home_is_A=True, city=city)
    fav, adv = (A, advA) if advA >= advB else (B, advB)
    total = exA + exB
    notes = []
    if adv < 0.60: notes.append("COIN-FLIP")
    if total < 2.2: notes.append("low-scoring (one goal decides it)")
    if pD > 0.30 and adv < 0.72: notes.append("likely ET/penalties")
    tier = "LOCK" if adv >= 0.70 else ("LEAN" if adv >= 0.60 else "STAY AWAY")
    rows.append({"date": date, "A": A, "B": B, "fav": fav, "adv": adv, "neutral": neutral,
                 "pA": pA, "pD": pD, "pB": pB, "total": total, "tier": tier, "notes": notes})

def show(title, sel):
    print("\n" + "=" * 76); print(title); print("=" * 76)
    if not sel: print("  (none)"); return
    for r in sel:
        host = "" if r["neutral"] else "  (host adv)"
        nt = ("   <- " + ", ".join(r["notes"])) if r["notes"] else ""
        print(f"  {r['date']}  {r['A']} vs {r['B']}{host}")
        print(f"      pick: {r['fav']} to advance {r['adv']*100:4.0f}%   |   90': "
              f"{r['A']} {r['pA']*100:.0f}% / draw {r['pD']*100:.0f}% / {r['B']} {r['pB']*100:.0f}%   "
              f"|   exp goals {r['total']:.1f}{nt}")

rows.sort(key=lambda r: -r["adv"])
print(f"\nWC 2026 ROUND OF 32 — BETTING BOARD ({len(rows)} games)   ratings through current data")
show("✅ LOCKS — most obvious, safest picks (favorite advances >= 70%)", [r for r in rows if r["tier"] == "LOCK"])
show("🟡 LEANS — solid but not safe (60-70%)", [r for r in rows if r["tier"] == "LEAN"])
show("⛔ STAY AWAY — upset-prone / coin-flips (don't bet the result)", [r for r in rows if r["tier"] == "STAY AWAY"])
