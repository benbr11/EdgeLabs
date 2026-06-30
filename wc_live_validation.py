# -*- coding: utf-8 -*-
"""
LIVE validation: how has the model actually done in the 2026 World Cup?

Predicts every real WC 2026 match from PRE-TOURNAMENT ratings (built with
WC_BUILD_CUTOFF so the tournament's own results don't leak in), then compares to
what actually happened. Mirrors simulate.py's compressed Dixon-Coles math.

Usage: python wc_live_validation.py [ratings_file]   (default ratings_pretourney.csv)
"""
import csv, math, os, sys, datetime
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass
BASE = os.path.dirname(os.path.abspath(__file__))
RATINGS = sys.argv[1] if len(sys.argv) > 1 else "ratings_pretourney.csv"
COMPRESS = 0.60; RHO = -0.12; VAR_BASE, VAR_SLOPE = 6.0, 0.34

R = {r["team"]: r for r in csv.DictReader(open(os.path.join(BASE, RATINGS), encoding="utf-8"))}
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

def nb_pmf(mu, r, mg):
    return [math.exp(math.lgamma(k+r)-math.lgamma(r)-math.lgamma(k+1)+r*math.log(r/(r+mu))+k*math.log(mu/(r+mu))) for k in range(mg+1)]
def predict(A, B, neutral, home_is_A):
    lamA = avg*(R[A]["am"]**COMPRESS)*(R[B]["dm"]**COMPRESS)/C2
    lamB = avg*(R[B]["am"]**COMPRESS)*(R[A]["dm"]**COMPRESS)/C2
    if not neutral:
        if home_is_A: lamA *= HADV
        else: lamB *= HADV
    dA = VAR_BASE+VAR_SLOPE*((R[A]["a"]+R[A]["d"])/2); dB = VAR_BASE+VAR_SLOPE*((R[B]["a"]+R[B]["d"])/2)
    mg = max(12, int(lamA+lamB)+8)
    ph = nb_pmf(lamA, dA, mg); pa = nb_pmf(lamB, dB, mg)
    M = [[ph[i]*pa[j] for j in range(mg+1)] for i in range(mg+1)]
    M[0][0]*=max(0.,1-lamA*lamB*RHO); M[0][1]*=max(0.,1+lamA*RHO); M[1][0]*=max(0.,1+lamB*RHO); M[1][1]*=max(0.,1-RHO)
    s=sum(sum(r) for r in M); rng=range(mg+1)
    pA=sum(M[i][j] for i in rng for j in rng if i>j)/s
    pB=sum(M[i][j] for i in rng for j in rng if j>i)/s
    pD=sum(M[i][i] for i in rng)/s
    return pA, pD, pB

# load actual WC 2026 matches
matches = []
for r in csv.DictReader(open(os.path.join(BASE, "results.csv"), encoding="utf-8")):
    if "World Cup" not in r.get("tournament", ""): continue
    try:
        d = datetime.date.fromisoformat(r["date"])
        hs = int(r["home_score"]); as_ = int(r["away_score"])
    except (ValueError, KeyError): continue
    if d < datetime.date(2026, 1, 1): continue
    h, a = r["home_team"], r["away_team"]
    if h not in R or a not in R: continue
    neutral = r.get("neutral", "").strip().upper() == "TRUE"
    matches.append((d, h, a, hs, as_, neutral))
matches.sort()
print(f"Ratings: {RATINGS}   WC 2026 matches found: {len(matches)}"
      + (f"  ({matches[0][0]} -> {matches[-1][0]})" if matches else ""))

hit1x2 = favn = favhit = 0; ll = 0.0; n = 0
buckets = {}
print("\n  date        match                              pred        model%   actual   ok")
for d, h, a, hs, as_, neutral in matches:
    pA, pD, pB = predict(h, a, neutral, home_is_A=True)
    actual = "H" if hs > as_ else ("A" if as_ > hs else "D")
    probs = {"H": pA, "D": pD, "A": pB}
    pred = max(probs, key=probs.get)
    ok = pred == actual
    hit1x2 += 1 if ok else 0; n += 1
    ll += -math.log(max(probs[actual], 1e-12))
    if actual in ("H", "A"):
        fav = "H" if pA >= pB else "A"; favn += 1; favhit += 1 if fav == actual else 0
    conf = probs[pred]; b = min(int(conf*10), 9); buckets.setdefault(b, [0,0.0,0])
    buckets[b][0]+=1; buckets[b][1]+=conf; buckets[b][2]+= 1 if ok else 0
    predname = {"H": h, "A": a, "D": "Draw"}[pred]
    res = f"{h[:14]} {hs}-{as_} {a[:14]}"
    print(f"  {d}  {res:34}{predname[:11]:>12}{conf*100:>7.0f}%   {actual:>4}    {'Y' if ok else '.'}")

if n:
    print("\n" + "="*60)
    print(f"  WC 2026 matches scored : {n}")
    print(f"  1X2 hit-rate (H/D/A)   : {hit1x2/n*100:.1f}%  ({hit1x2}/{n})")
    print(f"  favorite hit (decisive): {favhit/favn*100:.1f}%  ({favhit}/{favn})" if favn else "  (no decisive games)")
    print(f"  log-loss               : {ll/n:.4f}")
    print(f"  calibration:  {'bucket':>8}{'n':>5}{'pred%':>8}{'actual%':>9}")
    for b in sorted(buckets):
        cnt, sp, ok = buckets[b]
        print(f"               {b/10:.1f}-{(b+1)/10:.1f}{cnt:>5}{sp/cnt*100:>7.0f}%{ok/cnt*100:>8.0f}%")
    print("\n  benchmark: WC group/knockout 1X2 is ~50-55% for a good model;")
    print("  draws (~25% of group games) are the hardest single outcome to call.")
