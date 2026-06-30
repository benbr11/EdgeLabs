# -*- coding: utf-8 -*-
"""
WORLD CUP ACCURACY AUDIT — find every "Morocco-type" issue at scale.

Three systematic checks against the cross-confederation benchmark (World Football
Elo + FIFA, from intl_benchmark.csv):

  1. TEAM divergence: model overall rank vs the Elo/FIFA consensus rank. Big gaps =
     a team the model over- or under-rates (Morocco was the poster child).
  2. DEFENSE-tilt extremes: defense_100 - attack_100, the padded-defense signature.
  3. MATCHUP disagreements: for every pair, the model's 2-way win prob (compressed
     Dixon-Coles, mirrors simulate.py) vs the Elo-implied win prob. Biggest gaps =
     matchups where the model and the market/Elo most disagree -> audit candidates.

Read-only. Flags issues; does not change the model.
"""
import csv, math, os, sys
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass
BASE = os.path.dirname(os.path.abspath(__file__))

COMPRESS = 0.60; RHO = -0.12; VAR_BASE, VAR_SLOPE = 6.0, 0.34

R = {r["team"]: r for r in csv.DictReader(open(os.path.join(BASE, "ratings.csv"), encoding="utf-8"))}
for t in R:
    R[t]["a"] = float(R[t]["attack_100"]); R[t]["d"] = float(R[t]["defense_100"])
    R[t]["am"] = float(R[t]["attack_mult"]); R[t]["dm"] = float(R[t]["defense_mult"])
avg = float(next(iter(R.values()))["league_avg_goals"])
TEAMS = list(R.keys())

bench = {}
bpath = os.path.join(BASE, "intl_benchmark.csv")
if os.path.exists(bpath):
    for r in csv.DictReader(open(bpath, encoding="utf-8")):
        bench[r["team"]] = r

# C2 re-level (mirror simulate.py)
C2 = 0.0; n2 = 0
for a in TEAMS:
    apa = R[a]["am"] ** COMPRESS
    for b in TEAMS:
        if a == b: continue
        C2 += apa * (R[b]["dm"] ** COMPRESS); n2 += 1
C2 /= n2

def nb_pmf(mu, r, mg):
    return [math.exp(math.lgamma(k+r)-math.lgamma(r)-math.lgamma(k+1)+r*math.log(r/(r+mu))+k*math.log(mu/(r+mu))) for k in range(mg+1)]
def model_2way(A, B):
    lamA = avg*(R[A]["am"]**COMPRESS)*(R[B]["dm"]**COMPRESS)/C2
    lamB = avg*(R[B]["am"]**COMPRESS)*(R[A]["dm"]**COMPRESS)/C2
    dA = VAR_BASE+VAR_SLOPE*((R[A]["a"]+R[A]["d"])/2); dB = VAR_BASE+VAR_SLOPE*((R[B]["a"]+R[B]["d"])/2)
    mg = max(12, int(lamA+lamB)+8)
    ph = nb_pmf(lamA, dA, mg); pa = nb_pmf(lamB, dB, mg)
    M = [[ph[i]*pa[j] for j in range(mg+1)] for i in range(mg+1)]
    M[0][0]*=max(0.,1-lamA*lamB*RHO); M[0][1]*=max(0.,1+lamA*RHO); M[1][0]*=max(0.,1+lamB*RHO); M[1][1]*=max(0.,1-RHO)
    s=sum(sum(r) for r in M); rng=range(mg+1)
    pA=sum(M[i][j] for i in rng for j in rng if i>j)/s; pB=sum(M[i][j] for i in rng for j in rng if j>i)/s
    return pA/(pA+pB) if (pA+pB)>0 else 0.5

# ---- 1. TEAM divergence: model overall rank vs Elo/FIFA consensus ----
model_rank = {t: i+1 for i, t in enumerate(sorted(TEAMS, key=lambda t: -(R[t]["a"]+R[t]["d"])))}
out_div = []
if bench:
    elo_rank = {t: i+1 for i, t in enumerate(sorted(TEAMS, key=lambda t: -float(bench[t]["elo"])))}
    fifa_rank = {t: i+1 for i, t in enumerate(sorted(TEAMS, key=lambda t: -float(bench[t]["fifa_points"])))}
    for t in TEAMS:
        cons = (elo_rank[t]+fifa_rank[t])/2.0
        out_div.append((t, model_rank[t], elo_rank[t], fifa_rank[t], model_rank[t]-cons))
    print("="*78)
    print("1. TEAM RATING vs BENCHMARK  (model overall rank vs Elo/FIFA consensus rank)")
    print("   negative gap = model rates HIGHER than consensus (over-rates); positive = under-rates")
    print("="*78)
    print(f"   {'team':18}{'model#':>7}{'elo#':>6}{'fifa#':>6}{'gap':>7}")
    print("   -- model OVER-rates vs consensus (potential Morocco-type inflation) --")
    for t, mr, er, fr, g in sorted(out_div, key=lambda x: x[4])[:8]:
        print(f"   {t:18}{mr:>7}{er:>6}{fr:>6}{g:>+7.1f}")
    print("   -- model UNDER-rates vs consensus --")
    for t, mr, er, fr, g in sorted(out_div, key=lambda x: -x[4])[:8]:
        print(f"   {t:18}{mr:>7}{er:>6}{fr:>6}{g:>+7.1f}")
else:
    print("(intl_benchmark.csv not found -- skipping benchmark divergence)")

# ---- 2. DEFENSE-tilt extremes ----
print("\n"+"="*78)
print("2. DEFENSE-OVER-ATTACK TILT  (defense_100 - attack_100; padded-defense signature)")
print("="*78)
gap = sorted(TEAMS, key=lambda t: -(R[t]["d"]-R[t]["a"]))
print(f"   {'team':18}{'ATK':>6}{'DEF':>6}{'gap':>6}")
for t in gap[:8]:
    print(f"   {t:18}{R[t]['a']:>6.0f}{R[t]['d']:>6.0f}{R[t]['d']-R[t]['a']:>+6.0f}")

# ---- 3. MATCHUP disagreements vs Elo ----
if bench:
    print("\n"+"="*78)
    print("3. BIGGEST MODEL vs ELO MATCHUP DISAGREEMENTS  (2-way win prob, neutral)")
    print("="*78)
    elo = {t: float(bench[t]["elo"]) for t in TEAMS}
    pairs = []
    for i, A in enumerate(TEAMS):
        for B in TEAMS[i+1:]:
            mA = model_2way(A, B)
            eA = 1.0/(1.0+10**((elo[B]-elo[A])/400.0))
            pairs.append((abs(mA-eA), A, B, mA, eA))
    pairs.sort(reverse=True)
    print(f"   {'matchup':36}{'model A%':>9}{'elo A%':>8}{'gap':>7}")
    for d, A, B, mA, eA in pairs[:15]:
        print(f"   {A+' vs '+B:36}{mA*100:>8.0f}%{eA*100:>7.0f}%{(mA-eA)*100:>+6.0f}")
    print(f"\n   mean |model-elo| over all {len(pairs)} pairs: {sum(p[0] for p in pairs)/len(pairs)*100:.1f} pts")
