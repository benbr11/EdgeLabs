# -*- coding: utf-8 -*-
"""
Does weighting in-tournament games by OPPONENT STRENGTH (quality of wins) improve
knockout prediction?  (User hypothesis: combine in-tournament form with the historical
strength of the teams beaten.)

Same clean OOS setup as wc_form_weight_test.py: build ratings from GROUP-STAGE-only data
(cutoff = first knockout day) at several WC_OPP_GAMMA levels (0 = current model), predict
the 21 real knockout games, score log-loss / Brier / hit-rate. gamma>0 weights each WC
game by the opponent's FIFA strength (on top of the goals model's existing opp-rate adj).

Usage: python wc_oppstrength_test.py
"""
import csv, math, os, sys, subprocess, datetime
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass
BASE = os.path.dirname(os.path.abspath(__file__))
CUTOFF = "2026-06-29"
COMPRESS = 0.60; RHO = -0.12; VAR_BASE, VAR_SLOPE = 6.0, 0.34
GAMMAS = [0.0, 0.5, 1.0, 2.0, 3.0]

def build(gamma):
    out = f"_tmp_ratings_g{gamma}.csv"
    env = dict(os.environ, WC_BUILD_CUTOFF=CUTOFF, WC_OPP_GAMMA=str(gamma),
               WC_BOOST="1.5", FRANCE_NO1="0", WC_RATINGS_OUT=out)
    subprocess.run([sys.executable, "build_ratings.py"], cwd=BASE, env=env,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    return os.path.join(BASE, out)

def load(path):
    R = {r["team"]: r for r in csv.DictReader(open(path, encoding="utf-8"))}
    for t in R:
        R[t]["a"]=float(R[t]["attack_100"]); R[t]["d"]=float(R[t]["defense_100"])
        R[t]["am"]=float(R[t]["attack_mult"]); R[t]["dm"]=float(R[t]["defense_mult"])
    avg=float(next(iter(R.values()))["league_avg_goals"]); HADV=float(next(iter(R.values()))["home_adv_mult"])
    TEAMS=list(R); C2=0.0; n=0
    for a in TEAMS:
        apa=R[a]["am"]**COMPRESS
        for b in TEAMS:
            if a!=b: C2+=apa*(R[b]["dm"]**COMPRESS); n+=1
    return R, avg, HADV, C2/n

def nb_pmf(mu,r,mg):
    return [math.exp(math.lgamma(k+r)-math.lgamma(r)-math.lgamma(k+1)+r*math.log(r/(r+mu))+k*math.log(mu/(r+mu))) for k in range(mg+1)]
def predict(R,avg,HADV,C2,A,B,neutral):
    lamA=avg*(R[A]["am"]**COMPRESS)*(R[B]["dm"]**COMPRESS)/C2
    lamB=avg*(R[B]["am"]**COMPRESS)*(R[A]["dm"]**COMPRESS)/C2
    if not neutral: lamA*=HADV
    dA=VAR_BASE+VAR_SLOPE*((R[A]["a"]+R[A]["d"])/2); dB=VAR_BASE+VAR_SLOPE*((R[B]["a"]+R[B]["d"])/2)
    mg=max(12,int(lamA+lamB)+8); ph=nb_pmf(lamA,dA,mg); pa=nb_pmf(lamB,dB,mg)
    M=[[ph[i]*pa[j] for j in range(mg+1)] for i in range(mg+1)]
    M[0][0]*=max(0.,1-lamA*lamB*RHO);M[0][1]*=max(0.,1+lamA*RHO);M[1][0]*=max(0.,1+lamB*RHO);M[1][1]*=max(0.,1-RHO)
    s=sum(sum(r) for r in M); rng=range(mg+1)
    return (sum(M[i][j] for i in rng for j in rng if i>j)/s,
            sum(M[i][i] for i in rng)/s,
            sum(M[i][j] for i in rng for j in rng if j>i)/s)

ko=[]; cut=datetime.date.fromisoformat(CUTOFF)
for r in csv.DictReader(open(os.path.join(BASE,"results.csv"),encoding="utf-8")):
    if "World Cup" not in r.get("tournament",""): continue
    try:
        d=datetime.date.fromisoformat(r["date"]); hs=int(r["home_score"]); as_=int(r["away_score"])
    except (ValueError,KeyError): continue
    if d<cut: continue
    ko.append((d,r["home_team"],r["away_team"],hs,as_,r.get("neutral","").strip().upper()=="TRUE"))
print(f"Scoring {len(ko)} knockout games (>= {CUTOFF}) built from group-stage-only ratings.\n")
print(f"{'OPP_GAMMA':>10}{'1X2 hit':>9}{'fav hit':>9}{'log-loss':>10}{'Brier':>8}   (0.0 = current model)")
print("="*62)
results=[]
for g in GAMMAS:
    R,avg,HADV,C2 = load(build(g))
    n=hit=favn=favhit=0; ll=0.0; brier=0.0
    for d,h,a,hs,as_,neutral in ko:
        if h not in R or a not in R: continue
        pA,pD,pB=predict(R,avg,HADV,C2,h,a,neutral)
        actual="H" if hs>as_ else ("A" if as_>hs else "D")
        probs={"H":pA,"D":pD,"A":pB}; pred=max(probs,key=probs.get)
        n+=1; hit+=pred==actual; ll+=-math.log(max(probs[actual],1e-12))
        y={"H":0,"D":0,"A":0}; y[actual]=1
        brier+=(pA-y["H"])**2+(pD-y["D"])**2+(pB-y["A"])**2
        if actual in ("H","A"):
            fav="H" if pA>=pB else "A"; favn+=1; favhit+=fav==actual
    results.append((g, hit/n, favhit/favn if favn else 0, ll/n, brier/n))
    print(f"{g:>10.1f}{hit/n*100:>8.0f}%{favhit/favn*100:>8.0f}%{ll/n:>10.4f}{brier/n:>8.4f}")
best=min(results,key=lambda r:r[3])
print("="*62)
print(f"best log-loss at WC_OPP_GAMMA={best[0]} (ll {best[3]:.4f}). Current model = gamma 0.0.")
for g in GAMMAS:
    p=os.path.join(BASE,f"_tmp_ratings_g{g}.csv")
    if os.path.exists(p): os.remove(p)
