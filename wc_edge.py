# -*- coding: utf-8 -*-
"""
EDGE PROTOTYPE (the operational path the research endorses).

Our model is well-calibrated to actual results (validated: temperature T*=1.02).
So the ONLY place value can live is where our calibrated probability disagrees
with the de-vigged market price by enough to clear the vig. This tool flags those
spots on the upcoming matches and frames them honestly as HYPOTHESES, not proven
bets — the WC market is efficient (closing-line R^2=0.997), so a disagreement is a
candidate to be PROVEN via closing-line value, not an automatic bet.

It also stubs the CLV log: record (date, match, outcome, our_prob, price_taken).
Later, fill in the closing price and check whether we consistently beat it. Positive
CLV over many bets = real edge; anything less = stand down. Needs a live odds feed
(the-odds-api) + lineup-news ingestion to run for real — this is the framework.
"""
import csv, math, os, sys
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass
BASE=os.path.dirname(os.path.abspath(__file__)); COMPRESS=0.60; RHO=-0.12; VAR_BASE,VAR_SLOPE=6.0,0.34
HOSTS={"United States","Canada","Mexico"}; VALUE_THR=0.05   # flag if model prob exceeds fair market by >=5pp

R={r["team"]:r for r in csv.DictReader(open(os.path.join(BASE,"ratings.csv"),encoding="utf-8"))}
for t in R:
    R[t]["a"]=float(R[t]["attack_100"]);R[t]["d"]=float(R[t]["defense_100"])
    R[t]["am"]=float(R[t]["attack_mult"]);R[t]["dm"]=float(R[t]["defense_mult"])
avg=float(next(iter(R.values()))["league_avg_goals"]);HADV=float(next(iter(R.values()))["home_adv_mult"]);TEAMS=list(R.keys())
AVAIL={}
try:
    for r in csv.DictReader(open(os.path.join(BASE,"wc_availability.csv"),encoding="utf-8")): AVAIL[r["team"]]=float(r["avail"])
except FileNotFoundError: pass
def c2(P):
    s=0.0;n=0
    for a in TEAMS:
        ap=R[a]["am"]**P
        for b in TEAMS:
            if a!=b: s+=ap*(R[b]["dm"]**P);n+=1
    return s/n
C2=c2(COMPRESS)
def nb(mu,r,mg): return [math.exp(math.lgamma(k+r)-math.lgamma(r)-math.lgamma(k+1)+r*math.log(r/(r+mu))+k*math.log(mu/(r+mu))) for k in range(mg+1)]
def model3(A,B):
    avA=AVAIL.get(A,1.0);avB=AVAIL.get(B,1.0)
    amA,dmA=R[A]["am"]*avA,R[A]["dm"]/avA; amB,dmB=R[B]["am"]*avB,R[B]["dm"]/avB
    lA=avg*(amA**COMPRESS)*(dmB**COMPRESS)/C2; lB=avg*(amB**COMPRESS)*(dmA**COMPRESS)/C2
    if A not in HOSTS and B in HOSTS: lB*=HADV
    elif A in HOSTS: lA*=HADV
    dA=VAR_BASE+VAR_SLOPE*((R[A]["a"]+R[A]["d"])/2);dB=VAR_BASE+VAR_SLOPE*((R[B]["a"]+R[B]["d"])/2)
    mg=max(12,int(lA+lB)+8);ph=nb(lA,dA,mg);pa=nb(lB,dB,mg)
    M=[[ph[i]*pa[j] for j in range(mg+1)] for i in range(mg+1)]
    M[0][0]*=max(0.,1-lA*lB*RHO);M[0][1]*=max(0.,1+lA*RHO);M[1][0]*=max(0.,1+lB*RHO);M[1][1]*=max(0.,1-RHO)
    sm=sum(sum(r) for r in M);rng=range(mg+1)
    pH=sum(M[i][j] for i in rng for j in rng if i>j)/sm;pA=sum(M[i][j] for i in rng for j in rng if j>i)/sm;pD=sum(M[i][i] for i in rng)/sm
    return pH,pD,pA
# de-vigged market 3-way (home/draw/away), FanDuel June 27-29 (from odds research)
MKT={("Brazil","Japan"):(55.4,26.4,18.3),("Germany","Paraguay"):(72.9,18.4,8.7),
 ("Ivory Coast","Norway"):(26.3,28.7,45.1),("France","Sweden"):(75.2,16.1,8.6),
 ("Mexico","Ecuador"):(43.1,33.2,23.7),("England","DR Congo"):(75.5,17.7,6.8),
 ("Belgium","Senegal"):(44.1,29.6,26.3),("United States","Bosnia and Herzegovina"):(70.8,18.7,10.6),
 ("Spain","Austria"):(73.2,18.2,8.6),("Portugal","Croatia"):(54.2,26.6,19.2),
 ("Switzerland","Algeria"):(46.4,29.8,23.8),("Australia","Egypt"):(28.1,33.6,38.3),
 ("Argentina","Cape Verde"):(82.7,12.6,4.7),("Colombia","Ghana"):(62.3,24.2,13.5)}

flags=[]
for (A,B),(mh,md,ma) in MKT.items():
    if A not in R or B not in R: continue
    pH,pD,pA=model3(A,B); model=[pH*100,pD*100,pA*100]; mkt=[mh,md,ma]; lbl=[A,"Draw",B]
    for i in range(3):
        edge=model[i]-mkt[i]
        if edge>=VALUE_THR*100:
            flags.append((edge,f"{A} vs {B}",lbl[i],model[i],mkt[i]))
flags.sort(reverse=True)
print("="*78)
print("MODEL-vs-MARKET VALUE CANDIDATES  (our calibrated prob exceeds the fair market")
print(f"price by >= {VALUE_THR*100:.0f}pp). HYPOTHESES to prove via closing-line value — NOT bets yet.")
print("="*78)
print(f"  {'outcome':28}{'our%':>7}{'mkt fair%':>11}{'edge':>7}")
for edge,match,out,mp,kp in flags:
    print(f"  {(out+' ('+match+')')[:28]:28}{mp:>6.0f}%{kp:>10.0f}%{edge:>+6.0f}")
print(f"\n  {len(flags)} candidates. Note the pattern: a calibrated model that is less")
print("  bullish on favorites than the market will surface DRAWS and UNDERDOGS as value.")
print("  That is a hypothesis about a favorite-longshot lean, NOT a proven edge.")
print("\nNEXT (to make it real): 1) pull LIVE odds at bet time (the-odds-api), not these")
print("  stale lines;  2) log CLV = our price vs the closing price;  3) only stake after")
print("  a positive-CLV sample. Without that, treat every flag above as unproven.")
