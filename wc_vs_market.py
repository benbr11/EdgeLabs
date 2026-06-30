# -*- coding: utf-8 -*-
"""
Measure the model against the live R32 market odds (the "Vegas" benchmark), and
diagnose the systematic favorite gap by sweeping the compression parameter P.

Market = de-vigged FanDuel lines (June 27-29 2026), from the odds-research agent.
For each compression level we compare the model's favorite win/advance prob to the
market's. If a LESS-compressed model (higher P) matches the market's favorite
confidence better, compression (tuned on qualifier-heavy data) over-shrank the
strong-favorite edge for the WC field.
"""
import csv, math, os, sys
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass
BASE = os.path.dirname(os.path.abspath(__file__))
RHO = -0.12; VAR_BASE, VAR_SLOPE = 6.0, 0.34
HOSTS = {"United States", "Canada", "Mexico"}

# market de-vigged probs: home, away -> (m_home, m_draw, m_away, m_adv_home)
# (Netherlands-Morocco 3-way not published cleanly; advance only.)
MKT = {
 ("Brazil","Japan"):(55.4,26.4,18.3,72.0),
 ("Germany","Paraguay"):(72.9,18.4,8.7,84.4),
 ("Netherlands","Morocco"):(None,None,None,56.3),
 ("Ivory Coast","Norway"):(26.3,28.7,45.1,38.8),
 ("France","Sweden"):(75.2,16.1,8.6,86.4),
 ("Mexico","Ecuador"):(43.1,33.2,23.7,61.4),
 ("England","DR Congo"):(75.5,17.7,6.8,85.3),
 ("Belgium","Senegal"):(44.1,29.6,26.3,61.0),
 ("United States","Bosnia and Herzegovina"):(70.8,18.7,10.6,83.9),
 ("Spain","Austria"):(73.2,18.2,8.6,86.4),
 ("Portugal","Croatia"):(54.2,26.6,19.2,70.7),
 ("Switzerland","Algeria"):(46.4,29.8,23.8,64.0),
 ("Australia","Egypt"):(28.1,33.6,38.3,44.1),
 ("Argentina","Cape Verde"):(82.7,12.6,4.7,92.0),
 ("Colombia","Ghana"):(62.3,24.2,13.5,79.9),
}

R = {r["team"]: r for r in csv.DictReader(open(os.path.join(BASE, "ratings.csv"), encoding="utf-8"))}
for t in R:
    R[t]["a"]=float(R[t]["attack_100"]); R[t]["d"]=float(R[t]["defense_100"])
    R[t]["am"]=float(R[t]["attack_mult"]); R[t]["dm"]=float(R[t]["defense_mult"])
avg=float(next(iter(R.values()))["league_avg_goals"]); HADV=float(next(iter(R.values()))["home_adv_mult"])
TEAMS=list(R.keys())

def c2_for(P):
    s=0.0;n=0
    for a in TEAMS:
        ap=R[a]["am"]**P
        for b in TEAMS:
            if a==b: continue
            s+=ap*(R[b]["dm"]**P); n+=1
    return s/n
def nb(mu,r,mg):
    return [math.exp(math.lgamma(k+r)-math.lgamma(r)-math.lgamma(k+1)+r*math.log(r/(r+mu))+k*math.log(mu/(r+mu))) for k in range(mg+1)]
def mat(lA,lB,dA,dB):
    mg=max(12,int(lA+lB)+8); ph=nb(lA,dA,mg); pa=nb(lB,dB,mg)
    M=[[ph[i]*pa[j] for j in range(mg+1)] for i in range(mg+1)]
    M[0][0]*=max(0.,1-lA*lB*RHO);M[0][1]*=max(0.,1+lA*RHO);M[1][0]*=max(0.,1+lB*RHO);M[1][1]*=max(0.,1-RHO)
    s=sum(sum(r) for r in M);rng=range(mg+1)
    pA=sum(M[i][j] for i in rng for j in rng if i>j)/s;pB=sum(M[i][j] for i in rng for j in rng if j>i)/s;pD=sum(M[i][i] for i in rng)/s
    return pA,pD,pB
def predict(A,B,P,C2):
    neutral = A not in HOSTS
    lA=avg*(R[A]["am"]**P)*(R[B]["dm"]**P)/C2; lB=avg*(R[B]["am"]**P)*(R[A]["dm"]**P)/C2
    if not neutral: lA*=HADV
    dA=VAR_BASE+VAR_SLOPE*((R[A]["a"]+R[A]["d"])/2); dB=VAR_BASE+VAR_SLOPE*((R[B]["a"]+R[B]["d"])/2)
    pA,pD,pB=mat(lA,lB,dA,dB)
    petA,petD,petB=mat(lA/3,lB/3,dA,dB)
    share=pA/(pA+pB) if pA+pB>0 else .5; psA=min(.55,max(.45,.5+(share-.5)*.2))
    advA=pA+pD*(petA+petD*psA)
    return pA,pD,pB,advA

print("Comparing model to de-vigged market on the R32 favorites.\n")
print("Sweep: mean model favorite WIN% (90') and ADVANCE% vs the market's, by compression P")
print("="*72)
print(f"{'P':>5}{'model fav win%':>16}{'mkt fav win%':>14}{'model adv%':>12}{'mkt adv%':>10}{'win gap':>9}")
for P in [1.00, 0.85, 0.75, 0.60]:
    C2=c2_for(P)
    mwins=[]; mkt_wins=[]; madv=[]; mkt_adv=[]
    for (A,B),(mh,md,ma,madvh) in MKT.items():
        pA,pD,pB,advA=predict(A,B,P,C2)
        # favorite per market 3-way (or advance if 3-way missing)
        if mh is not None:
            fav_home = mh>=ma
        else:
            fav_home = madvh>=50
        mwins.append(pA if fav_home else pB)
        mkt_wins.append((mh if fav_home else ma) if mh is not None else None)
        madv.append(advA if fav_home else 1-advA)
        mkt_adv.append(madvh if fav_home else 100-madvh)
    valid=[(m,k) for m,k in zip(mwins,mkt_wins) if k is not None]
    mw=sum(m for m,_ in valid)/len(valid)*100; kw=sum(k for _,k in valid)/len(valid)
    ma_=sum(madv)/len(madv)*100; ka=sum(mkt_adv)/len(mkt_adv)
    print(f"{P:>5.2f}{mw:>15.1f}%{kw:>13.1f}%{ma_:>11.1f}%{ka:>9.1f}%{mw-kw:>+8.1f}")

# per-match detail at current production P=0.60 and at P=0.85
print("\n" + "="*72)
print("PER-MATCH: model (P=0.60 current) vs market — favorite to ADVANCE")
print("="*72)
C2c=c2_for(0.60); C2b=c2_for(0.85)
print(f"  {'match':34}{'mdl0.60':>9}{'mdl0.85':>9}{'market':>8}")
for (A,B),(mh,md,ma,madvh) in MKT.items():
    _,_,_,advA6=predict(A,B,0.60,C2c); _,_,_,advA8=predict(A,B,0.85,C2b)
    fav_home = (mh>=ma) if mh is not None else (madvh>=50)
    m6=advA6 if fav_home else 1-advA6; m8=advA8 if fav_home else 1-advA8
    favname=A if fav_home else B
    print(f"  {(favname+' adv'):34}{m6*100:>8.0f}%{m8*100:>8.0f}%{madvh if fav_home else 100-madvh:>7.0f}%")
