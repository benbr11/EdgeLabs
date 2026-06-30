# -*- coding: utf-8 -*-
"""
Calibration layer for the WC model's W/D/L probabilities.

Fits a single-parameter TEMPERATURE on the model's out-of-sample predictions vs
actual results (low overfit risk; the safe choice for small samples per the
literature). T<1 sharpens (model was under-confident), T>1 softens (over-confident),
T~=1 means already calibrated. Validated TIME-WISE: fit T on the earlier period,
measure log-loss improvement on a later held-out period (no leakage). Only worth
integrating if the held-out log-loss actually improves.

Mirrors the production model: goals+Elo core, compression P=0.60, half-life 600.
"""
import csv, math, datetime, itertools, sys, collections, os
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass
BASE = os.path.dirname(os.path.abspath(__file__))
GOALS_CUTOFF_YEARS=11; HALFLIFE_DAYS=600.0; BASE_ELO=1500.0; HFA_ELO=65.0
RHO=-0.12; VAR_BASE,VAR_SLOPE=6.0,0.34; GOALS_ITERS=40; COMPRESS=0.60
W_GOALS_REL,W_ELO_REL=0.22,0.16; _w=W_GOALS_REL+W_ELO_REL
W_GOALS_CONS=W_GOALS_REL/_w; W_ELO_CONS=W_ELO_REL/_w
TEST_START=datetime.date(2024,1,1); TEST_END=datetime.date(2026,12,31)
SPLIT=datetime.date(2025,9,1)   # fit T before this; validate on/after
COMPETITIVE={"FIFA World Cup qualification","FIFA World Cup","UEFA Euro","UEFA Euro qualification",
 "Copa América","African Cup of Nations","African Cup of Nations qualification","AFC Asian Cup",
 "AFC Asian Cup qualification","CONCACAF Nations League","Gold Cup","UEFA Nations League","CONMEBOL","Copa America"}
MINP=15

ALL=[]
for r in csv.DictReader(open(os.path.join(BASE,"results.csv"),encoding="utf-8")):
    try:
        d=datetime.date.fromisoformat(r["date"]); hs=int(r["home_score"]); as_=int(r["away_score"])
    except (ValueError,KeyError): continue
    ALL.append((d,r["home_team"],r["away_team"],hs,as_,r["neutral"].strip().upper()=="TRUE",r["tournament"]))
ALL.sort(key=lambda x:x[0])
TESTS=[m for m in ALL if TEST_START<=m[0]<=TEST_END and m[6] in COMPETITIVE]

def zsc(d):
    v=list(d.values()); m=sum(v)/len(v); sd=(sum((x-m)**2 for x in v)/len(v))**0.5 or 1.0
    return {k:(x-m)/sd for k,x in d.items()}
def snap(asof):
    hist=[m for m in ALL if m[0]<asof]
    if len(hist)<1000: return None
    md=hist[-1][0]; elo={}
    def kf(gd): return 30.0*(1.0 if gd<=1 else (1.5 if gd==2 else (11+gd)/8.0))
    for d,h,a,hs,as_,n,tn in hist:
        eh=elo.get(h,BASE_ELO);ea=elo.get(a,BASE_ELO);adj=0 if n else HFA_ELO
        e=1/(1+10**((ea-(eh+adj))/400));res=1.0 if hs>as_ else(0.5 if hs==as_ else 0)
        dl=kf(abs(hs-as_))*(res-e);elo[h]=eh+dl;elo[a]=ea-dl
    cut=datetime.date(md.year-GOALS_CUTOFF_YEARS,1,1);rec=[m for m in hist if m[0]>=cut]
    def wt(d): return 0.5**((md-d).days/HALFLIFE_DAYS)
    tg=tw=0.0
    for d,h,a,hs,as_,n,tn in rec: w=wt(d);tg+=w*(hs+as_);tw+=w*2
    if tw==0: return None
    AVG=tg/tw; att={};dfn={}
    for d,h,a,hs,as_,n,tn in rec:
        for t in(h,a): att.setdefault(t,1.0);dfn.setdefault(t,1.0)
    nc=collections.Counter()
    for d,h,a,hs,as_,n,tn in rec: nc[h]+=1;nc[a]+=1
    for _ in range(GOALS_ITERS):
        na={t:0.0 for t in att};da={t:0.0 for t in att};nd={t:0.0 for t in att};dd={t:0.0 for t in att}
        for d,h,a,hs,as_,n,tn in rec:
            w=wt(d);na[h]+=w*hs;da[h]+=w*AVG*dfn[a];nd[a]+=w*hs;dd[a]+=w*AVG*att[h]
            na[a]+=w*as_;da[a]+=w*AVG*dfn[h];nd[h]+=w*as_;dd[h]+=w*AVG*att[a]
        for t in att:
            if da[t]>0: att[t]=max(na[t]/da[t],1e-6)
            if dd[t]>0: dfn[t]=max(nd[t]/dd[t],1e-6)
        for dc in(att,dfn):
            gm=math.exp(sum(math.log(max(v,1e-6)) for v in dc.values())/len(dc))
            for t in dc: dc[t]/=gm
    tm=list(att.keys())
    AL={t:math.log(att[t]) for t in tm};DL={t:-math.log(dfn[t]) for t in tm}
    g={t:AL[t]+DL[t] for t in tm};ti={t:AL[t]-DL[t] for t in tm}
    zg=zsc(g);ze=zsc({t:elo.get(t,BASE_ELO) for t in tm})
    cz={t:W_GOALS_CONS*zg[t]+W_ELO_CONS*ze[t] for t in tm}
    gm=sum(g.values())/len(tm);gs=(sum((v-gm)**2 for v in g.values())/len(tm))**0.5 or 1.0
    G={t:gm+gs*cz[t] for t in tm};As={t:(G[t]+ti[t])/2 for t in tm};Ds={t:(G[t]-ti[t])/2 for t in tm}
    am={t:math.exp(As[t]) for t in tm};dm={t:math.exp(-Ds[t]) for t in tm}
    for dc in(am,dm):
        gmv=math.exp(sum(math.log(max(v,1e-9)) for v in dc.values())/len(dc))
        for t in dc: dc[t]/=gmv
    ts=tm if len(tm)<=80 else sorted(tm,key=lambda t:-cz[t])[:80]
    pr=list(itertools.permutations(ts,2));k=(sum(am[a]*dm[b] for a,b in pr)/len(pr))**0.5
    for t in tm: am[t]/=k;dm[t]/=k
    zA=zsc(As);zD=zsc(Ds);s100={t:100/(1+math.exp(-1.15*(zA[t]+zD[t])/2)) for t in tm}
    # c2 for compression relevel
    c2=sum((am[a]**COMPRESS)*(dm[b]**COMPRESS) for a,b in pr)/len(pr)
    return {"AVG":AVG,"nc":nc,"tm":set(tm),"am":am,"dm":dm,"s100":s100,"c2":c2}
def nb(mu,r,mg): return [math.exp(math.lgamma(k+r)-math.lgamma(r)-math.lgamma(k+1)+r*math.log(r/(r+mu))+k*math.log(mu/(r+mu))) for k in range(mg+1)]
def wdl(s,h,a,neutral):
    amh,dmh,sh=s["am"][h],s["dm"][h],s["s100"][h];ama,dma,sa=s["am"][a],s["dm"][a],s["s100"][a]
    lA=s["AVG"]*(amh**COMPRESS)*(dma**COMPRESS)/s["c2"];lB=s["AVG"]*(ama**COMPRESS)*(dmh**COMPRESS)/s["c2"]
    if not neutral: lA*=1.30
    lA=max(.05,lA);lB=max(.05,lB);rA=VAR_BASE+VAR_SLOPE*sh;rB=VAR_BASE+VAR_SLOPE*sa
    mg=max(12,int(lA+lB)+8);ph=nb(lA,rA,mg);pa=nb(lB,rB,mg)
    M=[[ph[i]*pa[j] for j in range(mg+1)] for i in range(mg+1)]
    M[0][0]*=max(0.,1-lA*lB*RHO);M[0][1]*=max(0.,1+lA*RHO);M[1][0]*=max(0.,1+lB*RHO);M[1][1]*=max(0.,1-RHO)
    sm=sum(sum(r) for r in M);rng=range(mg+1)
    pH=sum(M[i][j] for i in rng for j in rng if i>j)/sm;pA=sum(M[i][j] for i in rng for j in rng if j>i)/sm;pD=sum(M[i][i] for i in rng)/sm
    return pH,pD,pA
def moiter(s,e):
    y,m=s.year,s.month
    while datetime.date(y,m,1)<=e:
        yield datetime.date(y,m,1);m+=1
        if m>12:m=1;y+=1
preds=[]  # (pH,pD,pA, actualidx(0=H,1=D,2=A), date)
months=list(moiter(TEST_START,TEST_END))
for i,ms in enumerate(months):
    me=months[i+1] if i+1<len(months) else datetime.date(TEST_END.year+1,1,1)
    if not any(ms<=m[0]<me for m in TESTS): continue
    s=snap(ms)
    if not s: continue
    for d,h,a,hs,as_,n,tn in [m for m in TESTS if ms<=m[0]<me]:
        if h not in s["tm"] or a not in s["tm"]: continue
        if s["nc"][h]<MINP or s["nc"][a]<MINP: continue
        pH,pD,pA=wdl(s,h,a,n);ai=0 if hs>as_ else(1 if hs==as_ else 2)
        preds.append((pH,pD,pA,ai,d))
print(f"OOS predictions: {len(preds)}")
def ll_at(T,rows):
    tot=0.0
    for pH,pD,pA,ai,d in rows:
        p=[max(pH,1e-9)**(1/T),max(pD,1e-9)**(1/T),max(pA,1e-9)**(1/T)];z=sum(p)
        tot+=-math.log(max(p[ai]/z,1e-12))
    return tot/len(rows)
train=[r for r in preds if r[4]<SPLIT];test=[r for r in preds if r[4]>=SPLIT]
# fit T on train
bestT=1.0;bestll=1e9
T=0.5
while T<=1.6:
    l=ll_at(T,train)
    if l<bestll: bestll=l;bestT=T
    T+=0.01
print(f"\nTrain n={len(train)}  Test n={len(test)}  (split {SPLIT})")
print(f"Fitted temperature T* = {bestT:.2f}   (T<1 => model was UNDER-confident; T>1 => over-confident)")
print("\n           train log-loss   test(held-out) log-loss")
print(f"  raw (T=1.00)   {ll_at(1.0,train):.4f}           {ll_at(1.0,test):.4f}")
print(f"  calibrated T*  {ll_at(bestT,train):.4f}           {ll_at(bestT,test):.4f}")
imp=(ll_at(1.0,test)-ll_at(bestT,test))/ll_at(1.0,test)*100
print(f"\n  held-out log-loss change from calibration: {imp:+.1f}%  ({'KEEP' if imp>0.3 else 'NOT WORTH IT'})")
# calibration curve (favorite confidence) raw vs calibrated, full sample
print("\nCalibration of the model's PICK (confidence bucket -> actual win-rate of pick):")
def curve(T):
    b=collections.defaultdict(lambda:[0,0.0,0])
    for pH,pD,pA,ai,d in preds:
        p=[max(pH,1e-9)**(1/T),max(pD,1e-9)**(1/T),max(pA,1e-9)**(1/T)];z=sum(p);p=[x/z for x in p]
        pick=p.index(max(p));conf=p[pick];k=min(int(conf*10),9)
        b[k][0]+=1;b[k][1]+=conf;b[k][2]+=1 if pick==ai else 0
    return b
print(f"  {'bucket':>8}{'n':>5}{'rawPred':>9}{'calPred':>9}{'actual':>9}")
br=curve(1.0);bc=curve(bestT)
for k in range(4,10):
    if br[k][0]<8: continue
    print(f"  {k/10:.1f}-{(k+1)/10:.1f}{br[k][0]:>5}{br[k][1]/br[k][0]*100:>8.0f}%{bc[k][1]/max(bc[k][0],1)*100:>8.0f}%{br[k][2]/br[k][0]*100:>8.0f}%")
