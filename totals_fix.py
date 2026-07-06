# -*- coding: utf-8 -*-
"""
Fix the goal-over-prediction bug via a MISMATCH-COMPRESSION parameter P.

Diagnosis (from totals_backtest.py): the multiplicative attack x defense model
inflates expected goals in lopsided games -- a strong team's lambda balloons to
3-4 when in reality the favorite wins 2-0 and coasts -- so totals are over-
predicted (model avg 66% over-2.5 vs 50% actual; negative skill).

Fix: compress each team's multiplier toward 1 before forming the matchup lambda:
    lambda_A = AVG * (att_A^P) * (dfn_B^P) / c2(P)
with P in (0,1].  P=1 = current model.  P<1 shrinks the spread of expected goals
in mismatches.  c2(P) re-levels so the MEAN lambda over pairings stays = AVG (the
overall goal level is unchanged and leakage-free -- computed from snapshot ratings
only, never from test outcomes).  Note att_A^P * dfn_B^P / c2 has pairing-mean 1.

This sweep finds the P that flattens totals calibration & turns skill positive
WITHOUT hurting match-result (W/D/L) accuracy.  W/D/L depends on the att/dfn RATIO,
which P shrinks only mildly, so the favorite is still picked -- we verify hit-rate
and log-loss hold.
"""
import csv, math, datetime, itertools, sys, collections, os
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass
PROJ = os.path.dirname(os.path.abspath(__file__))

GOALS_CUTOFF_YEARS = 11; HALFLIFE_DAYS = 730.0
BASE_ELO = 1500.0; HFA_ELO = 65.0
RHO = -0.12; VAR_BASE, VAR_SLOPE = 6.0, 0.34; GOALS_ITERS = 40
W_GOALS_REL, W_ELO_REL = 0.22, 0.16
_wsum = W_GOALS_REL + W_ELO_REL
W_GOALS_CONS = W_GOALS_REL/_wsum; W_ELO_CONS = W_ELO_REL/_wsum
TEST_START = datetime.date(2024,1,1); TEST_END = datetime.date(2026,12,31)
COMPETITIVE = {"FIFA World Cup qualification","FIFA World Cup","UEFA Euro",
    "UEFA Euro qualification","Copa América","African Cup of Nations",
    "African Cup of Nations qualification","AFC Asian Cup","AFC Asian Cup qualification",
    "CONCACAF Nations League","Gold Cup","UEFA Nations League","CONMEBOL","Copa America"}
MIN_PRIOR_MATCHES = 15
P_GRID = [1.00, 0.90, 0.80, 0.70, 0.60, 0.50]

ALL = []
with open(PROJ + r"\results.csv", encoding="utf-8") as f:
    for r in csv.DictReader(f):
        try:
            d = datetime.date.fromisoformat(r["date"])
            hs = int(r["home_score"]); as_ = int(r["away_score"])
        except (ValueError, KeyError): continue
        ALL.append((d,r["home_team"],r["away_team"],hs,as_,
                    r["neutral"].strip().upper()=="TRUE",r["tournament"]))
ALL.sort(key=lambda x: x[0])
TESTS = [m for m in ALL if TEST_START<=m[0]<=TEST_END and m[6] in COMPETITIVE]
print(f"Loaded {len(ALL):,} matches; competitive test candidates: {len(TESTS):,}")

def zscores(d):
    vals=list(d.values()); m=sum(vals)/len(vals)
    sd=(sum((v-m)**2 for v in vals)/len(vals))**0.5 or 1.0
    return {k:(v-m)/sd for k,v in d.items()}

def build_snapshot(asof):
    hist=[m for m in ALL if m[0]<asof]
    if len(hist)<1000: return None
    maxdate=hist[-1][0]; elo={}
    def kfac(gd):
        if gd<=1: g=1.0
        elif gd==2: g=1.5
        else: g=(11+gd)/8.0
        return 30.0*g
    for d,h,a,hs,as_,neutral,tn in hist:
        eh=elo.get(h,BASE_ELO); ea=elo.get(a,BASE_ELO)
        adj=0.0 if neutral else HFA_ELO
        exp_h=1.0/(1.0+10**((ea-(eh+adj))/400.0))
        res_h=1.0 if hs>as_ else (0.5 if hs==as_ else 0.0)
        delta=kfac(abs(hs-as_))*(res_h-exp_h); elo[h]=eh+delta; elo[a]=ea-delta
    cutoff=datetime.date(maxdate.year-GOALS_CUTOFF_YEARS,1,1)
    recent=[m for m in hist if m[0]>=cutoff]
    def weight(d): return 0.5**((maxdate-d).days/HALFLIFE_DAYS)
    tot_g=tot_w=0.0
    for d,h,a,hs,as_,neutral,tn in recent:
        w=weight(d); tot_g+=w*(hs+as_); tot_w+=w*2
    if tot_w==0: return None
    AVG=tot_g/tot_w
    att={}; dfn={}
    for d,h,a,hs,as_,neutral,tn in recent:
        for t in (h,a): att.setdefault(t,1.0); dfn.setdefault(t,1.0)
    ncount=collections.Counter()
    for d,h,a,hs,as_,neutral,tn in recent: ncount[h]+=1; ncount[a]+=1
    for _ in range(GOALS_ITERS):
        na={t:0.0 for t in att}; da={t:0.0 for t in att}
        nd={t:0.0 for t in att}; dd={t:0.0 for t in att}
        for d,h,a,hs,as_,neutral,tn in recent:
            w=weight(d)
            na[h]+=w*hs; da[h]+=w*AVG*dfn[a]; nd[a]+=w*hs; dd[a]+=w*AVG*att[h]
            na[a]+=w*as_; da[a]+=w*AVG*dfn[h]; nd[h]+=w*as_; dd[h]+=w*AVG*att[a]
        for t in att:
            if da[t]>0: att[t]=max(na[t]/da[t],1e-6)
            if dd[t]>0: dfn[t]=max(nd[t]/dd[t],1e-6)
        for dct in (att,dfn):
            gm=math.exp(sum(math.log(max(v,1e-6)) for v in dct.values())/len(dct))
            for t in dct: dct[t]/=gm
    teams=list(att.keys())
    A_log={t:math.log(att[t]) for t in teams}; D_log={t:-math.log(dfn[t]) for t in teams}
    g_str={t:A_log[t]+D_log[t] for t in teams}; tilt={t:A_log[t]-D_log[t] for t in teams}
    zg=zscores(g_str); ze=zscores({t:elo.get(t,BASE_ELO) for t in teams})
    cons_z={t:W_GOALS_CONS*zg[t]+W_ELO_CONS*ze[t] for t in teams}
    gmean=sum(g_str.values())/len(teams)
    gsd=(sum((v-gmean)**2 for v in g_str.values())/len(teams))**0.5 or 1.0
    G_star={t:gmean+gsd*cons_z[t] for t in teams}
    A_star={t:(G_star[t]+tilt[t])/2 for t in teams}; D_star={t:(G_star[t]-tilt[t])/2 for t in teams}
    am={t:math.exp(A_star[t]) for t in teams}; dm={t:math.exp(-D_star[t]) for t in teams}
    for dct in (am,dm):
        gm=math.exp(sum(math.log(max(v,1e-9)) for v in dct.values())/len(dct))
        for t in dct: dct[t]/=gm
    strength={t:A_star[t]+D_star[t] for t in teams}
    ts80=sorted(teams,key=lambda t:-strength[t])[:80]
    # base k (P=1) calibration over the 80-team field
    pr=list(itertools.permutations(ts80,2))
    k=(sum(am[a]*dm[b] for a,b in pr)/len(pr))**0.5
    for t in teams: am[t]/=k; dm[t]/=k
    zA=zscores(A_star); zD=zscores(D_star)
    s100={t:100.0/(1.0+math.exp(-1.15*(zA[t]+zD[t])/2)) for t in teams}
    # c2(P) re-level constants over the 80-team field
    c2={}
    for P in P_GRID:
        s=0.0; n=0
        for x in ts80:
            axP=am[x]**P
            for y in ts80:
                if x==y: continue
                s+=axP*(dm[y]**P); n+=1
        c2[P]=s/n
    return {"AVG":AVG,"ncount":ncount,"teams":set(teams),"am":am,"dm":dm,"s100":s100,"c2":c2}

def nb_pmf(mu,r,mg):
    return [math.exp(math.lgamma(k+r)-math.lgamma(r)-math.lgamma(k+1)
            +r*math.log(r/(r+mu))+k*math.log(mu/(r+mu))) for k in range(mg+1)]

def predict(avg,amh,dmh,sh,ama,dma,sa,neutral,P,c2):
    HOME=1.30
    lamA=avg*(amh**P)*(dma**P)/c2
    lamB=avg*(ama**P)*(dmh**P)/c2
    if not neutral: lamA*=HOME    # home side is A (home_is_A=True for all tests)
    lamA=max(0.05,lamA); lamB=max(0.05,lamB)
    rA=VAR_BASE+VAR_SLOPE*sh; rB=VAR_BASE+VAR_SLOPE*sa
    mg=max(14,int(lamA+lamB)+10)
    ph=nb_pmf(lamA,rA,mg); pa=nb_pmf(lamB,rB,mg)
    M=[[ph[i]*pa[j] for j in range(mg+1)] for i in range(mg+1)]
    M[0][0]*=max(0.0,1-lamA*lamB*RHO); M[0][1]*=max(0.0,1+lamA*RHO)
    M[1][0]*=max(0.0,1+lamB*RHO); M[1][1]*=max(0.0,1-RHO)
    s=sum(sum(r) for r in M); M=[[v/s for v in r] for r in M]
    rng=range(mg+1)
    p_o25=sum(M[i][j] for i in rng for j in rng if i+j>=3)
    pA=sum(M[i][j] for i in rng for j in rng if i>j)
    pB=sum(M[i][j] for i in rng for j in rng if j>i)
    pD=sum(M[i][i] for i in rng)
    return p_o25,pA,pD,pB

def month_iter(s,e):
    y,m=s.year,s.month
    while datetime.date(y,m,1)<=e:
        yield datetime.date(y,m,1); m+=1
        if m>12: m=1; y+=1

def main():
    acc={P:{"opred":[],"oact":[],"wn":0,"whit":0,"wll":0.0,"favn":0,"favhit":0} for P in P_GRID}
    months=list(month_iter(TEST_START,TEST_END)); n_snap=0; skipped=0
    for i,mstart in enumerate(months):
        mend=(months[i+1] if i+1<len(months) else datetime.date(TEST_END.year+1,1,1))
        if not any(mstart<=m[0]<mend for m in TESTS): continue
        snap=build_snapshot(mstart)
        if snap is None: continue
        n_snap+=1
        for (d,h,a,hs,as_,neutral,tn) in [m for m in TESTS if mstart<=m[0]<mend]:
            if h not in snap["teams"] or a not in snap["teams"]: skipped+=1; continue
            if snap["ncount"][h]<MIN_PRIOR_MATCHES or snap["ncount"][a]<MIN_PRIOR_MATCHES: skipped+=1; continue
            over=1 if hs+as_>=3 else 0
            actual="H" if hs>as_ else ("A" if as_>hs else "D")
            for P in P_GRID:
                po,pA,pD,pB=predict(snap["AVG"],snap["am"][h],snap["dm"][h],snap["s100"][h],
                                    snap["am"][a],snap["dm"][a],snap["s100"][a],neutral,P,snap["c2"][P])
                tot=pA+pD+pB; pA,pD,pB=pA/tot,pD/tot,pB/tot
                A=acc[P]
                A["opred"].append(po); A["oact"].append(over)
                probs={"H":pA,"D":pD,"A":pB}; pred=max(probs,key=probs.get)
                A["wn"]+=1
                if pred==actual: A["whit"]+=1
                A["wll"]+=-math.log(max(probs[actual],1e-12))
                if actual in ("H","A"):
                    fav="H" if pA>=pB else "A"; A["favn"]+=1
                    if fav==actual: A["favhit"]+=1
    print(f"Snapshots {n_snap}  scored {acc[1.0]['wn']:,}  skipped {skipped}\n")

    # base rate for totals skill
    oact=acc[1.0]["oact"]; n=len(oact); base=sum(oact)/n
    b_brier=sum((base-y)**2 for y in oact)/n
    print(f"OVER-2.5 actual rate: {base*100:.1f}%   (base-rate Brier {b_brier:.4f})")
    print("="*86)
    print(f"{'P':>5}{'O25 meanPred':>14}{'O25 Brier':>11}{'skill%':>8}{'cal .7-.8':>11}{'cal .9+':>9}"
          f"{'WDL hit':>9}{'WDL ll':>9}{'fav hit':>9}")
    print("="*86)
    for P in P_GRID:
        A=acc[P]; preds=A["opred"]
        mp=sum(preds)/n
        brier=sum((p-y)**2 for p,y in zip(preds,oact))/n
        skill=(b_brier-brier)/b_brier*100
        # calibration realized in two high buckets
        def realized(lo,hi):
            idx=[j for j,p in enumerate(preds) if lo<=p<hi]
            return (sum(oact[j] for j in idx)/len(idx)*100) if len(idx)>=8 else float('nan')
        c78=realized(0.7,0.8); c9=realized(0.9,1.01)
        whit=A["whit"]/A["wn"]*100; wll=A["wll"]/A["wn"]; favr=A["favhit"]/A["favn"]*100
        print(f"{P:>5.2f}{mp*100:>13.1f}%{brier:>11.4f}{skill:>+7.1f}%{c78:>10.1f}%{c9:>8.1f}%"
              f"{whit:>8.1f}%{wll:>9.4f}{favr:>8.1f}%")
    print("\nGoal: meanPred -> ~actual, Brier skill -> positive, .7-.8 & .9+ realized -> close to")
    print("the bucket label, while WDL hit-rate / log-loss / fav-hit stay ~flat (match prediction")
    print("unharmed). Pick the largest P that achieves it (least intervention).")

if __name__ == "__main__":
    main()
