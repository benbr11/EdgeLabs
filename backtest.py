# -*- coding: utf-8 -*-
"""
Walk-forward backtest + calibration of the goals+Elo Dixon-Coles model.

For each past tournament we build ratings using ONLY matches before it started,
predict every group-stage match, and score the predictions vs what actually
happened. We grid-tune the recency half-life, the goals-vs-Elo blend weight, and
the Dixon-Coles rho, and report calibration. (FIFA points are a current-only
overlay in the live model and can't be reconstructed historically, so the
backtest validates the core goals+Elo engine.)
"""
import csv, math, datetime, itertools, collections

PROJ = r"C:\Users\bbraudo\Desktop\Claude Output\World Cup Model"
TEST = ([("FIFA World Cup", y) for y in (2006,2010,2014,2018,2022)] +
        [("UEFA Euro",     y) for y in (2012,2016,2021,2024)])

rows = []
with open(PROJ + r"\results.csv", encoding="utf-8") as f:
    for r in csv.DictReader(f):
        try:
            d = datetime.date.fromisoformat(r["date"]); hs=int(r["home_score"]); a_=int(r["away_score"])
        except (ValueError, KeyError):
            continue
        rows.append((d, r["home_team"], r["away_team"], hs, a_,
                     r["neutral"].strip().upper()=="TRUE", r["tournament"]))
rows.sort(key=lambda x: x[0])

def elo_asof(cutoff):
    elo = {}
    for d,h,a,hs,a_,neu,trn in rows:
        if d >= cutoff: break
        eh=elo.get(h,1500.0); ea=elo.get(a,1500.0); adj=0.0 if neu else 65.0
        exp_h=1.0/(1.0+10**((ea-(eh+adj))/400.0)); res=1.0 if hs>a_ else (0.5 if hs==a_ else 0.0)
        gd=abs(hs-a_); g=1.0 if gd<=1 else (1.5 if gd==2 else (11+gd)/8.0)
        dl=30.0*g*(res-exp_h); elo[h]=eh+dl; elo[a]=ea-dl
    return elo

def goals_asof(cutoff, halflife):
    lo = datetime.date(cutoff.year-8, 1, 1)
    w = [m for m in rows if lo <= m[0] < cutoff]
    def wt(d): return 0.5 ** ((cutoff - d).days / halflife)
    tg=tw=0.0
    for d,h,a,hs,a_,neu,trn in w: x=wt(d); tg+=x*(hs+a_); tw+=x*2
    AVG = tg/tw if tw else 1.3
    att={}; dfn={}
    for d,h,a,hs,a_,neu,trn in w:
        for t in (h,a): att.setdefault(t,1.0); dfn.setdefault(t,1.0)
    for _ in range(40):
        na={t:0.0 for t in att}; da={t:0.0 for t in att}; nd={t:0.0 for t in att}; dd={t:0.0 for t in att}
        for d,h,a,hs,a_,neu,trn in w:
            x=wt(d)
            na[h]+=x*hs; da[h]+=x*AVG*dfn[a]; nd[a]+=x*hs; dd[a]+=x*AVG*att[h]
            na[a]+=x*a_; da[a]+=x*AVG*dfn[h]; nd[h]+=x*a_; dd[h]+=x*AVG*att[a]
        for t in att:
            if da[t]>0: att[t]=na[t]/da[t]
            if dd[t]>0: dfn[t]=nd[t]/dd[t]
        for dct in (att,dfn):
            gm=math.exp(sum(math.log(max(v,1e-6)) for v in dct.values())/len(dct))
            for t in dct: dct[t]/=gm
    return att, dfn, AVG

def find_group_matches(instance):
    games=collections.defaultdict(list)
    for m in sorted(instance, key=lambda x:x[0]):
        games[m[1]].append(m); games[m[2]].append(m)
    opp={}
    for t,gs in games.items():
        opp[t]={(m[2] if m[1]==t else m[1]) for m in gs[:3]}
    adj={t:{u for u in opp[t] if t in opp.get(u,set())} for t in opp}
    used=set(); gm=[]
    for t in list(adj):
        if t in used: continue
        nb=[u for u in adj[t] if u not in used]
        for combo in itertools.combinations(nb,3):
            four=(t,)+combo
            if all(b in adj[a] for a,b in itertools.combinations(four,2)):
                grp=set(four); used.update(four)
                seen=set()
                for m in sorted(instance,key=lambda x:x[0]):
                    if m[1] in grp and m[2] in grp:
                        k=frozenset((m[1],m[2]))
                        if k not in seen: seen.add(k); gm.append(m)
                break
    return gm

def zmap(d):
    v=list(d.values()); m=sum(v)/len(v); sd=(sum((x-m)**2 for x in v)/len(v))**0.5 or 1.0
    return {k:(x-m)/sd for k,x in d.items()}, m, sd

def dc_probs(lh, la, rho):
    mg=max(12,int(lh+la)+8)
    ph=[math.exp(-lh)*lh**i/math.factorial(i) for i in range(mg+1)]
    pa=[math.exp(-la)*la**j/math.factorial(j) for j in range(mg+1)]
    M=[[ph[i]*pa[j] for j in range(mg+1)] for i in range(mg+1)]
    M[0][0]*=max(0.,1-lh*la*rho); M[0][1]*=max(0.,1+lh*rho)
    M[1][0]*=max(0.,1+la*rho);   M[1][1]*=max(0.,1-rho)
    s=sum(sum(r) for r in M)
    pH=sum(M[i][j] for i in range(mg+1) for j in range(mg+1) if i>j)/s
    pD=sum(M[i][i] for i in range(mg+1))/s
    return pH, pD, 1-pH-pD

HLS=[730,1095,1460,1825]; WS=[0.0,0.3,0.5,0.7,1.0]; RHOS=[0.0,-0.05,-0.08,-0.12]
results=collections.defaultdict(list)   # (hl,w,rho) -> list of (pH,pD,pA, outcome)

for name,year in TEST:
    inst=[m for m in rows if m[6]==name and m[0].year==year]
    if not inst: continue
    cutoff=min(m[0] for m in inst)
    gms=find_group_matches(inst)
    if not gms: continue
    elo=elo_asof(cutoff)
    teams={t for m in gms for t in (m[1],m[2])}
    for hl in HLS:
        att,dfn,AVG=goals_asof(cutoff,hl)
        A_log={t:math.log(att.get(t,1.0)) for t in teams}
        D_log={t:-math.log(dfn.get(t,1.0)) for t in teams}
        gstr={t:A_log[t]+D_log[t] for t in teams}; tilt={t:A_log[t]-D_log[t] for t in teams}
        zg,_,_=zmap(gstr); ze,_,_=zmap({t:elo.get(t,1500.0) for t in teams})
        gm_mean=sum(gstr.values())/len(teams); gm_sd=(sum((v-gm_mean)**2 for v in gstr.values())/len(teams))**0.5 or 1.0
        for w in WS:
            cons={t:(1-w)*zg[t]+w*ze[t] for t in teams}
            Gs={t:gm_mean+gm_sd*cons[t] for t in teams}
            astar={t:math.exp((Gs[t]+tilt[t])/2) for t in teams}
            dstar={t:math.exp(-(Gs[t]-tilt[t])/2) for t in teams}
            for rho in RHOS:
                for d,h,a,hs,a_,neu,trn in gms:
                    hf=1.0 if neu else 1.30
                    lh=AVG*astar[h]*dstar[a]*hf; la=AVG*astar[a]*dstar[h]
                    pH,pD,pA=dc_probs(lh,la,rho)
                    out=0 if hs>a_ else (1 if hs==a_ else 2)
                    results[(hl,w,rho)].append((pH,pD,pA,out))

def score(rec):
    n=len(rec); ll=0.0; br=0.0; acc=0
    for pH,pD,pA,out in rec:
        ps=[max(pH,1e-9),max(pD,1e-9),max(pA,1e-9)]
        ll-=math.log(ps[out]);
        ind=[1 if k==out else 0 for k in range(3)]
        br+=sum((ps[k]-ind[k])**2 for k in range(3))
        if max(range(3),key=lambda k:ps[k])==out: acc+=1
    return ll/n, br/n, acc/n

best=min(results, key=lambda k: score(results[k])[0])
n=len(next(iter(results.values())))
print(f"Backtest over {len(TEST)} tournaments, {n} group matches\n")
print(f"{'half-life':>9}{'wElo':>6}{'rho':>7}{'logloss':>9}{'brier':>8}{'acc':>7}")
# show a representative slice + the best
seen=set()
for hl in HLS:
    for w in WS:
        k=(hl,w,-0.08)
        if k in results:
            ll,br,ac=score(results[k]); print(f"{hl:>9}{w:>6}{-0.08:>7}{ll:>9.4f}{br:>8.4f}{ac:>7.1%}")
ll,br,ac=score(results[best])
print(f"\nBEST: half-life={best[0]}d  wElo={best[1]}  rho={best[2]}  -> logloss {ll:.4f}, brier {br:.4f}, acc {ac:.1%}")

# baseline: constant base rates
base=collections.Counter(out for rec in results.values() for (_,_,_,out) in [rec[0]] ) # placeholder
allrec=results[best]; from_counter=collections.Counter(o for *_,o in allrec)
br_rate=[from_counter[k]/len(allrec) for k in range(3)]
bll=-sum(math.log(br_rate[o]) for *_,o in allrec)/len(allrec)
print(f"Baseline (predict base rates {['%.2f'%x for x in br_rate]}): logloss {bll:.4f}")

# calibration of the BEST model (pooled over all 3 classes)
print("\nCalibration (best model): predicted prob vs actual frequency")
bins=collections.defaultdict(lambda:[0.0,0,0])
for pH,pD,pA,out in allrec:
    for k,p in enumerate((pH,pD,pA)):
        b=min(9,int(p*10)); bins[b][0]+=p; bins[b][1]+= (1 if out==k else 0); bins[b][2]+=1
print(f"  {'bin':>8}{'pred':>8}{'actual':>8}{'n':>7}")
for b in sorted(bins):
    s,h,c=bins[b]
    print(f"  {b/10:.1f}-{b/10+0.1:.1f}{s/c:>8.2f}{h/c:>8.2f}{c:>7}")
