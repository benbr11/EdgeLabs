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

import os
PROJ = os.path.dirname(os.path.abspath(__file__))
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
    cnt={t:0.0 for t in att}                 # effective (recency-weighted) match count per team
    for d,h,a,hs,a_,neu,trn in w:
        x=wt(d); cnt[h]+=x; cnt[a]+=x
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
    return att, dfn, AVG, cnt

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
KAPPAS=[0,20,40]   # thin-sample shrinkage prior (effective matches); 0 = no shrinkage (baseline)
# Host-nation advantage applied to the host's goal rate (asymmetric, same as simulate.py).
# Lowered 1.30 -> 1.15 after the HOME-ADVANTAGE experiment below: across 52 host group
# matches, host-only log-loss is minimised at hf~1.10-1.20 (0.847) vs 1.30 (0.853), and
# every leave-one-tournament-out fold votes below 1.30. Mirror of simulate.HOME_ADV.
HOME_ADV=1.15
# Deployed engine config (validated by LOTO grid-select, NOT by pooled in-sample):
# half-life, wElo, rho are the robust LOTO pick. KAPPA_DEPLOY=0: shrinkage was TESTED
# (kappa=20 lowers raw log-loss to 0.9785) but the HEAD-TO-HEAD below shows that once
# each engine gets its LOTO-validated temperature, plain kappa=0 + T=1.3 (OOS 0.9779)
# BEATS kappa=20 + T~0.9 (OOS 0.9808) -- shrinkage and temperature are substitutes and
# temperature is the more robust one. So shrinkage is NOT deployed (kept only as a
# diagnostic in the grid). The calibration/subset reporting keys off THIS fixed config.
HL_DEPLOY, W_DEPLOY, RHO_DEPLOY, KAPPA_DEPLOY = 730, 0.5, -0.12, 0
DEPLOY=(HL_DEPLOY, W_DEPLOY, RHO_DEPLOY, KAPPA_DEPLOY)
results=collections.defaultdict(list)   # (hl,w,rho,kappa) -> list of (pH,pD,pA, outcome)
results_yr=collections.defaultdict(list)  # parallel list of tournament year (for out-of-time calibration split)

for name,year in TEST:
    inst=[m for m in rows if m[6]==name and m[0].year==year]
    if not inst: continue
    cutoff=min(m[0] for m in inst)
    gms=find_group_matches(inst)
    if not gms: continue
    elo=elo_asof(cutoff)
    teams={t for m in gms for t in (m[1],m[2])}
    for hl in HLS:
        att,dfn,AVG,cnt=goals_asof(cutoff,hl)
        A_log={t:math.log(att.get(t,1.0)) for t in teams}
        D_log={t:-math.log(dfn.get(t,1.0)) for t in teams}
        gstr={t:A_log[t]+D_log[t] for t in teams}; tilt={t:A_log[t]-D_log[t] for t in teams}
        zg,_,_=zmap(gstr); ze,_,_=zmap({t:elo.get(t,1500.0) for t in teams})
        gm_mean=sum(gstr.values())/len(teams); gm_sd=(sum((v-gm_mean)**2 for v in gstr.values())/len(teams))**0.5 or 1.0
        for w in WS:
            cons0={t:(1-w)*zg[t]+w*ze[t] for t in teams}
            for kappa in KAPPAS:
                # shrink thin-sample teams' consensus z toward the field average (0).
                # shrink factor cnt/(cnt+kappa): minnows (low effective match count) get
                # pulled toward average; well-sampled teams are barely touched. kappa=0 = off.
                if kappa>0:
                    cons={t:cons0[t]*(cnt.get(t,0.0)/(cnt.get(t,0.0)+kappa)) for t in teams}
                else:
                    cons=cons0
                Gs={t:gm_mean+gm_sd*cons[t] for t in teams}
                astar={t:math.exp((Gs[t]+tilt[t])/2) for t in teams}
                dstar={t:math.exp(-(Gs[t]-tilt[t])/2) for t in teams}
                for rho in RHOS:
                    for d,h,a,hs,a_,neu,trn in gms:
                        hf=1.0 if neu else HOME_ADV
                        lh=AVG*astar[h]*dstar[a]*hf; la=AVG*astar[a]*dstar[h]
                        pH,pD,pA=dc_probs(lh,la,rho)
                        out=0 if hs>a_ else (1 if hs==a_ else 2)
                        results[(hl,w,rho,kappa)].append((pH,pD,pA,out))
                        results_yr[(hl,w,rho,kappa)].append(year)

def score(rec):
    n=len(rec); ll=0.0; br=0.0; acc=0
    for pH,pD,pA,out in rec:
        ps=[max(pH,1e-9),max(pD,1e-9),max(pA,1e-9)]
        ll-=math.log(ps[out]);
        ind=[1 if k==out else 0 for k in range(3)]
        br+=sum((ps[k]-ind[k])**2 for k in range(3))
        if max(range(3),key=lambda k:ps[k])==out: acc+=1
    return ll/n, br/n, acc/n

def subset_metrics(rec):
    """Report the subsets where 60%+ is genuinely achievable.
      fav2 : favourite-vs-field 2-way accuracy on DECISIVE games (draws excluded
             from the denominator); pick the higher of pH/pA, score vs the winner.
             This is also the knockout 'who advances' analog: in a knockout the
             draw mass redistributes ~by relative strength, so the decisive-game
             favourite is the side the model would back to advance.
      hi65/hi70 : among ALL per-outcome probabilities (H, D or A) the model put at
             >= the threshold, the hit-rate and count (high-confidence picks).
      kcal : calibration of the conditional 2-way prob pH/(pH+pA) on decisive games."""
    # favourite 2-way on decisive games
    dec=0; favhit=0
    for pH,pD,pA,out in rec:
        if out==1: continue              # drawn game: excluded from 2-way denominator
        dec+=1
        pick=0 if pH>=pA else 2          # model's favourite between the two sides
        if pick==out: favhit+=1
    fav2 = favhit/dec if dec else float('nan')
    # high-confidence buckets over all three per-outcome probs
    def hi(th):
        h=c=0
        for pH,pD,pA,out in rec:
            for k,p in enumerate((pH,pD,pA)):
                if p>=th: c+=1; h+= (1 if out==k else 0)
        return (h/c if c else float('nan')), c
    hr65,n65=hi(0.65); hr70,n70=hi(0.70)
    return fav2, dec, hr65, n65, hr70, n70

def print_subsets(rec, label="best model"):
    fav2,dec,hr65,n65,hr70,n70 = subset_metrics(rec)
    print(f"\nSubset metrics ({label}):")
    print(f"  favourite 2-way acc (draws excluded): {fav2:.1%}   (N decisive={dec})")
    print(f"  high-conf >=65%: hit {hr65:.1%}   (N picks={n65})")
    print(f"  high-conf >=70%: hit {hr70:.1%}   (N picks={n70})")
    # conditional 2-way calibration (knockout 'to advance' analog) on decisive games
    print("  conditional 2-way calibration (knockout 'to advance' analog, decisive games):")
    kb=collections.defaultdict(lambda:[0.0,0,0])
    for pH,pD,pA,out in rec:
        if out==1: continue
        s=pH+pA
        if s<=0: continue
        cph=pH/s
        for k,p in ((0,cph),(2,1-cph)):
            b=min(9,int(p*10)); kb[b][0]+=p; kb[b][1]+=(1 if out==k else 0); kb[b][2]+=1
    print(f"    {'bin':>8}{'pred':>8}{'actual':>8}{'n':>7}")
    for b in sorted(kb):
        s,h,c=kb[b]
        if c: print(f"    {b/10:.1f}-{b/10+0.1:.1f}{s/c:>8.2f}{h/c:>8.2f}{c:>7}")

ALLYR=results_yr[next(iter(results))]            # tournament year per match index (same ordering for every key)
YEARS=sorted(set(ALLYR))
def _ll_idx(key, idxs):
    rec=results[key]; s=0.0
    for i in idxs: pH,pD,pA,out=rec[i]; s-=math.log(max((pH,pD,pA)[out],1e-9))
    return s/len(idxs)
def loto_grid_select():
    """Honest out-of-sample hyperparameter selection: for each held-out tournament,
    pick the (hl,w,rho) minimising log-loss on the OTHER tournaments, then score the
    held-out tournament with that pick. Pool the held-out losses. This is the OOS
    number that should decide whether widening the grid actually helps -- it cannot
    overfit the held-out tournament because that tournament never influenced its own
    config choice."""
    by_year={y:[i for i,yy in enumerate(ALLYR) if yy==y] for y in YEARS}
    s=0.0; ntot=0; picks=collections.Counter()
    for hy in YEARS:
        tr=[i for i,yy in enumerate(ALLYR) if yy!=hy]
        k=min(results, key=lambda kk:_ll_idx(kk,tr))
        picks[k]+=1
        te=by_year[hy]
        s+=_ll_idx(k,te)*len(te); ntot+=len(te)
    return s/ntot, picks

best=min(results, key=lambda k: score(results[k])[0])
n=len(next(iter(results.values())))
print(f"Backtest over {len(TEST)} tournaments, {n} group matches\n")
print(f"{'half-life':>9}{'wElo':>6}{'rho':>7}{'kappa':>7}{'logloss':>9}{'brier':>8}{'acc':>7}")
# show a representative slice (rho=-0.08, kappa=0) + the best
for hl in HLS:
    for w in WS:
        k=(hl,w,-0.08,0)
        if k in results:
            ll,br,ac=score(results[k]); print(f"{hl:>9}{w:>6}{-0.08:>7}{0:>7}{ll:>9.4f}{br:>8.4f}{ac:>7.1%}")
ll,br,ac=score(results[best])
print(f"\nBEST (pooled in-sample): half-life={best[0]}d  wElo={best[1]}  rho={best[2]}  kappa={best[3]}  -> logloss {ll:.4f}, brier {br:.4f}, acc {ac:.1%}")
_cvll,_picks=loto_grid_select()
_top=_picks.most_common(3)
print(f"LOTO grid-select OOS log-loss = {_cvll:.4f}  (vs pooled-best in-sample {ll:.4f}); "
      f"per-fold picks: {', '.join('hl%d/w%g/rho%g/k%g x%d'%(k[0],k[1],k[2],k[3],c) for k,c in _top)}")

# ISOLATED shrinkage test: hold the validated engine fixed (hl=730,w=0.5,rho=-0.12)
# and compare kappa head-to-head on POOLED data (all params identical except kappa,
# so this is a clean one-variable comparison; pooled is fine because nothing is being
# *selected* -- we just read off each fixed kappa's log-loss).
print("\nIsolated shrinkage (engine fixed at hl=730,w=0.5,rho=-0.12):")
for kp in KAPPAS:
    kk=(730,0.5,-0.12,kp)
    if kk in results:
        l,b,a=score(results[kk]); print(f"  kappa={kp:>3}: logloss {l:.4f}  brier {b:.4f}  acc {a:.1%}")

# baseline: constant base rates
base=collections.Counter(out for rec in results.values() for (_,_,_,out) in [rec[0]] ) # placeholder
# calibration / subset / temperature analysis all key off the FIXED deployed config
allrec=results[DEPLOY]; allyr_d=results_yr[DEPLOY]
from_counter=collections.Counter(o for *_,o in allrec)
br_rate=[from_counter[k]/len(allrec) for k in range(3)]
bll=-sum(math.log(br_rate[o]) for *_,o in allrec)/len(allrec)
print(f"\n[Deployed config for calibration/subset analysis: hl={HL_DEPLOY},w={W_DEPLOY},rho={RHO_DEPLOY},kappa={KAPPA_DEPLOY}]")
dl,db,da=score(allrec); print(f"  deployed engine (pre-temperature): logloss {dl:.4f} brier {db:.4f} acc {da:.1%}")
print(f"Baseline (predict base rates {['%.2f'%x for x in br_rate]}): logloss {bll:.4f}")

# calibration of the deployed model (pooled over all 3 classes)
print("\nCalibration (deployed model, pre-temperature): predicted prob vs actual frequency")
bins=collections.defaultdict(lambda:[0.0,0,0])
for pH,pD,pA,out in allrec:
    for k,p in enumerate((pH,pD,pA)):
        b=min(9,int(p*10)); bins[b][0]+=p; bins[b][1]+= (1 if out==k else 0); bins[b][2]+=1
print(f"  {'bin':>8}{'pred':>8}{'actual':>8}{'n':>7}")
for b in sorted(bins):
    s,h,c=bins[b]
    print(f"  {b/10:.1f}-{b/10+0.1:.1f}{s/c:>8.2f}{h/c:>8.2f}{c:>7}")

print_subsets(allrec, "best model")

# =============================================================================
# OUT-OF-TIME PROBABILITY CALIBRATION EXPERIMENT (temperature scaling)
# Fit a single temperature T on EARLY tournaments, evaluate on HELD-OUT LATE
# tournaments. Temperature scaling: p_i -> p_i**(1/T) renormalised. T>1 softens
# (cures over-confidence), T<1 sharpens. Never fit and score on the same matches.
# =============================================================================
def temper(p3, T):
    q=[max(p,1e-12)**(1.0/T) for p in p3]; s=sum(q); return [x/s for x in q]
def ll_of(recs):
    n=len(recs); s=0.0
    for pH,pD,pA,out in recs: s-=math.log(max((pH,pD,pA)[out],1e-12))
    return s/n
def ll_tempered(recs, T):
    n=len(recs); s=0.0
    for pH,pD,pA,out in recs:
        q=temper((pH,pD,pA),T); s-=math.log(max(q[out],1e-12))
    return s/n
def fit_T(recs):
    # 1-D search for the temperature minimising log-loss on `recs`
    best_T=1.0; best=ll_tempered(recs,1.0)
    T=0.5
    while T<=3.0001:
        v=ll_tempered(recs,T)
        if v<best: best=v; best_T=T
        T+=0.05
    return best_T

allyr=allyr_d   # deployed config's per-match tournament years (allrec is results[DEPLOY])
yrs=sorted(set(allyr))
# chronological split: earliest ~half of distinct tournament-years = TRAIN, rest = TEST
split=yrs[len(yrs)//2 - 1] if len(yrs)>=4 else yrs[len(yrs)//2]
train=[r for r,y in zip(allrec,allyr) if y<=split]
held =[r for r,y in zip(allrec,allyr) if y> split]
if train and held:
    T_fit=fit_T(train)
    base_ll=ll_of(held); cal_ll=ll_tempered(held,T_fit)
    # Brier on held-out, base vs calibrated
    def brier(recs, T=None):
        n=len(recs); s=0.0
        for pH,pD,pA,out in recs:
            p=temper((pH,pD,pA),T) if T else (pH,pD,pA)
            ind=[1 if k==out else 0 for k in range(3)]
            s+=sum((p[k]-ind[k])**2 for k in range(3))
        return s/n
    print("\n" + "="*64)
    print("OUT-OF-TIME TEMPERATURE CALIBRATION")
    print(f"  train years <= {split} (N={len(train)}); held-out years > {split} (N={len(held)})")
    print(f"  fitted T on train = {T_fit:.2f}")
    print(f"  HELD-OUT log-loss : base {base_ll:.4f}  ->  calibrated {cal_ll:.4f}  (delta {cal_ll-base_ll:+.4f})")
    print(f"  HELD-OUT brier    : base {brier(held):.4f}  ->  calibrated {brier(held,T_fit):.4f}")
    # also report symmetric: fit on LATE, score on EARLY (so both halves are validated once)
    T_fit2=fit_T(held)
    base_ll2=ll_of(train); cal_ll2=ll_tempered(train,T_fit2)
    print(f"  [reverse] fit T on late={T_fit2:.2f}; EARLY held-out log-loss base {base_ll2:.4f} -> cal {cal_ll2:.4f} (delta {cal_ll2-base_ll2:+.4f})")
    print(f"  pooled out-of-sample log-loss delta (both halves): "
          f"{((cal_ll*len(held)+cal_ll2*len(train))-(base_ll*len(held)+base_ll2*len(train)))/(len(held)+len(train)):+.4f}")

    # LEAVE-ONE-TOURNAMENT-OUT cross-validation of a single temperature (most
    # data-efficient, least split-dependent). For each held-out tournament, fit T
    # on all OTHER tournaments, score the held-out one. Pool the held-out scores.
    pairs=list(zip(allrec,allyr))
    loto_base=loto_cal=0.0; ntot=0; Ts=[]
    for hy in yrs:
        tr=[r for r,y in pairs if y!=hy]; te=[r for r,y in pairs if y==hy]
        if not te: continue
        Tk=fit_T(tr); Ts.append(Tk)
        loto_base+=ll_of(te)*len(te); loto_cal+=ll_tempered(te,Tk)*len(te); ntot+=len(te)
    print(f"  [LOTO] per-fold fitted T range {min(Ts):.2f}-{max(Ts):.2f}; "
          f"pooled held-out log-loss base {loto_base/ntot:.4f} -> cal {loto_cal/ntot:.4f} (delta {(loto_cal-loto_base)/ntot:+.4f})")

    # Full diagnostic at the DEPLOYED temperature used by simulate.py (single source
    # of truth: simulate.CALIB_T), applied to the WHOLE pool to SHOW calibration/subset
    # effects (this pooled view is in-sample; the LOTO OOS verdict above decides KEEP/REVERT).
    import statistics as _st
    print(f"\n  LOTO median fitted T = {round(_st.median(Ts),2)} (this is what simulate.CALIB_T should track)")
    try:
        import simulate as _sim; T_dep=_sim.CALIB_T
    except Exception:
        T_dep=round(_st.median(Ts),2)
    cal_all=[(*temper((pH,pD,pA),T_dep),out) for pH,pD,pA,out in allrec]
    cll,cbr,cac=score(cal_all)
    print(f"  Deployment T={T_dep} (=simulate.CALIB_T) applied to full pool (illustrative, in-sample):")
    print(f"    3-way: logloss {cll:.4f} brier {cbr:.4f} acc {cac:.1%}  (acc unchanged by monotonic temper)")
    cb=collections.defaultdict(lambda:[0.0,0,0])
    for pH,pD,pA,out in cal_all:
        for k,p in enumerate((pH,pD,pA)):
            b=min(9,int(p*10)); cb[b][0]+=p; cb[b][1]+=(1 if out==k else 0); cb[b][2]+=1
    print(f"    {'bin':>8}{'pred':>8}{'actual':>8}{'n':>7}  (high buckets should now be closer)")
    for b in sorted(cb):
        s,h,c=cb[b]
        if c: print(f"    {b/10:.1f}-{b/10+0.1:.1f}{s/c:>8.2f}{h/c:>8.2f}{c:>7}")
    print_subsets(cal_all, f"calibrated T={T_dep}")

# =============================================================================
# HEAD-TO-HEAD of the two individually-validated improvements (shrinkage vs
# temperature), so the KEEP decision is on the table explicitly. For each, fit T
# by LOTO on that engine and report pooled held-out log-loss/brier + subsets.
# =============================================================================
def loto_T_and_metrics(cfg):
    rec=results[cfg]; yr=results_yr[cfg]
    ys=sorted(set(yr)); pairs=list(zip(rec,yr))
    s=0.0; nt=0; Tk=[]
    for hy in ys:
        tr=[r for r,y in pairs if y!=hy]; te=[r for r,y in pairs if y==hy]
        t=fit_T(tr); Tk.append(t)
        s+=ll_tempered(te,t)*len(te); nt+=len(te)
    import statistics as _st
    Tmed=round(_st.median(Tk),2)
    capp=[(*temper((pH,pD,pA),Tmed),out) for pH,pD,pA,out in rec]
    l,b,a=score(capp)
    fav2,dec,h65,n65,h70,n70=subset_metrics(capp)
    return s/nt, Tmed, l,b,a, fav2,h65,n65,h70,n70
print("\n" + "="*64)
print("HEAD-TO-HEAD candidate deployed configs (LOTO-validated temperature each):")
print(f"  {'config':<34}{'OOS-LL':>8}{'Tmed':>6}{'acc':>6}{'fav2':>6}{'h>=65(n)':>11}{'h>=70(n)':>11}")
for label,cfg in [("kappa=0  (temperature only)",(730,0.5,-0.12,0)),
                  ("kappa=20 (shrinkage)",       (730,0.5,-0.12,20))]:
    if cfg in results:
        oos,Tm,l,b,a,f2,h65,n65,h70,n70=loto_T_and_metrics(cfg)
        print(f"  {label:<34}{oos:>8.4f}{Tm:>6}{a:>6.1%}{f2:>6.1%}{('%.0f%%(%d)'%(h65*100,n65)):>11}{('%.0f%%(%d)'%(h70*100,n70)):>11}")

# =============================================================================
# HOME-ADVANTAGE experiment (deployed engine). Re-derive the deployed-config ratings
# per tournament, then re-predict under different host-advantage magnitudes & two
# formulations:
#   'asym'  (current live model): only the host's lambda is multiplied by hf.
#   'sym'   : host lambda *= sqrt(hf), opponent lambda /= sqrt(hf) (total-goals-preserving,
#             so the host both scores more AND concedes fewer -- a 'cleaner' home effect).
# Reported on (a) ALL group matches and (b) ONLY non-neutral host matches (high power).
# Temperature T=1.3 (deployed) is applied so this matches the live probabilities.
# =============================================================================
deploy_state=[]   # (year, gms_with_predcontext)
for name,year in TEST:
    inst=[m for m in rows if m[6]==name and m[0].year==year]
    if not inst: continue
    cutoff=min(m[0] for m in inst); gms=find_group_matches(inst)
    if not gms: continue
    elo=elo_asof(cutoff); teams={t for m in gms for t in (m[1],m[2])}
    att,dfn,AVG,cnt=goals_asof(cutoff,HL_DEPLOY)
    A_log={t:math.log(att.get(t,1.0)) for t in teams}; D_log={t:-math.log(dfn.get(t,1.0)) for t in teams}
    gstr={t:A_log[t]+D_log[t] for t in teams}; tilt={t:A_log[t]-D_log[t] for t in teams}
    zg,_,_=zmap(gstr); ze,_,_=zmap({t:elo.get(t,1500.0) for t in teams})
    gm_mean=sum(gstr.values())/len(teams); gm_sd=(sum((v-gm_mean)**2 for v in gstr.values())/len(teams))**0.5 or 1.0
    cons={t:(1-W_DEPLOY)*zg[t]+W_DEPLOY*ze[t] for t in teams}
    Gs={t:gm_mean+gm_sd*cons[t] for t in teams}
    astar={t:math.exp((Gs[t]+tilt[t])/2) for t in teams}; dstar={t:math.exp(-(Gs[t]-tilt[t])/2) for t in teams}
    deploy_state.append((year, AVG, astar, dstar, gms))

def home_eval(hf, mode):
    all_recs=[]; host_recs=[]
    for year, AVG, astar, dstar, gms in deploy_state:
        for d,h,a,hs,a_,neu,trn in gms:
            if neu: hfa=hfb=1.0
            elif mode=="asym": hfa, hfb = hf, 1.0
            else: hfa, hfb = math.sqrt(hf), 1.0/math.sqrt(hf)   # symmetric
            lh=AVG*astar[h]*dstar[a]*hfa; la=AVG*astar[a]*dstar[h]*hfb
            pH,pD,pA=dc_probs(lh,la,RHO_DEPLOY)
            q=temper((pH,pD,pA),1.3)
            out=0 if hs>a_ else (1 if hs==a_ else 2)
            all_recs.append((*q,out))
            if not neu: host_recs.append((*q,out))
    return ll_of(all_recs), ll_of(host_recs), len(host_recs)
print("\n" + "="*64)
print("HOME-ADVANTAGE experiment (deployed engine + T=1.3):")
print(f"  {'formulation':<10}{'hf':>6}{'LL(all)':>10}{'LL(host-only)':>16}")
for mode in ("asym","sym"):
    for hf in (1.0,1.10,1.15,1.20,1.30,1.45):
        lla,llh,nh=home_eval(hf,mode)
        tag=" <-live" if (mode=="asym" and abs(hf-HOME_ADV)<1e-9) else ""
        print(f"  {mode:<10}{hf:>6}{lla:>10.4f}{llh:>16.4f}{tag}   (host N={nh})")

# LOTO over hosting tournaments: fit hf on other tournaments' host games, score the
# held-out tournament's host games. Tells us if a data-driven hf generalises or is noise.
HF_GRID=[1.0,1.05,1.10,1.15,1.20,1.25,1.30,1.35,1.45]
def host_recs_by_year(hf, mode):
    by={}
    for year, AVG, astar, dstar, gms in deploy_state:
        rr=[]
        for d,h,a,hs,a_,neu,trn in gms:
            if neu: continue
            if mode=="asym": hfa,hfb=hf,1.0
            else: hfa,hfb=math.sqrt(hf),1.0/math.sqrt(hf)
            lh=AVG*astar[h]*dstar[a]*hfa; la=AVG*astar[a]*dstar[h]*hfb
            pH,pD,pA=dc_probs(lh,la,RHO_DEPLOY); q=temper((pH,pD,pA),1.3)
            rr.append((*q, 0 if hs>a_ else (1 if hs==a_ else 2)))
        if rr: by.setdefault(year,[]).extend(rr)
    return by
for mode in ("asym","sym"):
    cache={hf:host_recs_by_year(hf,mode) for hf in HF_GRID}
    hostyears=sorted({y for hf in HF_GRID for y in cache[hf]})
    s=0.0; nt=0; picks=[]
    for hy in hostyears:
        def trll(hf):
            recs=[r for y,rs in cache[hf].items() if y!=hy for r in rs]
            return ll_of(recs) if recs else 9e9
        bhf=min(HF_GRID, key=trll); picks.append(bhf)
        te=cache[bhf].get(hy,[])
        if te: s+=ll_of(te)*len(te); nt+=len(te)
    print(f"  [LOTO {mode}] per-tournament fitted hf={picks}; pooled held-out host LL={s/nt:.4f}")

# =============================================================================
# ISOTONIC vs TEMPERATURE calibration (LOTO). Tests whether a more flexible
# (monotone, non-parametric) recalibration beats the single-parameter temperature.
# Isotonic is fit on the pooled (prob, hit) pairs across all 3 classes of the TRAIN
# folds via pool-adjacent-violators, applied to held-out probs, then the 3-way is
# renormalised. With only ~330 train matches this can overfit -- LOTO is the judge.
# =============================================================================
def isotonic_fit(pairs):
    # pairs: list of (x, y) sorted by x; returns step function via PAV
    pts=sorted(pairs); xs=[p[0] for p in pts]; ys=[p[1] for p in pts]
    w=[1.0]*len(ys); yhat=ys[:]
    i=0
    while i<len(yhat)-1:
        if yhat[i]>yhat[i+1]+1e-12:
            # pool i and i+1
            tot=w[i]+w[i+1]; val=(yhat[i]*w[i]+yhat[i+1]*w[i+1])/tot
            yhat[i]=val; w[i]=tot; del yhat[i+1]; del w[i+1]; del xs[i+1]
            if i>0: i-=1
        else: i+=1
    # build lookup: xs are left edges of pooled blocks
    return xs, yhat
def isotonic_apply(model, x):
    xs,yh=model
    import bisect
    j=bisect.bisect_right(xs,x)-1
    return yh[max(0,min(j,len(yh)-1))]
pairs_all=list(zip(allrec,allyr))
ys_set=sorted(set(allyr))
iso_s=tmp_s=base_s=0.0; nt=0
for hy in ys_set:
    tr=[r for r,y in pairs_all if y!=hy]; te=[r for r,y in pairs_all if y==hy]
    # temperature
    Tk=fit_T(tr)
    # isotonic over flattened classes
    flat=[]
    for pH,pD,pA,out in tr:
        for k,p in enumerate((pH,pD,pA)): flat.append((p, 1.0 if out==k else 0.0))
    iso=isotonic_fit(flat)
    for pH,pD,pA,out in te:
        # base
        base_s-=math.log(max((pH,pD,pA)[out],1e-12))
        # temperature
        q=temper((pH,pD,pA),Tk); tmp_s-=math.log(max(q[out],1e-12))
        # isotonic + renormalise
        r=[max(isotonic_apply(iso,p),1e-9) for p in (pH,pD,pA)]; sr=sum(r); r=[x/sr for x in r]
        iso_s-=math.log(max(r[out],1e-12))
    nt+=len(te)
print("\n" + "="*64)
print("CALIBRATION METHOD COMPARISON (LOTO held-out log-loss, deployed engine):")
print(f"  uncalibrated : {base_s/nt:.4f}")
print(f"  temperature  : {tmp_s/nt:.4f}   <- deployed (simulate.CALIB_T)")
print(f"  isotonic     : {iso_s/nt:.4f}")
print(f"  -> {'isotonic beats temperature' if iso_s<tmp_s-1e-4 else 'temperature >= isotonic; keep temperature (simpler, no overfit)'}")

# =============================================================================
# DRAW-SPECIFIC calibration & a draw-inflation test. Is the draw probability mass
# (governed by rho + the engine) well-calibrated on its own? Then test whether a
# multiplicative draw-inflation gamma (pD->pD*gamma, renormalise) helps OOS via LOTO.
# =============================================================================
deployed_T=[(*temper((pH,pD,pA),1.3),out) for pH,pD,pA,out in allrec]
# observed draw rate vs mean predicted draw prob (deployed + T)
mpd=sum(r[1] for r in deployed_T)/len(deployed_T); adr=sum(1 for r in deployed_T if r[3]==1)/len(deployed_T)
print("\nDRAW calibration (deployed + T=1.3):")
print(f"  mean predicted draw prob {mpd:.3f}  vs  actual draw rate {adr:.3f}")
db=collections.defaultdict(lambda:[0.0,0,0])
for pH,pD,pA,out in deployed_T:
    b=min(9,int(pD*10)); db[b][0]+=pD; db[b][1]+=(1 if out==1 else 0); db[b][2]+=1
print(f"  {'binD':>8}{'pred':>8}{'actual':>8}{'n':>7}")
for b in sorted(db):
    s,h,c=db[b]
    if c: print(f"  {b/10:.1f}-{b/10+0.1:.1f}{s/c:>8.2f}{h/c:>8.2f}{c:>7}")
# draw-inflation gamma via LOTO (fit on train, apply to held-out, after temperature)
def apply_gamma(p3, g):
    pH,pD,pA=p3; pD2=pD*g; s=pH+pD2+pA; return [pH/s,pD2/s,pA/s]
def fit_gamma(recs):
    bg=1.0; bv=ll_of(recs); g=0.7
    while g<=1.40001:
        s=0.0
        for pH,pD,pA,out in recs:
            q=apply_gamma((pH,pD,pA),g); s-=math.log(max(q[out],1e-12))
        v=s/len(recs)
        if v<bv: bv=v; bg=g
        g+=0.05
    return bg
pairsT=list(zip(deployed_T,allyr))
g_s=base2=0.0; nt=0; gs=[]
for hy in ys_set:
    tr=[r for r,y in pairsT if y!=hy]; te=[r for r,y in pairsT if y==hy]
    g=fit_gamma(tr); gs.append(round(g,2))
    for pH,pD,pA,out in te:
        base2-=math.log(max((pH,pD,pA)[out],1e-12))
        q=apply_gamma((pH,pD,pA),g); g_s-=math.log(max(q[out],1e-12))
    nt+=len(te)
print(f"  draw-inflation gamma LOTO: per-fold g={gs}; held-out LL {base2/nt:.4f} -> {g_s/nt:.4f} "
      f"({'KEEP' if g_s<base2-1e-4 else 'no gain; gamma=1 (draws already well-modeled by rho)'})")
