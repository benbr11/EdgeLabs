# -*- coding: utf-8 -*-
"""
Is the model under-rating FAVORITES, and does the mismatch-compression P cause it?

Motivation: the market rates elite teams (e.g. France) far higher per game than the
model does, and wc_calibrate.py hinted the model is under-confident at the top end
(predicts 94% -> happens 98%). COMPRESS=0.60 shrinks every favorite toward the field.
Now that goal TOTALS are handled independently by GOAL_SCALE (simulate.py), the
RESULT-market compression is free to be retuned purely for W/D/L calibration.

This sweep reuses totals_fix.py's exact point-in-time rating math (build_snapshot /
predict) and, for each P, reports out-of-sample how well the FAVORITE's 90' win
probability is calibrated -- overall and by confidence bucket -- plus WDL log-loss.
Pick the P whose favorite calibration is flattest (predicted ~= actual) at the strong-
favorite end WITHOUT materially hurting aggregate log-loss.

Usage:  python wc_favorite_calibration.py
"""
import math, sys
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass
import totals_fix as T

P_GRID = [0.60, 0.70, 0.80, 0.90, 1.00]

# collect per-match (favorite win prob, did favorite win, log-loss pieces) per P
# snapshot is independent of P, so build once per month and evaluate all P.
recs = []   # list of dicts: {P: (favp, favwon, actual, pA,pD,pB)} plus decisive flag
months = list(T.month_iter(T.TEST_START, T.TEST_END))
for i, mstart in enumerate(months):
    mend = months[i+1] if i+1 < len(months) else __import__("datetime").date(T.TEST_END.year+1,1,1)
    batch = [m for m in T.TESTS if mstart <= m[0] < mend]
    if not batch: continue
    snap = T.build_snapshot(mstart)
    if snap is None: continue
    for (d,h,a,hs,as_,neutral,tn) in batch:
        if h not in snap["teams"] or a not in snap["teams"]: continue
        if snap["ncount"][h] < T.MIN_PRIOR_MATCHES or snap["ncount"][a] < T.MIN_PRIOR_MATCHES: continue
        actual = "H" if hs>as_ else ("A" if as_>hs else "D")
        per = {}
        for P in P_GRID:
            po,pA,pD,pB = T.predict(snap["AVG"],snap["am"][h],snap["dm"][h],snap["s100"][h],
                                    snap["am"][a],snap["dm"][a],snap["s100"][a],neutral,P,snap["c2"][P])
            tot=pA+pD+pB; pA,pD,pB=pA/tot,pD/tot,pB/tot
            favp = max(pA,pB)                       # model's pick confidence (90')
            favside = "H" if pA>=pB else "A"
            per[P] = (favp, 1 if actual==favside else 0, actual, pA,pD,pB)
        recs.append(per)

n = len(recs)
print(f"Out-of-sample matches: {n:,}\n")
BUCKETS = [(0.40,0.50),(0.50,0.60),(0.60,0.70),(0.70,0.80),(0.80,1.01)]
print(f"{'P':>5}{'favMeanPred':>13}{'favActual':>11}{'gap':>7}{'WDL_ll':>9}   calibration by pick-confidence bucket (pred->actual, n)")
print("="*130)
for P in P_GRID:
    favps = [r[P][0] for r in recs]
    wons  = [r[P][1] for r in recs]
    mp = sum(favps)/n; act = sum(wons)/n
    ll = 0.0
    for r in recs:
        _,_,actual,pA,pD,pB = r[P]
        p = {"H":pA,"D":pD,"A":pB}[actual]
        ll += -math.log(max(p,1e-12))
    ll /= n
    cells = []
    for lo,hi in BUCKETS:
        idx = [k for k,fp in enumerate(favps) if lo<=fp<hi]
        if len(idx) >= 10:
            bp = sum(favps[k] for k in idx)/len(idx)*100
            ba = sum(wons[k] for k in idx)/len(idx)*100
            cells.append(f"{lo:.1f}-{hi:.1f}:{bp:.0f}->{ba:.0f}(n{len(idx)})")
        else:
            cells.append(f"{lo:.1f}-{hi:.1f}: --")
    print(f"{P:>5.2f}{mp*100:>12.1f}%{act*100:>10.1f}%{(mp-act)*100:>+7.1f}{ll:>9.4f}   " + "  ".join(cells))
print("="*130)
print("gap>0 => model OVER-states favorites; gap<0 => UNDER-states (favorites win more than predicted).")
print("Want: bucket pred->actual close (esp. the 0.7-0.8 / 0.8+ strong-favorite buckets), WDL_ll not much above its min.")
