# -*- coding: utf-8 -*-
"""
Calibrate the goal LEVEL for competitive matches (fix the totals over-prediction).

totals_fix.py showed that at every mismatch-compression P the model still predicts
~63-66% Over-2.5 while reality is ~50% -- a uniform LEVEL bias, not a mismatch-shape
one (compression barely moves it). Root cause: `AVG` (goals/team/match) is fit on ALL
recent internationals, but the model is applied to tighter COMPETITIVE fixtures.

Fix under test: a single multiplicative GOAL_SCALE on both lambdas
    lambda_A, lambda_B  ->  GOAL_SCALE * lambda_A, GOAL_SCALE * lambda_B
which shifts the whole goal distribution without changing the attack/defense RATIO,
so W/D/L should stay ~flat. We reuse totals_fix's EXACT point-in-time rating math
(build_snapshot/predict) at the live P=0.60, and -- crucially -- FIT the scale on a
TRAIN window and VALIDATE on a later held-out window so it is not over-fit or leaky.

Usage:  python wc_totals_calibrate.py
"""
import math, datetime, sys
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass

import totals_fix as T   # importable now (its run block is guarded under main())

P_LIVE   = 0.60
GS_GRID  = [1.00, 0.95, 0.92, 0.90, 0.88, 0.85, 0.82, 0.80, 0.75]
SPLIT    = datetime.date(2025, 9, 1)   # train < SPLIT ; held-out test >= SPLIT

# ---- collect per-match predictions once (snapshot is independent of GS) ----
# record = (date, over_actual(0/1), actual "H"/"D"/"A", {GS: (p_over, pA,pD,pB)})
records = []
months = list(T.month_iter(T.TEST_START, T.TEST_END))
for i, mstart in enumerate(months):
    mend = months[i+1] if i+1 < len(months) else datetime.date(T.TEST_END.year+1, 1, 1)
    batch = [m for m in T.TESTS if mstart <= m[0] < mend]
    if not batch: continue
    snap = T.build_snapshot(mstart)
    if snap is None: continue
    c2 = snap["c2"][P_LIVE]
    for (d, h, a, hs, as_, neutral, tn) in batch:
        if h not in snap["teams"] or a not in snap["teams"]: continue
        if snap["ncount"][h] < T.MIN_PRIOR_MATCHES or snap["ncount"][a] < T.MIN_PRIOR_MATCHES: continue
        over = 1 if hs + as_ >= 3 else 0
        actual = "H" if hs > as_ else ("A" if as_ > hs else "D")
        by_gs = {}
        for gs in GS_GRID:
            po, pA, pD, pB = T.predict(snap["AVG"]*gs, snap["am"][h], snap["dm"][h], snap["s100"][h],
                                       snap["am"][a], snap["dm"][a], snap["s100"][a], neutral, P_LIVE, c2)
            tot = pA + pD + pB
            by_gs[gs] = (po, pA/tot, pD/tot, pB/tot)
        records.append((d, over, actual, by_gs))

train = [r for r in records if r[0] <  SPLIT]
test  = [r for r in records if r[0] >= SPLIT]
print(f"Collected {len(records):,} competitive predictions   train={len(train):,}  held-out={len(test):,}")
print(f"(fixed live mismatch-compression P={P_LIVE};  fitting a single goal-level scale)\n")

def metrics(rows, gs):
    """Over-2.5 skill/calibration + WDL log-loss for one GS on a row set."""
    n = len(rows)
    oact = [r[1] for r in rows]
    base = sum(oact)/n
    b_brier = sum((base-y)**2 for y in oact)/n
    preds = [r[3][gs][0] for r in rows]
    mp = sum(preds)/n
    brier = sum((p-y)**2 for p, y in zip(preds, oact))/n
    skill = (b_brier - brier)/b_brier*100
    # WDL log-loss
    wll = 0.0
    for r in rows:
        po, pA, pD, pB = r[3][gs]
        p = {"H": pA, "D": pD, "A": pB}[r[2]]
        wll += -math.log(max(p, 1e-12))
    wll /= n
    # high-bucket calibration (0.7-0.8, 0.9+)
    def realized(lo, hi):
        idx = [j for j, p in enumerate(preds) if lo <= p < hi]
        return (sum(oact[j] for j in idx)/len(idx)*100) if len(idx) >= 8 else float("nan")
    return {"mp": mp*100, "brier": brier, "skill": skill, "wll": wll,
            "over_rate": base*100, "c78": realized(0.7, 0.8), "c9": realized(0.9, 1.01)}

def table(title, rows):
    print("="*82); print(title); print("="*82)
    print(f"{'scale':>6}{'O25 pred':>10}{'O25 skill':>11}{'cal .7-.8':>11}{'cal .9+':>9}{'WDL ll':>10}")
    for gs in GS_GRID:
        m = metrics(rows, gs)
        print(f"{gs:>6.2f}{m['mp']:>9.1f}%{m['skill']:>+10.1f}%{m['c78']:>10.1f}%{m['c9']:>8.1f}%{m['wll']:>10.4f}")
    print(f"  (Over-2.5 actual rate here: {sum(r[1] for r in rows)/len(rows)*100:.1f}%)\n")

table("TRAIN  (< 2025-09-01) — pick the scale, largest that maximizes Over-2.5 skill", train)

# choose scale on TRAIN: maximize Over-2.5 Brier skill (tie -> largest scale = least intervention)
best_gs = max(GS_GRID, key=lambda gs: (round(metrics(train, gs)["skill"], 4), gs))
print(f">>> chosen GOAL_SCALE (fit on train) = {best_gs:.2f}\n")

table("HELD-OUT  (>= 2025-09-01) — did the chosen scale generalize?", test)

mt = metrics(test, best_gs); m1 = metrics(test, 1.00)
print("="*82)
print(f"HELD-OUT verdict at GOAL_SCALE={best_gs:.2f}  vs  no-scale (1.00):")
print(f"  Over-2.5 mean pred : {m1['mp']:5.1f}%  ->  {mt['mp']:5.1f}%   (actual {mt['over_rate']:.1f}%)")
print(f"  Over-2.5 Brier skill: {m1['skill']:+5.1f}% ->  {mt['skill']:+5.1f}%   (positive = beats base rate)")
print(f"  WDL log-loss        : {m1['wll']:.4f} ->  {mt['wll']:.4f}   (want ~unchanged)")
print("="*82)
