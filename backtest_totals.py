# -*- coding: utf-8 -*-
"""
Walk-forward validation of TOTAL-GOALS / BTTS calibration, with vs without the
goal-level fix. For each past tournament, build ratings from prior data only,
predict every group match's total + BTTS, compare to what actually happened.
Confirms the geometric-mean inflation fix generalizes (not just the 48 WC matches).
"""
import csv, math, datetime, itertools, collections
PROJ = r"C:\Users\bbraudo\Desktop\Claude Output\World Cup Model"
TEST = ([("FIFA World Cup", y) for y in (2010, 2014, 2018, 2022)] +
        [("UEFA Euro", y) for y in (2012, 2016, 2021, 2024)])
rows = []
for r in csv.DictReader(open(PROJ + r"\results.csv", encoding="utf-8")):
    try:
        d = datetime.date.fromisoformat(r["date"]); hs = int(r["home_score"]); a_ = int(r["away_score"])
    except (ValueError, KeyError): continue
    rows.append((d, r["home_team"], r["away_team"], hs, a_, r["neutral"].strip().upper() == "TRUE", r["tournament"]))
rows.sort(key=lambda x: x[0])

def elo_asof(cut):
    e = {}
    for d, h, a, hs, a_, neu, trn in rows:
        if d >= cut: break
        eh = e.get(h, 1500.0); ea = e.get(a, 1500.0); adj = 0 if neu else 65
        exp = 1/(1+10**((ea-(eh+adj))/400)); res = 1.0 if hs > a_ else (0.5 if hs == a_ else 0.0)
        gd = abs(hs-a_); g = 1 if gd <= 1 else (1.5 if gd == 2 else (11+gd)/8)
        dl = 30*g*(res-exp); e[h] = eh+dl; e[a] = ea-dl
    return e
def goals_asof(cut, hl=730):
    lo = datetime.date(cut.year-8, 1, 1); w = [m for m in rows if lo <= m[0] < cut]
    wt = lambda d: 0.5 ** ((cut-d).days/hl)
    tg = tw = 0.0
    for d, h, a, hs, a_, n, t in w: x = wt(d); tg += x*(hs+a_); tw += 2*x
    AVG = tg/tw if tw else 1.3
    att = {}; dfn = {}
    for d, h, a, hs, a_, n, t in w:
        for q in (h, a): att.setdefault(q, 1.0); dfn.setdefault(q, 1.0)
    for _ in range(30):
        na = {t:0. for t in att}; da = {t:0. for t in att}; nd = {t:0. for t in att}; dd = {t:0. for t in att}
        for d, h, a, hs, a_, n, t in w:
            x = wt(d)
            na[h] += x*hs; da[h] += x*AVG*dfn[a]; nd[a] += x*hs; dd[a] += x*AVG*att[h]
            na[a] += x*a_; da[a] += x*AVG*dfn[h]; nd[h] += x*a_; dd[h] += x*AVG*att[a]
        for t in att:
            if da[t] > 0: att[t] = na[t]/da[t]
            if dd[t] > 0: dfn[t] = nd[t]/dd[t]
        for dct in (att, dfn):
            g = math.exp(sum(math.log(max(v, 1e-6)) for v in dct.values())/len(dct))
            for t in dct: dct[t] /= g
    return att, dfn, AVG
def find_groups(inst):
    games = collections.defaultdict(list)
    for m in sorted(inst, key=lambda x: x[0]): games[m[1]].append(m); games[m[2]].append(m)
    opp = {t: {(m[2] if m[1] == t else m[1]) for m in gs[:3]} for t, gs in games.items()}
    adj = {t: {u for u in opp[t] if t in opp.get(u, set())} for t in opp}
    used = set(); gm = []
    for t in list(adj):
        if t in used: continue
        nb = [u for u in adj[t] if u not in used]
        for combo in itertools.combinations(nb, 3):
            four = (t,)+combo
            if all(b in adj[a] for a, b in itertools.combinations(four, 2)):
                grp = set(four); used.update(four); seen = set()
                for m in sorted(inst, key=lambda x: x[0]):
                    if m[1] in grp and m[2] in grp:
                        k = frozenset((m[1], m[2]))
                        if k not in seen: seen.add(k); gm.append(m)
                break
    return gm
def btts_dc(lh, la, rho=-0.12):
    f = [math.factorial(i) for i in range(9)]
    P = lambda k, l: math.exp(-l)*l**k/f[k]
    pa0 = pb0 = p00 = 0
    for i in range(9):
        for j in range(9):
            tau = (1-lh*la*rho) if i == 0 and j == 0 else (1+lh*rho) if i == 0 and j == 1 else (1+la*rho) if i == 1 and j == 0 else (1-rho) if i == 1 and j == 1 else 1
            m = P(i, lh)*P(j, la)*tau
            if j == 0: pa0 += m
            if i == 0: pb0 += m
            if i == 0 and j == 0: p00 += m
    return max(0., 1-pa0-pb0+p00)

raw_tot = []; cal_tot = []; act_tot = []; raw_btts = []; cal_btts = []; act_btts = []
for name, yr in TEST:
    inst = [m for m in rows if m[6] == name and m[0].year == yr]
    if not inst: continue
    cut = min(m[0] for m in inst); gms = find_groups(inst)
    if not gms: continue
    elo = elo_asof(cut); att, dfn, AVG = goals_asof(cut)
    teams = {t for m in gms for t in (m[1], m[2])}
    A = {t: math.log(att.get(t, 1.0)) for t in teams}; D = {t: -math.log(dfn.get(t, 1.0)) for t in teams}
    gstr = {t: A[t]+D[t] for t in teams}; tilt = {t: A[t]-D[t] for t in teams}
    zg = {t: v for t, v in gstr.items()}  # use goals strength (w=0.5 blend w/ elo z)
    em = sum(elo.get(t, 1500) for t in teams)/len(teams); es = (sum((elo.get(t, 1500)-em)**2 for t in teams)/len(teams))**.5 or 1
    gm_, gs_ = sum(gstr.values())/len(teams), (sum((v-sum(gstr.values())/len(teams))**2 for v in gstr.values())/len(teams))**.5 or 1
    cons = {t: 0.5*((gstr[t]-gm_)/gs_) + 0.5*((elo.get(t, 1500)-em)/es) for t in teams}
    Gs = {t: gm_+gs_*cons[t] for t in teams}
    astar = {t: math.exp((Gs[t]+tilt[t])/2) for t in teams}; dstar = {t: math.exp(-(Gs[t]-tilt[t])/2) for t in teams}
    infl = sum(astar[a]*dstar[b] for a, b in itertools.permutations(teams, 2))/(len(teams)*(len(teams)-1)); k = infl**.5
    for d, h, a, hs, a_, neu, trn in gms:
        hf = 1.0 if neu else 1.30
        lh = AVG*astar[h]*dstar[a]*hf; la = AVG*astar[a]*dstar[h]
        lhc = AVG*(astar[h]/k)*(dstar[a]/k)*hf; lac = AVG*(astar[a]/k)*(dstar[h]/k)
        raw_tot.append(lh+la); cal_tot.append(lhc+lac); act_tot.append(hs+a_)
        raw_btts.append(btts_dc(lh, la)); cal_btts.append(btts_dc(lhc, lac))
        act_btts.append(1 if hs > 0 and a_ > 0 else 0)
n = len(act_tot)
print(f"walk-forward group matches: {n}")
print(f"\nAVG TOTAL GOALS:   actual {sum(act_tot)/n:.2f}")
print(f"  model RAW (old):       {sum(raw_tot)/n:.2f}   (error {sum(raw_tot)/n-sum(act_tot)/n:+.2f})")
print(f"  model CALIBRATED(new): {sum(cal_tot)/n:.2f}   (error {sum(cal_tot)/n-sum(act_tot)/n:+.2f})")
print(f"\nBTTS:   actual {sum(act_btts)/n*100:.0f}%")
print(f"  model RAW (old):       {sum(raw_btts)/n*100:.0f}%   (error {(sum(raw_btts)-sum(act_btts))/n*100:+.0f})")
print(f"  model CALIBRATED(new): {sum(cal_btts)/n*100:.0f}%   (error {(sum(cal_btts)-sum(act_btts))/n*100:+.0f})")
# log-loss of BTTS prediction (calibrated should beat raw)
def bll(pr): return -sum(y*math.log(min(max(p,1e-9),1-1e-9))+(1-y)*math.log(min(max(1-p,1e-9),1-1e-9)) for p,y in zip(pr,act_btts))/n
print(f"\nBTTS log-loss:  RAW {bll(raw_btts):.4f}   CALIBRATED {bll(cal_btts):.4f}   baseline {bll([sum(act_btts)/n]*n):.4f}")
