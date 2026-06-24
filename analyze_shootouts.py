# -*- coding: utf-8 -*-
"""
Calibrate the knockout penalty-shootout model against 678 real shootouts.
The live model treats a shootout as ~a coin flip with a tiny favourite edge.
Is that right? Compute how often the higher-Elo team actually wins the shootout,
by pre-shootout Elo gap. Elo is reconstructed as-of each shootout date.
"""
import csv, os, datetime, collections
PROJ = os.path.dirname(os.path.abspath(__file__))

rows = []
for r in csv.DictReader(open(PROJ + r"\results.csv", encoding="utf-8")):
    try:
        d = datetime.date.fromisoformat(r["date"]); hs = int(r["home_score"]); a_ = int(r["away_score"])
    except (ValueError, KeyError):
        continue
    rows.append((d, r["home_team"], r["away_team"], hs, a_, r["neutral"].strip().upper() == "TRUE"))
rows.sort(key=lambda x: x[0])

sh_by_date = collections.defaultdict(list)
for s in csv.DictReader(open(PROJ + r"\shootouts.csv", encoding="utf-8")):
    try: d = datetime.date.fromisoformat(s["date"])
    except ValueError: continue
    sh_by_date[d].append((s["home_team"], s["away_team"], s["winner"]))

elo = {}; records = []                                  # (elo_gap, higher_elo_team_won_shootout)
i = 0
by_date = collections.OrderedDict()
for m in rows: by_date.setdefault(m[0], []).append(m)
for d, ms in by_date.items():
    for (h, a, w) in sh_by_date.get(d, []):             # snapshot Elo BEFORE today's matches
        eh = elo.get(h, 1500.0); ea = elo.get(a, 1500.0)
        if w not in (h, a): continue
        higher = h if eh >= ea else a
        records.append((abs(eh - ea), w == higher))
    for (dd, h, a, hs, a_, neu) in ms:                  # then update Elo with today's results
        eh = elo.get(h, 1500.0); ea = elo.get(a, 1500.0); adj = 0.0 if neu else 65.0
        exp_h = 1.0/(1.0+10**((ea-(eh+adj))/400.0)); res = 1.0 if hs > a_ else (0.5 if hs == a_ else 0.0)
        gd = abs(hs-a_); g = 1.0 if gd <= 1 else (1.5 if gd == 2 else (11+gd)/8.0)
        dl = 30.0*g*(res-exp_h); elo[h] = eh+dl; elo[a] = ea-dl

n = len(records); won = sum(1 for _, w in records if w)
print(f"shootouts with Elo history: {n}")
print(f"higher-Elo team won the shootout: {won}/{n} = {won/n*100:.1f}%  (50% = pure coin flip)")
for lo, hi in [(0,50),(50,100),(100,200),(200,9999)]:
    g = [w for gap, w in records if lo <= gap < hi]
    if g: print(f"  Elo gap {lo:>3}-{hi:<4}: higher-rated won {sum(g)}/{len(g)} = {sum(g)/len(g)*100:.0f}%")
# implied: map a within-match win-prob favourite to a shootout win-prob (logistic-ish)
print("\nInterpretation: if >~52-55%, the favourite has a real (small) shootout edge;")
print("if ~50%, the coin-flip assumption is correct and shouldn't be changed.")
