# -*- coding: utf-8 -*-
"""
Accuracy proof for the UNIVERSAL player layer (goalscorers.csv), leak-free.

The StatsBomb match-backtest can't test the universal rates (goalscorers has no
"played but didn't score" rows). So we validate them the only honest way the data
allows: leave-FUTURE-out, year over year. For each player+year Y, predict their
open-play scoring RATE from recency-weighted goals STRICTLY BEFORE Y, then compare
to what they actually did in Y. If past rate predicts future scoring, the layer has
real signal (not just hindsight).

Metrics: Pearson/Spearman (predicted rate vs actual), separation (top vs bottom
predicted quartile actual rates), and per-(team,year) "did we name the top scorer".
"""
import csv, os, collections, math

PROJ = os.path.dirname(os.path.abspath(__file__))
HALF = 2.5                                              # recency half-life (years), matches build_player_intl
def yr(s): return int(s[:4]) if s[:4].isdigit() else None

grows = list(csv.DictReader(open(PROJ + r"\goalscorers.csv", encoding="utf-8")))
rrows = list(csv.DictReader(open(PROJ + r"\results.csv", encoding="utf-8")))

tm_year = collections.Counter()                         # (team, year) -> matches played
for r in rrows:
    y = yr(r["date"])
    if y:
        tm_year[(r["home_team"], y)] += 1; tm_year[(r["away_team"], y)] += 1

opg = collections.defaultdict(float)                    # (team, scorer, year) -> open-play goals
for r in grows:
    if (r.get("own_goal") or "").upper() == "TRUE" or (r.get("penalty") or "").upper() == "TRUE":
        continue
    y = yr(r["date"])
    if y: opg[(r["team"], r["scorer"], y)] += 1

players = set((t, s) for (t, s, y) in opg)
TARGETS = range(2010, 2027)
xs = []; ys = []; per_year = collections.defaultdict(list)
for (team, scorer) in players:
    for Y in TARGETS:
        mY = tm_year.get((team, Y), 0)
        if mY < 4:                                      # need enough opportunities in target year
            continue
        num = den = 0.0                                 # predicted rate from STRICTLY BEFORE Y
        for yy in range(Y - 12, Y):
            w = 0.5 ** ((Y - yy) / HALF)
            num += w * opg.get((team, scorer, yy), 0.0)
            den += w * tm_year.get((team, yy), 0)
        if den < 3 or num < 0.5:                        # need prior team history AND prior scoring
            continue                                    # (universe = established scorers, not every name ever)
        pred = num / den
        actual = opg.get((team, scorer, Y), 0.0) / mY
        xs.append(pred); ys.append(actual); per_year[Y].append((pred, actual, team, scorer))

n = len(xs)
def pearson(a, b):
    ma = sum(a)/len(a); mb = sum(b)/len(b)
    num = sum((x-ma)*(y-mb) for x, y in zip(a, b))
    da = math.sqrt(sum((x-ma)**2 for x in a)); db = math.sqrt(sum((y-mb)**2 for y in b))
    return num/(da*db) if da and db else float("nan")
def ranks(v):
    order = sorted(range(len(v)), key=lambda i: v[i]); rk = [0]*len(v); i = 0
    while i < len(order):
        j = i
        while j < len(order) and v[order[j]] == v[order[i]]: j += 1
        for t in range(i, j): rk[order[t]] = (i+j-1)/2
        i = j
    return rk
spearman = pearson(ranks(xs), ranks(ys))

print(f"samples (player-years, leak-free): {n:,}")
print(f"Pearson  r (predicted rate vs actual): {pearson(xs, ys):.3f}")
print(f"Spearman r (ordering):                 {spearman:.3f}")

# separation: actual scoring of the top vs bottom predicted quartile
paired = sorted(zip(xs, ys)); q = n // 4
botQ = paired[:q]; topQ = paired[-q:]
print(f"\nactual open-play rate, BOTTOM predicted quartile: {sum(b for _,b in botQ)/len(botQ):.3f} /game")
print(f"actual open-play rate, TOP    predicted quartile: {sum(b for _,b in topQ)/len(topQ):.3f} /game")
print(f"  -> top-quartile players actually score {(sum(b for _,b in topQ)/len(topQ))/max(sum(b for _,b in botQ)/len(botQ),1e-9):.1f}x more")

# per (team, year): did the model's top predicted scorer turn out to be a real top scorer?
hit = tot = 0
for Y, lst in per_year.items():
    byteam = collections.defaultdict(list)
    for pred, act, team, scorer in lst: byteam[team].append((pred, act, scorer))
    for team, ps in byteam.items():
        if len(ps) < 3: continue
        amax = max(a for _, a, _ in ps)
        if amax <= 0: continue
        pred_top = max(ps, key=lambda x: x[0])[2]
        real_top = set(s for _, a, s in ps if a == amax)
        tot += 1; hit += 1 if pred_top in real_top else 0
print(f"\nmodel's #1 predicted scorer WAS a team's actual top scorer that year: {hit}/{tot} ({hit/tot*100:.0f}%)")
print("(leak-free: every prediction uses only goals from before the year scored)")
