# -*- coding: utf-8 -*-
"""
check_acceptance.py -- the 4 acceptance tests for the UFC model.
  1. Islam Makhachev is division-wide #1 (top by rank_elo in his division).
  2. Khamzat Chimaev is #1 Middleweight (top MW by rank_elo) AND beats Strickland (p>0.5).
  3. Arman Tsarukyan is favored over Justin Gaethje (p>0.5).
  4. Ciryl Gane vs (a top HW, e.g. Aspinall/Jones) is ~coin-flip (0.40<=p<=0.60).
"""
import sys
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass
import pandas as pd
import ufc_model as M

ratings = pd.read_csv("ufc_ratings.csv")
fighters = pd.read_csv("ufc_fighters.csv")
log = pd.read_csv("ufc_fight_log.csv", parse_dates=["date"])
elo = M.compute_elo(log, fighters)

def top_in_div(div_name, n=3):
    g = ratings[ratings["division"] == div_name].sort_values("rank_elo", ascending=False)
    return list(zip(g["fighter"].head(n), g["rank_elo"].head(n)))

def p_a_beats_b(a, b, rounds=5):
    sa, sb = M.get_stats(fighters, a), M.get_stats(fighters, b)
    wp = M.win_probability(sa, sb, elo)
    return wp["p_a"]

results = []

# find Islam's division
islam_div = ratings[ratings["fighter"].str.contains("Makhachev", case=False, na=False)]
idiv = islam_div.iloc[0]["division"] if len(islam_div) else "Lightweight"
top_lw = top_in_div(idiv)
t1 = top_lw[0][0].lower().find("makhachev") >= 0 if top_lw else False
results.append(("1. Islam #1 in %s" % idiv, t1, str(top_lw)))

# Chimaev #1 MW
top_mw = top_in_div("Middleweight")
t2a = top_mw[0][0].lower().find("chimaev") >= 0 if top_mw else False
p_chim_strick = p_a_beats_b("Khamzat Chimaev", "Sean Strickland")
t2b = p_chim_strick > 0.5
results.append(("2a. Chimaev #1 MW", t2a, str(top_mw)))
results.append(("2b. Chimaev beats Strickland", t2b, "p=%.3f" % p_chim_strick))

# Arman > Gaethje
p_arman = p_a_beats_b("Arman Tsarukyan", "Justin Gaethje")
t3 = p_arman > 0.5
results.append(("3. Arman > Gaethje", t3, "p=%.3f" % p_arman))

# Gane ~coin-flip vs a top HW (canonical superfight framing: Gane vs Jones).
# Gane is the #1 HW by Elo here; a coin-flip vs the other elite is the intended read.
p_gane = p_a_beats_b("Ciryl Gane", "Jon Jones")
t4 = 0.40 <= p_gane <= 0.60
results.append(("4. Gane vs Jones coin-flip", t4, "p=%.3f" % p_gane))

allpass = all(r[1] for r in results)
print("=" * 60)
print("ACCEPTANCE TESTS")
print("=" * 60)
for name, ok, detail in results:
    print(f"  [{'PASS' if ok else 'FAIL'}]  {name:32s} {detail}")
print("=" * 60)
print("ALL PASS" if allpass else "SOME FAILED")
sys.exit(0 if allpass else 1)
