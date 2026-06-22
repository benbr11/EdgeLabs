# -*- coding: utf-8 -*-
"""Show which teams have over/under-performed their xG so far this World Cup -- i.e.
where actual goals were lucky/unlucky. The model now regresses these toward xG."""
import csv
PROJ = r"C:\Users\bbraudo\Desktop\Claude Output\World Cup Model"
XG_NAME = {"Cabo Verde":"Cape Verde","Congo DR":"DR Congo","Czechia":"Czech Republic",
           "Côte d'Ivoire":"Ivory Coast","IR Iran":"Iran","Türkiye":"Turkey","USA":"United States"}
agg = {}
for r in csv.DictReader(open(PROJ + r"\wc2026_xg.csv", encoding="utf-8")):
    if r["status"] != "Completed" or not r["home_xg"]: continue
    h = XG_NAME.get(r["home_team_name"], r["home_team_name"])
    a = XG_NAME.get(r["away_team_name"], r["away_team_name"])
    hx, ax = float(r["home_xg"]), float(r["away_xg"]); hs, as_ = int(r["home_score"]), int(r["away_score"])
    for t, xf, xa, gf, ga in [(h, hx, ax, hs, as_), (a, ax, hx, as_, hs)]:
        d = agg.setdefault(t, [0.,0.,0,0,0]); d[0]+=xf; d[1]+=xa; d[2]+=gf; d[3]+=ga; d[4]+=1

print(f"{'team':<22}{'GP':>3}{'GF':>4}{'xGF':>6}{'+/-':>6}   {'GA':>3}{'xGA':>6}{'+/-':>6}")
for t in sorted(agg, key=lambda t: (agg[t][2]-agg[t][0]), reverse=True):
    xf, xa, gf, ga, gp = agg[t]
    print(f"{t:<22}{gp:>3}{gf:>4}{xf:>6.1f}{gf-xf:>+6.1f}   {ga:>3}{xa:>6.1f}{ga-xa:>+6.1f}")
print("\n+/- on GF: positive = scored MORE than chances (finishing luck -> model tempers).")
print("+/- on GA: positive = conceded MORE than chances faced.")
