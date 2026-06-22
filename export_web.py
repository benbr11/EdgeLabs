# -*- coding: utf-8 -*-
"""Bake the live model into web/data.js for the static web app.
Run after build_ratings.py (and --refresh) so the site shows current numbers:
    python build_ratings.py --refresh
    python export_web.py
"""
import csv, json
import groups as G

import os
PROJ = os.path.dirname(os.path.abspath(__file__))
RHO = -0.12
MODS = {
    "avail_floor": 0.60, "fatigue_per_day": 0.025,
    "stakes": {"clinched": 0.93, "eliminated": 1.0, "must-win": 1.0, "normal": 1.0},
    "alt_pen_per_km": 0.05, "alt_buffer": 500, "heat_pen_per_c": 0.005, "heat_buffer": 8,
    "weather": {"clear": 1.0, "rain": 0.90, "cold": 0.95, "heat": 0.93},
}

# --- ratings + params --------------------------------------------------------
R = {}; avg = home_adv = None
for r in csv.DictReader(open(PROJ + r"\ratings.csv", encoding="utf-8")):
    R[r["team"]] = {"att_mult": float(r["attack_mult"]), "dfn_mult": float(r["defense_mult"]),
                    "att100": float(r["attack_100"]), "def100": float(r["defense_100"]),
                    "elo": round(float(r["elo"])), "fifa": round(float(r["fifa_points"]))}
    avg = float(r["league_avg_goals"]); home_adv = float(r["home_adv_mult"])

for r in csv.DictReader(open(PROJ + r"\context.csv", encoding="utf-8")):
    if r["team"] in R:
        R[r["team"]].update({"home_temp": float(r["home_temp_c"]), "home_alt": float(r["home_alt_m"]),
                             "pop": float(r["population_m"]), "climate": r["climate"]})

# --- groups, standings, situations ------------------------------------------
groups, played, sched = G.get_groups()
group_complete = sum(len(G.group_state(g, played, sched)[4]) for g in groups) == 0
groups_sorted = sorted(groups, key=lambda grp: max(R[t]["att100"] + R[t]["def100"] for t in grp if t in R),
                       reverse=True)
group_of = {}; groups_out = []
for i, grp in enumerate(groups_sorted, 1):
    name = f"Group {i}"
    pts, gf, ga, pl, rem = G.group_state(grp, played, sched)
    table = sorted(grp, key=lambda t: (-pts[t], -(gf[t] - ga[t]), -gf[t]))
    for t in grp:
        group_of[t] = name
        if t in R:
            lab, stk = G.situation(t, grp, pts, rem)
            R[t]["stakes"] = stk; R[t]["stakes_label"] = lab
    groups_out.append({"name": name,
                       "table": [{"team": t, "P": pl[t], "pts": pts[t], "gf": gf[t],
                                  "ga": ga[t], "gd": gf[t] - ga[t]} for t in table],
                       "remaining": [[h, a] for h, a in rem]})
for t in R:
    R[t].setdefault("group", group_of.get(t, "?"))
    R[t].setdefault("stakes", "normal"); R[t].setdefault("stakes_label", "")

def stage_of(h, a):
    if group_of.get(h) and group_of.get(h) == group_of.get(a):
        return "group"
    return "knockout" if group_complete else "unknown"

# --- xG for played matches ---------------------------------------------------
XG_NAME = {"Cabo Verde": "Cape Verde", "Congo DR": "DR Congo", "Czechia": "Czech Republic",
           "Côte d'Ivoire": "Ivory Coast", "IR Iran": "Iran", "Türkiye": "Turkey", "USA": "United States"}
mxg = {}
try:
    for r in csv.DictReader(open(PROJ + r"\wc2026_xg.csv", encoding="utf-8")):
        if r["status"] != "Completed" or not r["home_xg"]:
            continue
        h = XG_NAME.get(r["home_team_name"], r["home_team_name"])
        a = XG_NAME.get(r["away_team_name"], r["away_team_name"])
        mxg[(r["date"], frozenset((h, a)))] = {h: float(r["home_xg"]), a: float(r["away_xg"])}
except FileNotFoundError:
    pass

# --- fixtures ----------------------------------------------------------------
fixtures = []
for (d, h, a, hs, as_) in played:
    xg = mxg.get((d, frozenset((h, a))), {})
    fixtures.append({"date": d, "home": h, "away": a, "status": "played",
                     "hs": hs, "as": as_, "hxg": xg.get(h), "axg": xg.get(a), "stage": stage_of(h, a)})
for (d, h, a) in sched:
    fixtures.append({"date": d, "home": h, "away": a, "status": "scheduled",
                     "stage": stage_of(h, a)})
fixtures.sort(key=lambda x: x["date"])
generated = max((f["date"] for f in fixtures if f["status"] == "played"), default="")

DATA = {"params": {"avg": avg, "home_adv": home_adv, "rho": RHO, "hosts": sorted(G.HOSTS),
                   "group_complete": group_complete, "generated": generated},
        "mods": MODS, "teams": R, "groups": groups_out, "fixtures": fixtures}

with open(PROJ + r"\web\data.js", "w", encoding="utf-8") as f:
    f.write("window.WC_DATA = " + json.dumps(DATA, ensure_ascii=False) + ";\n")
print(f"Wrote web/data.js: {len(R)} teams, {len(groups_out)} groups, "
      f"{len(fixtures)} fixtures, generated {generated}, group_complete={group_complete}")
