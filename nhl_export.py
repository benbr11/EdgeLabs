# -*- coding: utf-8 -*-
"""Bake the NHL model into web/nhl_data.js (window.NHL_DATA) for the web app.
Run after build_nhl_xg.py + build_nhl.py. Mirrors export_web.py (soccer)."""
import csv, json, os, datetime, urllib.request
PROJ = os.path.dirname(os.path.abspath(__file__))
def get(url, t=20):
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    return json.loads(urllib.request.urlopen(req, timeout=t).read())

# team full names + logos from the live API (fallback to abbrev)
NAMES = {}; LOGOS = {}
try:
    for s in get("https://api-web.nhle.com/v1/standings/now")["standings"]:
        ab = {"ARI": "UTA"}.get(s["teamAbbrev"]["default"], s["teamAbbrev"]["default"])
        NAMES[ab] = s["teamName"]["default"]; LOGOS[ab] = s.get("teamLogo", "")
except Exception as e:
    print("  standings fetch failed (names fallback to abbrev):", e)

teams = {}; params = {}
for r in csv.DictReader(open(PROJ + r"\nhl_ratings.csv", encoding="utf-8")):
    t = r["team"]
    teams[t] = {"name": NAMES.get(t, t), "logo": LOGOS.get(t, ""),
                "att": float(r["attack_mult"]), "dfn": float(r["defense_mult"]),
                "att100": float(r["attack_100"]), "def100": float(r["defense_100"]),
                "elo": round(float(r["elo"])), "xgf": float(r["xgf_pg"]), "xga": float(r["xga_pg"]),
                "gsax": float(r["gsax_per_shot"]), "pp": float(r["pp_pct"]), "pk": float(r["pk_pct"])}
    # winprob_temp: logit temperature that softens the systematically overconfident raw
    # Poisson favourite win-prob (calibrated on the OOS walk-forward backtest; see
    # nhl_predict.py WINPROB_TEMP). 1.0 = off; 2.0 = production-calibrated.
    params = {"avg": float(r["avg_goals"]), "home_adv": float(r["home_adv"]), "sog": 29.0,
              "winprob_temp": 2.0}

goalies = []
try:
    for r in csv.DictReader(open(PROJ + r"\nhl_goalies.csv", encoding="utf-8")):
        if int(r["games"]) >= 15 and r["team"] in teams:
            goalies.append({"n": r["goalie"], "team": r["team"], "gsax": float(r["gsax_per_shot"]),
                            "sv": float(r["sv_pct"]), "g": int(r["games"])})
except FileNotFoundError:
    pass
goalies.sort(key=lambda x: (x["team"], -x["g"]))

skaters = []
try:
    rows = list(csv.DictReader(open(PROJ + r"\nhl_skaters.csv", encoding="utf-8")))
    rows.sort(key=lambda r: -float(r["ixg"]))
    for r in rows[:150]:
        if r["team"] in teams:
            skaters.append({"n": r["skater"], "team": r["team"], "pos": r["pos"],
                            "ixg_pg": float(r["ixg_pg"]), "g_pg": float(r["g_pg"])})
except FileNotFoundError:
    pass

generated = datetime.date.today().isoformat()
DATA = {"params": params, "teams": teams, "goalies": goalies, "skaters": skaters, "generated": generated}
with open(PROJ + r"\web\nhl_data.js", "w", encoding="utf-8") as f:
    f.write("window.NHL_DATA = " + json.dumps(DATA, ensure_ascii=False) + ";\n")
print(f"Wrote web/nhl_data.js: {len(teams)} teams, {len(goalies)} goalies, {len(skaters)} skaters")
