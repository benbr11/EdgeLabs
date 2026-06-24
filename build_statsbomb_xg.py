# -*- coding: utf-8 -*-
"""
Compute shot-level expected goals (xG) per team from StatsBomb Open Data — the
same kind of data the TikTok model uses (every shot + its xG). Historical major
international tournaments only (that's what StatsBomb publishes free). Run ONCE;
output statsbomb_xg.csv is committed and blended into build_ratings.py.
"""
import json, csv, os, urllib.request, collections

PROJ = os.path.dirname(os.path.abspath(__file__))
BASE = "https://raw.githubusercontent.com/statsbomb/open-data/master/data"
COMPS = [(43,106,"World Cup 2022"), (55,282,"Euro 2024"),
         (223,282,"Copa America 2024"), (1267,107,"AFCON 2023")]
SB_NAME = {"Korea Republic":"South Korea","IR Iran":"Iran","Côte d'Ivoire":"Ivory Coast",
           "Czechia":"Czech Republic","Türkiye":"Turkey","Cabo Verde":"Cape Verde",
           "Congo DR":"DR Congo","United States of America":"United States"}

def fetch(url):
    with urllib.request.urlopen(url, timeout=40) as r:
        return json.loads(r.read())

xgf = collections.defaultdict(float); xga = collections.defaultdict(float)
games = collections.defaultdict(int); shots = collections.defaultdict(int)
for comp, season, label in COMPS:
    try:
        matches = fetch(f"{BASE}/matches/{comp}/{season}.json")
    except Exception as e:
        print(f"skip {label}: {e}", flush=True); continue
    print(f"{label}: {len(matches)} matches", flush=True)
    for i, m in enumerate(matches):
        mid = m["match_id"]
        home = m["home_team"]["home_team_name"]; away = m["away_team"]["away_team_name"]
        h = SB_NAME.get(home, home); a = SB_NAME.get(away, away)
        try:
            events = fetch(f"{BASE}/events/{mid}.json")
        except Exception:
            continue
        hx = ax = 0.0
        for ev in events:
            if ev.get("type", {}).get("name") == "Shot":
                xg = ev.get("shot", {}).get("statsbomb_xg", 0) or 0
                tm = ev.get("team", {}).get("name")
                if tm == home: hx += xg; shots[h] += 1
                elif tm == away: ax += xg; shots[a] += 1
        xgf[h] += hx; xga[h] += ax; xgf[a] += ax; xga[a] += hx
        games[h] += 1; games[a] += 1
        if i % 10 == 0: print(f"  {label} {i+1}/{len(matches)}", flush=True)

with open(PROJ + r"\statsbomb_xg.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f); w.writerow(["team","games","xg_for","xg_against","xgf_pg","xga_pg"])
    for t in sorted(games, key=lambda t: -games[t]):
        if games[t] == 0: continue
        w.writerow([t, games[t], round(xgf[t],2), round(xga[t],2),
                    round(xgf[t]/games[t],3), round(xga[t]/games[t],3)])
print(f"\nWrote statsbomb_xg.csv: {len(games)} teams", flush=True)
