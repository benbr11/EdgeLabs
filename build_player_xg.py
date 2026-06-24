# -*- coding: utf-8 -*-
"""
Per-player shot-level xG + set-piece profile from StatsBomb open data.
For each player (recency-weighted across recent tournaments) we extract:
  apps, non-penalty xG, shots, goals, penalties taken/scored, direct free-kicks taken/xG/scored.
Each shot's xG is StatsBomb's coordinate-based statsbomb_xg. Output: player_xg.csv.
Run once (heavy); committed and merged into the player model by export_web.py.
"""
import json, csv, os, urllib.request, collections

PROJ = os.path.dirname(os.path.abspath(__file__))
BASE = "https://raw.githubusercontent.com/statsbomb/open-data/master/data"
# (competition_id, season_id, label, recency_weight)
COMPS = [(43,106,"World Cup 2022",0.70), (55,282,"Euro 2024",1.00), (55,43,"Euro 2020",0.55),
         (223,282,"Copa America 2024",1.00), (1267,107,"AFCON 2023",0.85), (43,3,"World Cup 2018",0.45)]

def fetch(url):
    with urllib.request.urlopen(url, timeout=45) as r:
        return json.loads(r.read())

apps=collections.defaultdict(float); team_of={}
npxg=collections.defaultdict(float); goals=collections.defaultdict(float); shots=collections.defaultdict(float)
pen_sh=collections.defaultdict(float); pen_g=collections.defaultdict(float)
fk_sh=collections.defaultdict(float); fk_xg=collections.defaultdict(float); fk_g=collections.defaultdict(float)
assists=collections.defaultdict(float); xa=collections.defaultdict(float); keyp=collections.defaultdict(float)  # REAL assists + expected assists

for comp, season, label, w in COMPS:
    try:
        matches = fetch(f"{BASE}/matches/{comp}/{season}.json")
    except Exception as e:
        print(f"skip {label}: {e}", flush=True); continue
    print(f"{label}: {len(matches)} matches (w={w})", flush=True)
    for i, m in enumerate(matches):
        try:
            events = fetch(f"{BASE}/events/{m['match_id']}.json")
        except Exception:
            continue
        seen = set(); id2pl = {}
        for ev in events:
            et = ev.get("type", {}).get("name"); tm = ev.get("team", {}).get("name")
            pid = ev.get("id"); _pl = ev.get("player", {}).get("name")
            if pid and _pl: id2pl[pid] = (_pl, tm)               # map event id -> passer (for assist credit)
            if et == "Starting XI":
                for pl in ev.get("tactics", {}).get("lineup", []):
                    nm = pl.get("player", {}).get("name")
                    if nm: seen.add((nm, tm))
            elif et == "Substitution":
                rep = ev.get("substitution", {}).get("replacement", {}).get("name")
                if rep: seen.add((rep, tm))
            elif et == "Shot":
                nm = ev.get("player", {}).get("name")
                if not nm: continue
                key = (nm, tm); seen.add(key); team_of[key] = tm
                sh = ev.get("shot", {}); sxg = sh.get("statsbomb_xg", 0) or 0
                stype = sh.get("type", {}).get("name"); goal = sh.get("outcome", {}).get("name") == "Goal"
                shots[key] += w
                if goal: goals[key] += w
                if stype == "Penalty":
                    pen_sh[key] += w
                    if goal: pen_g[key] += w
                else:
                    npxg[key] += w * sxg
                    kp = sh.get("key_pass_id")                   # the pass that created this shot -> credit assister
                    if kp and kp in id2pl:
                        ak = id2pl[kp]; keyp[ak] += w; xa[ak] += w * sxg
                        if goal: assists[ak] += w
                    if stype == "Free Kick":
                        fk_sh[key] += w; fk_xg[key] += w * sxg
                        if goal: fk_g[key] += w
        for key in seen:
            apps[key] += w; team_of.setdefault(key, key[1])
        if i % 10 == 0: print(f"  {label} {i+1}/{len(matches)}", flush=True)

with open(PROJ + r"\player_xg.csv", "w", newline="", encoding="utf-8") as f:
    wr = csv.writer(f)
    wr.writerow(["player","team","apps","npxg","goals","shots","pen_sh","pen_g","fk_sh","fk_xg","fk_g","assists","xa","key_passes"])
    for key in sorted(apps, key=lambda k: -npxg[k]):
        if apps[key] < 0.5: continue
        nm, tm = key
        wr.writerow([nm, tm, round(apps[key],2), round(npxg[key],2), round(goals[key],2), round(shots[key],1),
                     round(pen_sh[key],2), round(pen_g[key],2), round(fk_sh[key],2), round(fk_xg[key],3), round(fk_g[key],2),
                     round(assists[key],2), round(xa[key],3), round(keyp[key],1)])
print(f"\nWrote player_xg.csv: {len(apps)} players", flush=True)
