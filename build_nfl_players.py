# -*- coding: utf-8 -*-
"""NFL player values from nflverse player_stats (recent seasons). Value = total EPA
(passing+rushing+receiving) — expected points the player has added — plus stat line.
Used to surface who drives each team and (later) a starting-QB availability toggle."""
import csv, io, os, urllib.request, datetime
from collections import defaultdict
PROJ = os.path.dirname(os.path.abspath(__file__)); csv.field_size_limit(10**7)
def get(url, t=60):
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    return urllib.request.urlopen(req, timeout=t).read().decode("utf-8", "replace")
def nfl_seasons(n=2, today=None):
    d = today or datetime.date.today(); sy = d.year if d.month >= 8 else d.year - 1
    return list(range(sy - n + 1, sy + 1))
SEASONS = nfl_seasons(2); CUR = max(SEASONS); HL = 1.2
RELO = {"OAK": "LV", "SD": "LAC", "STL": "LA", "LAR": "LA"}
def fix(t): return RELO.get(t, t)
def fl(x):
    try: return float(x)
    except (TypeError, ValueError): return 0.0
def first(r, *names):
    for n in names:
        if n in r and r[n] not in (None, "", "NA"): return r[n]
    return ""
print(f"player seasons: {SEASONS}", flush=True)
P = {}
hdr_shown = False
for yr in SEASONS:
    w = 0.5 ** ((CUR - yr) / HL)
    urls = [f"https://github.com/nflverse/nflverse-data/releases/download/stats_player/stats_player_week_{yr}.csv",
            f"https://github.com/nflverse/nflverse-data/releases/download/player_stats/player_stats_{yr}.csv"]
    rows = None
    for u in urls:
        try:
            rows = list(csv.DictReader(io.StringIO(get(u)))); break
        except Exception:
            continue
    if not rows:
        print(f"  {yr} player_stats unavailable (tried both paths)", flush=True); continue
    if not hdr_shown and rows:
        print("cols sample:", [c for c in rows[0].keys()][:40]); hdr_shown = True
    seen_this = set()
    for r in rows:
        pid = first(r, "player_id", "gsis_id")
        nm = first(r, "player_display_name", "player_name")
        pos = first(r, "position", "position_group")
        tm = fix(first(r, "recent_team", "team"))
        if not pid or not tm: continue
        seen_this.add(pid)
        e = w * (fl(r.get("passing_epa")) + fl(r.get("rushing_epa")) + fl(r.get("receiving_epa")))
        d = P.setdefault(pid, dict(nm=nm, pos=pos, tm=tm, epa=0.0, py=0.0, ry=0.0, recy=0.0,
                                   ptd=0.0, rtd=0.0, rectd=0.0, g=0.0, sw=0.0))
        d["nm"] = nm or d["nm"]; d["pos"] = pos or d["pos"]; d["tm"] = tm
        d["epa"] += e
        d["py"] += w * fl(r.get("passing_yards")); d["ry"] += w * fl(r.get("rushing_yards")); d["recy"] += w * fl(r.get("receiving_yards"))
        d["ptd"] += w * fl(r.get("passing_tds")); d["rtd"] += w * fl(r.get("rushing_tds")); d["rectd"] += w * fl(r.get("receiving_tds"))
        d["g"] += w
    for pid in seen_this:
        if pid in P: P[pid]["sw"] += w        # total season-weight (to convert sums -> per-season)
# rank within team, keep top contributors
by_team = defaultdict(list)
for pid, d in P.items():
    if d["g"] < 0.4: continue
    by_team[d["tm"]].append(d)
def per(d, k): return d[k] / d["sw"] if d["sw"] > 0 else 0.0      # convert recency-weighted sums -> per-season
def val(d): return per(d, "epa")                                  # player value = EPA per season
with open(os.path.join(PROJ, "nfl_players.csv"), "w", newline="", encoding="utf-8") as f:
    wr = csv.writer(f); wr.writerow(["team", "player", "pos", "value", "pass_yds", "rush_yds", "rec_yds", "tds"])
    for tm in sorted(by_team):
        ps = sorted(by_team[tm], key=lambda d: -val(d))[:7]
        for d in ps:
            wr.writerow([tm, d["nm"], d["pos"], round(val(d), 1),
                         round(per(d, "py")), round(per(d, "ry")), round(per(d, "recy")),
                         round(per(d, "ptd") + per(d, "rtd") + per(d, "rectd"))])
allp = sorted([d for d in P.values() if d["g"] >= 0.4], key=lambda d: -val(d))[:12]
print("TOP 12 by value (EPA/season):", [(d["nm"], d["pos"], d["tm"], round(val(d), 1)) for d in allp])
print("Wrote nfl_players.csv", flush=True)
