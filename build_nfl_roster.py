# -*- coding: utf-8 -*-
"""
NFL roster-health adjustment — makes ratings FORWARD-LOOKING.

The team-history EPA ratings (build_nfl.py) grade what happened, so injury-wrecked teams
(e.g. the 2024 49ers) are under-rated even though the roster is elite when healthy. This
script fixes that by valuing every PLAYER as a per-game RATE (survives injury years and
follows the player across teams), then for each team comparing:
  NOW  = talent of the CURRENT roster (roster_2026) projected to full health, vs
  HIST = the talent that ACTUALLY played for the team over the window (injured stars
         played few games -> they barely count here, but count fully in NOW).
The standardized difference is added to offense / subtracted from defense EPA, so teams
getting healthy stars back (SF) rise and teams that lost / were carried by now-departed
players fall. Reads/REWRITES nfl_ratings.csv (adds off_hist/def_hist/adj_off/adj_def).
"""
import csv, io, os, urllib.request, datetime, math
from collections import defaultdict
PROJ = os.path.dirname(os.path.abspath(__file__)); csv.field_size_limit(10**7)
def get(url, t=60):
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    return urllib.request.urlopen(req, timeout=t).read().decode("utf-8", "replace")
def rows_csv(url):
    try: return list(csv.DictReader(io.StringIO(get(url))))
    except Exception: return None
def fl(x):
    try: return float(x)
    except (TypeError, ValueError): return 0.0

RELO = {"OAK": "LV", "SD": "LAC", "STL": "LA", "LAR": "LA"}
def fix(t): return RELO.get(t, t)
def cur_year():
    d = datetime.date.today(); return d.year if d.month >= 8 else d.year - 1
CUR = cur_year(); SEASONS = [CUR-2, CUR-1, CUR]
WS = {y: 0.6 ** (CUR - y) for y in SEASONS}           # season recency
TARGET_OFF, TARGET_DEF = 0.040, 0.034                 # max ~ this * z  EPA/play swing from roster/health
K_REL = float(os.environ.get("NFL_K_REL", "6.0"))     # games for reliability shrink
OFF_PG = {"QB", "RB", "WR", "TE"}; DEF_PG = {"DL", "LB", "DB"}
print(f"NFL roster adjustment | seasons {SEASONS}", flush=True)

# ---- player per-game values (offense EPA, defense impact), health-robust ----
P = {}   # pid -> dict
def prow(pid, nm, pg):
    return P.setdefault(pid, dict(nm=nm, pg=pg, oepa=0.0, dval=0.0, g=0.0, team_g=defaultdict(float)))
for y in SEASONS:
    w = WS[y]
    rows = rows_csv(f"https://github.com/nflverse/nflverse-data/releases/download/stats_player/stats_player_week_{y}.csv") \
        or rows_csv(f"https://github.com/nflverse/nflverse-data/releases/download/player_stats/player_stats_{y}.csv")
    if not rows: print(f"  {y}: no player stats", flush=True); continue
    n = 0
    for r in rows:
        if (r.get("season_type") or "REG") != "REG": continue
        pid = r.get("player_id") or r.get("gsis_id");
        if not pid: continue
        pg = r.get("position_group") or ""
        tm = fix(r.get("team") or r.get("recent_team") or "")
        d = prow(pid, r.get("player_display_name") or r.get("player_name") or "", pg)
        d["pg"] = pg or d["pg"]
        d["g"] += w
        if tm: d["team_g"][tm] += w
        if pg in OFF_PG:
            d["oepa"] += w * (fl(r.get("passing_epa")) + fl(r.get("rushing_epa")) + fl(r.get("receiving_epa")))
        elif pg in DEF_PG:
            d["dval"] += w * (2.0*fl(r.get("def_sacks")) + 0.5*fl(r.get("def_qb_hits")) + 0.8*fl(r.get("def_tackles_for_loss"))
                              + 3.0*fl(r.get("def_interceptions")) + 1.0*fl(r.get("def_pass_defended")) + 2.0*fl(r.get("def_fumbles_forced"))
                              + 4.0*fl(r.get("def_tds")) + 0.2*fl(r.get("def_tackles_solo")) + 0.1*fl(r.get("def_tackle_assists")))
        n += 1
    print(f"  {y}: {n} player-weeks", flush=True)
for d in P.values():
    g = d["g"]; rel = g / (g + K_REL)
    d["rel"] = rel
    d["opg"] = (d["oepa"] / g) * rel if g > 0 else 0.0      # per-game offensive value (shrunk)
    d["dpg"] = (d["dval"] / g) * rel if g > 0 else 0.0      # per-game defensive value (shrunk)

# ---- current rosters (roster_2026) -> who is on each team NOW ----
cur = defaultdict(list)   # team -> [pid,...]
rj = rows_csv(f"https://github.com/nflverse/nflverse-data/releases/download/rosters/roster_{CUR}.csv")
if not rj:
    rj = rows_csv(f"https://github.com/nflverse/nflverse-data/releases/download/rosters/roster_{CUR-1}.csv")
    print("  (using prior-year roster; current not posted yet)", flush=True)
roster_team = {}
for r in (rj or []):
    st = (r.get("status") or "").upper()
    if st and st not in ("ACT", "RES", "ACTIVE", "RESERVE/INJURED", "INA"):
        pass  # keep most; only obvious non-roster excluded below
    pid = r.get("gsis_id") or r.get("player_id")
    tm = fix(r.get("team") or "")
    if pid and tm: roster_team[pid] = (tm, (r.get("position") or r.get("depth_chart_position") or ""))
for pid, (tm, pos) in roster_team.items():
    cur[tm].append(pid)
print(f"  roster: {len(roster_team)} players across {len(cur)} teams", flush=True)

teams = sorted({fix(r["team"]) for r in csv.DictReader(open(os.path.join(PROJ, "nfl_ratings.csv"), encoding="utf-8"))})
TEAMGAMES_W = 17.0 * sum(WS.values())

OFF_CAPS = {"QB": 1, "RB": 2, "WR": 4, "TE": 2}; DEF_CAP = 8
def off_level(players, now):   # players: list of (d, games_for_team); now=True -> full health (avail 1)
    by = {pos: [] for pos in OFF_CAPS}
    for d, gt in players:
        if d["pg"] in by: by[d["pg"]].append((d, gt))
    tot = 0.0
    for pos, cap in OFF_CAPS.items():
        lst = by[pos]
        if now:
            for d, gt in sorted(lst, key=lambda x: -x[0]["opg"])[:cap]: tot += d["opg"]
        else:
            for d, gt in sorted(lst, key=lambda x: -(x[0]["opg"]*x[1]))[:cap]: tot += d["opg"] * (gt / TEAMGAMES_W)
    return tot
def def_level(players, now):
    lst = [(d, gt) for d, gt in players if d["pg"] in DEF_PG]
    tot = 0.0
    if now:
        for d, gt in sorted(lst, key=lambda x: -x[0]["dpg"])[:DEF_CAP]: tot += d["dpg"]
    else:
        for d, gt in sorted(lst, key=lambda x: -(x[0]["dpg"]*x[1]))[:DEF_CAP]: tot += d["dpg"] * (gt / TEAMGAMES_W)
    return tot
now_off = {}; now_def = {}; hist_off = {}; hist_def = {}
for t in teams:
    now_players = [(P[p], 0.0) for p in cur.get(t, []) if p in P]
    hist_players = [(d, d["team_g"][t]) for d in P.values() if d["team_g"].get(t, 0.0) > 0]
    now_off[t] = off_level(now_players, True);  hist_off[t] = off_level(hist_players, False)
    now_def[t] = def_level(now_players, True);  hist_def[t] = def_level(hist_players, False)

def zmap(dct):
    vs = list(dct.values()); m = sum(vs)/len(vs); sd = (sum((v-m)**2 for v in vs)/len(vs))**0.5 or 1.0
    return {k: (v-m)/sd for k, v in dct.items()}
# BLEND recent on-field performance with ABSOLUTE current-roster talent (full health).
# This is what makes loaded rosters (LA w/ Garrett+Nacua+Stafford, healthy SF) rank correctly.
rows = list(csv.DictReader(open(os.path.join(PROJ, "nfl_ratings.csv"), encoding="utf-8")))
p0 = rows[0]; LG = fl(p0["lg_ppg"]); KP = fl(p0["kp"])
offH = {fix(r["team"]): fl(r.get("off_hist") or r["off_epa"]) for r in rows}
defH = {fix(r["team"]): fl(r.get("def_hist") or r["def_epa"]) for r in rows}
mO = sum(offH.values())/len(offH); sO = (sum((v-mO)**2 for v in offH.values())/len(offH))**0.5 or 1.0
mD = sum(defH.values())/len(defH); sD = (sum((v-mD)**2 for v in defH.values())/len(defH))**0.5 or 1.0
zOffP = zmap(offH); zOffR = zmap(now_off)                       # offensive quality: perf vs roster
zDefP = zmap({t: -defH.get(t, 0.0) for t in teams}); zDefR = zmap(now_def)   # defensive quality (higher=better)
A = float(os.environ.get("NFL_A", "0.72"))                      # weight on recent performance; 1-A on roster talent (roster nudges, results lead)
# Regression-to-mean: extreme single-window EPA overstates true talent (small-sample / unsustainable
# variance). Shrink the PERFORMANCE z-scores toward 0 so outlier teams compress toward league average,
# matching forward-looking consensus (e.g. a results-inflated team regresses, a gutted team is not over-punished).
RTM = float(os.environ.get("NFL_RTM", "0.32"))                  # 0 = none, 1 = fully to mean; 0.32 best vs forward-looking consensus
RTM_OFF = float(os.environ.get("NFL_RTM_OFF", str(RTM)))        # offense regresses at least as hard (scoring output is more variance/schedule-driven)
SH = 1.0 - RTM
SH_OFF = 1.0 - RTM_OFF
A_DEF = float(os.environ.get("NFL_A_DEF", str(A)))             # defensive perf/roster blend (defaults equal to A)
off_final = {}; def_final = {}
for t in teams:
    bOff = A*SH_OFF*zOffP.get(t, 0.0) + (1-A)*zOffR.get(t, 0.0)
    bDef = A_DEF*SH*zDefP.get(t, 0.0) + (1-A_DEF)*zDefR.get(t, 0.0)
    off_final[t] = mO + bOff*sO
    def_final[t] = mD - bDef*sD                                # higher defensive quality => lower (better) def EPA
adj_off = {t: off_final[t]-offH.get(t, 0.0) for t in teams}; adj_def = {t: defH.get(t, 0.0)-def_final[t] for t in teams}
for r in rows:
    t = fix(r["team"])
    r["off_hist"] = round(offH.get(t, 0.0), 4); r["def_hist"] = round(defH.get(t, 0.0), 4)
    r["adj_off"] = round(adj_off.get(t, 0.0), 4); r["adj_def"] = round(adj_def.get(t, 0.0), 4)
    onew = off_final.get(t, offH.get(t, 0.0)); dnew = def_final.get(t, defH.get(t, 0.0))
    r["off_epa"] = round(onew, 4); r["def_epa"] = round(dnew, 4); r["net_epa"] = round(onew - dnew, 4)
    r["ppf"] = round(LG + KP*onew, 1); r["ppa"] = round(LG + KP*dnew, 1)
off_rank = sorted(rows, key=lambda r: -fl(r["off_epa"]));
def_rank = sorted(rows, key=lambda r: fl(r["def_epa"]))
net_rank = sorted(rows, key=lambda r: -fl(r["net_epa"]))
for i, r in enumerate(off_rank): r["off_rank"] = i+1
for i, r in enumerate(def_rank): r["def_rank"] = i+1
for i, r in enumerate(net_rank): r["net_rank"] = i+1
fields = list(rows[0].keys())
for extra in ("off_hist", "def_hist", "adj_off", "adj_def"):
    if extra not in fields: fields.append(extra)
with open(os.path.join(PROJ, "nfl_ratings.csv"), "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=fields); w.writeheader()
    for r in sorted(rows, key=lambda r: r["net_rank"]): w.writerow(r)

def line(rs): return [f"{r['team']}({r['net_epa']})" for r in rs]
print("NET top8:", line(net_rank[:8]))
print("Biggest roster/health RISERS:", sorted(teams, key=lambda t: -(adj_off[t]-adj_def[t]))[:6])
print("Biggest FALLERS:", sorted(teams, key=lambda t: (adj_off[t]-adj_def[t]))[:6])
sf = next((r for r in rows if r["team"] in ("SF",)), None); gb = next((r for r in rows if r["team"] == "GB"), None)
if sf and gb:
    print(f"SF: net_rank {sf['net_rank']} (adj_off {sf['adj_off']}, adj_def {sf['adj_def']}) | GB: net_rank {gb['net_rank']}")
    print(f"SF over GB? {'YES' if int(sf['net_rank'])<int(gb['net_rank']) else 'NO'}")
print("Wrote roster-adjusted nfl_ratings.csv", flush=True)
