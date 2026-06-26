# -*- coding: utf-8 -*-
"""
NHL shot-level expected goals (the hockey StatsBomb) from MoneyPuck shot data — every
shot's x/y-coordinate-based xG. Builds the three pillars the model needs:
  TEAM OFFENSE  -> xGF/game (chance quality created)
  TEAM DEFENSE  -> xGA/game (dangerous chances suppressed)
  GOALTENDING   -> GSAx = Sum(xGoal faced) - goals allowed (saving high-danger shots)
Also per-skater individual xG (ixG) for player props. Recency-weighted across seasons.
Outputs nhl_team_xg.csv, nhl_goalies.csv, nhl_skaters.csv.
"""
import urllib.request, zipfile, io, csv, os, collections, datetime
PROJ = os.path.dirname(os.path.abspath(__file__))
def latest_seasons(n=3, today=None):
    """NHL season START years, newest first. Auto-rolls: in-season starts in Oct, so
    months Sep-Dec belong to year Y's season, Jan-Aug to (Y-1)'s. Self-updating forever."""
    d = today or datetime.date.today()
    start = d.year if d.month >= 9 else d.year - 1
    return [start - i for i in range(n)]
# Recency weights per season (newest first). STEEPENED so the CURRENT season dominates the
# team xG aggregate: the walk-forward backtest validated that it is *current-form*,
# point-in-time xG (a ~70-day half-life) that adds out-of-sample signal -- the old near-flat
# 5-season blend (1.0/0.75/0.55/...) re-imported stale form and was the reason xG had been
# zeroed in build_nhl.py. These weights make the production xGF/xGA track the same current
# form the backtest measured. (Goalie/skater rows use the same weights; current-season
# emphasis is appropriate there too.)
_SY = latest_seasons(5); _WT = [1.0, 0.40, 0.18, 0.08, 0.04]
SEASONS = list(zip(_SY, _WT))                         # [(currentStartYear,1.0),(prev,.40),(prev2,.18),...]
CURRENT = _SY[0]                                      # in-progress season -> never cache (daily refresh)
print(f"NHL xG seasons (auto): {[f'{y}-{str(y+1)[2:]}' for y in _SY]}  current={CURRENT}", flush=True)
UNBLOCKED = {"SHOT", "GOAL", "MISS"}; ONGOAL = {"SHOT", "GOAL"}
RELOCATE = {"ARI": "UTA"}                             # unify relocated franchise (Arizona -> Utah 2024)
def fix(ab): return RELOCATE.get(ab, ab)

txgf = collections.defaultdict(float); txga = collections.defaultdict(float)
tgf = collections.defaultdict(float); tga = collections.defaultdict(float)
tgw = collections.defaultdict(float); seen_tg = set()                 # weighted games per team
g_xg = collections.defaultdict(float); g_ga = collections.defaultdict(float); g_sh = collections.defaultdict(float)
g_team = {}; g_gms = collections.defaultdict(set)
s_xg = collections.defaultdict(float); s_g = collections.defaultdict(float); s_sh = collections.defaultdict(float)
s_team = {}; s_pos = {}; s_gms = collections.defaultdict(set)

for season, w in SEASONS:
    url = f"https://moneypuck.com/moneypuck/playerData/shots/shots_{season}.zip"
    cache = PROJ + f"\\mp_shots_{season}.zip"
    try:
        # past seasons are static -> cache them; the in-progress season is re-pulled every run
        # so new games flow in daily.
        if season != CURRENT and os.path.exists(cache):
            data = open(cache, "rb").read(); print(f"cached {season}", flush=True)
        else:
            print(f"downloading {season} ...", flush=True)
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            data = urllib.request.urlopen(req, timeout=240).read()
            if season != CURRENT: open(cache, "wb").write(data)   # only cache completed seasons
    except Exception as e:
        print(f"  skip {season} (likely not posted yet): {e}", flush=True); continue
    z = zipfile.ZipFile(io.BytesIO(data))
    with z.open(z.namelist()[0]) as f:
        rd = csv.DictReader(io.TextIOWrapper(f, encoding="utf-8", errors="replace"))
        n = 0
        for r in rd:
            ev = r.get("event")
            if ev not in UNBLOCKED: continue
            try: xg = float(r.get("xGoal") or 0)
            except ValueError: xg = 0.0
            goal = 1.0 if ev == "GOAL" else 0.0
            shoot = fix(r.get("teamCode")); home = fix(r.get("homeTeamCode")); away = fix(r.get("awayTeamCode"))
            isHome = (r.get("isHomeTeam") in ("1", "1.0"))
            defend = away if isHome else home
            gid = r.get("game_id") or r.get("id")
            if shoot:
                txgf[shoot] += w*xg; tgf[shoot] += w*goal
                if (shoot, gid) not in seen_tg: seen_tg.add((shoot, gid)); tgw[shoot] += w
            if defend:
                txga[defend] += w*xg; tga[defend] += w*goal
            # goalie: xG faced summed over ALL UNBLOCKED shots (keeps GSAx calibrated to 0);
            # save% uses shots on goal only.
            gname = r.get("goalieNameForShot")
            if gname:
                g_xg[gname] += w*xg; g_ga[gname] += w*goal
                g_team[gname] = defend; g_gms[gname].add(gid)
                if ev in ONGOAL: g_sh[gname] += w
            # skater individual xG
            sname = r.get("shooterName")
            if sname:
                s_xg[sname] += w*xg; s_g[sname] += w*goal; s_sh[sname] += w
                s_team[sname] = shoot; s_pos[sname] = r.get("playerPositionThatDidEvent", ""); s_gms[sname].add(gid)
            n += 1
    print(f"  {season}: {n} unblocked shots", flush=True)

# ---- team xG ----
teams = sorted(tgw)
with open(PROJ + r"\nhl_team_xg.csv", "w", newline="", encoding="utf-8") as f:
    wr = csv.writer(f); wr.writerow(["team", "xgf_pg", "xga_pg", "gf_pg", "ga_pg", "xg_diff_pg"])
    for t in sorted(teams, key=lambda t: -((txgf[t]-txga[t])/tgw[t] if tgw[t] else 0)):
        gms = tgw[t] or 1
        wr.writerow([t, round(txgf[t]/gms, 2), round(txga[t]/gms, 2), round(tgf[t]/gms, 2),
                     round(tga[t]/gms, 2), round((txgf[t]-txga[t])/gms, 2)])
print(f"\nTeam xG diff/game leaders:")
for t in sorted(teams, key=lambda t: -((txgf[t]-txga[t])/tgw[t]))[:6]:
    print(f"  {t}: xGF {txgf[t]/tgw[t]:.2f} xGA {txga[t]/tgw[t]:.2f} diff {(txgf[t]-txga[t])/tgw[t]:+.2f}")

# ---- goalies: GSAx ----
with open(PROJ + r"\nhl_goalies.csv", "w", newline="", encoding="utf-8") as f:
    wr = csv.writer(f); wr.writerow(["goalie", "team", "shots_faced", "xg_faced", "goals_allowed",
                                     "gsax", "gsax_per_shot", "sv_pct", "games"])
    gl = [g for g in g_sh if g_sh[g] >= 8]               # min sample
    for g in sorted(gl, key=lambda g: -(g_xg[g]-g_ga[g])):
        gsax = g_xg[g]-g_ga[g]; sv = 1-g_ga[g]/g_sh[g] if g_sh[g] else 0
        wr.writerow([g, g_team.get(g, ""), round(g_sh[g], 1), round(g_xg[g], 1), round(g_ga[g], 1),
                     round(gsax, 1), round(gsax/g_sh[g], 4) if g_sh[g] else 0, round(sv, 4), len(g_gms[g])])
print("\nGoaltending (GSAx = goals saved above expected) leaders:")
for g in sorted([g for g in g_sh if g_sh[g] >= 200], key=lambda g: -(g_xg[g]-g_ga[g]))[:6]:
    print(f"  {g} ({g_team.get(g,'')}): GSAx {g_xg[g]-g_ga[g]:+.1f}  sv% {1-g_ga[g]/g_sh[g]:.3f}  ({len(g_gms[g])} gm)")

# ---- skaters: individual xG (props value) ----
with open(PROJ + r"\nhl_skaters.csv", "w", newline="", encoding="utf-8") as f:
    wr = csv.writer(f); wr.writerow(["skater", "team", "pos", "games", "shots", "ixg", "goals",
                                     "ixg_pg", "g_pg", "fin_vs_xg"])
    sk = [s for s in s_sh if len(s_gms[s]) >= 5]
    for s in sorted(sk, key=lambda s: -s_xg[s]):
        gp = len(s_gms[s]) or 1
        wr.writerow([s, s_team.get(s, ""), s_pos.get(s, ""), gp, round(s_sh[s], 1), round(s_xg[s], 2),
                     round(s_g[s], 1), round(s_xg[s]/gp, 3), round(s_g[s]/gp, 3), round(s_g[s]-s_xg[s], 1)])
print("\nTop skaters by individual xG (recency-weighted):")
for s in sorted([s for s in s_sh if len(s_gms[s]) >= 20], key=lambda s: -s_xg[s])[:6]:
    gp = len(s_gms[s])
    print(f"  {s} ({s_team.get(s,'')}): ixG {s_xg[s]:.1f} goals {s_g[s]:.0f} ixG/gm {s_xg[s]/gp:.2f}")
print("\nWrote nhl_team_xg.csv, nhl_goalies.csv, nhl_skaters.csv")
