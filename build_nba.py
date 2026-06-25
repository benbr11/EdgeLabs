# -*- coding: utf-8 -*-
"""
NBA ratings — Gaussian point-margin model (same engine as NFL; basketball margins are
roughly normal). Opponent-adjusted offensive/defensive point ratings + home court + SDs.
Data: ESPN public API (team schedules). Self-updating: dynamic recent seasons.
"""
import json, os, urllib.request, datetime, math
PROJ = os.path.dirname(os.path.abspath(__file__))
def get(url, t=30):
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    return json.loads(urllib.request.urlopen(req, timeout=t).read())
def nba_seasons(n=3, today=None):                      # ESPN labels NBA season by END year (2024-25 -> 2025)
    d = today or datetime.date.today(); endyr = d.year if d.month >= 10 else d.year  # season ending ~Jun
    # if before October, the most recent completed/inprogress season ends this calendar year
    return list(range(endyr, endyr - n, -1))
SEASONS = nba_seasons(5); CUR = SEASONS[0]; HALFLIFE = 160.0   # tuned: lower halflife = more recent-weighted (forward-looking, matches consensus)
print(f"NBA seasons (auto, ESPN end-year): {SEASONS}", flush=True)

ESPN = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba"
tj = get(f"{ESPN}/teams")
tlist = tj["sports"][0]["leagues"][0]["teams"]
id2ab = {t["team"]["id"]: t["team"]["abbreviation"] for t in tlist}
ab2name = {t["team"]["abbreviation"]: t["team"]["displayName"] for t in tlist}

games = {}
for tid in id2ab:
    for season in SEASONS:
        try: data = get(f"{ESPN}/teams/{tid}/schedule?season={season}&seasontype=2")
        except Exception: continue
        for ev in data.get("events", []):
            comp = (ev.get("competitions") or [{}])[0]
            if not comp.get("status", {}).get("type", {}).get("completed"): continue
            cs = comp.get("competitors", [])
            hm = next((c for c in cs if c.get("homeAway") == "home"), None)
            aw = next((c for c in cs if c.get("homeAway") == "away"), None)
            if not hm or not aw: continue
            def sc(c):
                s = c.get("score"); s = s.get("value") if isinstance(s, dict) else s
                try: return int(float(s))
                except (TypeError, ValueError): return None
            hs, as_ = sc(hm), sc(aw)
            if hs is None or as_ is None: continue
            try: d = datetime.date.fromisoformat((ev.get("date") or "")[:10])
            except ValueError: continue
            gid = ev.get("id")
            if gid in games: continue
            games[gid] = (d, id2ab.get(int(hm["team"]["id"]), hm["team"].get("abbreviation")),
                          id2ab.get(int(aw["team"]["id"]), aw["team"].get("abbreviation")), hs, as_)
G = list(games.values())
print(f"  {len(G)} games, {len(set(t for g in G for t in (g[1],g[2])))} teams", flush=True)
ref = max(g[0] for g in G); wt = lambda d: 0.5 ** ((ref - d).days / HALFLIFE)
teams = sorted({t for g in G for t in (g[1], g[2])})

tw = tp = hm_ = hw = 0.0
for d, h, a, hs, a_s in G:
    w = wt(d); tp += w*(hs+a_s); tw += 2*w; hm_ += w*(hs-a_s); hw += w
LG = tp/tw; HFA = hm_/hw
off = {t: 0.0 for t in teams}; dfn = {t: 0.0 for t in teams}
for _ in range(50):
    no = {t:[0.,0.] for t in teams}; nd = {t:[0.,0.] for t in teams}
    for d, h, a, hs, a_s in G:
        w = wt(d)
        no[h][0]+=w*((hs-HFA/2)-LG-dfn[a]); no[h][1]+=w
        no[a][0]+=w*((a_s+HFA/2)-LG-dfn[h]); no[a][1]+=w
        nd[h][0]+=w*((a_s+HFA/2)-LG-off[a]); nd[h][1]+=w
        nd[a][0]+=w*((hs-HFA/2)-LG-off[h]); nd[a][1]+=w
    for t in teams:
        if no[t][1]: off[t]=no[t][0]/no[t][1]
        if nd[t][1]: dfn[t]=nd[t][0]/nd[t][1]
    om=sum(off.values())/len(teams); dm=sum(dfn.values())/len(teams)
    for t in teams: off[t]-=om; dfn[t]-=dm
sm=st=sw=0.0
for d,h,a,hs,a_s in G:
    w=wt(d); pm=(off[h]-off[a])+(dfn[a]-dfn[h])+HFA; pt=2*LG+off[h]+off[a]+dfn[h]+dfn[a]
    sm+=w*((hs-a_s)-pm)**2; st+=w*((hs+a_s)-pt)**2; sw+=w
SD_M=(sm/sw)**0.5; SD_T=(st/sw)**0.5
print(f"LG {LG:.1f} pts/team | HFA {HFA:.1f} | SD margin {SD_M:.1f} total {SD_T:.1f}", flush=True)

# current teams only (in case of abbrev drift): all in id2ab are current 30
cur=set(id2ab.values())

# ---- player values (current-season per-game stats from ESPN byathlete) ----
# Computed BEFORE writing ratings so a roster/talent prior can be blended into the
# published "net" (overall) rating. Predictions read off/def (unchanged) — only the
# net ORDER is recomposed to be forward-looking, matching expert consensus.
players = []
try:
    base = "https://site.web.api.espn.com/apis/common/v3/sports/basketball/nba/statistics/byathlete"
    pj = get(f"{base}?region=us&lang=en&contentorigin=espn&isqualified=true&limit=320&sort=offensive.avgPoints%3Adesc&season={CUR}&seasontype=2")
    names = {c["name"]: c.get("names", []) for c in pj.get("categories", [])}
    for a in pj.get("athletes", []):
        info = a.get("athlete", {}); m = {}
        for c in a.get("categories", []):
            ns = names.get(c["name"], []); vals = c.get("values", [])
            for i, key in enumerate(ns):
                if i < len(vals): m[key] = vals[i]
        gp = m.get("gamesPlayed", 0) or 0
        if gp < 25: continue
        pts = m.get("avgPoints", 0) or 0; reb = m.get("avgRebounds", 0) or 0; ast = m.get("avgAssists", 0) or 0
        stl = m.get("avgSteals", 0) or 0; blk = m.get("avgBlocks", 0) or 0; to = m.get("avgTurnovers", 0) or 0
        pos = (info.get("position") or {}); pos = pos.get("abbreviation", "") if isinstance(pos, dict) else ""
        try: tab = id2ab.get(int(info.get("teamId", 0)), info.get("teamShortName", ""))
        except (ValueError, TypeError): tab = info.get("teamShortName", "")
        val = pts + 0.7 * reb + 0.7 * ast + stl + blk - 0.7 * to     # simple all-around impact value
        players.append((round(val, 1), info.get("displayName", ""), tab, pos, round(pts, 1), round(reb, 1), round(ast, 1), round(stl, 1), round(blk, 1)))
    players.sort(reverse=True)
    import csv as _csv
    with open(PROJ + r"\nba_players.csv", "w", newline="", encoding="utf-8") as f:
        w = _csv.writer(f); w.writerow(["player", "team", "pos", "value", "ppg", "rpg", "apg", "spg", "bpg"])
        for val, nm, tab, pos, pts, reb, ast, stl, blk in players:
            w.writerow([nm, tab, pos, val, pts, reb, ast, stl, blk])
    print(f"  players: {len(players)} | TOP 6:", [f"{p[1]} ({p[0]})" for p in players[:6]])
except Exception as e:
    print(f"  players: skipped ({e})", flush=True)

# ---- forward-looking NET = blend of recent on-court net rating with a current-roster
#      talent prior. Consensus is forward-looking (reflects offseason roster moves), so the
#      results-only net under-rates teams whose CURRENT roster is stronger than last season's
#      play (and vice-versa). We z-score each signal, blend, then map BACK onto the net
#      points scale (its own mean/SD) so the rating stays in points and off/def/SDs/HFA —
#      which drive every spread, win% and total — are completely untouched.
cur_teams = [t for t in teams if t in cur]
net_raw = {t: off[t] - dfn[t] for t in cur_teams}
# star-weighted roster talent: sum top-7 current player values per team, geometric decay
# (0.65) so true top-end stars dominate (consensus prizes star power), summed from the
# per-game player values already mapped to current teams in nba_players.csv.
TALENT_N = 7; TALENT_DECAY = 0.65; BLEND_W = 0.72   # BLEND_W on recent net, 1-BLEND_W on roster talent
tal_by_team = {t: [] for t in cur_teams}
for val, nm, tab, pos, *_ in players:
    if tab in tal_by_team: tal_by_team[tab].append(val)
talent = {t: sum(v * (TALENT_DECAY ** i) for i, v in enumerate(sorted(vs, reverse=True)[:TALENT_N]))
          for t, vs in tal_by_team.items()}
def _z(d):
    vs = list(d.values()); mu = sum(vs) / len(vs)
    sd = (sum((v - mu) ** 2 for v in vs) / len(vs)) ** 0.5 or 1.0
    return {k: (v - mu) / sd for k, v in d.items()}, mu, sd
zN, muN, sdN = _z(net_raw); zT, _, _ = _z(talent) if any(talent.values()) else ({t: 0.0 for t in cur_teams}, 0, 1)
net_blended = {t: muN + (BLEND_W * zN[t] + (1 - BLEND_W) * zT.get(t, 0.0)) * sdN for t in cur_teams}
order = sorted(cur_teams, key=lambda t: -net_blended[t])
with open(PROJ+r"\nba_ratings.csv","w",newline="",encoding="utf-8") as f:
    w=__import__("csv").writer(f); w.writerow(["team","name","off","def","net","ppg_for","ppg_against","lg_ppg","hfa","sd_margin","sd_total"])
    for t in order:
        w.writerow([t, ab2name.get(t,t), round(off[t],2), round(dfn[t],2), round(net_blended[t],2),
                    round(LG+off[t],1), round(LG+dfn[t],1), round(LG,2), round(HFA,2), round(SD_M,2), round(SD_T,2)])
print("TOP 6:", [f"{t} ({net_blended[t]:+.1f})" for t in order[:6]])
print("BOT 4:", order[-4:])
print("Wrote nba_ratings.csv" + (", nba_players.csv" if players else ""))
