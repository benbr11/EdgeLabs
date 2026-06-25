# -*- coding: utf-8 -*-
"""
NFL ratings v2 — built on nflverse PLAY-BY-PLAY (not just final scores).

Independent, opponent-adjusted OFFENSE and DEFENSE ratings (reported and modelled
separately — in football they are independent units). Core quality metric is
opponent-adjusted EPA/play (expected points added) — the public gold standard — plus a
full set of granular components, the way soccer uses shot xG:

  OFFENSE : EPA/play, success%, completion%, yards/attempt, yards/carry, 3rd-down%,
            % of drives that score, giveaways, return game, field position.
  DEFENSE : the mirror — suppressing every one of those, plus takeaways and sacks.
  SPECIAL : FG% by distance bucket (kicker), kickoff/punt return avg, drive start
            field position (pin-back), penalty yards.

Spatial: nflverse exposes yard line (yardline_100), field zone and play direction for
every play — used as field-position features. (True x,y player tracking is NGS-only /
not public; this is the best public proxy.)

EPA ratings are calibrated to points (scales KP/KT fit on real game margins/totals) so
the Gaussian win%/spread/total engine stays calibrated. Self-updating: dynamic season
window, re-fetches the current season each run, caches completed seasons on disk.
"""
import csv, io, os, math, urllib.request, datetime
from collections import defaultdict
PROJ = os.path.dirname(os.path.abspath(__file__)); csv.field_size_limit(10**7)

def get(url, t=60):
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    return urllib.request.urlopen(req, timeout=t).read().decode("utf-8", "replace")

def nfl_seasons(n=4, today=None):
    d = today or datetime.date.today(); sy = d.year if d.month >= 8 else d.year - 1
    return list(range(sy - n + 1, sy + 1))

SEASONS = nfl_seasons(4); CUR = max(SEASONS); HALFLIFE = float(os.environ.get("NFL_HALFLIFE", "230.0"))   # stronger recency: last season dominates current strength
print(f"NFL seasons (auto): {SEASONS}  (current={CUR})", flush=True)

def pbp_path(yr): return os.path.join(PROJ, f"pbp_{yr}.csv")
def ensure_pbp(yr):
    p = pbp_path(yr)
    if os.path.exists(p) and yr != CUR and os.path.getsize(p) > 1_000_000:
        return p
    url = f"https://github.com/nflverse/nflverse-data/releases/download/pbp/play_by_play_{yr}.csv"
    try:
        print(f"  downloading pbp {yr} ...", flush=True)
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=300) as r, open(p, "wb") as f:
            while True:
                c = r.read(1 << 20)
                if not c: break
                f.write(c)
        print(f"    cached {os.path.getsize(p)/1e6:.1f} MB", flush=True)
    except Exception as e:
        print(f"    pbp {yr} unavailable ({e}); using cache if present", flush=True)
    return p if os.path.exists(p) else None

RELO = {"OAK": "LV", "SD": "LAC", "STL": "LA", "LAR": "LA"}
def fix(t): return RELO.get(t, t)
def fl(x):
    try: return float(x)
    except (TypeError, ValueError): return None
def fg_bucket(d):
    if d is None: return None
    if d < 30: return "u30"
    if d < 40: return "30-39"
    if d < 50: return "40-49"
    return "50+"

ref_date = datetime.date.today()
def wt(dt): return 0.5 ** ((ref_date - dt).days / HALFLIFE)

# accumulators
Mepa = defaultdict(lambda: defaultdict(float)); Mn = defaultdict(lambda: defaultdict(float))   # off-vs-def EPA matrix
O = defaultdict(lambda: defaultdict(float)); D = defaultdict(lambda: defaultdict(float))        # offence / defence component sums
drives = {}                                                                                     # (game,drive)->[o,d,result,startyl,w]
FG = defaultdict(lambda: defaultdict(lambda: [0.0, 0.0]))                                        # team-> bucket -> [made,att]
KICKER = defaultdict(lambda: defaultdict(lambda: [0.0, 0.0]))                                    # kicker-> bucket -> [made,att]
KNAME = {}; GP = defaultdict(float); seen_gt = set(); teams = set()

def rows_season(yr):
    p = ensure_pbp(yr)
    if not p: return
    with open(p, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            yield r

total = 0
for yr in SEASONS:
    cnt = 0
    for r in rows_season(yr):
        cnt += 1
        gd = r.get("game_date") or ""
        try: dt = datetime.date.fromisoformat(gd)
        except ValueError: dt = datetime.date(yr, 11, 1)
        w = wt(dt)
        o = fix(r.get("posteam") or ""); d = fix(r.get("defteam") or ""); pt = r.get("play_type") or ""
        if not o or not d: continue
        teams.add(o); teams.add(d)
        gid = r.get("game_id")
        if gid:
            for tm in (o, d):
                if (gid, tm) not in seen_gt:
                    seen_gt.add((gid, tm)); GP[tm] += w
        epa = fl(r.get("epa")); succ = fl(r.get("success")); dr = r.get("fixed_drive")
        if gid and dr:
            key = (gid, dr)
            if key not in drives:
                res = r.get("fixed_drive_result") or ""
                syl = fl(r.get("drive_start_yard_line_100")) or fl(r.get("yardline_100"))
                drives[key] = [o, d, res, syl, w]
        if pt in ("pass", "run") and epa is not None:
            Mepa[o][d] += w * epa; Mn[o][d] += w
            O[o]["epa"] += w * epa; O[o]["epa_n"] += w
            D[d]["epa"] += w * epa; D[d]["epa_n"] += w
            if succ is not None:
                O[o]["succ"] += w * succ; O[o]["succ_n"] += w
                D[d]["succ"] += w * succ; D[d]["succ_n"] += w
        if r.get("pass_attempt") == "1":
            yg = fl(r.get("yards_gained")) or 0.0; cp = 1.0 if r.get("complete_pass") == "1" else 0.0
            O[o]["pa"] += w; O[o]["pyds"] += w * yg; O[o]["cmp"] += w * cp
            D[d]["pa"] += w; D[d]["pyds"] += w * yg; D[d]["cmp"] += w * cp
            cpoe = fl(r.get("cpoe"))
            if cpoe is not None: O[o]["cpoe"] += w * cpoe; O[o]["cpoe_n"] += w
            if r.get("interception") == "1": O[o]["gv"] += w; D[d]["tk"] += w
            if r.get("sack") == "1": D[d]["sack"] += w
        if r.get("rush_attempt") == "1":
            ry = fl(r.get("rushing_yards"))
            if ry is None: ry = fl(r.get("yards_gained")) or 0.0
            O[o]["ra"] += w; O[o]["ryds"] += w * ry
            D[d]["ra"] += w; D[d]["ryds"] += w * ry
        if r.get("fumble_lost") == "1": O[o]["gv"] += w; D[d]["tk"] += w
        if r.get("third_down_converted") == "1": O[o]["td3c"] += w; D[d]["td3c"] += w
        if r.get("third_down_failed") == "1": O[o]["td3f"] += w; D[d]["td3f"] += w
        if r.get("field_goal_attempt") == "1":
            b = fg_bucket(fl(r.get("kick_distance"))); made = 1.0 if r.get("field_goal_result") == "made" else 0.0
            if b:
                FG[o][b][1] += w; FG[o][b][0] += w * made
                k = r.get("kicker_player_name") or ""
                if k: KICKER[k][b][1] += w; KICKER[k][b][0] += w * made; KNAME[k] = o
        if r.get("kickoff_attempt") == "1":
            ry = fl(r.get("return_yards"))
            if ry is not None: O[d]["kr"] += w * ry; O[d]["kr_n"] += w
        if r.get("punt_attempt") == "1":
            ry = fl(r.get("return_yards"))
            if ry is not None: O[d]["pr"] += w * ry; O[d]["pr_n"] += w
        if r.get("penalty") == "1":
            pteam = fix(r.get("penalty_team") or ""); py = fl(r.get("penalty_yards")) or 0.0
            if pteam: O[pteam]["pen"] += w * py
    total += cnt
    print(f"  {yr}: {cnt} plays", flush=True)
teams = sorted(t for t in teams if t)
print(f"TOTAL {total} plays; {len(teams)} teams; ref {ref_date}", flush=True)

# opponent-adjusted offence / defence EPA (iterative): obs EPA/play for o vs d = off[o] + def[d]
off = {t: 0.0 for t in teams}; dfn = {t: 0.0 for t in teams}
for _ in range(60):
    no = {t: [0.0, 0.0] for t in teams}; nd = {t: [0.0, 0.0] for t in teams}
    for o in teams:
        for d, n in Mn[o].items():
            if n <= 0: continue
            avg = Mepa[o][d] / n
            no[o][0] += n * (avg - dfn[d]); no[o][1] += n
            nd[d][0] += n * (avg - off[o]); nd[d][1] += n
    for t in teams:
        if no[t][1]: off[t] = no[t][0] / no[t][1]
        if nd[t][1]: dfn[t] = nd[t][0] / nd[t][1]
    om = sum(off.values()) / len(teams); dm = sum(dfn.values()) / len(teams)
    for t in teams: off[t] -= om; dfn[t] -= dm

# drive scoring % + start field position
dscore = defaultdict(lambda: [0.0, 0.0]); dscore_all = defaultdict(lambda: [0.0, 0.0]); startfp = defaultdict(lambda: [0.0, 0.0])
for (gid, dr), (o, d, res, syl, w) in drives.items():
    scored = 1.0 if res in ("Touchdown", "Field goal") else 0.0
    dscore[o][0] += w * scored; dscore[o][1] += w
    dscore_all[d][0] += w * scored; dscore_all[d][1] += w
    if syl is not None: startfp[o][0] += w * syl; startfp[o][1] += w

def rate(num, den): return (num / den) if den else 0.0
comp = {}
for t in teams:
    o = O[t]; d = D[t]; gp = GP[t]
    comp[t] = dict(
        off_succ=rate(o["succ"], o["succ_n"]), def_succ=rate(d["succ"], d["succ_n"]),
        cmp_pct=rate(o["cmp"], o["pa"]), cmp_pct_allowed=rate(d["cmp"], d["pa"]),
        ypa=rate(o["pyds"], o["pa"]), ypa_allowed=rate(d["pyds"], d["pa"]),
        ypc=rate(o["ryds"], o["ra"]), ypc_allowed=rate(d["ryds"], d["ra"]),
        cpoe=rate(o["cpoe"], o["cpoe_n"]),
        third=rate(o["td3c"], o["td3c"] + o["td3f"]), third_allowed=rate(d["td3c"], d["td3c"] + d["td3f"]),
        drive_score=rate(dscore[t][0], dscore[t][1]), drive_score_allowed=rate(dscore_all[t][0], dscore_all[t][1]),
        giveaways=rate(o["gv"], gp), takeaways=rate(d["tk"], gp), sacks=rate(d["sack"], gp),
        kr=rate(o["kr"], o["kr_n"]), pr=rate(o["pr"], o["pr_n"]),
        start_fp=rate(startfp[t][0], startfp[t][1]), pen_pg=rate(o["pen"], gp),
        fg_pct=rate(sum(FG[t][b][0] for b in FG[t]), sum(FG[t][b][1] for b in FG[t])),
    )

# ---- calibrate EPA->points on real game margins/totals ----
grows = list(csv.DictReader(io.StringIO(get("https://raw.githubusercontent.com/nflverse/nfldata/master/data/games.csv"))))
gms = []
for r in grows:
    try: yr = int(r["season"])
    except (ValueError, KeyError): continue
    if yr not in SEASONS: continue
    try: hs, as_ = int(r["home_score"]), int(r["away_score"])
    except (ValueError, KeyError, TypeError): continue
    h, a = fix(r["home_team"]), fix(r["away_team"])
    try: dt = datetime.date.fromisoformat(r["gameday"])
    except (ValueError, KeyError, TypeError): dt = datetime.date(yr, 11, 1)
    if h in off and a in off: gms.append((dt, h, a, hs, as_))
tw = tp = hw = hm = 0.0
for dt, h, a, hs, as_ in gms:
    w = wt(dt); tp += w * (hs + as_); tw += 2 * w; hm += w * (hs - as_); hw += w
LG = tp / tw; HFA = hm / hw
sxy = sxx = sxyt = sxxt = 0.0
for dt, h, a, hs, as_ in gms:
    w = wt(dt)
    dE = (off[h] + dfn[a]) - (off[a] + dfn[h]); m = (hs - as_) - HFA
    sxy += w * dE * m; sxx += w * dE * dE
    sE = (off[h] + dfn[a]) + (off[a] + dfn[h]); tt = (hs + as_) - 2 * LG
    sxyt += w * sE * tt; sxxt += w * sE * sE
KP = sxy / sxx if sxx else 60.0
KT = sxyt / sxxt if sxxt else KP
sm = st = sw = 0.0
for dt, h, a, hs, as_ in gms:
    w = wt(dt)
    pm = KP * ((off[h] + dfn[a]) - (off[a] + dfn[h])) + HFA
    ptot = 2 * LG + KT * ((off[h] + dfn[a]) + (off[a] + dfn[h]))
    sm += w * ((hs - as_) - pm) ** 2; st += w * ((hs + as_) - ptot) ** 2; sw += w
SD_M = (sm / sw) ** 0.5; SD_T = (st / sw) ** 0.5
print(f"{len(gms)} games | LG {LG:.1f} | HFA {HFA:.1f} | KP {KP:.1f} | KT {KT:.1f} | SDm {SD_M:.1f} SDt {SD_T:.1f}", flush=True)

off_rank = sorted(teams, key=lambda t: -off[t])
def_rank = sorted(teams, key=lambda t: dfn[t])
net_rank = sorted(teams, key=lambda t: -(off[t] - dfn[t]))
orank = {t: i + 1 for i, t in enumerate(off_rank)}; drank = {t: i + 1 for i, t in enumerate(def_rank)}
nrank = {t: i + 1 for i, t in enumerate(net_rank)}

with open(os.path.join(PROJ, "nfl_ratings.csv"), "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["team", "off_epa", "def_epa", "net_epa", "off_rank", "def_rank", "net_rank",
                "ppf", "ppa", "off_succ", "def_succ", "cmp_pct", "cmp_pct_allowed", "ypa", "ypa_allowed",
                "ypc", "ypc_allowed", "cpoe", "third", "third_allowed", "drive_score", "drive_score_allowed",
                "giveaways", "takeaways", "sacks", "kr_avg", "pr_avg", "start_fp", "pen_pg", "fg_pct",
                "lg_ppg", "hfa", "kp", "kt", "sd_margin", "sd_total"])
    for t in net_rank:
        c = comp[t]
        w.writerow([t, round(off[t], 4), round(dfn[t], 4), round(off[t] - dfn[t], 4),
                    orank[t], drank[t], nrank[t],
                    round(LG + KP * off[t], 1), round(LG + KP * dfn[t], 1),
                    round(c["off_succ"], 4), round(c["def_succ"], 4),
                    round(c["cmp_pct"], 4), round(c["cmp_pct_allowed"], 4),
                    round(c["ypa"], 2), round(c["ypa_allowed"], 2), round(c["ypc"], 2), round(c["ypc_allowed"], 2),
                    round(c["cpoe"], 2), round(c["third"], 4), round(c["third_allowed"], 4),
                    round(c["drive_score"], 4), round(c["drive_score_allowed"], 4),
                    round(c["giveaways"], 2), round(c["takeaways"], 2), round(c["sacks"], 2),
                    round(c["kr"], 1), round(c["pr"], 1), round(c["start_fp"], 1), round(c["pen_pg"], 1),
                    round(c["fg_pct"], 4),
                    round(LG, 2), round(HFA, 2), round(KP, 2), round(KT, 2), round(SD_M, 2), round(SD_T, 2)])

with open(os.path.join(PROJ, "nfl_kickers.csv"), "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f); w.writerow(["kicker", "team", "u30", "30-39", "40-49", "50+", "att"])
    krows = []
    for k, bk in KICKER.items():
        att = sum(bk[b][1] for b in bk)
        if att < 10: continue
        row = [k, KNAME.get(k, "")]
        for b in ("u30", "30-39", "40-49", "50+"):
            row.append(round(rate(bk[b][0], bk[b][1]), 3) if bk[b][1] >= 2 else "")
        row.append(round(att, 0)); krows.append((att, row))
    for _, row in sorted(krows, reverse=True): w.writerow(row)

def cdf(x): return 0.5 * (1 + math.erf(x / 2 ** 0.5))
print("OFF top6:", [f"{t} ({off[t]:+.2f})" for t in off_rank[:6]])
print("DEF top6:", [f"{t} ({dfn[t]:+.2f})" for t in def_rank[:6]])
print("NET top6:", [f"{t} ({off[t]-dfn[t]:+.2f})" for t in net_rank[:6]])
print("NET bot4:", net_rank[-4:])
h, a = net_rank[0], net_rank[-1]
eH = LG + KP * (off[h] + dfn[a]) + HFA / 2; eA = LG + KP * (off[a] + dfn[h]) - HFA / 2
print(f"SAMPLE {h} vs {a}: {eH:.0f}-{eA:.0f} | {h} win {cdf((eH-eA)/SD_M)*100:.0f}% | total {eH+eA:.0f}")
print("Wrote nfl_ratings.csv + nfl_kickers.csv", flush=True)
