# -*- coding: utf-8 -*-
"""
NHL team ratings — the hockey analogue of build_ratings.py. Same philosophy as soccer:
a MULTI-SOURCE consensus feeding attack & defense, then Poisson + goal-level calibration.

Hockey metric sources (all from the official NHL APIs, no key):
  ATTACK  = goals-model attack + shots-for (Corsi-lite possession) + power-play%
  DEFENSE = goals-model defense + shots-against + penalty-kill% + goaltending (save%)
  + Elo (MOV-weighted) as an overall-strength source.
Outputs nhl_ratings.csv. Player layer (skaters/goalies) is build_nhl_players.py.
"""
import json, csv, os, urllib.request, datetime, math, collections, itertools, urllib.parse
PROJ = os.path.dirname(os.path.abspath(__file__))
_CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_nhl_api_cache")
_USE_CACHE = os.environ.get("NHL_CACHE") == "1"
def get(url, t=30):
    if _USE_CACHE:
        import hashlib
        os.makedirs(_CACHE, exist_ok=True)
        fp = os.path.join(_CACHE, hashlib.md5(url.encode()).hexdigest() + ".json")
        if os.path.exists(fp):
            return json.load(open(fp, encoding="utf-8"))
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    data = json.loads(urllib.request.urlopen(req, timeout=t).read())
    if _USE_CACHE:
        json.dump(data, open(fp, "w", encoding="utf-8"))
    return data
def latest_seasons(n=3, today=None):                  # NHL season start years, newest first (auto-rolls)
    d = today or datetime.date.today()
    start = d.year if d.month >= 9 else d.year - 1
    return [start - i for i in range(n)]
RELOCATE = {"ARI": "UTA"}                             # franchise relocations -> unify history (Arizona->Utah 2024)
def fix(ab): return RELOCATE.get(ab, ab)
_SY = latest_seasons(5)
def _envf(name, default):
    v = os.environ.get(name)
    return float(v) if v not in (None, "") else default
def _envlist(name, default):
    v = os.environ.get(name)
    return [float(x) for x in v.split(",")] if v else default
# Recency: heavily favor the CURRENT season so the rating tracks current form / offseason roster
# changes (the forward-looking expert consensus weights this season's roster, not 3-yr-old results).
SEASON_WEIGHTS = _envlist("SW", [1.0, 0.10, 0.03, 0.01, 0.005])
SEASONS = [(f"{y}{y+1}", w) for y, w in zip(_SY, SEASON_WEIGHTS)]   # NHL API format YYYYYYYY
SEASON_INTS = [int(s) for s, _ in SEASONS]
print(f"NHL seasons (auto): {[s for s,_ in SEASONS]}", flush=True)
HALFLIFE = _envf("HL", 70.0)                          # per-date recency (independent of season weights)

# ---- team identity map (fullName/id -> abbrev) ----
stand = get("https://api-web.nhle.com/v1/standings/now")["standings"]
name2ab = {s["teamName"]["default"]: fix(s["teamAbbrev"]["default"]) for s in stand}
name2ab["Arizona Coyotes"] = "UTA"                    # fold relocated franchise's team stats
abbrevs = sorted(set(name2ab.values()))

# ---- 1. game results -> goals model + Elo ----
print("Fetching schedules ...", flush=True)
games = {}
for ab in abbrevs:
    for season, _w in SEASONS:
        try: data = get(f"https://api-web.nhle.com/v1/club-schedule-season/{ab}/{season}")
        except Exception: continue
        for g in data.get("games", []):
            if g.get("gameState") not in ("OFF", "FINAL") or g.get("gameType") not in (2, 3): continue
            gid = g["id"]
            if gid in games: continue
            h, a = fix(g["homeTeam"]["abbrev"]), fix(g["awayTeam"]["abbrev"])
            hg, ag = g["homeTeam"].get("score"), g["awayTeam"].get("score")
            if hg is None or ag is None: continue
            games[gid] = (g["gameDate"], h, a, int(hg), int(ag))
G = list(games.values())
pdate = lambda s: datetime.date.fromisoformat(s[:10])
ref = max(pdate(g[0]) for g in G); wt = lambda d: 0.5 ** ((ref - pdate(d)).days / HALFLIFE)
teams = sorted({t for g in G for t in (g[1], g[2])})
tg = tw = hg_ = ag_ = 0.0
for d, h, a, hgl, agl in G:
    w = wt(d); tg += w*(hgl+agl); tw += 2*w; hg_ += w*hgl; ag_ += w*agl
AVG = tg/tw; HOME_ADV = min(1.12, max(1.0, hg_/ag_))
# Poisson att/def (venue-blind, recency)
att = {t: 1.0 for t in teams}; dfn = {t: 1.0 for t in teams}
for _ in range(60):
    na = {t: 0. for t in teams}; da = dict(na); nd = dict(na); dd = dict(na)
    for d, h, a, hgl, agl in G:
        w = wt(d)
        na[h] += w*hgl; da[h] += w*AVG*dfn[a]; nd[a] += w*hgl; dd[a] += w*AVG*att[h]
        na[a] += w*agl; da[a] += w*AVG*dfn[h]; nd[h] += w*agl; dd[h] += w*AVG*att[a]
    for t in teams:
        if da[t] > 0: att[t] = na[t]/da[t]
        if dd[t] > 0: dfn[t] = nd[t]/dd[t]
    for dct in (att, dfn):
        gmn = math.exp(sum(math.log(max(v, 1e-6)) for v in dct.values())/len(dct))
        for t in dct: dct[t] /= gmn
# Elo (MOV-weighted)
elo = {t: 1500.0 for t in teams}
for d, h, a, hgl, agl in sorted(G, key=lambda x: x[0]):
    eh, ea = elo[h], elo[a]; exp = 1/(1+10**((ea-(eh+50))/400))
    res = 1.0 if hgl > agl else 0.0 if hgl < agl else 0.5
    g = 1 + 0.5*abs(hgl-agl); dl = 6*g*(res-exp); elo[h] = eh+dl; elo[a] = ea-dl

# ---- 2. team season stats (shots, PP, PK) recency-weighted across seasons ----
def restget(kind, season):
    exp = urllib.parse.quote(f"seasonId={season} and gameTypeId=2")
    return get(f"https://api.nhle.com/stats/rest/en/{kind}/summary?limit=-1&cayenneExp={exp}").get("data", [])
SW = {s: w for s, w in SEASONS}
acc = collections.defaultdict(lambda: collections.defaultdict(float)); accw = collections.defaultdict(float)
for season, w in SEASONS:
    for r in restget("team", int(season)):
        ab = name2ab.get(r["teamFullName"])
        if not ab: continue
        gp = r.get("gamesPlayed") or 1; accw[ab] += w*gp
        for key in ("shotsForPerGame", "shotsAgainstPerGame", "powerPlayPct", "penaltyKillPct"):
            acc[ab][key] += w*gp*(r.get(key) or 0)
tstat = {ab: {k: acc[ab][k]/accw[ab] for k in acc[ab]} for ab in acc if accw[ab] > 0}
# ---- 3. goaltending: team weighted save% (by games started) ----
gsv = collections.defaultdict(float); gsw = collections.defaultdict(float)
for season, w in SEASONS:
    for r in restget("goalie", int(season)):
        for ab in str(r.get("teamAbbrevs", "")).split(","):
            gs = r.get("gamesStarted") or 0
            if gs and r.get("savePct"): gsv[ab] += w*gs*r["savePct"]; gsw[ab] += w*gs
teamsv = {ab: gsv[ab]/gsw[ab] for ab in gsw if gsw[ab] > 0}

# ---- shot-QUALITY (real xG) + goaltending (GSAx) from build_nhl_xg.py (x/y shot coords) ----
xgf = {}; xga = {}
try:
    for r in csv.DictReader(open(PROJ + r"\nhl_team_xg.csv", encoding="utf-8")):
        xgf[r["team"]] = float(r["xgf_pg"]); xga[r["team"]] = float(r["xga_pg"])
except FileNotFoundError:
    print("  WARN nhl_team_xg.csv missing -- run build_nhl_xg.py first")
tgn = collections.defaultdict(float); tgd = collections.defaultdict(float)
try:
    for r in csv.DictReader(open(PROJ + r"\nhl_goalies.csv", encoding="utf-8")):
        if r["team"]: tgn[r["team"]] += float(r["gsax"]); tgd[r["team"]] += float(r["shots_faced"])
except FileNotFoundError:
    print("  WARN nhl_goalies.csv missing -- run build_nhl_xg.py first")
team_gsax = {t: (tgn[t]/tgd[t] if tgd[t] else 0.0) for t in teams}      # GSAx per shot faced

# ---- 4. consensus attack & defense (z-blend of hockey metrics) ----
def z(d):
    v = list(d.values()); m = sum(v)/len(v); sd = (sum((x-m)**2 for x in v)/len(v))**0.5 or 1
    return {t: (x-m)/sd for t, x in d.items()}
zA_goals = z({t: math.log(att[t]) for t in teams}); zD_goals = z({t: -math.log(dfn[t]) for t in teams})
zElo = z(elo)
_xgfm = sum(xgf.values())/len(xgf) if xgf else 3.0; _xgam = sum(xga.values())/len(xga) if xga else 3.0
zXGF = z({t: xgf.get(t, _xgfm) for t in teams})                 # REAL chance quality created
zXGA = z({t: -xga.get(t, _xgam) for t in teams})                # fewer/softer chances allowed = better D
zPP = z({t: tstat.get(t, {}).get("powerPlayPct", 0) for t in teams})
zPK = z({t: tstat.get(t, {}).get("penaltyKillPct", 0) for t in teams})
zGSAx = z({t: team_gsax.get(t, 0.0) for t in teams})            # goaltending: goals saved above expected
# Composite metric weights (per side). Tuned so the rating ORDER matches the forward-looking
# expert consensus: Elo (MOV, recency-decayed) carries most signal; the recency-weighted goals
# model adds the off/def split; PP/PK small. NO consensus list or hand-set ranks enter here.
# xG (shot QUALITY): RE-ENABLED at 0.30. The honest walk-forward backtest (backtest_nhl.py,
# n=2792 OOS) showed a WINDOWED, point-in-time xGF/xGA snapshot adds real out-of-sample
# discrimination (AUC 0.5835 -> 0.5881, log-loss 0.6798 -> 0.6790, straight-up 55.48% ->
# 55.52% before the abstain band) -- it was the *stale 5-season-flat aggregate* that lagged,
# not xG itself. Weight 0.30 sits at the OOS Brier/log-loss optimum (higher weights keep
# lifting AUC but start eroding calibration). For this to match the validated signal the xG
# inputs must emphasise CURRENT form (see build_nhl_xg.py recency weights).
# GSAx (goaltending) stays at 0.0: adding it MEASURABLY HURT OOS (AUC 0.5869 -> 0.5853 at
# weight 0.20) -- single-team goaltending is noise at the win/loss level (the same reason the
# point-in-time starting-goalie adjustment did not transfer out of sample).
_EA = _envf("EA", 0.75); _ED = _envf("ED", 0.74)          # overall strength (Elo) — dominant signal
_GA = _envf("GA", 0.20); _GD = _envf("GD", 0.18)          # recency goals model (gives the off/def split)
_XF = _envf("XF", 0.30); _XA = _envf("XA", 0.30)          # windowed xG (shot quality) -- validated OOS
_GS = _envf("GS", 0.0)                                     # GSAx aggregate HURTS OOS -> kept off
_PP = _envf("PP", 0.16); _PK = _envf("PK", 0.16)          # special teams (real forward signal experts use)
attZ = {t: _GA*zA_goals[t] + _XF*zXGF[t] + _PP*zPP[t] + _EA*zElo[t] for t in teams}
defZ = {t: _GD*zD_goals[t] + _XA*zXGA[t] + _PK*zPK[t] + _GS*zGSAx[t] + _ED*zElo[t] for t in teams}
# put back on the goals log-scale (same spread as the goals model)
lA = [math.log(att[t]) for t in teams]; mA = sum(lA)/len(lA); sA = (sum((x-mA)**2 for x in lA)/len(lA))**.5
lD = [-math.log(dfn[t]) for t in teams]; mD = sum(lD)/len(lD); sD = (sum((x-mD)**2 for x in lD)/len(lD))**.5
att = {t: math.exp(mA + sA*attZ[t]) for t in teams}
dfn = {t: math.exp(-(mD + sD*defZ[t])) for t in teams}
# ---- 5. goal-level calibration (validated fix) ----
infl = sum(att[a]*dfn[b] for a, b in itertools.permutations(teams, 2))/(len(teams)*(len(teams)-1)); k = infl**.5
for t in teams: att[t] /= k; dfn[t] /= k

# ---- output ----
zAf = z({t: math.log(att[t]) for t in teams}); zDf = z({t: -math.log(dfn[t]) for t in teams})
to100 = lambda x: round(100/(1+math.exp(-1.1*x)), 1)
order = sorted(teams, key=lambda t: -(zAf[t]+zDf[t]))
with open(PROJ + r"\nhl_ratings.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f); w.writerow(["team", "attack_100", "defense_100", "attack_mult", "defense_mult",
        "elo", "xgf_pg", "xga_pg", "gsax_per_shot", "pp_pct", "pk_pct", "avg_goals", "home_adv"])
    for t in order:
        s = tstat.get(t, {})
        w.writerow([t, to100(zAf[t]), to100(zDf[t]), round(att[t], 4), round(dfn[t], 4), round(elo[t]),
            round(xgf.get(t, 0), 2), round(xga.get(t, 0), 2), round(team_gsax.get(t, 0), 4),
            round(100*s.get("powerPlayPct", 0), 1), round(100*s.get("penaltyKillPct", 0), 1),
            round(AVG, 3), round(HOME_ADV, 3)])
print(f"{len(G)} games | AVG {AVG:.2f}/team | home {HOME_ADV:.3f} | "
      f"model avg total {2*AVG*sum(att[a]*dfn[b] for a,b in itertools.permutations(teams,2))/(len(teams)*(len(teams)-1)):.2f}")
print("TOP 8:", [f"{t}" for t in order[:8]])
print("BOT 5:", order[-5:])
for t in order[:6]:
    print(f"  {t}: ATK {to100(zAf[t])} DEF {to100(zDf[t])} | Elo {elo[t]:.0f} | "
          f"xGF {xgf.get(t,0):.2f} xGA {xga.get(t,0):.2f} GSAx/shot {team_gsax.get(t,0):+.4f} | "
          f"PP {100*tstat.get(t,{}).get('powerPlayPct',0):.1f} PK {100*tstat.get(t,{}).get('penaltyKillPct',0):.1f}")
print("Wrote nhl_ratings.csv")
