# -*- coding: utf-8 -*-
"""
MLB ratings — Poisson run model (runs are low-count like soccer/hockey) + the dominant
single-game factor, the STARTING PITCHER (modeled like the NHL goalie via run-prevention).
Independent team OFFENSE (run scoring) and PITCHING/DEFENSE (run prevention) ratings,
per-pitcher RA9, and per-hitter value (OPS + line). Data: statsapi.mlb.com (official).
Self-updating: dynamic recent seasons (5, smoothly recency-weighted); re-fetches each run.
"""
import json, csv, os, urllib.request, datetime, math, itertools, collections
PROJ = os.path.dirname(os.path.abspath(__file__))
def get(url, t=45):
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    return json.loads(urllib.request.urlopen(req, timeout=t).read())
NAMEFIX = {"Oakland Athletics": "Athletics"}
fn = lambda n: NAMEFIX.get(n, n)
def seasons(n=5, today=None): d = today or datetime.date.today(); return list(range(d.year, d.year - n, -1))
SEASONS = seasons(5); CUR = SEASONS[0]
_E = lambda k, d: float(os.environ.get(k, d))
SW_HL = _E("MLB_SW_HL", 0.70)                           # season-recency halflife (smaller -> 2026 dominates harder)
SW = {y: 0.5 ** ((CUR - y) / SW_HL) for y in SEASONS}   # sharp season recency: 2026 (in progress) dominates
HALFLIFE = _E("MLB_HALFLIFE", 230.0)                    # game-level decay (shorter -> recent games / current form matter more)
# --- rating-ORDER composition (does NOT touch AVG/HOME/SD/points scale; only how att/dfn are composed) ---
# The consensus is current-form driven; a team's recency-weighted WINNING PCT tracks it far better
# than run differential alone (teams over/under-perform their run diff). We blend a win%-implied
# run-equivalent into the run-based att/dfn, keeping geo-mean 1.0 so the rating stays in run units.
WPCT_W = _E("MLB_WPCT_W", 0.80)    # weight on recency-weighted win%-implied rating vs raw run-diff
WPCT_SCALE = _E("MLB_WPCT_SCALE", 8.0)  # run-diff-per-game equivalent of a full win% swing (0->1)
REG = _E("MLB_REG", 0.40)          # regression of the composed multiplier toward league mean 1.0 (MLB high-variance)
PRIOR_W = _E("MLB_PRIOR_W", 0.0)   # optional roster-talent nudge (OPS/RA9); off by default (added noise vs win%)
N_HIT = 9          # top hitters used for a team's offense prior
N_SP = 5           # top starters used for a team's run-prevention prior
PRIOR_SPREAD = _E("MLB_PRIOR_SPREAD", 0.7) # how strongly prior z-scores map into att/dfn multiplier space
print(f"MLB seasons (auto): {SEASONS}", flush=True)

# ---- 1. game results -> team run ratings ----
games = []
for y in SEASONS:
    try: j = get(f"https://statsapi.mlb.com/api/v1/schedule?sportId=1&startDate={y}-03-15&endDate={y}-11-15&gameType=R")
    except Exception as e: print(f"  schedule {y}: skip {e}", flush=True); continue
    for dd in j.get("dates", []):
        for g in dd.get("games", []):
            if g.get("status", {}).get("detailedState") != "Final": continue
            t = g.get("teams", {}); h, a = t.get("home", {}), t.get("away", {})
            try: hs, as_ = int(h["score"]), int(a["score"])
            except (KeyError, ValueError, TypeError): continue
            try: d = datetime.date.fromisoformat(g.get("officialDate") or g["gameDate"][:10])
            except (ValueError, KeyError): continue
            games.append((d, fn(h["team"]["name"]), fn(a["team"]["name"]), hs, as_, y))
print(f"  {len(games)} games", flush=True)
ref = max(g[0] for g in games)
# Game weight = game-level recency * SEASON recency. Consensus is current-form driven, and the
# in-progress 2026 standings track it ~0.84-0.96; folding SW[season] in lets 2026 dominate the
# att/dfn solve while older seasons only stabilize small early-season samples.
wt = lambda d, y: 0.5 ** ((ref - d).days / HALFLIFE) * SW.get(y, 0.05)
teams = sorted({t for g in games for t in (g[1], g[2])})
tw = tr = hr = ar = 0.0
twsum = collections.defaultdict(float); twin = collections.defaultdict(float)  # recency-weighted games / wins per team
for d, h, a, hs, as_, y in games:
    w = wt(d, y); tr += w*(hs+as_); tw += 2*w; hr += w*hs; ar += w*as_
    twsum[h] += w; twsum[a] += w
    twin[h] += w*(1 if hs > as_ else 0); twin[a] += w*(1 if as_ > hs else 0)
AVG = tr/tw; HOME = min(1.10, max(1.0, hr/ar))
att = {t: 1.0 for t in teams}; dfn = {t: 1.0 for t in teams}
for _ in range(60):
    na={t:0. for t in teams}; da=dict(na); nd=dict(na); dd=dict(na)
    for d, h, a, hs, as_, y in games:
        w=wt(d, y)
        na[h]+=w*hs; da[h]+=w*AVG*dfn[a]; nd[a]+=w*hs; dd[a]+=w*AVG*att[h]
        na[a]+=w*as_; da[a]+=w*AVG*dfn[h]; nd[h]+=w*as_; dd[h]+=w*AVG*att[a]
    for t in teams:
        if da[t]>0: att[t]=na[t]/da[t]
        if dd[t]>0: dfn[t]=nd[t]/dd[t]
    for dct in (att,dfn):
        gm=math.exp(sum(math.log(max(v,1e-6)) for v in dct.values())/len(dct))
        for t in dct: dct[t]/=gm
infl=sum(att[a]*dfn[b] for a,b in itertools.permutations(teams,2))/(len(teams)*(len(teams)-1)); k=infl**.5
for t in teams: att[t]/=k; dfn[t]/=k
print(f"  AVG {AVG:.2f} runs/team | home {HOME:.3f} | model avg total {2*AVG*sum(att[a]*dfn[b] for a,b in itertools.permutations(teams,2))/(len(teams)*(len(teams)-1)):.2f}", flush=True)

# ---- 2. starting-pitcher run prevention (RA9), recency-weighted, regressed ----
pw_ip = collections.defaultdict(float); pw_runs = collections.defaultdict(float); pw_sw = collections.defaultdict(float); p_team = {}; lg_ip=lg_runs=0.0
for y in SEASONS:
    w = SW.get(y, 0.1)
    try: j = get(f"https://statsapi.mlb.com/api/v1/stats?stats=season&group=pitching&season={y}&sportId=1&gameType=R&limit=400&sortStat=inningsPitched&order=desc")
    except Exception: continue
    for s in j["stats"][0].get("splits", []):
        nm = s["player"]["fullName"]; st = s["stat"]
        try: ip = float(st.get("inningsPitched", 0)); runs = float(st.get("runs", 0))
        except (ValueError, TypeError): continue
        if ip < 10: continue
        pw_ip[nm] += w*ip; pw_runs[nm] += w*runs; pw_sw[nm] += w; p_team[nm] = fn(s.get("team", {}).get("name", p_team.get(nm,"")))
        lg_ip += w*ip; lg_runs += w*runs
LG_RA9 = lg_runs/lg_ip*9 if lg_ip else 4.3
pitchers = {}
for nm in pw_ip:
    ip = pw_ip[nm]                                       # total weighted IP (sample size -> regression)
    ra9 = pw_runs[nm]/ip*9
    ra9_reg = (ip*ra9 + 60*LG_RA9)/(ip+60)
    ip_pg = ip/pw_sw[nm] if pw_sw[nm] else ip            # per-season IP (display)
    pitchers[nm] = {"team": p_team.get(nm,""), "ra9": round(ra9_reg,2), "factor": round(ra9_reg/LG_RA9,3), "ip": round(ip_pg)}
print(f"  league RA9 {LG_RA9:.2f} | {len(pitchers)} pitchers", flush=True)

# ---- 2b. hitters (player value = OPS + line), recency-weighted, per-season counting stats ----
HC = collections.defaultdict(lambda: collections.defaultdict(float)); h_team = {}
for y in SEASONS:
    w = SW.get(y, 0.1)
    try: j = get(f"https://statsapi.mlb.com/api/v1/stats?stats=season&group=hitting&season={y}&sportId=1&gameType=R&limit=350&sortStat=plateAppearances&order=desc")
    except Exception: continue
    for s in j["stats"][0].get("splits", []):
        nm = s["player"]["fullName"]; st = s["stat"]
        def gi(key):
            try: return float(st.get(key, 0) or 0)
            except (ValueError, TypeError): return 0.0
        if gi("plateAppearances") < 20: continue
        d = HC[nm]
        for key in ("atBats","hits","doubles","triples","homeRuns","baseOnBalls","hitByPitch","sacFlies","rbi","runs","plateAppearances","totalBases","stolenBases"):
            d[key] += w*gi(key)
        d["sw"] += w; h_team[nm] = fn(s.get("team", {}).get("name", h_team.get(nm,"")))
hitters = {}
for nm, d in HC.items():
    ab=d["atBats"]; pa=d["plateAppearances"]
    if pa < 300: continue
    tb = d["totalBases"] or (d["hits"]+d["doubles"]+2*d["triples"]+3*d["homeRuns"])
    obp_den = ab+d["baseOnBalls"]+d["hitByPitch"]+d["sacFlies"]
    obp = (d["hits"]+d["baseOnBalls"]+d["hitByPitch"])/obp_den if obp_den else 0
    slg = tb/ab if ab else 0; avg = d["hits"]/ab if ab else 0
    sw = d["sw"] or 1.0
    hitters[nm] = {"team": h_team.get(nm,""), "ops": round(obp+slg,3), "avg": round(avg,3),
                   "hr": round(d["homeRuns"]/sw), "rbi": round(d["rbi"]/sw), "r": round(d["runs"]/sw),
                   "sb": round(d["stolenBases"]/sw), "pa": round(pa/sw)}
print(f"  {len(hitters)} qualified hitters", flush=True)

# ---- 3. compose rating ORDER: blend run-diff att/dfn with a recency-weighted WIN%-implied rating ----
# Why: consensus tracks W-L far more than pure run differential (teams over/under-perform their run
# diff via close-game record / blowout distribution). Both terms come from the SAME recency-weighted
# games already used above (no extra data, no consensus leakage). We keep everything in run-diff units
# and geo-mean-normalize att/dfn to 1.0 so AVG*att stays in runs -> the rating->points scale is UNCHANGED.
import statistics as _st
# (a) raw run-diff rating per team (in runs/game): AVG*(att - dfn)
rd_rating = {t: AVG*(att[t] - dfn[t]) for t in teams}
# (b) win%-implied rating: center win% at .500, scale to a runs/game equivalent
wp = {t: (twin[t]/twsum[t] if twsum[t] else 0.5) for t in teams}
wp_rating = {t: (wp[t]-0.5)*WPCT_SCALE for t in teams}
# (c) blended target rating (still runs/game), then regress toward 0
tgt = {t: (1-REG)*((1-WPCT_W)*rd_rating[t] + WPCT_W*wp_rating[t]) for t in teams}
# (optional) tiny roster-talent nudge from current OPS / rotation RA9 (off by default)
if PRIOR_W > 0:
    lg_ops = _st.mean([h["ops"] for h in hitters.values()]) if hitters else 0.72
    team_hit = collections.defaultdict(list); team_sp = collections.defaultdict(list)
    for h in hitters.values():
        if h["team"] in teams: team_hit[h["team"]].append(h["ops"])
    for nm, p in pitchers.items():
        if p["team"] in teams and pw_ip[nm] >= 60: team_sp[p["team"]].append(p["ra9"])
    def _z(d):
        vals=list(d.values()); m=_st.mean(vals); sd=_st.pstdev(vals) or 1e-9
        return {t:(d[t]-m)/sd for t in d}
    zoff=_z({t:_st.mean(sorted(team_hit.get(t,[]),reverse=True)[:N_HIT]) if team_hit.get(t) else lg_ops for t in teams})
    zpit=_z({t:_st.mean(sorted(team_sp.get(t,[]))[:N_SP]) if team_sp.get(t) else LG_RA9 for t in teams})
    for t in teams: tgt[t] += PRIOR_W * PRIOR_SPREAD * (zoff[t] - zpit[t])
# (d) map the target runs/game rating back into att/dfn multipliers, split symmetrically around AVG,
#     then geo-mean-normalize to 1.0 (preserves the runs scale exactly).
for t in teams:
    half = tgt[t]/2.0
    att[t] = max(1e-6, (AVG + half)/AVG)
    dfn[t] = max(1e-6, (AVG - half)/AVG)
for dct in (att, dfn):
    gm = math.exp(sum(math.log(v) for v in dct.values())/len(dct))
    for t in dct: dct[t] /= gm
print(f"  composed rating: WPCT_W={WPCT_W} REG={REG} PRIOR_W={PRIOR_W}", flush=True)

order = sorted(teams, key=lambda t: -(att[t]/dfn[t]))
with open(PROJ+r"\mlb_ratings.csv","w",newline="",encoding="utf-8") as f:
    w=csv.writer(f); w.writerow(["team","att","dfn","rpg_for","rpg_against","avg_runs","home_adv","lg_ra9"])
    for t in order:
        w.writerow([t, round(att[t],4), round(dfn[t],4), round(AVG*att[t],2), round(AVG*dfn[t],2), round(AVG,3), round(HOME,3), round(LG_RA9,2)])
with open(PROJ+r"\mlb_pitchers.csv","w",newline="",encoding="utf-8") as f:
    w=csv.writer(f); w.writerow(["pitcher","team","ra9","factor","ip"])
    for nm in sorted(pitchers, key=lambda n: pitchers[n]["ra9"]):
        p=pitchers[nm]; w.writerow([nm, p["team"], p["ra9"], p["factor"], p["ip"]])
with open(PROJ+r"\mlb_hitters.csv","w",newline="",encoding="utf-8") as f:
    w=csv.writer(f); w.writerow(["hitter","team","ops","avg","hr","rbi","r","sb","pa"])
    for nm in sorted(hitters, key=lambda n: -hitters[n]["ops"]):
        h=hitters[nm]; w.writerow([nm, h["team"], h["ops"], h["avg"], h["hr"], h["rbi"], h["r"], h["sb"], h["pa"]])
print("TOP 6 teams (run diff):", [f"{t}" for t in order[:6]])
print("BEST 5 SP (RA9, >=120ip-total):", [f"{n} {pitchers[n]['ra9']}" for n in sorted([n for n in pitchers if pw_ip[n]>=120], key=lambda n:pitchers[n]['ra9'])[:5]])
print("TOP 6 hitters (OPS):", [f"{n} {hitters[n]['ops']}" for n in sorted(hitters, key=lambda n:-hitters[n]['ops'])[:6]])
print("Wrote mlb_ratings.csv, mlb_pitchers.csv, mlb_hitters.csv")
