# -*- coding: utf-8 -*-
"""
MLB ratings — Poisson run model (runs are low-count like soccer/hockey) + the dominant
single-game factor, the STARTING PITCHER (modeled like the NHL goalie via run-prevention).
Team offense/defense run ratings + per-pitcher RA9. Data: statsapi.mlb.com (official).
Self-updating: dynamic recent seasons; re-fetches each run (2026 fills in as it's played).
"""
import json, csv, os, urllib.request, datetime, math, itertools, collections
PROJ = os.path.dirname(os.path.abspath(__file__))
def get(url, t=45):
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    return json.loads(urllib.request.urlopen(req, timeout=t).read())
NAMEFIX = {"Oakland Athletics": "Athletics"}           # unify the relocated/renamed franchise
fn = lambda n: NAMEFIX.get(n, n)
def seasons(n=3, today=None): d = today or datetime.date.today(); return list(range(d.year, d.year - n, -1))
SEASONS = seasons(3); SW = {SEASONS[0]:1.0, SEASONS[1]:0.6, SEASONS[2]:0.4} if len(SEASONS)>=3 else {}
HALFLIFE = 400.0
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
            games.append((d, fn(h["team"]["name"]), fn(a["team"]["name"]), hs, as_))
print(f"  {len(games)} games", flush=True)
ref = max(g[0] for g in games); wt = lambda d: 0.5 ** ((ref - d).days / HALFLIFE)
teams = sorted({t for g in games for t in (g[1], g[2])})
tw = tr = hr = ar = 0.0
for d, h, a, hs, as_ in games:
    w = wt(d); tr += w*(hs+as_); tw += 2*w; hr += w*hs; ar += w*as_
AVG = tr/tw; HOME = min(1.10, max(1.0, hr/ar))
att = {t: 1.0 for t in teams}; dfn = {t: 1.0 for t in teams}
for _ in range(60):
    na={t:0. for t in teams}; da=dict(na); nd=dict(na); dd=dict(na)
    for d, h, a, hs, as_ in games:
        w=wt(d)
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
pw_ip = collections.defaultdict(float); pw_runs = collections.defaultdict(float); p_team = {}; lg_ip=lg_runs=0.0
for y in SEASONS:
    w = SW.get(y, 0.5)
    try: j = get(f"https://statsapi.mlb.com/api/v1/stats?stats=season&group=pitching&season={y}&sportId=1&gameType=R&limit=400&sortStat=inningsPitched&order=desc")
    except Exception: continue
    for s in j["stats"][0].get("splits", []):
        nm = s["player"]["fullName"]; st = s["stat"]
        try: ip = float(st.get("inningsPitched", 0)); runs = float(st.get("runs", 0))
        except (ValueError, TypeError): continue
        if ip < 10: continue
        pw_ip[nm] += w*ip; pw_runs[nm] += w*runs; p_team[nm] = fn(s.get("team", {}).get("name", p_team.get(nm,"")))
        lg_ip += w*ip; lg_runs += w*runs
LG_RA9 = lg_runs/lg_ip*9 if lg_ip else 4.3
pitchers = {}
for nm in pw_ip:
    ip = pw_ip[nm]
    ra9 = pw_runs[nm]/ip*9
    ra9_reg = (ip*ra9 + 60*LG_RA9)/(ip+60)             # regress small samples to league
    pitchers[nm] = {"team": p_team.get(nm,""), "ra9": round(ra9_reg,2), "factor": round(ra9_reg/LG_RA9,3), "ip": round(ip)}
print(f"  league RA9 {LG_RA9:.2f} | {len(pitchers)} pitchers", flush=True)

order = sorted(teams, key=lambda t: -(att[t]/dfn[t]))
with open(PROJ+r"\mlb_ratings.csv","w",newline="",encoding="utf-8") as f:
    w=csv.writer(f); w.writerow(["team","att","dfn","rpg_for","rpg_against","avg_runs","home_adv","lg_ra9"])
    for t in order:
        w.writerow([t, round(att[t],4), round(dfn[t],4), round(AVG*att[t],2), round(AVG*dfn[t],2), round(AVG,3), round(HOME,3), round(LG_RA9,2)])
with open(PROJ+r"\mlb_pitchers.csv","w",newline="",encoding="utf-8") as f:
    w=csv.writer(f); w.writerow(["pitcher","team","ra9","factor","ip"])
    for nm in sorted(pitchers, key=lambda n: pitchers[n]["ra9"]):
        p=pitchers[nm]; w.writerow([nm, p["team"], p["ra9"], p["factor"], p["ip"]])
print("TOP 6 teams (run diff):", [f"{t}" for t in order[:6]])
print("BEST 5 SP (RA9, >=120ip):", [f"{n} {pitchers[n]['ra9']}" for n in sorted([n for n in pitchers if pitchers[n]['ip']>=120], key=lambda n:pitchers[n]['ra9'])[:5]])
print("Wrote mlb_ratings.csv, mlb_pitchers.csv")
