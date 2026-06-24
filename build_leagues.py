# -*- coding: utf-8 -*-
"""
Club-league ratings (Premier League, La Liga, Serie A, Bundesliga, Ligue 1) — the same
Poisson/Dixon-Coles engine as the World Cup model, on club results from football-data.co.uk.
Self-updating: dynamic season detection (auto-rolls each August) + re-fetches the in-progress
season every run. Outputs web/leagues_data.js (window.LEAGUES_DATA). MLS intentionally excluded.
"""
import csv, io, json, os, urllib.request, datetime, math, itertools
PROJ = os.path.dirname(os.path.abspath(__file__))
LEAGUES = [("epl","Premier League","E0"), ("laliga","La Liga","SP1"), ("seriea","Serie A","I1"),
           ("bundesliga","Bundesliga","D1"), ("ligue1","Ligue 1","F1")]
HALFLIFE = 400.0; RHO = -0.12
def get(url, t=30):
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    return urllib.request.urlopen(req, timeout=t).read().decode("utf-8", "replace")
def euro_seasons(n=3, today=None):                    # football-data season codes, newest first
    d = today or datetime.date.today(); sy = d.year if d.month >= 7 else d.year - 1
    return [f"{str(y)[2:]}{str(y+1)[2:]}" for y in range(sy, sy - n, -1)]
def pdate(s):
    for f in ("%d/%m/%Y", "%d/%m/%y"):
        try: return datetime.datetime.strptime(s, f).date()
        except (ValueError, TypeError): pass
    return None
SEASONS = euro_seasons(3); CURRENT = SEASONS[0]
print(f"League seasons (auto): {SEASONS}", flush=True)

OUT = {}
for code, name, fdc in LEAGUES:
    games = []; season_teams = {}
    for season in SEASONS:
        try:
            rows = list(csv.DictReader(io.StringIO(get(f"https://www.football-data.co.uk/mmz4281/{season}/{fdc}.csv"))))
        except Exception as e:
            print(f"  {name} {season}: skip ({e})", flush=True); continue
        st = set()
        for r in rows:
            h, a = (r.get("HomeTeam") or "").strip(), (r.get("AwayTeam") or "").strip()
            d = pdate(r.get("Date", ""))
            try: hg, ag = int(r["FTHG"]), int(r["FTAG"])
            except (ValueError, KeyError, TypeError): continue
            if h and a and d: games.append((d, h, a, hg, ag)); st |= {h, a}
        season_teams[season] = st
    if len(games) < 100:
        print(f"  {name}: only {len(games)} games — skipping", flush=True); continue
    ref = max(g[0] for g in games); wt = lambda d: 0.5 ** ((ref - d).days / HALFLIFE)
    teams = sorted({t for g in games for t in (g[1], g[2])})
    tg = tw = hgs = ags = 0.0
    for d, h, a, hg, ag in games:
        w = wt(d); tg += w*(hg+ag); tw += 2*w; hgs += w*hg; ags += w*ag
    AVG = tg/tw; HOME = min(1.45, max(1.05, hgs/ags))
    # Poisson attack/defence (venue-blind, recency-weighted)
    att = {t: 1.0 for t in teams}; dfn = {t: 1.0 for t in teams}
    for _ in range(60):
        na = {t:0. for t in teams}; da = dict(na); nd = dict(na); dd = dict(na)
        for d, h, a, hg, ag in games:
            w = wt(d)
            na[h] += w*hg; da[h] += w*AVG*dfn[a]; nd[a] += w*hg; dd[a] += w*AVG*att[h]
            na[a] += w*ag; da[a] += w*AVG*dfn[h]; nd[h] += w*ag; dd[h] += w*AVG*att[a]
        for t in teams:
            if da[t] > 0: att[t] = na[t]/da[t]
            if dd[t] > 0: dfn[t] = nd[t]/dd[t]
        for dct in (att, dfn):
            gm = math.exp(sum(math.log(max(v,1e-6)) for v in dct.values())/len(dct))
            for t in dct: dct[t] /= gm
    # Elo (MOV-weighted)
    elo = {t: 1500.0 for t in teams}
    for d, h, a, hg, ag in sorted(games, key=lambda x: x[0]):
        eh, ea = elo[h], elo[a]; exp = 1/(1+10**((ea-(eh+60))/400))
        res = 1.0 if hg > ag else 0.0 if hg < ag else 0.5
        g = 1 + 0.4*abs(hg-ag); dl = 8*g*(res-exp); elo[h] = eh+dl; elo[a] = ea-dl
    # light consensus: goals backbone + Elo stabiliser, on the goals log-scale
    def z(d): v=list(d.values()); m=sum(v)/len(v); sd=(sum((x-m)**2 for x in v)/len(v))**.5 or 1; return {t:(x-m)/sd for t,x in d.items()}
    zAg=z({t:math.log(att[t]) for t in teams}); zDg=z({t:-math.log(dfn[t]) for t in teams}); zE=z(elo)
    attZ={t:0.8*zAg[t]+0.2*zE[t] for t in teams}; defZ={t:0.8*zDg[t]+0.2*zE[t] for t in teams}
    lA=[math.log(att[t]) for t in teams]; mA=sum(lA)/len(lA); sA=(sum((x-mA)**2 for x in lA)/len(lA))**.5
    lD=[-math.log(dfn[t]) for t in teams]; mD=sum(lD)/len(lD); sD=(sum((x-mD)**2 for x in lD)/len(lD))**.5
    att={t:math.exp(mA+sA*attZ[t]) for t in teams}; dfn={t:math.exp(-(mD+sD*defZ[t])) for t in teams}
    # goal-level calibration (validated fix)
    infl=sum(att[a]*dfn[b] for a,b in itertools.permutations(teams,2))/(len(teams)*(len(teams)-1)); k=infl**.5
    for t in teams: att[t]/=k; dfn[t]/=k
    # OUTPUT only the CURRENT season's clubs (drop relegated; the fit still used all 3 seasons).
    # current = newest season with a real slate (>=12 teams); falls back gracefully in preseason.
    current=set()
    for season in SEASONS:
        if len(season_teams.get(season,set()))>=12: current=season_teams[season]; break
    if not current: current=set(teams)
    out=[t for t in teams if t in current]
    to100=lambda x: round(100/(1+math.exp(-1.05*x)),1)
    zAf=z({t:math.log(att[t]) for t in out}); zDf=z({t:-math.log(dfn[t]) for t in out})  # 0-100 over current clubs
    order=sorted(out, key=lambda t:-(zAf[t]+zDf[t]))
    OUT[code]={"name":name, "params":{"avg":round(AVG,3),"home_adv":round(HOME,3),"rho":RHO},
        "teams":{t:{"att":round(att[t],4),"dfn":round(dfn[t],4),"att100":to100(zAf[t]),
                    "def100":to100(zDf[t]),"elo":round(elo[t])} for t in out}}
    print(f"  {name}: {len(games)} games, {len(out)} current clubs (of {len(teams)} seen) | AVG {AVG:.2f} home {HOME:.2f} | top: {order[:4]}", flush=True)

with open(PROJ + r"\web\leagues_data.js", "w", encoding="utf-8") as f:
    f.write("window.LEAGUES_DATA = " + json.dumps({"leagues": OUT, "generated": datetime.date.today().isoformat()}, ensure_ascii=False) + ";\n")
print(f"Wrote web/leagues_data.js: {len(OUT)} leagues")
