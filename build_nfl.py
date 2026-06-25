# -*- coding: utf-8 -*-
"""
NFL ratings — Gaussian point-margin model (the right engine for football: scores are
points with a roughly-normal margin, not low-count Poisson goals). Opponent-adjusted
offensive/defensive point ratings + home-field + margin/total SDs -> win %, spread, total.
Self-updating: dynamic recent-seasons window, re-fetches nflverse games.csv each run.
"""
import csv, io, json, os, urllib.request, datetime, math
PROJ = os.path.dirname(os.path.abspath(__file__))
def get(url, t=40):
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    return urllib.request.urlopen(req, timeout=t).read().decode("utf-8", "replace")
def nfl_seasons(n=4, today=None):                      # NFL season year (starts Sep; runs into Feb)
    d = today or datetime.date.today(); sy = d.year if d.month >= 8 else d.year - 1
    return set(range(sy - n + 1, sy + 1))
SEASONS = nfl_seasons(4); HALFLIFE = 330.0
print(f"NFL seasons (auto): {sorted(SEASONS)}", flush=True)

rows = list(csv.DictReader(io.StringIO(get("https://raw.githubusercontent.com/nflverse/nfldata/master/data/games.csv"))))
games = []
for r in rows:
    try: yr = int(r["season"])
    except (ValueError, KeyError): continue
    if yr not in SEASONS: continue
    try: hs, a_s = int(r["home_score"]), int(r["away_score"])
    except (ValueError, KeyError, TypeError): continue
    try: d = datetime.date.fromisoformat(r["gameday"])
    except (ValueError, KeyError, TypeError): continue
    games.append((d, r["home_team"], r["away_team"], hs, a_s))
ref = max(g[0] for g in games); wt = lambda d: 0.5 ** ((ref - d).days / HALFLIFE)
teams = sorted({t for g in games for t in (g[1], g[2])})

tw = tp = hm = hw = 0.0
for d, h, a, hs, a_s in games:
    w = wt(d); tp += w*(hs+a_s); tw += 2*w; hm += w*(hs-a_s); hw += w
LG = tp/tw                                             # league avg points / team / game
HFA = hm/hw                                            # avg home margin (points)
# opponent-adjusted offensive (points scored) & defensive (points allowed) ratings, venue-neutral
off = {t: 0.0 for t in teams}; dfn = {t: 0.0 for t in teams}
for _ in range(50):
    no = {t: [0.0, 0.0] for t in teams}; nd = {t: [0.0, 0.0] for t in teams}
    for d, h, a, hs, a_s in games:
        w = wt(d)
        # neutralise home edge: home perf - HFA/2, away perf + HFA/2
        no[h][0] += w*((hs - HFA/2) - LG - dfn[a]); no[h][1] += w
        no[a][0] += w*((a_s + HFA/2) - LG - dfn[h]); no[a][1] += w
        nd[h][0] += w*((a_s + HFA/2) - LG - off[a]); nd[h][1] += w
        nd[a][0] += w*((hs - HFA/2) - LG - off[h]); nd[a][1] += w
    for t in teams:
        if no[t][1]: off[t] = no[t][0]/no[t][1]
        if nd[t][1]: dfn[t] = nd[t][0]/nd[t][1]
    om = sum(off.values())/len(teams); dm = sum(dfn.values())/len(teams)
    for t in teams: off[t] -= om; dfn[t] -= dm
# margin & total SDs from residuals
sm = st = sw = 0.0
for d, h, a, hs, a_s in games:
    w = wt(d)
    pm = (off[h]-off[a]) + (dfn[a]-dfn[h]) + HFA
    pt = 2*LG + off[h]+off[a] + dfn[h]+dfn[a]
    sm += w*((hs-a_s)-pm)**2; st += w*((hs+a_s)-pt)**2; sw += w
SD_M = (sm/sw)**0.5; SD_T = (st/sw)**0.5
print(f"{len(games)} games | LG {LG:.1f} pts/team | HFA {HFA:.1f} | SD margin {SD_M:.1f} total {SD_T:.1f}", flush=True)

def cdf(x): return 0.5*(1+math.erf(x/2**0.5))
order = sorted(teams, key=lambda t: -(off[t]-dfn[t]))
with open(PROJ + r"\nfl_ratings.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f); w.writerow(["team","off","def","net","ppg_for","ppg_against","lg_ppg","hfa","sd_margin","sd_total"])
    for t in order:
        w.writerow([t, round(off[t],2), round(dfn[t],2), round(off[t]-dfn[t],2),
                    round(LG+off[t],1), round(LG+dfn[t],1), round(LG,2), round(HFA,2), round(SD_M,2), round(SD_T,2)])
print("TOP 6:", [f"{t} ({off[t]-dfn[t]:+.1f})" for t in order[:6]])
print("BOT 4:", order[-4:])
# sample: best home vs worst, and a close one
def pred(h,a):
    expH=LG+off[h]+dfn[a]+HFA/2; expA=LG+off[a]+dfn[h]-HFA/2
    m=expH-expA; winH=cdf(m/SD_M); tot=expH+expA
    return expH,expA,winH,tot
h,a=order[0],order[-1]; eH,eA,wH,tt=pred(h,a)
print(f"SAMPLE {h} (home) vs {a}: {eH:.0f}-{eA:.0f} | {h} win {wH*100:.0f}% | spread {h} -{eH-eA:.1f} | total {tt:.0f}")
print("Wrote nfl_ratings.csv")
