# -*- coding: utf-8 -*-
"""
Backtest our power ratings against OFFICIAL results, per sport.
For each sport: (1) data completeness (games/team — catches missing data),
(2) rank correlation of our rating vs the official standings, (3) the biggest
team-by-team disparities (so we can diagnose whether each gap is justified
[injuries/roster] or a data/model bug). Run: python backtest_rankings.py nfl
"""
import csv, io, os, sys, math, urllib.request, datetime
from collections import defaultdict
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass
PROJ = os.path.dirname(os.path.abspath(__file__)); csv.field_size_limit(10**7)
def get(url, t=60):
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    return urllib.request.urlopen(req, timeout=t).read().decode("utf-8", "replace")
def fl(x):
    try: return float(x)
    except (TypeError, ValueError): return None
def ranks(vals):                       # vals: dict team->number (higher=better). returns team->rank(1=best)
    order = sorted(vals, key=lambda t: -vals[t]); return {t: i+1 for i, t in enumerate(order)}
def pearson(xs, ys):
    n = len(xs); mx = sum(xs)/n; my = sum(ys)/n
    cov = sum((x-mx)*(y-my) for x, y in zip(xs, ys)); vx = sum((x-mx)**2 for x in xs); vy = sum((y-my)**2 for y in ys)
    return cov/((vx*vy)**0.5) if vx > 0 and vy > 0 else 0.0
def spearman(a, b):                    # a,b: dict team->value (higher=better). rank-correlation
    ra, rb = ranks(a), ranks(b); ts = [t for t in a if t in b]
    return pearson([ra[t] for t in ts], [rb[t] for t in ts])

def report(sport, our_net, official, extra_cols, games_per_team, expected_games):
    """our_net: team->our rating (roster-adjusted). official: team->(win%, diff, label).
       extra_cols: team-> list of (name,val) for the table. """
    ts = [t for t in our_net if t in official]
    win = {t: official[t][0] for t in ts}; diff = {t: official[t][1] for t in ts}
    sp_win = spearman(our_net, win); sp_diff = spearman(our_net, diff)
    pe_diff = pearson([our_net[t] for t in ts], [diff[t] for t in ts])
    print(f"\n===== {sport.upper()} backtest =====")
    print(f"teams={len(ts)} | expected games/team≈{expected_games}")
    bad = {t: g for t, g in games_per_team.items() if abs(g-expected_games) > max(3, expected_games*0.15)}
    print("DATA COMPLETENESS issues (games/team off expected):", bad if bad else "none")
    print(f"rank-corr (Spearman) our-rating vs official:  win%={sp_win:.3f}  pointdiff={sp_diff:.3f}")
    print(f"Pearson our-rating vs official point-diff: {pe_diff:.3f}")
    our_rank = ranks(our_net); off_rank = ranks(win)
    disp = sorted(ts, key=lambda t: -abs(our_rank[t]-off_rank[t]))
    print("BIGGEST DISPARITIES (our_rank vs official_rank):")
    for t in disp[:10]:
        ex = "  ".join(f"{n}={v}" for n, v in extra_cols.get(t, []))
        print(f"  {t:4s} ourRank {our_rank[t]:2d}  officialRank {off_rank[t]:2d}  (Δ{our_rank[t]-off_rank[t]:+d})  win%={win[t]:.3f} diff={diff[t]:+.0f}  {ex}")
    return sp_win, sp_diff

# ---------------- NFL ----------------
def backtest_nfl():
    RELO = {"OAK": "LV", "SD": "LAC", "STL": "LA", "LAR": "LA"}; fix = lambda t: RELO.get(t, t)
    rows = list(csv.DictReader(io.StringIO(get("https://raw.githubusercontent.com/nflverse/nfldata/master/data/games.csv"))))
    # most recent completed season
    seasons = sorted({int(r["season"]) for r in rows if r.get("season", "").isdigit()})
    # find latest season with finals
    done = [s for s in seasons if any(r.get("home_score") not in (None, "", "NA") and int(r["season"]) == s for r in rows)]
    SEA = max(done)
    W = defaultdict(float); L = defaultdict(float); PF = defaultdict(float); PA = defaultdict(float); GP = defaultdict(int)
    for r in rows:
        if r.get("season", "") != str(SEA): continue
        if (r.get("game_type") or "REG") != "REG": continue
        hs, as_ = fl(r.get("home_score")), fl(r.get("away_score"))
        if hs is None or as_ is None: continue
        h, a = fix(r["home_team"]), fix(r["away_team"])
        GP[h] += 1; GP[a] += 1; PF[h] += hs; PA[h] += as_; PF[a] += as_; PA[a] += hs
        if hs > as_: W[h] += 1; L[a] += 1
        elif as_ > hs: W[a] += 1; L[h] += 1
        else: W[h] += .5; W[a] += .5; L[h] += .5; L[a] += .5
    teams = sorted(GP)
    winp = {t: W[t]/(W[t]+L[t]) for t in teams}; diff = {t: PF[t]-PA[t] for t in teams}
    official = {t: (winp[t], diff[t], "") for t in teams}
    # our ratings
    our = list(csv.DictReader(open(os.path.join(PROJ, "nfl_ratings.csv"), encoding="utf-8")))
    adj_net = {fix(r["team"]): fl(r["net_epa"]) for r in our}
    hist_net = {fix(r["team"]): (fl(r.get("off_hist") or r["off_epa"]) - fl(r.get("def_hist") or r["def_epa"])) for r in our}
    extra = {fix(r["team"]): [("histNet", round(fl(r.get("off_hist") or r["off_epa"])-fl(r.get("def_hist") or r["def_epa"]),3)),
                              ("adjOff", r.get("adj_off","")), ("adjDef", r.get("adj_def",""))] for r in our}
    print(f"(official = {SEA} regular-season standings)")
    # how well does PURE HISTORY rating match this season's results? (data-integrity check)
    print(f"DATA-INTEGRITY: Spearman(history-net, {SEA} win%) = {spearman(hist_net, winp):.3f} | Pearson(history-net,{SEA} diff)={pearson([hist_net[t] for t in teams],[diff[t] for t in teams]):.3f}")
    report("nfl", adj_net, official, extra, dict(GP), 17)

if __name__ == "__main__":
    which = (sys.argv[1] if len(sys.argv) > 1 else "nfl").lower()
    {"nfl": backtest_nfl}.get(which, backtest_nfl)()
