# -*- coding: utf-8 -*-
"""Print the FULL ranking comparison (every team) for each sport: our rank vs consensus rank."""
import csv, os, sys
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass
PROJ = os.path.dirname(os.path.abspath(__file__))
def load(name):
    with open(os.path.join(PROJ, name), encoding="utf-8") as f:
        return list(csv.DictReader(f))
def cons(code):
    d = {}
    try:
        for r in load(f"consensus_{code}.csv"):
            d[r["team"]] = int(float(r["consensus_rank"]))
    except FileNotFoundError: pass
    return d
def spearman(pairs):   # pairs: list of (our_rank, cons_rank)
    n = len(pairs)
    if n < 2: return 0.0
    ds = sum((a-b)**2 for a, b in pairs)
    return 1 - 6*ds/(n*(n*n-1))
SPORTS = [("NFL","nfl_ratings.csv","nfl","team"),
          ("NBA","nba_ratings.csv","nba","team"),
          ("NHL","nhl_ratings.csv","nhl","team"),
          ("MLB","mlb_ratings.csv","mlb","team")]
for NAME, fn, code, key in SPORTS:
    rows = load(fn); cmap = cons(code)
    print(f"\n================ {NAME}  (our rank | team | consensus | gap) ================")
    pairs = []; big = 0
    for i, r in enumerate(rows):
        t = r[key]; orank = i+1; cr = cmap.get(t)
        if cr is None:
            print(f"  {orank:2d}  {t:24s}  cons=?   (no consensus match)"); continue
        gap = orank - cr; pairs.append((orank, cr))
        flag = "  <<<" if abs(gap) >= 7 else ("  <<" if abs(gap) >= 4 else "")
        print(f"  {orank:2d}  {t:24s}  {cr:2d}   {gap:+d}{flag}")
        if abs(gap) >= 7: big += 1
    print(f"  -> Spearman={spearman(pairs):.3f}  teams_off_by>=7={big}  (n={len(pairs)})")
