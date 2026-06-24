# -*- coding: utf-8 -*-
"""Probe: what match/goal/shot data do we have vs what can we fetch? Plan the expansion."""
import csv, json, os, urllib.request, collections

PROJ = os.path.dirname(os.path.abspath(__file__))
def fetch(url, timeout=60):
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()

def csv_rows(text):
    return list(csv.DictReader(text.splitlines()))

print("="*70); print("LOCAL results.csv"); print("="*70)
try:
    rows = list(csv.DictReader(open(PROJ+r"\results.csv", encoding="utf-8")))
    dates = [r.get("date","") for r in rows if r.get("date")]
    print(f"rows: {len(rows):,}   date range: {min(dates)} .. {max(dates)}")
    print(f"columns: {list(rows[0].keys())}")
    teams = collections.Counter()
    for r in rows:
        teams[r.get("home_team","")] += 1; teams[r.get("away_team","")] += 1
    for t in ("Bosnia and Herzegovina","Qatar","France","Brazil"):
        recent = [r for r in rows if (r.get("home_team")==t or r.get("away_team")==t) and r.get("date","")>="2023-01-01"]
        print(f"  {t}: {teams[t]} total matches, {len(recent)} since 2023-01-01")
except Exception as e:
    print("ERR", e)

# --- martj42 international datasets (GitHub) ---
M = "https://raw.githubusercontent.com/martj42/international_results/master"
for name in ("results.csv","goalscorers.csv","shootouts.csv"):
    print("="*70); print(f"REMOTE martj42/{name}"); print("="*70)
    try:
        txt = fetch(f"{M}/{name}").decode("utf-8", "replace")
        rows = csv_rows(txt)
        dates = [r.get("date","") for r in rows if r.get("date")]
        print(f"rows: {len(rows):,}   date range: {min(dates)} .. {max(dates)}")
        print(f"columns: {list(rows[0].keys())}")
        if name == "goalscorers.csv":
            scorers = collections.Counter(r.get("scorer","") for r in rows)
            dem = [r for r in rows if "Demirov" in r.get("scorer","")]
            print(f"  distinct scorers: {len(scorers):,}; sample top: {scorers.most_common(3)}")
            print(f"  Demirovic goal rows: {len(dem)} -> {[ (r['date'],r['team'],r.get('penalty')) for r in dem[:6] ]}")
            bos = sum(1 for r in rows if r.get("team")=="Bosnia and Herzegovina")
            qat = sum(1 for r in rows if r.get("team")=="Qatar")
            print(f"  Bosnia goal-rows: {bos}   Qatar goal-rows: {qat}")
    except Exception as e:
        print("ERR", e)

# --- StatsBomb: enumerate ALL competitions, flag internationals ---
print("="*70); print("STATSBOMB competitions.json (what shot data is available)"); print("="*70)
try:
    comps = json.loads(fetch("https://raw.githubusercontent.com/statsbomb/open-data/master/data/competitions.json").decode("utf-8"))
    by_comp = collections.defaultdict(list)
    for c in comps:
        by_comp[(c["competition_id"], c["competition_name"], c.get("competition_international", False))].append((c["season_id"], c["season_name"]))
    intl = [(k,v) for k,v in by_comp.items() if k[2]]
    club = [(k,v) for k,v in by_comp.items() if not k[2]]
    print(f"total competition-seasons: {len(comps)};  international comps: {len(intl)};  club comps: {len(club)}")
    print("\nINTERNATIONAL competitions (id | name | #seasons | seasons):")
    for (cid,name,_),seasons in sorted(intl, key=lambda x:x[0][1]):
        print(f"  {cid:>4} | {name:<28} | {len(seasons)} | {', '.join(s[1] for s in sorted(seasons))}")
    print("\nCLUB comps with likely WC players (sample):")
    for (cid,name,_),seasons in sorted(club, key=lambda x:x[0][1]):
        print(f"  {cid:>4} | {name:<34} | {len(seasons)} seasons")
except Exception as e:
    print("ERR", e)
print("\nDONE probe")
