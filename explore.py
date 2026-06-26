import csv, collections, datetime

import os
PROJ = os.path.dirname(os.path.abspath(__file__))
rows = list(csv.DictReader(open(PROJ + r"\results.csv", encoding="utf-8")))
print("total matches:", len(rows))
print("date range:", rows[0]["date"], "->", rows[-1]["date"])

# count matches per team
cnt = collections.Counter()
recent = collections.Counter()
for r in rows:
    y = int(r["date"][:4])
    for t in (r["home_team"], r["away_team"]):
        cnt[t] += 1
        if y >= 2022:
            recent[t] += 1

teams_48 = [
    "Canada","Mexico","United States","Australia","Iran","Iraq","Japan","Jordan",
    "Qatar","Saudi Arabia","South Korea","Uzbekistan","Algeria","Cape Verde",
    "DR Congo","Egypt","Ghana","Ivory Coast","Morocco","Senegal","South Africa",
    "Tunisia","Curaçao","Haiti","Panama","Argentina","Brazil","Colombia","Ecuador",
    "Paraguay","Uruguay","New Zealand","Austria","Belgium","Bosnia and Herzegovina",
    "Croatia","Czech Republic","England","France","Germany","Netherlands","Norway",
    "Portugal","Scotland","Spain","Sweden","Switzerland","Turkey",
]
print("\n--- coverage for the 48 (name as-tried | total | since2022) ---")
missing = []
for t in teams_48:
    if t in cnt:
        print(f"{t:28s} {cnt[t]:5d} {recent[t]:4d}")
    else:
        missing.append(t)
print("\nNOT FOUND under that exact name:", missing)

# show candidate names for the missing ones
allnames = set(cnt)
for m in missing:
    cands = [n for n in allnames if m.split()[0].lower() in n.lower() or n.lower() in m.lower()]
    print(f"  candidates for '{m}': {cands[:8]}")
