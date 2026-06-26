# -*- coding: utf-8 -*-
"""Show recent matches in the dataset (default: since 2026-06-01) so you can see
how results have gone up to the match you're about to simulate.
Usage:  python recent.py            (since 2026-06-01)
        python recent.py 2026-06-15 (custom cutoff)
"""
import csv, sys
import os
PROJ = os.path.dirname(os.path.abspath(__file__))
cutoff = sys.argv[1] if len(sys.argv) > 1 else "2026-06-01"
rows = [r for r in csv.DictReader(open(PROJ + r"\results.csv", encoding="utf-8"))
        if r["date"] >= cutoff]
print(f"{len(rows)} matches since {cutoff} (latest first):\n")
for r in sorted(rows, key=lambda r: r["date"], reverse=True):
    wc = "  << WC" if "World Cup" in r["tournament"] else ""
    print(f"  {r['date']}  {r['home_team']:>16} {r['home_score']}-{r['away_score']} "
          f"{r['away_team']:<16} [{r['tournament']}]{wc}")
