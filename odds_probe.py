# -*- coding: utf-8 -*-
"""Probe what bookmaker-odds data is reachable offline (for a model-vs-book proof)."""
import csv, json, os, urllib.request
PROJ = os.path.dirname(os.path.abspath(__file__))
def fetch(url, t=25):
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    return urllib.request.urlopen(req, timeout=t).read()

print("=== WC2026 dataset (mominullptr) file list ===")
try:
    j = json.loads(fetch("https://api.github.com/repos/mominullptr/FIFA-World-Cup-2026-Dataset/contents/"))
    for f in j:
        print(f"  {f['name']}  ({f.get('size','?')} B)")
except Exception as e:
    print("  ERR", e)

print("=== local matches_detailed (wc2026_xg.csv) columns ===")
try:
    hdr = next(csv.reader(open(PROJ + r"\wc2026_xg.csv", encoding="utf-8")))
    print(" ", hdr)
    odds_cols = [c for c in hdr if any(k in c.lower() for k in ("odd","prob","book","b365","implied","price"))]
    print("  odds-like columns:", odds_cols or "NONE")
except Exception as e:
    print("  ERR", e)

print("=== egress tests (can Python reach non-GitHub hosts?) ===")
for url in ["https://www.football-data.co.uk/mmz4281/2425/E0.csv",
            "https://raw.githubusercontent.com/martj42/international_results/master/former_names.csv",
            "https://api.the-odds-api.com/v4/sports/"]:
    try:
        d = fetch(url, 15)
        print(f"  OK   {url[:55]:55} {len(d)} B")
    except Exception as e:
        print(f"  FAIL {url[:55]:55} {type(e).__name__}: {str(e)[:50]}")
print("DONE")
