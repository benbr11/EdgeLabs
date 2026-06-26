# -*- coding: utf-8 -*-
"""
mlb_odds.py -- parse the free GitHub historical MLB odds dump into a clean, de-vigged
CLOSING-moneyline table joined to games by date + team names.

Source (free, no auth): ArnavSaraogi/mlb-odds-scraper release asset
  https://github.com/ArnavSaraogi/mlb-odds-scraper/releases/download/dataset/mlb_odds_dataset.json
  -> saved locally as mlb_odds_raw.json (run download first, see fetch() below).
Coverage: all of 2024 + 2025 through 2025-08-16 (late-2025 NOT in the dump). Per-book
opening + currentLine (closing) home/away American odds.

We take the MEDIAN closing line across books (robust consensus), de-vig multiplicatively
(p_home_true = p_home_imp / (p_home_imp + p_away_imp)), and write mlb_odds.csv:
  date, home, away, home_ml, away_ml, p_home_close, p_away_close, vig, n_books
keyed by date|home|away (doubleheaders: one game keeps the line, a known minor limitation).

This file is MEASUREMENT-ONLY market data; it never feeds build_mlb.py predictions.
"""
import json, os, csv, statistics, urllib.request

PROJ = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(PROJ, "mlb_odds_raw.json")
OUT = os.path.join(PROJ, "mlb_odds.csv")
URL = "https://github.com/ArnavSaraogi/mlb-odds-scraper/releases/download/dataset/mlb_odds_dataset.json"

NAMEFIX = {"Oakland Athletics": "Athletics"}
fn = lambda n: NAMEFIX.get(n, n)


def fetch():
    if os.path.exists(RAW):
        return
    print("Downloading odds dump (~80MB)...", flush=True)
    req = urllib.request.Request(URL, headers={"User-Agent": "Mozilla/5.0"})
    data = urllib.request.urlopen(req, timeout=180).read()
    with open(RAW, "wb") as f:
        f.write(data)


def american_to_prob(ml):
    """American moneyline -> implied probability (with vig)."""
    if ml is None:
        return None
    ml = float(ml)
    if ml < 0:
        return -ml / (-ml + 100.0)
    return 100.0 / (ml + 100.0)


def build():
    fetch()
    with open(RAW, encoding="utf-8") as f:
        o = json.load(f)
    rows = {}
    for date, games in o.items():
        for g in games:
            gv = g.get("gameView", {})
            if gv.get("gameType") != "R":
                continue
            home = fn(gv.get("homeTeam", {}).get("fullName", ""))
            away = fn(gv.get("awayTeam", {}).get("fullName", ""))
            d = gv.get("startDate", "")[:10]
            if not (home and away and d):
                continue
            ml = g.get("odds", {}).get("moneyline", []) or []
            home_lines, away_lines = [], []
            for m in ml:
                cl = m.get("currentLine", {}) or {}
                ho, ao = cl.get("homeOdds"), cl.get("awayOdds")
                if ho is not None and ao is not None:
                    home_lines.append(float(ho))
                    away_lines.append(float(ao))
            if not home_lines:
                continue
            home_ml = statistics.median(home_lines)
            away_ml = statistics.median(away_lines)
            ph = american_to_prob(home_ml)
            pa = american_to_prob(away_ml)
            if ph is None or pa is None or (ph + pa) <= 0:
                continue
            vig = ph + pa - 1.0
            # SANITY FILTER: the source dump contains scraping artifacts (e.g. a -1, -2, -8
            # moneyline) that yield absurd implied probs and a huge fake vig. Every real MLB
            # moneyline has |odds| >= 100 (favorites <= -100, underdogs >= +100), and a real
            # two-way market has a small positive vig. Reject anything else.
            if abs(home_ml) < 100 or abs(away_ml) < 100:
                continue
            if not (0.0 <= vig <= 0.15):
                continue
            ph_true = ph / (ph + pa)               # multiplicative de-vig
            pa_true = pa / (ph + pa)
            key = f"{d}|{home}|{away}"
            # keep the FIRST game of a doubleheader (can't disambiguate g1/g2 reliably);
            # prefer one with more books if a duplicate appears.
            if key in rows and rows[key]["n_books"] >= len(home_lines):
                continue
            rows[key] = {
                "date": d, "home": home, "away": away,
                "home_ml": round(home_ml), "away_ml": round(away_ml),
                "p_home_close": round(ph_true, 5), "p_away_close": round(pa_true, 5),
                "vig": round(vig, 5), "n_books": len(home_lines),
            }
    with open(OUT, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["date", "home", "away", "home_ml", "away_ml",
                                          "p_home_close", "p_away_close", "vig", "n_books"])
        w.writeheader()
        for k in sorted(rows):
            w.writerow(rows[k])
    print(f"Wrote {len(rows)} de-vigged closing-line rows to mlb_odds.csv", flush=True)
    # quick coverage / vig summary
    import collections
    by_yr = collections.Counter(r["date"][:4] for r in rows.values())
    vigs = [r["vig"] for r in rows.values()]
    print(f"  by year: {dict(by_yr)}", flush=True)
    print(f"  median vig: {statistics.median(vigs)*100:.2f}%  mean books/game: "
          f"{statistics.mean(r['n_books'] for r in rows.values()):.1f}", flush=True)
    return rows


if __name__ == "__main__":
    build()
