# -*- coding: utf-8 -*-
"""
Universal per-player international scoring profile from martj42 goalscorers.csv.
Covers EVERY national team (fixes the StatsBomb coverage gap for teams like Bosnia/Qatar).

For each (team, scorer): recency-weighted open-play goals, penalty goals, total goals,
last goal date -- plus each team's recency-weighted match count (opportunities) from
results.csv. Output player_intl.csv with a recency-weighted goals-per-match rate.

Light enough to run in the auto-update Action (unlike the heavy StatsBomb crunch).
Run after build_ratings.py --refresh (which downloads goalscorers.csv + results.csv).
"""
import csv, os, urllib.request, collections, datetime

PROJ = os.path.dirname(os.path.abspath(__file__))
GOALS_URL = "https://raw.githubusercontent.com/martj42/international_results/master/goalscorers.csv"
HALFLIFE_Y = 2.5                                   # recent form half-life (years)

def _read_local_or_fetch(path, url):
    if os.path.exists(path):
        return open(path, encoding="utf-8").read()
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    data = urllib.request.urlopen(req, timeout=60).read().decode("utf-8", "replace")
    open(path, "w", encoding="utf-8", newline="").write(data)        # cache for the Action
    return data

goals_txt = _read_local_or_fetch(PROJ + r"\goalscorers.csv", GOALS_URL)
grows = list(csv.DictReader(goals_txt.splitlines()))
rrows = list(csv.DictReader(open(PROJ + r"\results.csv", encoding="utf-8")))

def pdate(s):
    try: return datetime.datetime.strptime(s, "%Y-%m-%d").date()
    except Exception: return None

# reference = latest date in the data (no wall-clock dependency -> deterministic)
ref = max([d for d in (pdate(r["date"]) for r in grows) if d] +
          [d for d in (pdate(r["date"]) for r in rrows) if d])
def weight(d):
    if not d: return 0.0
    yrs = (ref - d).days / 365.25
    return 0.5 ** (yrs / HALFLIFE_Y) if yrs >= 0 else 1.0

# team recency-weighted match count (opportunities denominator) + recent volume, universal
matches_w = collections.defaultdict(float); matches_n = collections.Counter()
recent3 = collections.Counter()                          # matches in last 3 years -> schedule volume
for r in rrows:
    d = pdate(r["date"]); w = weight(d)
    for t in (r["home_team"], r["away_team"]):
        matches_w[t] += w; matches_n[t] += 1
        if d and (ref - d).days <= 1095: recent3[t] += 1

# per (team, scorer): weighted open-play / penalty goals
op_w = collections.defaultdict(float); pen_w = collections.defaultdict(float)
tot = collections.Counter(); last = {}
for r in grows:
    if (r.get("own_goal") or "").upper() == "TRUE":          # own goals: no attacking credit
        continue
    team = r["team"]; scorer = (r.get("scorer") or "").strip()
    if not scorer: continue
    d = pdate(r["date"]); w = weight(d); k = (team, scorer)
    if (r.get("penalty") or "").upper() == "TRUE": pen_w[k] += w
    else: op_w[k] += w
    tot[k] += 1
    if k not in last or (d and d > last[k]): last[k] = d

with open(PROJ + r"\player_intl.csv", "w", newline="", encoding="utf-8") as f:
    wr = csv.writer(f)
    wr.writerow(["team", "scorer", "goals_total", "op_goals_w", "pen_goals_w",
                 "matches_w", "op_rate", "pen_rate", "last_date"])
    keys = sorted(set(op_w) | set(pen_w), key=lambda k: -(op_w[k] + pen_w[k]))
    for k in keys:
        team, scorer = k
        mw = matches_w.get(team, 0.0)
        if mw < 1.0: continue                                # too few team matches to rate
        op_rate = op_w[k] / mw; pen_rate = pen_w[k] / mw
        wr.writerow([team, scorer, tot[k], round(op_w[k], 3), round(pen_w[k], 3),
                     round(mw, 1), round(op_rate, 4), round(pen_rate, 4),
                     last[k].isoformat() if last.get(k) else ""])

# per-team appearance fraction: a regular plays ~APP of the team's matches. High-volume
# schedules (CONCACAF: Gold Cup + Nations League + friendlies) -> stars sit more -> lower
# fraction -> a bigger per-team-match -> per-appearance correction downstream.
with open(PROJ + r"\team_appfrac.csv", "w", newline="", encoding="utf-8") as f:
    wr = csv.writer(f); wr.writerow(["team", "matches_3y", "per_year", "app_frac"])
    for t in sorted(matches_n):
        mpy = recent3.get(t, 0) / 3.0
        app = min(0.90, max(0.60, 8.5 / mpy)) if mpy > 0 else 0.75
        wr.writerow([t, recent3.get(t, 0), round(mpy, 1), round(app, 3)])

print(f"goalscorers rows: {len(grows):,}   results rows: {len(rrows):,}   ref date: {ref}")
print(f"Wrote player_intl.csv: {sum(1 for k in (set(op_w)|set(pen_w)) if matches_w.get(k[0],0)>=1)} player-team rows, "
      f"{len(matches_w)} teams; team_appfrac.csv: {len(matches_n)} teams")
# quick sanity
for tm, nm in [("Bosnia and Herzegovina","Demirovic"),("Qatar","Afif"),("France","Mbappe"),("England","Kane")]:
    hit = [(k, op_w[k], pen_w[k], matches_w[k[0]]) for k in (set(op_w)|set(pen_w)) if k[0]==tm and nm.lower() in k[1].lower()]
    for (t,s),o,p,mw in hit[:2]:
        print(f"  {t} / {s}: op_w={o:.2f} pen_w={p:.2f} team_matches_w={mw:.1f} -> op_rate={o/mw:.3f} pen_rate={p/mw:.3f}")
