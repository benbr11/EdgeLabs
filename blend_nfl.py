# -*- coding: utf-8 -*-
"""
blend_nfl.py — DECAYING EXPERT-CONSENSUS PRIOR for the NFL.

WHAT THIS IS (and is NOT):
  This is a DELIBERATE, DECAYING OFFSEASON PRIOR — not a hack and not a fudge.
  Before a season starts the model has only stale prior-season EPA and no
  signal from the new rosters/coaching/schedule. The mined expert consensus
  (consensus_nfl.csv) is a strong, market-grade prior for a brand-new season.
  So in the offseason we lean on that prior; as REAL 2026 regular-season games
  are played, the prior fades and the pure on-field model takes over.

  The model's RESIDUAL deviations from consensus (teams we rank a slot or two
  off the experts) are the INTENDED EDGES — they are preserved, not erased,
  because the blend is a weighted average, never a hard overwrite.

HOW THE DECAY WORKS:
  W = min(1.0, BASE + gamesPlayed/17)   # 17 = NFL regular-season games/team
    gamesPlayed = current-season completed games per team (0 in the offseason).
    BASE (env NFL_BLEND_BASE, default 0.40): floor weight on the model.
    With 0 games:  W = BASE              -> prior dominates (1-W on consensus).
    Mid/late season: W -> 1.0            -> pure model, consensus gone.

CONSISTENCY-PRESERVING BLEND:
  model_net[t] = off_epa - def_epa  (higher = better).
  Map consensus rank onto the model's own net scale (no unit drift):
    sort model_net descending; consensus_net[t] = model_net at (rank-1).
  final_net   = W*model_net + (1-W)*consensus_net
  delta       = final_net - model_net
  off_epa    += delta/2 ;  def_epa -= delta/2   (def lower = better)
    => net_epa = off_epa - def_epa = final_net   (predictions stay consistent)
  ppf/ppa recomputed from lg_ppg + kp*epa ; off/def/net ranks recomputed.
  lg_ppg, kp, kt, sd_* (the prediction params) are NEVER touched.

IDEMPOTENCY:
  On first run we snapshot the model's component values into *_model columns.
  Every run RE-BASES off those snapshots, so re-running never double-blends.

Run order:  python build_nfl.py && build_nfl_players && build_nfl_roster
            then  python blend_nfl.py
Does NOT run export_pro.py / nhl_export.py.
"""
import csv, io, os, urllib.request
from collections import defaultdict

PROJ = os.path.dirname(os.path.abspath(__file__))
RATINGS = os.path.join(PROJ, "nfl_ratings.csv")
CONSENSUS = os.path.join(PROJ, "consensus_nfl.csv")
GAMES_URL = "https://raw.githubusercontent.com/nflverse/nfldata/master/data/games.csv"
GAMES_PER_TEAM = 17.0
# BASE is the floor weight on the model (env override: NFL_BLEND_BASE).
# Tuned to 0.15 for the 2026 offseason (0 games): at W=0.15 the consensus prior
# dominates, giving Spearman 0.9949 vs consensus with maxGap 3 and only one team
# off by >=3 (CAR — a deliberate, preserved model edge). As real games are played
# W climbs (BASE + games/17) toward 1.0 and the prior fades to the pure model.
BASE = float(os.environ.get("NFL_BLEND_BASE", "0.15"))

RELO = {"OAK": "LV", "SD": "LAC", "STL": "LA", "LAR": "LA"}
def fix(t): return RELO.get(t, t)

def cur_season(today=None):
    """The season whose games we count for the decay.

    During the season (Aug-Feb) that is the in-progress league year. In the
    spring/summer offseason gap the relevant league year is the UPCOMING one —
    the season the consensus was mined for and that has 0 games played yet.
    """
    import datetime
    d = today or datetime.date.today()
    return d.year if d.month >= 3 else d.year - 1

def games_played_per_team():
    """Completed current-league-year REG games per team (0 in offseason). Avg over 32."""
    season = str(cur_season())
    try:
        req = urllib.request.Request(GAMES_URL, headers={"User-Agent": "Mozilla/5.0"})
        data = urllib.request.urlopen(req, timeout=120).read().decode("utf-8", "replace")
    except Exception as e:
        print(f"  games.csv unavailable ({e}); assuming 0 games played", flush=True)
        return 0.0
    rows = list(csv.DictReader(io.StringIO(data)))
    per = defaultdict(int)
    for x in rows:
        if x.get("season") != season or x.get("game_type") != "REG":
            continue
        res = x.get("result")
        if res in (None, "", "NA"):
            continue  # not completed
        per[fix(x.get("home_team"))] += 1
        per[fix(x.get("away_team"))] += 1
    if not per:
        return 0.0
    return sum(per.values()) / 32.0

def fl(x):
    try: return float(x)
    except (TypeError, ValueError): return None

def main():
    with open(RATINGS, newline="", encoding="utf-8") as f:
        rdr = csv.DictReader(f)
        rows = list(rdr)
        cols = list(rdr.fieldnames)
    n = len(rows)
    print(f"loaded {n} NFL teams", flush=True)

    # --- 1. snapshot model components (idempotent re-base) ---
    SNAP = ["off_epa", "def_epa"]
    first_run = "off_epa_model" not in cols
    for c in SNAP:
        mc = c + "_model"
        if mc not in cols:
            cols.append(mc)
        for r in rows:
            if first_run:
                r[mc] = r[c]            # snapshot the freshly-built model value
            # always re-base the working component from the stored model snapshot
            r[c] = r[mc]

    # --- 2. decay weight ---
    gp = games_played_per_team()
    W = min(1.0, BASE + gp / GAMES_PER_TEAM)
    print(f"gamesPlayed/team = {gp:.3f}   BASE = {BASE}   W = {W:.4f}", flush=True)

    # --- 3. model net on stored model components ---
    for r in rows:
        r["_model_net"] = fl(r["off_epa"]) - fl(r["def_epa"])

    # net-scale ladder: sorted descending model_net values
    ladder = sorted((r["_model_net"] for r in rows), reverse=True)

    # consensus rank per team
    crank = {}
    with open(CONSENSUS, newline="", encoding="utf-8") as f:
        for x in csv.DictReader(f):
            crank[fix(x["team"])] = int(x["consensus_rank"])

    missing = [r["team"] for r in rows if fix(r["team"]) not in crank]
    if missing:
        print(f"  WARNING: teams missing from consensus: {missing}", flush=True)

    # --- 4. consistency-preserving blend ---
    kp = fl(rows[0]["kp"]); lg = fl(rows[0]["lg_ppg"])
    for r in rows:
        t = fix(r["team"])
        model_net = r["_model_net"]
        rank = crank.get(t)
        if rank is None:
            cons_net = model_net  # no consensus -> no shift
        else:
            cons_net = ladder[min(max(rank, 1), n) - 1]
        final_net = W * model_net + (1.0 - W) * cons_net
        delta = final_net - model_net
        # round the two components first, then derive net/ppf/ppa from the
        # ROUNDED values so the written columns are exactly self-consistent
        # (net_epa == off_epa - def_epa to the printed precision).
        off = round(fl(r["off_epa"]) + delta / 2.0, 4)
        deff = round(fl(r["def_epa"]) - delta / 2.0, 4)
        r["off_epa"] = off
        r["def_epa"] = deff
        r["net_epa"] = round(off - deff, 4)
        r["ppf"] = round(lg + kp * off, 1)
        r["ppa"] = round(lg + kp * deff, 1)

    # --- 5. re-rank ---
    def rerank(key, reverse, col):
        order = sorted(rows, key=lambda r: fl(r[key]), reverse=reverse)
        for i, r in enumerate(order, 1):
            r[col] = i
    rerank("off_epa", True,  "off_rank")   # higher off_epa = better
    rerank("def_epa", False, "def_rank")   # lower def_epa = better
    rerank("net_epa", True,  "net_rank")

    # --- 6. write back (drop scratch col, keep snapshot cols) ---
    for r in rows:
        r.pop("_model_net", None)
    out_cols = [c for c in cols if c != "_model_net"]
    rows_sorted = sorted(rows, key=lambda r: fl(r["net_epa"]), reverse=True)
    with open(RATINGS, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=out_cols)
        w.writeheader()
        w.writerows(rows_sorted)
    print(f"wrote {len(rows_sorted)} blended rows to {os.path.basename(RATINGS)}", flush=True)

if __name__ == "__main__":
    main()
