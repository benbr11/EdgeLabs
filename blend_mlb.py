# -*- coding: utf-8 -*-
"""
blend_mlb.py - DECAYING EXPERT-CONSENSUS PRIOR for MLB (post-build, run AFTER build_mlb.py).

WHY THIS EXISTS (this is a deliberate, decaying offseason prior, NOT a hack):
  In the offseason / very early season the model has almost no current-season signal, so we
  anchor the team ordering to a mined EXPERT CONSENSUS (consensus_mlb.csv: ESPN/CBS/MLB.com).
  As real 2026 games are played, a games-decay weight W moves the rating smoothly back to the
  PURE MODEL. The model's residual deviations from consensus are the INTENDED EDGES -- a few
  teams sitting 1-2 spots off the consensus is the signal we want to bet, not error to remove.

  MLB 2026 is ~half played (~80 games/team), so W is high (~0.89) and the MODEL DOMINATES
  (~89%). That is correct and intended: mid-season we trust the model, not the preseason prior.

HOW IT STAYS HONEST (prediction consistency):
  We never touch the prediction machinery's scale params (avg_runs, home_adv, lg_ra9). We only
  shift each team's net run rating toward the blended target and re-derive the att/dfn run
  multipliers from that shift, so simulate.py keeps producing internally-consistent run lines.

IDEMPOTENT:
  On first run we snapshot the model's component values into *_model columns. Every run RE-BASES
  from those stored model columns before blending, so re-running never double-blends.

Reads:  mlb_ratings.csv (freshly built), consensus_mlb.csv
Writes: mlb_ratings.csv (blended, re-ranked, all 30 teams; adds *_model snapshot columns)
Does NOT run export_pro.py / nhl_export.py.
"""
import csv, os

PROJ = os.path.dirname(os.path.abspath(__file__))
RATINGS = os.path.join(PROJ, "mlb_ratings.csv")
CONSENSUS = os.path.join(PROJ, "consensus_mlb.csv")

# --- tunables ---
BASE = float(os.environ.get("MLB_BLEND_BASE", "0.40"))   # W=BASE at 0 games (prior dominates in offseason)
# Current-season games played per team (2026), measured from statsapi (final games / team ~= 80).
GAMES_PLAYED = float(os.environ.get("MLB_GAMES_PLAYED", "80"))
SEASON_LEN = 162.0

# Model component columns we snapshot/re-base from (idempotency anchor).
COMP = ["att", "dfn", "rpg_for", "rpg_against"]
MODELCOL = {c: c + "_model" for c in COMP}


def main():
    with open(RATINGS, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise SystemExit("mlb_ratings.csv is empty")

    have_model_cols = all(MODELCOL[c] in rows[0] for c in COMP)

    # 1. RE-BASE: ensure each row carries the pristine MODEL component values.
    for r in rows:
        if have_model_cols:
            # Re-running: restore working columns FROM the stored model snapshot (no double-blend).
            for c in COMP:
                r[c] = r[MODELCOL[c]]
        else:
            # First run: snapshot the freshly-built model values.
            for c in COMP:
                r[MODELCOL[c]] = r[c]

    # Scale params are identical across teams; never modified here.
    avg_runs = float(rows[0]["avg_runs"])
    home_adv = rows[0]["home_adv"]
    lg_ra9 = rows[0]["lg_ra9"]

    # 2. games-decay weight: W = min(1, BASE + gamesPlayed/162). 0 games -> BASE (prior dominates),
    #    mid/late season -> 1.0 (pure model).
    W = min(1.0, BASE + GAMES_PLAYED / SEASON_LEN)

    # 3. model net run rating (run units) from the stored model snapshot.
    for r in rows:
        r["_model_net"] = float(r[MODELCOL["rpg_for"]]) - float(r[MODELCOL["rpg_against"]])

    # 4. Map consensus rank -> the model_net scale: sort model_net DESC, the consensus_net for a
    #    team ranked k is the value sitting at sorted position (k-1). This makes the consensus a
    #    target on the SAME run scale, so the blend is in run units.
    sorted_net = sorted((r["_model_net"] for r in rows), reverse=True)

    crank = {}
    with open(CONSENSUS, newline="", encoding="utf-8") as f:
        for c in csv.DictReader(f):
            crank[c["team"]] = int(c["consensus_rank"])

    n = len(rows)
    for r in rows:
        k = crank.get(r["team"])
        if k is None:
            raise SystemExit(f"team missing from consensus: {r['team']}")
        idx = max(0, min(n - 1, k - 1))
        cons_net = sorted_net[idx]
        model_net = r["_model_net"]
        final_net = W * model_net + (1.0 - W) * cons_net
        delta = final_net - model_net                       # runs
        att = float(r[MODELCOL["att"]]) + delta / (2.0 * avg_runs)
        dfn = float(r[MODELCOL["dfn"]]) - delta / (2.0 * avg_runs)
        r["att"] = round(att, 4)
        r["dfn"] = round(dfn, 4)
        r["rpg_for"] = round(avg_runs * att, 2)
        r["rpg_against"] = round(avg_runs * dfn, 2)
        r["avg_runs"] = round(avg_runs, 3)
        r["home_adv"] = home_adv
        r["lg_ra9"] = lg_ra9

    # 5. re-rank by (att - dfn) desc
    rows.sort(key=lambda r: (float(r["att"]) - float(r["dfn"])), reverse=True)

    fields = ["team", "att", "dfn", "rpg_for", "rpg_against", "avg_runs", "home_adv", "lg_ra9"]
    fields += [MODELCOL[c] for c in COMP]
    with open(RATINGS, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r[k] for k in fields})

    print(f"blend_mlb: gamesPlayed={GAMES_PLAYED:.0f} BASE={BASE} W={W:.4f} "
          f"({W*100:.1f}% model / {(1-W)*100:.1f}% consensus) -> wrote {n} teams")
    return W


if __name__ == "__main__":
    main()
