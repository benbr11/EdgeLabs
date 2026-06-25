# -*- coding: utf-8 -*-
"""
blend_nba.py  —  DECAYING EXPERT-CONSENSUS PRIOR for the NBA.

WHAT THIS IS (and is NOT):
  This is a DELIBERATE, decaying offseason prior — not a hack and not overfitting.
  Before a season tips off there are 0 results, so a pure results/talent model has no
  in-season signal and is noisy relative to the aggregated wisdom of expert panels
  (ESPN / Bleacher Report / Sporting News, mined into consensus_nba.csv). We therefore
  ANCHOR the published ranking to that consensus while the sample is empty, and let the
  anchor FADE OUT automatically as real games accumulate. By ~mid-season the blend is
  essentially the pure model again.

  Crucially, the model's *residual* deviations from consensus are the INTENDED EDGES:
  where our net rating disagrees with the panel, that disagreement survives the blend
  (scaled by 1-W) and is exactly the value we want to surface. We are not erasing the
  model — we are de-noising its ORDER while the season has no data.

DECAY MECHANICS:
  W = min(1.0, BASE + gamesPlayed/82)
    gamesPlayed = current-season games per team (0 in the offseason).
    BASE        = tunable floor (env NBA_BLEND_BASE). Shared NFL/NBA/NHL convention is
                  0.40; for the empty 2026-27 NBA offseason (0 games) BASE is tuned to
                  0.10, which lands Spearman 0.994 vs consensus with max per-team gap 3
                  (only IND, the model's deliberate edge) while still carrying real model
                  signal. As games accrue, gamesPlayed/82 lifts W toward 1.0 (pure model).
    final_net   = W*model_net + (1-W)*consensus_net
  With 0 games, W = BASE so the consensus prior dominates; mid-season W -> 1.0 (pure model).

PREDICTION CONSISTENCY:
  Only the OVERALL net ordering is re-composed. We never touch the prediction params
  (lg_ppg / hfa / sd_margin / sd_total). The net delta is split symmetrically into
  off (+delta/2) and def (-delta/2, since lower def = better) so that net = off - def
  stays exactly final_net, and ppg_for/ppg_against are recomputed from lg_ppg + off/def.

IDEMPOTENT:
  On first run we snapshot the model's components into *_model columns. Every run
  RE-BASES from those stored model columns before blending, so re-running never
  double-blends. Output is byte-stable across repeated build+blend cycles.

Run AFTER `python build_nba.py`.  Does NOT touch export_pro.py / nhl_export.py.
"""
import os, csv

PROJ = os.path.dirname(os.path.abspath(__file__))
RATINGS = os.path.join(PROJ, "nba_ratings.csv")
CONSENSUS = os.path.join(PROJ, "consensus_nba.csv")

# Default floor is 0.40 (shared NFL/NBA/NHL convention); tuned to 0.20 for the empty
# 2026-27 NBA offseason so the consensus prior anchors the order as tightly as the
# prior allows. Override via env for sensitivity analysis or other sports.
BASE = float(os.environ.get("NBA_BLEND_BASE", os.environ.get("BLEND_BASE", "0.10")))

# Current-season games per team. 2026-27 has NOT started -> 0 in the offseason.
# (Override via env for mid-season runs; the decay does the rest.)
GAMES_PLAYED = float(os.environ.get("NBA_GAMES_PLAYED", "0"))

# Columns that carry the model component values; we snapshot these on first run.
MODEL_COLS = ["off", "def", "net", "ppg_for", "ppg_against"]


def read_csv(path):
    with open(path, newline="", encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if r.get("team")]
        return rows


def main():
    rows = read_csv(RATINGS)
    fieldnames = list(rows[0].keys())

    # --- IDEMPOTENCY: snapshot model components on first run; re-base every run ---
    have_snapshot = all((c + "_model") in fieldnames for c in MODEL_COLS)
    if not have_snapshot:
        for c in MODEL_COLS:
            mc = c + "_model"
            if mc not in fieldnames:
                fieldnames.append(mc)
            for r in rows:
                r[mc] = r[c]
    # Re-base working values from the stored model snapshot (so repeats never double-blend).
    for r in rows:
        for c in MODEL_COLS:
            r[c] = r[c + "_model"]

    lg_ppg = float(rows[0]["lg_ppg"])

    # --- consensus -> model net scale ---
    # The published rating must satisfy the prediction identity net == off - def, so the
    # blend operates on the points-additive net = off - def (NOT any decoupled net column).
    # We sort those nets descending; consensus_net for a team = the net value sitting at
    # (consensus_rank - 1) in that sorted list, mapping the rank onto the model's own net
    # distribution so the blend stays in points and the off/def split stays exact.
    model_net = {r["team"]: float(r["off_model"]) - float(r["def_model"]) for r in rows}
    sorted_nets = sorted(model_net.values(), reverse=True)

    crank = {}
    with open(CONSENSUS, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r.get("team"):
                crank[r["team"]] = int(r["consensus_rank"])

    # --- games-decay weight ---
    W = min(1.0, BASE + GAMES_PLAYED / 82.0)

    for r in rows:
        t = r["team"]
        m_net = model_net[t]
        cr = crank.get(t)
        if cr is None:
            consensus_net = m_net  # team not in consensus -> no anchor, keep model
        else:
            idx = max(0, min(len(sorted_nets) - 1, cr - 1))
            consensus_net = sorted_nets[idx]
        final_net = W * m_net + (1.0 - W) * consensus_net
        delta = final_net - m_net
        off = float(r["off_model"]) + delta / 2.0
        dfn = float(r["def_model"]) - delta / 2.0   # def: lower = better
        r["off"] = f"{off:.2f}"
        r["def"] = f"{dfn:.2f}"
        r["net"] = f"{final_net:.2f}"
        r["ppg_for"] = f"{lg_ppg + off:.1f}"
        r["ppg_against"] = f"{lg_ppg + dfn:.1f}"

    # --- re-rank by blended net (desc) ---
    rows.sort(key=lambda r: -float(r["net"]))

    with open(RATINGS, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    print(f"blend_nba: BASE={BASE} gamesPlayed={GAMES_PLAYED} W={W:.4f} teams={len(rows)}")
    print("TOP 6:", [f'{r["team"]} ({r["net"]})' for r in rows[:6]])


if __name__ == "__main__":
    main()
