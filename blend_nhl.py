# -*- coding: utf-8 -*-
"""
blend_nhl.py  --  DECAYING EXPERT-CONSENSUS PRIOR for the NHL team ratings.

WHAT THIS IS (and is NOT):
  This is a DELIBERATE, DECAYING OFFSEASON PRIOR, not a hack and not a fudge.
  Before a season has been played the only forward-looking information that beats
  a stale on-ice model is the aggregated expert consensus (ESPN / BR / CBS power
  rankings, mined into consensus_nhl.csv). So in the offseason we anchor the
  ranking to that consensus. As REAL current-season games accumulate, the blend
  weight W slides from BASE toward 1.0 and the rating becomes the PURE on-ice
  model. The model's residual deviations from the consensus (a few teams off by
  1-2 spots) are the INTENDED EDGES -- they are where our model disagrees with the
  crowd, which is exactly the value we want to surface, not noise to be erased.

PIPELINE POSITION:
  Runs AFTER `python build_nhl.py`. It does NOT modify the build's prediction
  parameters and does NOT run build_nhl_xg.py / export_pro.py / nhl_export.py.
  It reads the freshly-built nhl_ratings.csv plus consensus_nhl.csv and rewrites
  nhl_ratings.csv in place with blended, re-ranked rows.

IDEMPOTENCY:
  On first run it snapshots the model's component values into *_model columns
  (attack_mult_model, defense_mult_model). On EVERY run it RE-BASES from those
  stored model columns before blending, so re-running never double-blends:
  build+blend run twice produces byte-identical output.

PREDICTION CONSISTENCY:
  The blend is applied in goals/game NET space so the Poisson predictions stay
  self-consistent:
    model_net[t]   = attack_mult_model[t] - defense_mult_model[t]
    consensus_net  = model_net sorted desc, value at (consensus_rank-1)
    final_net      = W*model_net + (1-W)*consensus_net
    delta          = final_net - model_net
    attack_mult   += delta/2 ; defense_mult -= delta/2   (defense lower = better)
  avg_goals / home_adv / elo / xg / pp / pk are left UNCHANGED. attack_100 and
  defense_100 are recomputed as 0-100 percentiles of the new mults so the
  displayed composite and the row order reflect the blend; rows are re-sorted
  best-first by (attack_mult - defense_mult).

GAMES-DECAY WEIGHT:
    W = min(1.0, BASE + gamesPlayed/82)
  gamesPlayed = current-season games per team (0 in the offseason). BASE is the
  env var NHL_BLEND_BASE (default 0.40). With 0 games W=BASE so the prior
  dominates; mid-season W rises toward 1.0 (pure model).
"""
import csv, os

PROJ = os.path.dirname(os.path.abspath(__file__))
RATINGS = os.path.join(PROJ, "nhl_ratings.csv")
CONSENSUS = os.path.join(PROJ, "consensus_nhl.csv")

# BASE default: the multi-sport convention is 0.40, but for the NHL offseason
# (0 games) that left a few teams 4-5 spots off consensus. TUNED to 0.25 so the
# prior dominates tightly (Spearman ~0.993, max gap 3, only one team off by 3)
# while still leaving 25% model weight -> the small 1-2 spot residuals are the
# intended edges. Override with NHL_BLEND_BASE. As games are played the
# games-decay term raises W regardless of BASE.
BASE = float(os.environ.get("NHL_BLEND_BASE", "0.25"))

# Current-season games per team. 2026-27 NHL season has NOT started -> 0.
# (Override with NHL_GAMES_PLAYED for mid-season runs; the games-decay then
#  raises W toward 1.0 / pure model automatically.)
GAMES_PLAYED = float(os.environ.get("NHL_GAMES_PLAYED", "0"))
SEASON_LEN = 82.0


def to100_percentile(values, x):
    """0-100 percentile rank of x within values (higher value = higher score)."""
    n = len(values)
    if n <= 1:
        return 50.0
    below = sum(1 for v in values if v < x)
    equal = sum(1 for v in values if v == x)
    pr = (below + 0.5 * equal) / n
    return round(100.0 * pr, 1)


def main():
    with open(RATINGS, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fields = list(reader.fieldnames)
        rows = list(reader)

    # ---- idempotency: snapshot model components on first run; re-base every run ----
    have_model_cols = "attack_mult_model" in fields and "defense_mult_model" in fields
    for r in rows:
        if not have_model_cols:
            r["attack_mult_model"] = r["attack_mult"]
            r["defense_mult_model"] = r["defense_mult"]
        # RE-BASE: always start the blend from the stored model values
        r["attack_mult"] = r["attack_mult_model"]
        r["defense_mult"] = r["defense_mult_model"]
    if not have_model_cols:
        # keep the snapshot columns at the end of the header
        fields = fields + ["attack_mult_model", "defense_mult_model"]

    model_net = {r["team"]: float(r["attack_mult_model"]) - float(r["defense_mult_model"])
                 for r in rows}

    # ---- consensus target mapped onto the model_net scale ----
    cons_rank = {}
    with open(CONSENSUS, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            cons_rank[r["team"]] = int(r["consensus_rank"])

    # sorted model nets, descending; consensus value at (rank-1)
    net_desc = sorted(model_net.values(), reverse=True)
    missing = [t for t in model_net if t not in cons_rank]
    if missing:
        raise SystemExit(f"Teams missing from consensus_nhl.csv: {missing}")

    # ---- weight ----
    W = min(1.0, BASE + GAMES_PLAYED / SEASON_LEN)

    # ---- apply blend in net space ----
    for r in rows:
        t = r["team"]
        cnet = net_desc[cons_rank[t] - 1]
        fnet = W * model_net[t] + (1.0 - W) * cnet
        delta = fnet - model_net[t]
        am = float(r["attack_mult_model"]) + delta / 2.0
        dm = float(r["defense_mult_model"]) - delta / 2.0
        r["attack_mult"] = round(am, 4)
        r["defense_mult"] = round(dm, 4)

    # ---- recompute attack_100/defense_100 as 0-100 percentiles of new mults ----
    new_att = [float(r["attack_mult"]) for r in rows]
    new_dfn = [float(r["defense_mult"]) for r in rows]
    for r in rows:
        r["attack_100"] = to100_percentile(new_att, float(r["attack_mult"]))
        # defense: LOWER mult is better -> percentile on negated values
        r["defense_100"] = to100_percentile([-v for v in new_dfn], -float(r["defense_mult"]))

    # ---- re-sort best-first by (attack_mult - defense_mult) ----
    rows.sort(key=lambda r: -(float(r["attack_mult"]) - float(r["defense_mult"])))

    with open(RATINGS, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    print(f"blend_nhl: BASE={BASE} gamesPlayed={GAMES_PLAYED:.0f} W={W:.4f} "
          f"({'prior-dominant' if W < 1.0 else 'pure model'})")
    print(f"  re-based={'no (first run, snapshotted model cols)' if not have_model_cols else 'yes (from *_model)'}")
    print("  TOP 8:", [r["team"] for r in rows[:8]])
    print("  BOT 5:", [r["team"] for r in rows[-5:]])
    print("Wrote nhl_ratings.csv")


if __name__ == "__main__":
    main()
