# -*- coding: utf-8 -*-
"""
mlb_segments.py -- GOAL 2 edge-hunt. Find the SEGMENTS where the MLB model BEATS the real
de-vigged CLOSING market out-of-sample, so we can bet only those.

Pipeline
--------
1. Walk-forward (backtest_mlb.run_backtest) -> per-game model home-win prob (WITH the kept
   bullpen signal) + rich point-in-time metadata (venue, day/night, handedness, fatigue, park
   factor, division/interleague, month, confidence).
2. Join real de-vigged CLOSING moneylines from mlb_odds.csv (date|home|away).
3. For every segment, compute on the games that HAVE odds:
     - model hit-rate (model's pick = side with higher model prob)
     - market hit-rate (market's pick = side the closing price favors)
     - model edge in probability vs market (mean |model - market| on the model's pick side)
     - N (sample size)
     - flat-stake ROI: bet $1 on the MODEL's pick at the actual CLOSING American odds
     - CLV proxy: mean (model_prob_on_bet_side - market_prob_on_bet_side); >0 => we price
       the bet better than the closing line ("positive expected closing-line value")
4. Cross-check: split 2024 (tune) vs 2025 (validate). A segment is only "real" if it shows
   positive ROI / beat-the-market in BOTH the full sample AND each season independently
   (held-out confirmation) with adequate N.

Honest framing: the de-vigged closing line is the sharpest public benchmark; positive flat ROI
at the close + positive CLV is the gold standard for "beat the market". Where odds are missing
(2025 after the dataset cutoff, doubleheaders) those games are simply excluded from market
comparison; we report the odds-covered N for every segment.
"""
import csv, os, math, collections
import backtest_mlb as B

PROJ = os.path.dirname(os.path.abspath(__file__))
ODDS = os.path.join(PROJ, "mlb_odds.csv")

# AL / NL league membership (for division-vs-interleague segmentation).
AL = {"Baltimore Orioles", "Boston Red Sox", "New York Yankees", "Tampa Bay Rays",
      "Toronto Blue Jays", "Chicago White Sox", "Cleveland Guardians", "Detroit Tigers",
      "Kansas City Royals", "Minnesota Twins", "Houston Astros", "Los Angeles Angels",
      "Athletics", "Seattle Mariners", "Texas Rangers"}
NL = {"Atlanta Braves", "Miami Marlins", "New York Mets", "Philadelphia Phillies",
      "Washington Nationals", "Chicago Cubs", "Cincinnati Reds", "Milwaukee Brewers",
      "Pittsburgh Pirates", "St. Louis Cardinals", "Arizona Diamondbacks", "Colorado Rockies",
      "Los Angeles Dodgers", "San Diego Padres", "San Francisco Giants"}


def american_payout(ml):
    """Profit on a $1 win at American odds ml (e.g. +150 -> 1.5, -120 -> 0.8333)."""
    ml = float(ml)
    return ml / 100.0 if ml > 0 else 100.0 / (-ml)


def load_odds():
    odds = {}
    if not os.path.exists(ODDS):
        return odds
    with open(ODDS, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            odds[f"{r['date']}|{r['home']}|{r['away']}"] = r
    return odds


def get_preds():
    """Run the walk-forward (kept signals on) and return the enriched per-game preds."""
    test_seasons = {2024, 2025}
    fetch_seasons = list(range(2026, 2019, -1))
    games, plogs = B.load_cache(fetch_seasons, test_seasons)
    crank = B.load_consensus()
    preds, skipped = B.run_backtest(games, plogs, test_seasons, crank)
    return preds


def join_market(preds, odds):
    """Attach market fields to each pred; return only those with a usable closing line."""
    out = []
    for p in preds:
        r = odds.get(f"{p['date']}|{p['home']}|{p['away']}")
        if not r:
            continue
        q = dict(p)
        q["p_home_mkt"] = float(r["p_home_close"])
        q["home_ml"] = float(r["home_ml"])
        q["away_ml"] = float(r["away_ml"])
        out.append(q)
    return out


def bet_outcome(p):
    """For the MODEL's pick, return (won, profit_at_close, model_prob_side, mkt_prob_side, ml_side).
    Bet $1 flat on whichever side the model favors, settled at the real closing American odds."""
    model_home = p["winp"] >= 0.5
    if model_home:
        ml = p["home_ml"]; won = bool(p["home_won"])
        mp, kp = p["winp"], p["p_home_mkt"]
    else:
        ml = p["away_ml"]; won = not bool(p["home_won"])
        mp, kp = 1 - p["winp"], 1 - p["p_home_mkt"]
    profit = american_payout(ml) if won else -1.0
    return won, profit, mp, kp, ml


def seg_stats(rows):
    """Aggregate a list of joined preds into the segment metric bundle."""
    n = len(rows)
    if n == 0:
        return None
    model_hits = mkt_hits = 0
    profit = 0.0; clv = 0.0; edge = 0.0
    for p in rows:
        won, pr, mp, kp, ml = bet_outcome(p)
        model_hits += 1 if won else 0
        # market's own pick correctness (side the closing price favors)
        mkt_home = p["p_home_mkt"] >= 0.5
        mkt_hits += 1 if ((mkt_home and p["home_won"]) or (not mkt_home and not p["home_won"])) else 0
        profit += pr
        clv += (mp - kp)            # model prob minus market prob on the bet side
        edge += (mp - kp)
    return {
        "N": n,
        "model_hit": model_hits / n * 100,
        "mkt_hit": mkt_hits / n * 100,
        "edge_pp": edge / n * 100,           # mean prob edge (percentage points) on the bet side
        "roi": profit / n * 100,             # flat-stake ROI %
        "clv_pp": clv / n * 100,             # mean CLV (pp); >0 = positive closing-line value
        "profit": profit,
    }


def by_season(rows):
    s24 = [p for p in rows if p["season"] == 2024]
    s25 = [p for p in rows if p["season"] == 2025]
    return seg_stats(s24), seg_stats(s25)


# ----------------------------------------------------------------------------
# Segment definitions: name -> function(pred)->bucket_label (or None to exclude).
# ----------------------------------------------------------------------------
def conf_band(p):
    c = max(p["winp"], 1 - p["winp"])
    for thr in (0.80, 0.75, 0.70, 0.65, 0.60, 0.55):
        if c >= thr:
            return f">={int(thr*100)}%"
    return "<55%"


def fav_dog(p):
    return "model_fav_home" if p["winp"] >= 0.5 else "model_fav_away"


def mkt_side(p):
    return "bet_favorite" if (p["p_home_mkt"] >= 0.5) == (p["winp"] >= 0.5) and \
        max(p["p_home_mkt"], 1 - p["p_home_mkt"]) >= 0.5 else "bet_underdog"


def disagreement(p):
    """Bucket by how far the model's home prob sits from the market's home prob."""
    d = p["winp"] - p["p_home_mkt"]
    ad = abs(d)
    if ad < 0.03:
        return "agree(<3pp)"
    if ad < 0.06:
        return "edge 3-6pp"
    if ad < 0.10:
        return "edge 6-10pp"
    return "edge >10pp"


def div_inter(p):
    same_league = (p["home"] in AL and p["away"] in AL) or (p["home"] in NL and p["away"] in NL)
    return "intraleague" if same_league else "interleague"


def month_seg(p):
    return p["date"][:7]


def fav_underdog_bet(p):
    """Is the model BETTING a market underdog (model likes a dog) or a market favorite?"""
    model_home = p["winp"] >= 0.5
    ml = p["home_ml"] if model_home else p["away_ml"]
    return "bet_dog(+)" if ml > 0 else "bet_fav(-)"


SEGMENTS = {
    "confidence_band": conf_band,
    "model_side": fav_dog,
    "home_away_pick": lambda p: "pick_home" if p["winp"] >= 0.5 else "pick_away",
    "bet_fav_vs_dog": fav_underdog_bet,
    "model_vs_market_gap": disagreement,
    "division_vs_interleague": div_inter,
    "day_night": lambda p: p.get("dayNight") or "unknown",
    "month": month_seg,
    "roof": lambda p: "roofed" if p.get("roof") else "outdoor",
    "sp_hand_matchup": lambda p: f"away_sp_{p.get('away_sp_hand') or '?'}_home_sp_{p.get('home_sp_hand') or '?'}",
    "home_pen_fatigue": lambda p: ("home_pen_tired" if (p.get("home_pen_fat") or 0) > 1.2
                                   else "home_pen_rested" if (p.get("home_pen_fat") or 0) < 0.6 else "home_pen_mid"),
}


def report():
    odds = load_odds()
    preds = get_preds()
    joined = join_market(preds, odds)
    print(f"\n{'='*92}")
    print("GOAL 2 -- SEGMENT EDGE ANALYSIS vs REAL DE-VIGGED CLOSING MARKET")
    print(f"Model preds: {len(preds)}   |   with closing odds joined: {len(joined)} "
          f"(2024={sum(1 for p in joined if p['season']==2024)}, 2025={sum(1 for p in joined if p['season']==2025)})")
    print("Bet = $1 flat on the MODEL's pick at the actual CLOSING price. ROI/CLV vs that close.")
    print(f"{'='*92}")

    overall = seg_stats(joined)
    print(f"\nOVERALL (all odds-covered games):")
    print(f"  N={overall['N']}  model SU {overall['model_hit']:.2f}%  market SU {overall['mkt_hit']:.2f}%"
          f"  flat-ROI {overall['roi']:+.2f}%  mean CLV {overall['clv_pp']:+.2f}pp")

    # Collect candidate segments (N>=100 in full sample) ranked by ROI, with held-out check.
    candidates = []
    for seg_name, fnc in SEGMENTS.items():
        print(f"\n--- {seg_name} ---")
        groups = collections.defaultdict(list)
        for p in joined:
            lbl = fnc(p)
            if lbl is not None:
                groups[lbl].append(p)
        hdr = f"  {'bucket':<26}{'N':>6}{'mdlSU':>8}{'mktSU':>8}{'edge':>8}{'ROI%':>9}{'CLVpp':>8}"
        print(hdr)
        for lbl in sorted(groups, key=lambda k: -seg_stats(groups[k])["roi"]):
            st = seg_stats(groups[lbl])
            print(f"  {lbl:<26}{st['N']:>6}{st['model_hit']:>7.1f}%{st['mkt_hit']:>7.1f}%"
                  f"{st['edge_pp']:>7.1f}{st['roi']:>+8.2f}%{st['clv_pp']:>+7.1f}")
            if st["N"] >= 100 and st["roi"] > 0:
                s24, s25 = by_season(groups[lbl])
                candidates.append({"seg": seg_name, "bucket": lbl, **st,
                                   "roi24": s24["roi"] if s24 and s24["N"] >= 25 else None,
                                   "n24": s24["N"] if s24 else 0,
                                   "roi25": s25["roi"] if s25 and s25["N"] >= 25 else None,
                                   "n25": s25["N"] if s25 else 0})

    # ---- ranked deliverable: positive-ROI, N>=100, held-out-confirmed ----
    print(f"\n{'='*92}")
    print("RANKED 'BET HERE' CANDIDATES (full-sample N>=100, positive flat-ROI at the close)")
    print("Held-out confirmation: ROI must also be POSITIVE in BOTH 2024 and 2025 (each N>=25).")
    print(f"{'='*92}")
    print(f"  {'segment / bucket':<42}{'N':>6}{'ROI%':>9}{'CLVpp':>8}{'2024 ROI':>11}{'2025 ROI':>11}  conf")
    candidates.sort(key=lambda c: -c["roi"])
    confirmed = []
    for c in candidates:
        ok = (c["roi24"] is not None and c["roi25"] is not None
              and c["roi24"] > 0 and c["roi25"] > 0)
        r24 = f"{c['roi24']:+.1f}%({c['n24']})" if c["roi24"] is not None else f"n/a({c['n24']})"
        r25 = f"{c['roi25']:+.1f}%({c['n25']})" if c["roi25"] is not None else f"n/a({c['n25']})"
        tag = "CONFIRMED" if ok else ""
        print(f"  {c['seg']+': '+c['bucket']:<42}{c['N']:>6}{c['roi']:>+8.2f}%{c['clv_pp']:>+7.1f}"
              f"{r24:>11}{r25:>11}  {tag}")
        if ok:
            confirmed.append(c)

    print(f"\n{'='*92}")
    print(f"HELD-OUT-CONFIRMED EDGES (positive ROI full-sample AND both seasons, N>=100): "
          f"{len(confirmed)}")
    for c in confirmed:
        print(f"  * {c['seg']}={c['bucket']}: N={c['N']}, ROI {c['roi']:+.2f}%, CLV {c['clv_pp']:+.2f}pp, "
              f"model SU {c['model_hit']:.1f}% vs market {c['mkt_hit']:.1f}% "
              f"(2024 {c['roi24']:+.1f}% / 2025 {c['roi25']:+.1f}%)")
    print(f"{'='*92}")
    return confirmed


if __name__ == "__main__":
    report()
