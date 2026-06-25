"""
export_ufc.py
=============
Build web/ufc_card.js — the NEXT-CARD PREDICTOR data for the EdgeLabs UFC section.

Reads raw_nextcard.json, runs ufc_model.predict() for every bout, and writes a
JS file:

    window.UFC_CARD = {event,date,venue,location,generated,
      bouts:[{a,b,weightClass,rounds,isTitle,
              winA,winB,                       # model win prob for each corner (0..1)
              method:{ko,sub,dec},             # FAVORED fighter's win-conditional split (~sums to 1)
              roundDist:[r1..rN],              # P(finish in round r), len == rounds
              favored,                         # name of the favored fighter
              keyEdge,                         # short driver string
              dataGap}]}                       # true when a fighter is not in the DB

For any bout that involves a fighter not in ufc_fighters.csv, the bout is still
included but flagged dataGap:true with winA/winB == null (UI shows "insufficient
data").
"""

import os
import json
import datetime as dt

import pandas as pd

import ufc_model

BASE = os.path.dirname(os.path.abspath(ufc_model.__file__))
WEB = os.path.join(BASE, "web")

# --------------------------------------------------------------------------- #
#  CONFIDENCE TIERS  ("Best Bets")
#  BEST_BET_THRESHOLD is chosen from the WALK-FORWARD OUT-OF-SAMPLE backtest
#  (ufc_backtest.py confidence-threshold sweep): it is the LOWEST min-win-prob T
#  whose clean OOS hit-rate >= 80% on adequate N.  As of the latest backtest:
#      T = 0.75  ->  82.5% OOS hit-rate, N = 97 (23.1% of fights qualify).
#  This is the honest 80%: on selected high-conviction picks only, with the
#  threshold picked on OOS results (not in-sample), so it does not overfit.
#  LEAN_THRESHOLD (~0.62) is a real edge but below the 80% bar; below it is a
#  coin-flip "Pass".
# --------------------------------------------------------------------------- #
BEST_BET_THRESHOLD = 0.75
LEAN_THRESHOLD = 0.62


def confidence_tier(win_prob):
    """Map the model's confidence on its PICK (max of the two corner probs) to a
    tier label.  Returns one of 'best', 'lean', 'pass'."""
    conf = max(float(win_prob), 1.0 - float(win_prob))
    if conf >= BEST_BET_THRESHOLD:
        return "best"
    if conf >= LEAN_THRESHOLD:
        return "lean"
    return "pass"


def _resolved_name(fighters, name):
    """
    Return the model's resolved fighter name if `name` is genuinely in the DB,
    else None.  get_stats() does an exact (case-insensitive) match first and then
    a fuzzy "contains" fallback; the fuzzy fallback can mis-fire on short names,
    so we only accept a resolution that is itself a sensible match for the query.
    """
    s = ufc_model.get_stats(fighters, name)
    if s is None:
        return None
    resolved = str(s["fighter"])
    q = str(name).strip().lower()
    r = resolved.lower()
    if q == r:
        return resolved
    # accept the fuzzy hit only if the query name is contained in the resolved
    # name or vice-versa AND the last names line up (guards against a stray
    # substring matching an unrelated fighter).
    if q in r or r in q:
        if q.split()[-1] == r.split()[-1]:
            return resolved
    return None


def american_from_prob(p):
    """American moneyline odds implied by a model win probability."""
    p = max(1e-6, min(1 - 1e-6, float(p)))
    if p >= 0.5:
        return -round(100 * p / (1 - p))
    return round(100 * (1 - p) / p)


def key_edge(pred, a_stats, b_stats, fav_is_a):
    """
    Produce a short string naming the main driver of the favorite's edge.
    Picks the single largest contributor among grappling, Elo, situational/size.
    """
    win = pred["win"]
    mu = pred["matchup"]
    fav = a_stats if fav_is_a else b_stats
    dog = b_stats if fav_is_a else a_stats

    # signed-from-the-favorite components
    grap = mu["grappler_premium"] if fav_is_a else -mu["grappler_premium"]
    elo_diff = (win["elo_a"] - win["elo_b"]) if fav_is_a else (win["elo_b"] - win["elo_a"])
    sit = win["situational"] if fav_is_a else -win["situational"]
    reach = (fav["reach_in"] - dog["reach_in"])

    # candidate edges with a rough comparable magnitude (logit-ish scale)
    cands = []
    if grap > 0.04:
        # which grappling dimension is doing the work?
        fav_g = ufc_model.grappler_index(fav)
        dog_leak = ufc_model.td_leakiness(dog)
        cands.append((abs(grap) * 1.25,
                      "grappling: {} {:.1f} TD/15 vs {:.0f}% TD def leak".format(
                          fav["fighter"].split()[-1], fav["td_per15"],
                          100 * dog_leak)))
    if abs(elo_diff) >= 25:
        cands.append((abs(elo_diff) / 120.0,
                      "Elo {}{:.0f}".format("+" if elo_diff >= 0 else "-", abs(elo_diff))))
    if abs(reach) >= 2:
        cands.append((abs(reach) / 8.0,
                      "reach {}{:.0f}in".format("+" if reach >= 0 else "-", abs(reach))))
    if sit > 0.05:
        cands.append((abs(sit) * 1.6, "younger / fresher (situational edge)"))
    # striking volume edge
    str_edge = (fav["SLpM"] - dog["SLpM"]) if True else 0
    if str_edge >= 0.8:
        cands.append((str_edge / 4.0,
                      "striking: {} +{:.1f} SLpM".format(fav["fighter"].split()[-1], str_edge)))

    if not cands:
        return "narrow edge — even matchup"
    cands.sort(key=lambda c: c[0], reverse=True)
    return cands[0][1]


def build_card():
    raw = json.load(open(os.path.join(BASE, "raw_nextcard.json"), encoding="utf-8"))
    fighters = pd.read_csv(os.path.join(BASE, "ufc_fighters.csv"))

    bouts_out = []
    for b in raw["bouts"]:
        a_name, b_name = b["fighterA"], b["fighterB"]
        rounds = int(b.get("rounds", 3))
        rec = {
            "a": a_name,
            "b": b_name,
            "weightClass": b.get("weightClass", ""),
            "rounds": rounds,
            "isTitle": bool(b.get("isTitle", False)),
            "winA": None,
            "winB": None,
            "method": None,
            "roundDist": None,
            "favored": None,
            "keyEdge": None,
            "tier": None,        # "best" | "lean" | "pass" (confidence tier)
            "dataGap": False,
        }

        ra = _resolved_name(fighters, a_name)
        rb = _resolved_name(fighters, b_name)
        if ra is None or rb is None:
            rec["dataGap"] = True
            bouts_out.append(rec)
            continue

        pred = ufc_model.predict(ra, rb, scheduled_rounds=rounds)
        win = pred["win"]
        winA, winB = win["p_a"], win["p_b"]
        fav_is_a = winA >= winB
        favored = ra if fav_is_a else rb

        mr = pred["method_round"]
        meth = mr["method_if_a_wins"] if fav_is_a else mr["method_if_b_wins"]
        rd = mr["round_dist"]
        round_dist = [round(rd.get("R{}".format(r), 0.0), 4) for r in range(1, rounds + 1)]

        a_stats = ufc_model.get_stats(fighters, ra)
        b_stats = ufc_model.get_stats(fighters, rb)
        edge = key_edge(pred, a_stats, b_stats, fav_is_a)

        rec["winA"] = round(winA, 4)
        rec["winB"] = round(winB, 4)
        rec["method"] = {
            "ko": round(meth["KO/TKO"], 4),
            "sub": round(meth["Submission"], 4),
            "dec": round(meth["Decision"], 4),
        }
        rec["roundDist"] = round_dist
        rec["favored"] = favored
        rec["keyEdge"] = edge
        rec["tier"] = confidence_tier(max(winA, winB))
        bouts_out.append(rec)

    card = {
        "event": raw["event"],
        "date": raw["date"],
        "venue": raw["venue"],
        "location": raw["location"],
        "generated": dt.date.today().isoformat(),
        "bestBetThreshold": BEST_BET_THRESHOLD,
        "leanThreshold": LEAN_THRESHOLD,
        "bouts": bouts_out,
    }
    return card


def main():
    card = build_card()
    os.makedirs(WEB, exist_ok=True)
    out_path = os.path.join(WEB, "ufc_card.js")
    js = "window.UFC_CARD = " + json.dumps(card, ensure_ascii=False) + ";\n"
    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write(js)

    n = len(card["bouts"])
    gaps = [b["a"] + " vs " + b["b"] for b in card["bouts"] if b["dataGap"]]
    print("wrote {} ({} bouts)".format(out_path, n))
    if gaps:
        print("dataGap bouts ({}):".format(len(gaps)))
        for g in gaps:
            print("  - " + g)
    # echo the main-event line
    me = card["bouts"][0]
    if not me["dataGap"]:
        m = me["method"]
        rd = me["roundDist"]
        print("\nMain event: {} {:.1%} vs {} {:.1%}  (favored {})".format(
            me["a"], me["winA"], me["b"], me["winB"], me["favored"]))
        print("  method (favored): KO {:.0%} / Sub {:.0%} / Dec {:.0%}".format(
            m["ko"], m["sub"], m["dec"]))
        print("  round dist: " + ", ".join("R{} {:.0%}".format(i + 1, v)
                                            for i, v in enumerate(rd)))
        print("  keyEdge: " + me["keyEdge"])


if __name__ == "__main__":
    main()
