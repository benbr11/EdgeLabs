# -*- coding: utf-8 -*-
"""
ufc_backtest.py
===============
HONEST walk-forward (out-of-sample) backtest for the UFC model in ufc_model.py.

The ONLY meaningful accuracy number is on fights the model has NOT seen.  This script
processes the fight log chronologically and, for each TEST fight, predicts it using
ONLY information available BEFORE that fight's date.  No look-ahead.

Three prediction tiers are reported, labelled honestly:

  (1) ELO-ONLY  -- fully clean out-of-sample floor.
      Pre-fight Elo (the rating going INTO the bout, before it is updated) is captured
      by REPLICATING ufc_model.compute_elo's exact update loop (same K / provisional /
      opponent-quality / robbery / dominance logic) and snapshotting both fighters'
      ratings just before each bout updates them.  p = Elo expectation, temperature-
      scaled by the model's LOGIT_TEMP on the Elo logit only.

  (2) FULL MODEL, AS-OF  -- clean(ish): every feature recomputed as-of the cutoff.
      * Pre-fight Elo as in (1).
      * Striking / grappling RATE stats (SLpM, SApM, str_acc/def, td_*, ctrl, kd, subs)
        re-aggregated from raw_fight_stats.csv over ONLY the fighter's bouts BEFORE the
        cutoff, then run through the SAME hygiene the model uses (hard caps + small-sample
        shrinkage toward the as-of division mean).
      * finish_rate / gets_finished_rate / pct_distance recomputed from the fight log
        before the cutoff (the model's recompute_finish_stats logic, as-of).
      * age and layoff_days computed from DOB + prior-fight dates as-of.
      * Time-invariant traits (reach, height, stance, style) taken from ufc_fighters.csv.
      Fed through the real ufc_model.win_probability via a patched stat dict.

  (3) FULL MODEL, CURRENT STATS  -- LEAKAGE-FLAGGED optimistic ceiling.
      Pre-fight Elo (clean) but career-aggregate stats read from the CURRENT
      ufc_fighters.csv (computed over ALL fights, incl. the test fight + future fights).
      This is look-ahead on the stat profile and is reported only as a contrast.

TEST SET: bouts on/after 2024-01-01 where BOTH fighters had >=3 prior UFC fights in the
log and both are in the model.  A separate evaluation covers the subset overlapping
raw_calibration_odds.csv (closing de-vigged book odds), incl. a vs-market comparison and
flat-stake ROI.

Run:
    python ufc_backtest.py
"""
import os
import sys
import math
import datetime as dt

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import numpy as np
import pandas as pd

import ufc_model as M

BASE = os.path.dirname(os.path.abspath(__file__))
TEST_START = pd.Timestamp("2024-01-01")
MIN_PRIOR_FIGHTS = 3


# --------------------------------------------------------------------------- #
#  Helpers for the as-of striking aggregation
# --------------------------------------------------------------------------- #
def _parse_landed(x):
    """'23 of 38' -> (23, 38).  Robust to '---', NaN."""
    s = str(x)
    if " of " in s:
        a, b = s.split(" of ")
        try:
            return float(a), float(b)
        except ValueError:
            return np.nan, np.nan
    return np.nan, np.nan


def _parse_ctrl(x):
    """'1:44' -> 104.0 seconds.  '---'/NaN -> 0.0 (no control logged)."""
    s = str(x)
    if ":" in s:
        try:
            m, sec = s.split(":")
            return float(m) * 60.0 + float(sec)
        except ValueError:
            return 0.0
    return 0.0


def _parse_pct(x):
    s = str(x).replace("%", "").strip()
    try:
        return float(s) / 100.0
    except ValueError:
        return np.nan


def load_raw_fight_stats_with_dates():
    """
    Return a long DataFrame of per-(fighter, bout) aggregated round stats with the bout
    DATE attached, so we can re-aggregate a fighter's profile AS-OF any cutoff.

    Aggregation is per bout (sum the rounds), tracking:
      sig landed/att, total time (sec), KD, TD landed/att, sub att, ctrl sec,
      str absorbed (opponent's sig landed), rounds.
    """
    st = pd.read_csv(os.path.join(BASE, "raw_fight_stats.csv"))
    ev = pd.read_csv(os.path.join(BASE, "raw_events.csv"))[["EVENT", "DATE"]]
    ev["DATE"] = pd.to_datetime(ev["DATE"], errors="coerce")
    st = st.merge(ev, on="EVENT", how="left")

    # per-round numeric parse
    sig = st["SIG.STR."].map(_parse_landed)
    st["sig_l"] = [t[0] for t in sig]
    st["sig_a"] = [t[1] for t in sig]
    td = st["TD"].map(_parse_landed)
    st["td_l"] = [t[0] for t in td]
    st["td_a"] = [t[1] for t in td]
    st["ctrl_s"] = st["CTRL"].map(_parse_ctrl)
    st["kd"] = pd.to_numeric(st["KD"], errors="coerce").fillna(0.0)
    st["subatt"] = pd.to_numeric(st["SUB.ATT"], errors="coerce").fillna(0.0)

    # opponent sig landed per round (for str absorbed / SApM + str_def)
    # within a (EVENT, BOUT, ROUND) there are exactly two fighter rows.
    opp_sig = (st.groupby(["EVENT", "BOUT", "ROUND"])["sig_l"]
                 .transform("sum")) - st["sig_l"]
    opp_siga = (st.groupby(["EVENT", "BOUT", "ROUND"])["sig_a"]
                  .transform("sum")) - st["sig_a"]
    st["opp_sig_l"] = opp_sig
    st["opp_sig_a"] = opp_siga
    st["one"] = 1.0

    agg = st.groupby(["FIGHTER", "BOUT", "EVENT", "DATE"]).agg(
        sig_l=("sig_l", "sum"), sig_a=("sig_a", "sum"),
        td_l=("td_l", "sum"), td_a=("td_a", "sum"),
        ctrl_s=("ctrl_s", "sum"), kd=("kd", "sum"), subatt=("subatt", "sum"),
        opp_sig_l=("opp_sig_l", "sum"), opp_sig_a=("opp_sig_a", "sum"),
        rounds=("one", "sum"),
    ).reset_index()
    agg["fighter_l"] = agg["FIGHTER"].astype(str).str.strip().str.lower()
    agg = agg.dropna(subset=["DATE"])
    return agg


def asof_strike_stats(fs_long, fighter_name, cutoff):
    """
    Aggregate a fighter's striking/grappling RATE profile from raw_fight_stats over only
    their bouts strictly BEFORE `cutoff`.  Mirrors the model's per-15 / per-minute / per-
    round rate definitions.  Returns a dict of raw (un-shrunk, un-capped) rate stats plus
    an effective fight count n, or None if no prior stat'd bouts.
    """
    key = str(fighter_name).strip().lower()
    g = fs_long[(fs_long["fighter_l"] == key) & (fs_long["DATE"] < cutoff)]
    if len(g) == 0:
        return None
    n = len(g)
    total_min = g["rounds"].sum() * 5.0            # round = 5 min (approx; matches per-15/per-min convention)
    total_rounds = g["rounds"].sum()
    if total_min <= 0:
        return None
    sig_l = g["sig_l"].sum()
    sig_a = g["sig_a"].sum()
    opp_sig_l = g["opp_sig_l"].sum()
    opp_sig_a = g["opp_sig_a"].sum()
    td_l = g["td_l"].sum()
    td_a = g["td_a"].sum()
    kd = g["kd"].sum()
    subatt = g["subatt"].sum()
    ctrl_s = g["ctrl_s"].sum()

    out = {
        "SLpM": sig_l / total_min,
        "SApM": opp_sig_l / total_min,
        "str_acc": (sig_l / sig_a) if sig_a > 0 else np.nan,
        "str_def": (1.0 - opp_sig_l / opp_sig_a) if opp_sig_a > 0 else np.nan,
        "td_per15": td_l / total_min * 15.0,
        "td_att_per_round": td_a / total_rounds,
        "td_acc": (td_l / td_a) if td_a > 0 else np.nan,
        "kd_per15": kd / total_min * 15.0,
        "sub_att_per15": subatt / total_min * 15.0,
        "ctrl_sec_per_round": ctrl_s / total_rounds,
        "_n_stat": n,
    }
    return out


def load_dob():
    tott = pd.read_csv(os.path.join(BASE, "raw_fighter_tott.csv"))[["FIGHTER", "DOB"]]
    tott["DOB"] = pd.to_datetime(tott["DOB"], errors="coerce")
    tott["fighter_l"] = tott["FIGHTER"].astype(str).str.strip().str.lower()
    return dict(zip(tott["fighter_l"], tott["DOB"]))


# --------------------------------------------------------------------------- #
#  Replicate compute_elo's update loop, snapshotting PRE-fight Elo per bout
# --------------------------------------------------------------------------- #
def elo_walk_forward(log):
    """
    Replicate ufc_model.compute_elo EXACTLY, but record, for every processed bout, the
    PRE-fight Elo of both corners (the rating going INTO the bout, before the update).

    Returns:
      pre_elo : dict keyed by (sorted_pair, date_ordinal) -> {fighter_name: pre_fight_elo}
      pre_n   : same key -> {fighter_name: prior_ufc_fight_count}
      final   : final {fighter: elo} (sanity-check vs compute_elo)
    """
    elo = {}
    nfights = {}

    log = log.sort_values("date").reset_index(drop=True)
    today = log["date"].max()

    # index every row by bout key so we can read the other corner (robbery logic)
    bout_rows = {}
    for _, rr in log.iterrows():
        o = rr["opponent"]
        if not isinstance(o, str) or o in ("", "nan"):
            continue
        bk = tuple(sorted([rr["fighter"], o])) + (rr["date"].toordinal(),)
        bout_rows.setdefault(bk, {})[rr["fighter"]] = rr

    # ---- the model's exact inner functions (copied verbatim in logic) ----
    def is_genuine_robbery(my_row, opp_row, my_result):
        if my_result != "L":
            return 0.0
        dtype = str(my_row.get("decision_type", "") or "")
        method = str(my_row.get("method", "") or "")
        is_close_dec = (dtype in ("Split", "Majority")) or ("Split" in method) or ("Majority" in method)
        if not is_close_dec:
            return 0.0
        my_margin = my_row.get("score_margin", np.nan)
        if pd.notna(my_margin) and float(my_margin) < -1.0:
            return 0.0
        strength = 0.9 if (dtype == "Split" or "Split" in method) else 0.5
        return strength

    def effective_score(my_row, opp_row, my_result, dom, controversy, my_elo, opp_elo):
        sa = 0.5 if my_result == "D" else (1.0 if my_result == "W" else 0.0)
        robbery = is_genuine_robbery(my_row, opp_row, my_result) if my_result == "L" else 0.0
        oq = max(0.0, min(1.0, (opp_elo - my_elo) / 150.0))
        if my_result == "W":
            perf = 0.5 + 0.5 * dom
            blend = 0.25
        elif my_result == "L":
            if robbery > 0:
                perf = 0.5 + 0.35 * robbery
                blend = 0.25 + 0.65 * robbery
            else:
                opp_dom = M.dominance_score(opp_row)[0] if opp_row is not None else 0.55
                comp = 0.12 * (1.0 - opp_dom)
                cushion = 0.55 * oq
                perf = comp + cushion
                blend = 0.25 + 0.55 * oq
        else:
            perf = 0.5
            blend = 0.25
        return (1 - blend) * sa + blend * perf

    pre_elo = {}
    pre_n = {}

    seen = set()
    for idx, row in log.iterrows():
        f, opp, date = row["fighter"], row["opponent"], row["date"]
        if not isinstance(opp, str) or opp in ("", "nan"):
            continue
        key = tuple(sorted([f, opp])) + (date.toordinal(),)
        if key in seen:
            continue
        seen.add(key)

        result = str(row["result"])
        if result == "NC" or str(row.get("method")) == "Overturned":
            continue

        ra = elo.get(f, M.START_ELO)
        rb = elo.get(opp, M.START_ELO)

        # ---- SNAPSHOT pre-fight Elo + prior fight counts (clean, out-of-sample) ----
        pre_elo[key] = {f: ra, opp: rb}
        pre_n[key] = {f: nfights.get(f, 0), opp: nfights.get(opp, 0)}

        opp_row = bout_rows.get(key, {}).get(opp)
        opp_result = str(opp_row["result"]) if opp_row is not None else (
            "L" if result == "W" else ("W" if result == "L" else "D"))

        dom_f, con_f = M.dominance_score(row)
        dom_o, con_o = (M.dominance_score(opp_row) if opp_row is not None else (dom_f, con_f))
        controversy = max(con_f, con_o)

        eff_f = effective_score(row, opp_row, result, dom_f, controversy, ra, rb)
        eff_o = effective_score(opp_row, row, opp_result, dom_o, controversy, rb, ra) if opp_row is not None else (1 - eff_f)

        ea = M._expected(ra, rb)
        eb = 1 - ea

        age_days = (today - date).days
        k_rec = 0.65 + 0.35 * math.exp(-age_days / (365.0 * 3.5))

        dom_night = max(dom_f, dom_o)

        def sched_q(opp_rating):
            return max(0.0, min(1.0, (opp_rating - 1420.0) / 230.0))
        q_f = sched_q(rb)
        q_o = sched_q(ra)
        k_dom_f = 0.75 + 0.85 * dom_night * (0.55 + 0.45 * q_f)
        k_dom_o = 0.75 + 0.85 * dom_night * (0.55 + 0.45 * q_o)

        def k_provisional(name):
            n = nfights.get(name, 0)
            if n >= M.PROVISIONAL_FIGHTS:
                return M.BASE_K
            frac = n / float(M.PROVISIONAL_FIGHTS)
            return M.PROVISIONAL_K + (M.BASE_K - M.PROVISIONAL_K) * frac

        def k_for(name, k_dom):
            return k_provisional(name) * k_dom * k_rec

        delta_f = k_for(f, k_dom_f) * (1.0 + 0.9 * abs(eff_f - ea)) * (eff_f - ea)
        delta_o = k_for(opp, k_dom_o) * (1.0 + 0.9 * abs(eff_o - eb)) * (eff_o - eb)

        elo[f] = ra + delta_f
        elo[opp] = rb + delta_o

        nfights[f] = nfights.get(f, 0) + 1
        nfights[opp] = nfights.get(opp, 0) + 1

    return pre_elo, pre_n, elo


# --------------------------------------------------------------------------- #
#  As-of finish-tendency stats (log-derived, before cutoff)
# --------------------------------------------------------------------------- #
def asof_finish_stats(log, fighter_name, cutoff):
    """recompute_finish_stats logic, restricted to a fighter's bouts before cutoff."""
    key = str(fighter_name).strip()
    g = log[(log["fighter"].astype(str).str.strip() == key) & (log["date"] < cutoff)].copy()
    if len(g) == 0:
        return None
    g["_cls"] = g["method"].map(M.classify_method)
    g["_res"] = g["result"].astype(str).str.upper().str[0]
    n = len(g)
    is_fin = g["_cls"].isin(["KO", "SUB"])
    fin_win = int((is_fin & (g["_res"] == "W")).sum())
    fin_loss = int((is_fin & (g["_res"] == "L")).sum())
    n_dec = int((g["_cls"] == "DEC").sum())
    return {
        "finish_rate": fin_win / n,
        "gets_finished_rate": fin_loss / n,
        "decision_rate": n_dec / n,
        "pct_distance": n_dec / n,
        "ko_tko_wins": fin_win,    # crude proxy used by method/striker index
        "_n": n,
    }


# --------------------------------------------------------------------------- #
#  Scoring helpers
# --------------------------------------------------------------------------- #
def _safe_log(p):
    return math.log(min(1 - 1e-12, max(1e-12, p)))


def score_set(preds):
    """
    preds: list of (p_model_winner_side, y, ...) where p is model's P(side A wins) and
    y is 1 if A actually won.  Returns hit-rate, log-loss, Brier, N.
    """
    if not preds:
        return None
    hits, ll, brier = 0, 0.0, 0.0
    n = 0
    for p, y in preds:
        if p == 0.5:
            hits += 0.5         # count a dead pick'em as half (rare)
        else:
            fav_win = (p > 0.5) == (y == 1)
            hits += 1 if fav_win else 0
        ll += -(y * _safe_log(p) + (1 - y) * _safe_log(1 - p))
        brier += (p - y) ** 2
        n += 1
    return hits / n, ll / n, brier / n, n


def calibration_table(preds):
    """preds: list of (p_fav, won) where p_fav = max(p,1-p) (model's confidence on its
    pick) and won = 1 if the model's pick won.  Buckets by confidence."""
    buckets = [(0.50, 0.60), (0.60, 0.70), (0.70, 0.80), (0.80, 1.001)]
    rows = []
    for lo, hi in buckets:
        sel = [(p, w) for p, w in preds if lo <= p < hi]
        if not sel:
            rows.append((f"{int(lo*100)}-{int(hi*100) if hi<=1 else 100}", 0, np.nan, np.nan))
            continue
        pred_mean = np.mean([p for p, _ in sel])
        act = np.mean([w for _, w in sel])
        rows.append((f"{int(lo*100)}-{min(100,int(round(hi*100)))}", len(sel), pred_mean, act))
    return rows


# --------------------------------------------------------------------------- #
#  Build the as-of stat dict for a fighter (used by tier 2)
# --------------------------------------------------------------------------- #
def build_asof_statdict(fighters, fs_long, dob_map, log, name, cutoff,
                        asof_div_means):
    """
    Construct a get_stats-style dict for `name` AS-OF `cutoff`:
      - rate stats from raw_fight_stats before cutoff (+ caps + as-of shrinkage)
      - finish stats from log before cutoff
      - age/layoff from DOB + prior fight dates
      - time-invariant traits (reach/height/stance/style/division) from fighters.csv
    Falls back to STAT_DEFAULTS for anything unavailable.  Returns None if the fighter
    is not in the model at all.
    """
    base = M.get_stats(fighters, name)
    if base is None:
        return None
    s = dict(base)                      # start from current (gives traits + defaults)
    div_code = s.get("division_code")

    # --- rate stats as-of (raw), then cap + shrink toward as-of division mean ---
    raw = asof_strike_stats(fs_long, name, cutoff)
    if raw is not None:
        n = float(raw["_n_stat"])
        for col in ["SLpM", "SApM", "str_acc", "str_def", "td_per15", "td_att_per_round",
                    "td_acc", "kd_per15", "sub_att_per15", "ctrl_sec_per_round"]:
            val = raw.get(col, np.nan)
            if val is None or (isinstance(val, float) and not np.isfinite(val)):
                continue
            # hard caps (same as model)
            cap = M.RATE_CAPS.get(col)
            if cap is not None:
                val = min(cap, max(0.0, val))
            # small-sample shrinkage toward as-of division mean (same form as model)
            if col in M.SHRINK_STATS:
                dmean = asof_div_means.get((div_code, col))
                if dmean is None or not np.isfinite(dmean):
                    dmean = M.STAT_DEFAULTS.get(col, val)
                val = (n * val + M.SHRINK_K * dmean) / (n + M.SHRINK_K)
            s[col] = val

    # --- finish stats as-of (log) ---
    fin = asof_finish_stats(log, name, cutoff)
    if fin is not None:
        for col in ["finish_rate", "gets_finished_rate", "decision_rate", "pct_distance"]:
            s[col] = fin[col]
        # ko_tko_wins proxy used by striker_index/method; keep as count
        s["ko_tko_wins"] = max(s.get("ko_tko_wins", 1.0), float(fin["ko_tko_wins"]))

    # --- age as-of ---
    dob = dob_map.get(str(name).strip().lower())
    if dob is not None and pd.notna(dob):
        s["age"] = (cutoff - dob).days / 365.25

    # --- layoff as-of (days since most recent prior fight) ---
    key = str(name).strip()
    prior = log[(log["fighter"].astype(str).str.strip() == key) & (log["date"] < cutoff)]
    if len(prior) > 0:
        last = prior["date"].max()
        s["layoff_days"] = (cutoff - last).days

    return s


def compute_asof_division_means(fighters, fs_long, cutoff):
    """
    Division means of the SHRINK_STATS computed AS-OF the cutoff, over fighters with >=3
    prior stat'd bouts.  Mirrors the model's division-mean shrinkage target (computed on
    reasonably-sampled fighters).  Cached per cutoff by the caller.
    """
    div_of = dict(zip(fighters["fighter"].astype(str).str.strip().str.lower(),
                      fighters["division_code"]))
    # restrict raw stats to before cutoff
    g = fs_long[fs_long["DATE"] < cutoff]
    if len(g) == 0:
        return {}
    # per fighter, prior aggregates
    means = {}
    # build per-fighter rate values
    recs = {}
    for name_l, sub in g.groupby("fighter_l"):
        n = len(sub)
        if n < 3:
            continue
        total_min = sub["rounds"].sum() * 5.0
        total_rounds = sub["rounds"].sum()
        if total_min <= 0:
            continue
        dc = div_of.get(name_l)
        if dc is None:
            continue
        rates = {
            "SLpM": sub["sig_l"].sum() / total_min,
            "SApM": sub["opp_sig_l"].sum() / total_min,
            "td_per15": sub["td_l"].sum() / total_min * 15.0,
            "td_att_per_round": sub["td_a"].sum() / total_rounds,
            "kd_per15": sub["kd"].sum() / total_min * 15.0,
            "sub_att_per15": sub["subatt"].sum() / total_min * 15.0,
            "ctrl_sec_per_round": sub["ctrl_s"].sum() / total_rounds,
        }
        for col, v in rates.items():
            cap = M.RATE_CAPS.get(col)
            if cap is not None:
                v = min(cap, max(0.0, v))
            recs.setdefault((dc, col), []).append(v)
    for k, vals in recs.items():
        means[k] = float(np.mean(vals))
    return means


# --------------------------------------------------------------------------- #
#  MAIN
# --------------------------------------------------------------------------- #
def main():
    print("Loading data ...")
    fighters = pd.read_csv(os.path.join(BASE, "ufc_fighters.csv"))
    log = pd.read_csv(os.path.join(BASE, "ufc_fight_log.csv"), parse_dates=["date"])
    log = log.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)

    model_names = set(fighters["fighter"].astype(str).str.strip().str.lower())

    # --- clean pre-fight Elo for every bout (replicated compute_elo) ---
    print("Walking Elo forward (capturing pre-fight ratings) ...")
    pre_elo, pre_n, final_elo = elo_walk_forward(log)

    # sanity: replicated final Elo should match compute_elo's
    ref_elo = M.compute_elo(log, fighters)
    diffs = [abs(final_elo.get(k, M.START_ELO) - v) for k, v in ref_elo.items()]
    print(f"  Elo replication check: max |diff| vs compute_elo = {max(diffs):.6f} "
          f"(over {len(diffs)} fighters)  [should be ~0]")

    # --- as-of striking stats source ---
    print("Loading per-fight stats for as-of features ...")
    fs_long = load_raw_fight_stats_with_dates()
    dob_map = load_dob()

    # --- assemble the TEST SET (one row per bout) ---
    # iterate bouts in chronological order using the pre_elo keys; pick the canonical
    # orientation A=fighter (the row's own fighter) using the log row that has result.
    # We need a single result-bearing row per bout to know who won.
    log_sorted = log.sort_values("date").reset_index(drop=True)
    bout_seen = set()

    # closing odds map (de-vigged) keyed by (sorted pair, date) -> dict
    odds = pd.read_csv(os.path.join(BASE, "raw_calibration_odds.csv"), parse_dates=["date"])
    odds_map = {}
    for _, r in odds.iterrows():
        k = tuple(sorted([str(r["fighter_a"]).strip(), str(r["fighter_b"]).strip()])) + (r["date"].toordinal(),)
        odds_map[k] = r

    # cache as-of division means per cutoff date (expensive)
    div_mean_cache = {}

    tests = []   # each: dict with everything we need to score
    for _, row in log_sorted.iterrows():
        f, opp, date = str(row["fighter"]).strip(), str(row["opponent"]).strip(), row["date"]
        if opp in ("", "nan") or pd.isna(date):
            continue
        result = str(row["result"])
        if result not in ("W", "L"):     # only decisive bouts with a clear winner
            continue
        key = tuple(sorted([f, opp])) + (date.toordinal(),)
        if key in bout_seen:
            continue
        bout_seen.add(key)
        if date < TEST_START:
            continue
        if key not in pre_elo:
            continue
        # both in model?
        if f.lower() not in model_names or opp.lower() not in model_names:
            continue
        # both >=3 prior UFC fights?
        nA = pre_n[key].get(f, 0)
        nB = pre_n[key].get(opp, 0)
        if nA < MIN_PRIOR_FIGHTS or nB < MIN_PRIOR_FIGHTS:
            continue

        # Orientation: A = the result-bearing row's fighter, y=1 if A won.
        winner = f if result == "W" else opp
        A, B = f, opp
        y = 1 if winner == A else 0
        tests.append({
            "key": key, "date": date, "A": A, "B": B, "y": y,
            "preA": pre_elo[key][A], "preB": pre_elo[key][B],
            "nA": nA, "nB": nB,
            "div": row.get("division_code"),
        })

    print(f"\nTEST SET: {len(tests)} bouts (>= {TEST_START.date()}, both fighters "
          f">= {MIN_PRIOR_FIGHTS} prior UFC fights, both in model)")

    # ---------------------------------------------------------------------- #
    #  Produce predictions for the three tiers
    # ---------------------------------------------------------------------- #
    elo_preds = []        # (p_fav, won)  + parallel arrays for log-loss
    elo_scor = []         # (p_A, y)
    asof_scor = []
    full_scor = []
    elo_cal, asof_cal, full_cal = [], [], []
    per_test = []         # keep details for division/style breakdowns + odds subset

    TEMP = M.LOGIT_TEMP

    for i, t in enumerate(tests):
        A, B, y = t["A"], t["B"], t["y"]
        cutoff = t["date"]

        # ---- tier 1: ELO-ONLY (clean) ----
        ra, rb = t["preA"], t["preB"]
        p_elo = M._expected(ra, rb)
        logit_elo = math.log(p_elo / (1 - p_elo)) if 0 < p_elo < 1 else 0.0
        p_elo_only = 1.0 / (1.0 + math.exp(-(TEMP * (1.33 * logit_elo))))
        elo_scor.append((p_elo_only, y))

        # ---- tier 3: FULL MODEL, CURRENT STATS (leakage-flagged) ----
        # uses clean pre-fight Elo but CURRENT career-aggregate stats from fighters.csv
        a_cur = M.get_stats(fighters, A)
        b_cur = M.get_stats(fighters, B)
        p_full_cur = None
        if a_cur is not None and b_cur is not None:
            m = M.matchup(a_cur, b_cur)
            sit = M._situational_edge(a_cur, b_cur)
            logit = TEMP * (1.33 * logit_elo + 1.12 * m["edge"] + 1.38 * sit)
            p_full_cur = 1.0 / (1.0 + math.exp(-logit))
            full_scor.append((p_full_cur, y))

        # ---- tier 2: FULL MODEL, AS-OF stats (clean) ----
        if cutoff not in div_mean_cache:
            div_mean_cache[cutoff] = compute_asof_division_means(fighters, fs_long, cutoff)
        dmeans = div_mean_cache[cutoff]
        a_asof = build_asof_statdict(fighters, fs_long, dob_map, log, A, cutoff, dmeans)
        b_asof = build_asof_statdict(fighters, fs_long, dob_map, log, B, cutoff, dmeans)
        p_asof = None
        if a_asof is not None and b_asof is not None:
            m2 = M.matchup(a_asof, b_asof)
            sit2 = M._situational_edge(a_asof, b_asof)
            logit2 = TEMP * (1.33 * logit_elo + 1.12 * m2["edge"] + 1.38 * sit2)
            p_asof = 1.0 / (1.0 + math.exp(-logit2))
            asof_scor.append((p_asof, y))

        # calibration (confidence on the model's PICK)
        def pick_conf(p):
            return (max(p, 1 - p), 1 if ((p > 0.5) == (y == 1)) else 0)
        elo_cal.append(pick_conf(p_elo_only))
        if p_asof is not None:
            asof_cal.append(pick_conf(p_asof))
        if p_full_cur is not None:
            full_cal.append(pick_conf(p_full_cur))

        # odds overlap
        orow = odds_map.get(t["key"])
        per_test.append({
            **t,
            "p_elo": p_elo_only, "p_asof": p_asof, "p_full": p_full_cur,
            "odds": orow,
        })

        if (i + 1) % 200 == 0:
            print(f"  ... {i+1}/{len(tests)} bouts predicted")

    # ---------------------------------------------------------------------- #
    #  REPORT: overall metrics for each tier
    # ---------------------------------------------------------------------- #
    print("\n" + "=" * 74)
    print("HEADLINE OUT-OF-SAMPLE RESULTS (test fights the model had NOT seen)")
    print("=" * 74)

    def report_tier(name, scor, cal, leakage=False):
        r = score_set(scor)
        if r is None:
            print(f"\n{name}: no predictions")
            return
        hit, ll, br, n = r
        tag = "  [LEAKAGE-FLAGGED: stats are current career aggregates]" if leakage else ""
        print(f"\n{name}{tag}")
        print(f"  N = {n}")
        print(f"  straight-up hit-rate : {hit:.4f}  ({hit*100:.1f}%)")
        print(f"  log-loss             : {ll:.4f}")
        print(f"  Brier                : {br:.4f}")
        print(f"  calibration (model confidence on its pick vs actual win-rate of pick):")
        print(f"    {'bucket':>8}  {'N':>5}  {'pred%':>7}  {'actual%':>8}")
        for b, nn, pm, ac in calibration_table(cal):
            if nn == 0:
                print(f"    {b:>8}  {nn:>5}  {'--':>7}  {'--':>8}")
            else:
                print(f"    {b:>8}  {nn:>5}  {pm*100:>6.1f}%  {ac*100:>7.1f}%")

    report_tier("(1) ELO-ONLY  (fully clean out-of-sample FLOOR)", elo_scor, elo_cal)
    report_tier("(2) FULL MODEL, AS-OF features  (clean; proper out-of-sample)", asof_scor, asof_cal)
    report_tier("(3) FULL MODEL, CURRENT stats  (optimistic CEILING)", full_scor, full_cal, leakage=True)

    print("\nBENCHMARKS: coin-flip = 50.0% | sharp closing market hits ~64-66% straight-up")

    # ---------------------------------------------------------------------- #
    #  VS MARKET on the odds-overlap subset
    # ---------------------------------------------------------------------- #
    print("\n" + "=" * 74)
    print("VS MARKET  (subset overlapping raw_calibration_odds.csv, de-vigged closing)")
    print("=" * 74)

    mkt_rows = [pt for pt in per_test if pt["odds"] is not None]
    print(f"  bouts with closing odds AND in test set: {len(mkt_rows)}")

    if mkt_rows:
        model_hits = 0
        book_hits = 0
        n_eval = 0
        ll_model, ll_book = [], []
        # collect per-bet records so we can sweep an EDGE THRESHOLD afterwards
        bet_recs = []   # (edge, dec_odds, won)  for the side the model would back

        def amer_to_dec(o):
            o = float(o)
            return 1 + (o / 100.0 if o > 0 else 100.0 / (-o))

        # choose which model prob to use for vs-market: tier-2 as-of (clean) if available,
        # else tier-1 elo-only.
        for pt in mkt_rows:
            orow = pt["odds"]
            A, B, y = pt["A"], pt["B"], pt["y"]
            fa = str(orow["fighter_a"]).strip()
            fb = str(orow["fighter_b"]).strip()
            # reconstruct de-vigged book P from the two American odds (normalize out vig)
            oa, ob = float(orow["odds_a"]), float(orow["odds_b"])
            raw_a = (1.0 / amer_to_dec(oa))
            raw_b = (1.0 / amer_to_dec(ob))
            dv_a = raw_a / (raw_a + raw_b)
            dv_b = 1 - dv_a
            if fa.lower() == A.lower():
                book_pA, dec_A, dec_B = dv_a, amer_to_dec(oa), amer_to_dec(ob)
            elif fb.lower() == A.lower():
                book_pA, dec_A, dec_B = dv_b, amer_to_dec(ob), amer_to_dec(oa)
            else:
                continue
            book_pB = 1.0 - book_pA

            p_model = pt["p_asof"] if pt["p_asof"] is not None else pt["p_elo"]
            n_eval += 1
            model_pick_A = p_model > 0.5
            book_pick_A = book_pA > 0.5
            model_hits += 1 if (model_pick_A == (y == 1)) else 0
            book_hits += 1 if (book_pick_A == (y == 1)) else 0
            ll_model.append(-(y * _safe_log(p_model) + (1 - y) * _safe_log(1 - p_model)))
            ll_book.append(-(y * _safe_log(book_pA) + (1 - y) * _safe_log(1 - book_pA)))

            # the model's biggest perceived edge is on whichever side it prices above book
            edge_A = p_model - book_pA
            edge_B = (1 - p_model) - book_pB
            if edge_A >= edge_B:
                bet_recs.append((edge_A, dec_A, 1 if y == 1 else 0))
            else:
                bet_recs.append((edge_B, dec_B, 1 if y == 0 else 0))

        if n_eval:
            print(f"\n  evaluable (names matched): {n_eval}")
            print(f"  MODEL favorite hit-rate : {model_hits/n_eval:.4f}  ({model_hits/n_eval*100:.1f}%)")
            print(f"  BOOK  favorite hit-rate : {book_hits/n_eval:.4f}  ({book_hits/n_eval*100:.1f}%)")
            print(f"  model log-loss          : {np.mean(ll_model):.4f}")
            print(f"  book  log-loss          : {np.mean(ll_book):.4f}")

            # Flat-stake ROI at several minimum-edge thresholds (model prob exceeds the
            # de-vigged book prob by at least `thr`).  thr=0 backs every disagreement.
            print(f"\n  FLAT-STAKE ROI by minimum model-vs-book edge (1u/bet, closing odds):")
            print(f"    {'min edge':>8}  {'bets':>5}  {'P/L (u)':>9}  {'ROI':>8}")
            for thr in [0.00, 0.03, 0.05, 0.08, 0.10]:
                sel = [(d, w) for e, d, w in bet_recs if e >= thr]
                if not sel:
                    print(f"    {thr*100:>6.0f}%   {'0':>5}  {'--':>9}  {'--':>8}")
                    continue
                pl = sum((d - 1.0) if w == 1 else -1.0 for d, w in sel)
                print(f"    {thr*100:>6.0f}%   {len(sel):>5}  {pl:>+8.2f}  {pl/len(sel)*100:>+7.1f}%")
            print("  (small N -- ROI here is dominated by variance; sign + magnitude only,")
            print("   not a tradeable estimate.  Model log-loss vs book log-loss is the more")
            print("   reliable read on whether the model adds information over the line.)")

    # ---------------------------------------------------------------------- #
    #  STRONG/WEAK breakdown (by division) on tier-2 as-of
    # ---------------------------------------------------------------------- #
    print("\n" + "=" * 74)
    print("WHERE THE MODEL IS STRONG / WEAK  (tier-2 as-of; by division)")
    print("=" * 74)
    by_div = {}
    for pt in per_test:
        p = pt["p_asof"] if pt["p_asof"] is not None else pt["p_elo"]
        if p is None:
            continue
        dc = pt["div"] or "?"
        won = 1 if ((p > 0.5) == (pt["y"] == 1)) else 0
        by_div.setdefault(dc, []).append(won)
    print(f"  {'division':>10}  {'N':>4}  {'hit%':>6}")
    for dc in sorted(by_div, key=lambda k: -len(by_div[k])):
        v = by_div[dc]
        print(f"  {dc:>10}  {len(v):>4}  {np.mean(v)*100:>5.1f}%")

    print("\nDone.")


if __name__ == "__main__":
    main()
