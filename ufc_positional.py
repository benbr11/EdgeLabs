"""
ufc_positional.py
=================
Shared positional / striking feature engineering mined from raw_fight_stats.csv.

raw_fight_stats.csv holds, PER FIGHTER PER ROUND, the breakdown that the rest of the
pipeline never used:

    HEAD / BODY / LEG          significant strikes by TARGET   ("landed of attempted")
    DISTANCE / CLINCH / GROUND significant strikes by POSITION ("landed of attempted")
    KD                         knockdowns
    SUB.ATT                    submission attempts
    CTRL                       control time ("m:ss")
    SIG.STR. / TD              overall sig-strike / takedown   ("landed of attempted")

This module turns that into PER-FIGHTER positional features that are computable
AS-OF a cutoff date, so the walk-forward backtest stays leakage-free: every feature
aggregates ONLY a fighter's bouts strictly BEFORE the cutoff.

It is the SINGLE source of truth used by both consumers:

  * ufc_model.merge_sources() / get_stats() / ufc_fighters.csv
        -> attach_positional_features(fighters)  (cutoff = None == "as of today")
  * ufc_backtest.py as-of feature builder
        -> asof_positional_stats(pos_long, name, cutoff)  (cutoff = the test bout date)

Both call paths run through the EXACT SAME aggregation (asof_positional_stats), so a
feature seen in the backtest is computed identically to the one shipped in
ufc_fighters.csv -- only the cutoff differs.  By construction nothing here can look
ahead: load_positional_long() attaches each bout's DATE, and every aggregator filters
DATE < cutoff before summing.
"""

import os
import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))

# Approx minutes per round used for per-15 / per-minute rates (matches the convention
# already used by ufc_backtest.asof_strike_stats: rounds * 5 min).
MIN_PER_ROUND = 5.0

# --------------------------------------------------------------------------- #
#  Canonical positional feature names + the hygiene the model applies to them.
#  These mirror the style of ufc_model.RATE_CAPS / SHRINK_STATS / STAT_DEFAULTS so
#  the new columns get the same small-sample treatment as the legacy rate stats.
# --------------------------------------------------------------------------- #
POSITIONAL_FEATURES = [
    # ---- offensive striking shape ----
    "head_share",            # head sig landed / total sig landed   (head-hunting)
    "head_acc",              # head landed / head attempted
    "ground_strike_per15",   # ground sig landed per 15 min         (ground-and-pound volume)
    "ground_share",          # ground sig landed / total sig landed (how ground-based the offense is)
    "ground_ctrl_share",     # share of fight time spent controlling on the mat
    "clinch_strike_per15",   # clinch sig landed per 15 min
    "clinch_share",          # clinch sig landed / total sig landed (clinch dominance proxy)
    "distance_strike_per15", # distance sig landed per 15 min
    "distance_strike_diff15",# (own - opp) distance sig landed per 15 (distance-striking differential)
    # ---- finishing threat ----
    "kd_per15",              # knockdowns per 15 (KO threat) -- positional re-derivation
    "finish_threat15",       # KD/15 + sub_att/15 (combined finishing threat)
    # ---- defensive mirrors ----
    "head_absorbed_per15",   # opponent head sig landed per 15 (chin / head exposure)
    "head_def",              # 1 - opp head landed / opp head attempted (head-strike defense)
    "kd_absorbed_per15",     # opponent knockdowns per 15 (knocked-down rate / durability)
    "ground_absorbed_per15", # opponent ground sig landed per 15 (g&p absorbed)
]

# Hard physical caps (same spirit as ufc_model.RATE_CAPS) to clip small-sample blowups.
POS_RATE_CAPS = {
    "head_share":             1.0,
    "head_acc":               1.0,
    "ground_strike_per15":    90.0,
    "ground_share":           1.0,
    "ground_ctrl_share":      1.0,
    "clinch_strike_per15":    60.0,
    "clinch_share":           1.0,
    "distance_strike_per15":  120.0,
    "distance_strike_diff15": 120.0,   # signed; clipped to +/- via the helper below
    "kd_per15":               3.0,
    "finish_threat15":        8.0,
    "head_absorbed_per15":    120.0,
    "head_def":               1.0,
    "kd_absorbed_per15":      3.0,
    "ground_absorbed_per15":  90.0,
}

# Rate features that get small-sample shrinkage toward the division mean.  Shares /
# accuracies / defenses are bounded 0..1 ratios and ALSO benefit from shrinkage toward
# a sane divisional prior on tiny samples.
POS_SHRINK_STATS = list(POSITIONAL_FEATURES)
POS_SHRINK_K = 4.0      # matches ufc_model.SHRINK_K

# Division-agnostic fallbacks (rough UFC-wide medians) used when no divisional prior
# exists.  Picked from the population means printed by this module's __main__.
POS_DEFAULTS = {
    "head_share":             0.63,
    "head_acc":               0.36,
    "ground_strike_per15":    6.1,
    "ground_share":           0.13,
    "ground_ctrl_share":      0.18,
    "clinch_strike_per15":    6.2,
    "clinch_share":           0.13,
    "distance_strike_per15":  34.8,
    "distance_strike_diff15": 0.0,
    "kd_per15":               0.27,
    "finish_threat15":        0.75,
    "head_absorbed_per15":    29.7,
    "head_def":               0.64,
    "kd_absorbed_per15":      0.27,
    "ground_absorbed_per15":  6.1,
}


# --------------------------------------------------------------------------- #
#  Parsing helpers (kept byte-compatible with ufc_backtest's private helpers)
# --------------------------------------------------------------------------- #
def parse_landed(x):
    """'23 of 38' -> (23.0, 38.0).  Robust to '---', NaN, blanks."""
    s = str(x)
    if " of " in s:
        a, b = s.split(" of ")
        try:
            return float(a), float(b)
        except ValueError:
            return np.nan, np.nan
    return np.nan, np.nan


def parse_ctrl(x):
    """'1:44' -> 104.0 seconds.  '---'/NaN -> 0.0 (no control logged)."""
    s = str(x)
    if ":" in s:
        try:
            m, sec = s.split(":")
            return float(m) * 60.0 + float(sec)
        except ValueError:
            return 0.0
    return 0.0


# --------------------------------------------------------------------------- #
#  Load per-(fighter, bout) positional aggregates WITH the bout date attached.
#  This is the leakage-free spine: nothing downstream sees a bout without its date.
# --------------------------------------------------------------------------- #
def load_positional_long(stats_path=None, events_path=None):
    """
    Return a long DataFrame: one row per (fighter, bout), columns summed over the bout's
    rounds, with the bout DATE attached so we can re-aggregate AS-OF any cutoff.

    Tracks, per bout (own + opponent mirror):
      head/body/leg landed+attempted, distance/clinch/ground landed+attempted,
      total sig landed+attempted, kd, sub attempts, control seconds, rounds.
    """
    sp = stats_path or os.path.join(BASE, "raw_fight_stats.csv")
    ep = events_path or os.path.join(BASE, "raw_events.csv")
    st = pd.read_csv(sp)
    ev = pd.read_csv(ep)[["EVENT", "DATE"]]
    ev["DATE"] = pd.to_datetime(ev["DATE"], errors="coerce")
    st = st.merge(ev, on="EVENT", how="left")

    # parse the "landed of attempted" pairs we need
    for col, prefix in [("SIG.STR.", "sig"), ("HEAD", "head"), ("BODY", "body"),
                        ("LEG", "leg"), ("DISTANCE", "dist"), ("CLINCH", "clin"),
                        ("GROUND", "grnd")]:
        pr = st[col].map(parse_landed)
        st[prefix + "_l"] = [t[0] for t in pr]
        st[prefix + "_a"] = [t[1] for t in pr]

    st["ctrl_s"] = st["CTRL"].map(parse_ctrl)
    st["kd"] = pd.to_numeric(st["KD"], errors="coerce").fillna(0.0)
    st["subatt"] = pd.to_numeric(st["SUB.ATT"], errors="coerce").fillna(0.0)
    st["one"] = 1.0

    # opponent mirror per round: within a (EVENT, BOUT, ROUND) there are exactly two
    # fighter rows, so (group sum) - (own) == opponent.  Same trick the backtest uses
    # for SApM / str_def, extended to the positional + kd columns.
    grp = st.groupby(["EVENT", "BOUT", "ROUND"])
    for base_col in ["sig_l", "sig_a", "head_l", "head_a", "grnd_l", "kd"]:
        st["opp_" + base_col] = grp[base_col].transform("sum") - st[base_col]

    agg_cols = {
        "sig_l": ("sig_l", "sum"), "sig_a": ("sig_a", "sum"),
        "head_l": ("head_l", "sum"), "head_a": ("head_a", "sum"),
        "body_l": ("body_l", "sum"), "leg_l": ("leg_l", "sum"),
        "dist_l": ("dist_l", "sum"), "clin_l": ("clin_l", "sum"),
        "grnd_l": ("grnd_l", "sum"),
        "ctrl_s": ("ctrl_s", "sum"), "kd": ("kd", "sum"), "subatt": ("subatt", "sum"),
        "opp_sig_l": ("opp_sig_l", "sum"),
        "opp_head_l": ("opp_head_l", "sum"), "opp_head_a": ("opp_head_a", "sum"),
        "opp_grnd_l": ("opp_grnd_l", "sum"), "opp_dist_l": ("opp_dist_l", "sum"),
        "opp_kd": ("opp_kd", "sum"),
        "rounds": ("one", "sum"),
    }
    # opp_dist_l needs its own mirror (not in the loop above) -- add it before agg
    st["opp_dist_l"] = grp["dist_l"].transform("sum") - st["dist_l"]

    agg = st.groupby(["FIGHTER", "BOUT", "EVENT", "DATE"]).agg(**agg_cols).reset_index()
    agg["fighter_l"] = agg["FIGHTER"].astype(str).str.strip().str.lower()
    agg = agg.dropna(subset=["DATE"])
    return agg


# --------------------------------------------------------------------------- #
#  AS-OF positional aggregation -- the single function both consumers call.
# --------------------------------------------------------------------------- #
def asof_positional_stats(pos_long, fighter_name, cutoff=None):
    """
    Aggregate `fighter_name`'s positional feature profile over ONLY their bouts strictly
    BEFORE `cutoff` (cutoff=None -> use all bouts == "as of today").

    Returns a dict of RAW (un-shrunk, un-capped) positional features plus an effective
    fight count "_n_pos", or None if the fighter has no qualifying stat'd bouts.
    """
    key = str(fighter_name).strip().lower()
    g = pos_long[pos_long["fighter_l"] == key]
    if cutoff is not None:
        g = g[g["DATE"] < cutoff]
    if len(g) == 0:
        return None

    n = len(g)
    total_rounds = g["rounds"].sum()
    total_min = total_rounds * MIN_PER_ROUND
    total_sec = total_min * 60.0
    if total_min <= 0:
        return None

    sig_l = g["sig_l"].sum()
    head_l = g["head_l"].sum()
    head_a = g["head_a"].sum()
    grnd_l = g["grnd_l"].sum()
    dist_l = g["dist_l"].sum()
    clin_l = g["clin_l"].sum()
    ctrl_s = g["ctrl_s"].sum()
    kd = g["kd"].sum()
    subatt = g["subatt"].sum()
    opp_head_l = g["opp_head_l"].sum()
    opp_head_a = g["opp_head_a"].sum()
    opp_grnd_l = g["opp_grnd_l"].sum()
    opp_dist_l = g["opp_dist_l"].sum()
    opp_kd = g["opp_kd"].sum()

    per15 = 15.0 / total_min       # multiply a count by this to get a per-15-min rate

    out = {
        # offensive shape
        "head_share":             (head_l / sig_l) if sig_l > 0 else np.nan,
        "head_acc":               (head_l / head_a) if head_a > 0 else np.nan,
        "ground_strike_per15":    grnd_l * per15,
        "ground_share":           (grnd_l / sig_l) if sig_l > 0 else np.nan,
        "ground_ctrl_share":      (ctrl_s / total_sec) if total_sec > 0 else np.nan,
        "clinch_strike_per15":    clin_l * per15,
        "clinch_share":           (clin_l / sig_l) if sig_l > 0 else np.nan,
        "distance_strike_per15":  dist_l * per15,
        "distance_strike_diff15": (dist_l - opp_dist_l) * per15,
        # finishing threat
        "kd_per15":               kd * per15,
        "finish_threat15":        (kd + subatt) * per15,
        # defensive mirrors
        "head_absorbed_per15":    opp_head_l * per15,
        "head_def":               (1.0 - opp_head_l / opp_head_a) if opp_head_a > 0 else np.nan,
        "kd_absorbed_per15":      opp_kd * per15,
        "ground_absorbed_per15":  opp_grnd_l * per15,
        "_n_pos":                 n,
    }
    return out


def _cap(col, val):
    """Apply the hard physical cap for a positional feature (signed for diff stats)."""
    cap = POS_RATE_CAPS.get(col)
    if cap is None or val is None or not np.isfinite(val):
        return val
    if col == "distance_strike_diff15":      # signed feature
        return max(-cap, min(cap, val))
    return max(0.0, min(cap, val))


# --------------------------------------------------------------------------- #
#  As-of division means (shrinkage target) -- mirrors
#  ufc_backtest.compute_asof_division_means but for the positional features.
# --------------------------------------------------------------------------- #
def compute_asof_pos_division_means(pos_long, div_of, cutoff=None, min_fights=3):
    """
    Division means of the positional features, AS-OF cutoff, over fighters with >=
    `min_fights` prior stat'd bouts.  `div_of` maps lowercase fighter name -> division_code.
    Returns {(division_code, feature): mean}.
    """
    g = pos_long if cutoff is None else pos_long[pos_long["DATE"] < cutoff]
    if len(g) == 0:
        return {}
    recs = {}
    for name_l, sub in g.groupby("fighter_l"):
        if len(sub) < min_fights:
            continue
        dc = div_of.get(name_l)
        if dc is None:
            continue
        feats = asof_positional_stats_from_group(sub)
        if feats is None:
            continue
        for col in POSITIONAL_FEATURES:
            v = _cap(col, feats.get(col, np.nan))
            if v is None or not np.isfinite(v):
                continue
            recs.setdefault((dc, col), []).append(v)
    return {k: float(np.mean(v)) for k, v in recs.items()}


def asof_positional_stats_from_group(g):
    """Same math as asof_positional_stats but on an already-filtered group (no name/cutoff)."""
    total_rounds = g["rounds"].sum()
    total_min = total_rounds * MIN_PER_ROUND
    total_sec = total_min * 60.0
    if total_min <= 0:
        return None
    sig_l = g["sig_l"].sum(); head_l = g["head_l"].sum(); head_a = g["head_a"].sum()
    grnd_l = g["grnd_l"].sum(); dist_l = g["dist_l"].sum(); clin_l = g["clin_l"].sum()
    ctrl_s = g["ctrl_s"].sum(); kd = g["kd"].sum(); subatt = g["subatt"].sum()
    opp_head_l = g["opp_head_l"].sum(); opp_head_a = g["opp_head_a"].sum()
    opp_grnd_l = g["opp_grnd_l"].sum(); opp_dist_l = g["opp_dist_l"].sum()
    opp_kd = g["opp_kd"].sum()
    per15 = 15.0 / total_min
    return {
        "head_share":             (head_l / sig_l) if sig_l > 0 else np.nan,
        "head_acc":               (head_l / head_a) if head_a > 0 else np.nan,
        "ground_strike_per15":    grnd_l * per15,
        "ground_share":           (grnd_l / sig_l) if sig_l > 0 else np.nan,
        "ground_ctrl_share":      (ctrl_s / total_sec) if total_sec > 0 else np.nan,
        "clinch_strike_per15":    clin_l * per15,
        "clinch_share":           (clin_l / sig_l) if sig_l > 0 else np.nan,
        "distance_strike_per15":  dist_l * per15,
        "distance_strike_diff15": (dist_l - opp_dist_l) * per15,
        "kd_per15":               kd * per15,
        "finish_threat15":        (kd + subatt) * per15,
        "head_absorbed_per15":    opp_head_l * per15,
        "head_def":               (1.0 - opp_head_l / opp_head_a) if opp_head_a > 0 else np.nan,
        "kd_absorbed_per15":      opp_kd * per15,
        "ground_absorbed_per15":  opp_grnd_l * per15,
        "_n_pos":                 len(g),
    }


def shrink_positional(raw, n, div_means, div_code):
    """
    Apply hard caps + small-sample shrinkage toward the as-of division mean to a RAW
    positional dict, returning {feature: value} ready to store.  Mirrors the model's
    (n*raw + K*mean)/(n+K) shrinkage exactly.
    """
    out = {}
    n = float(n)
    for col in POSITIONAL_FEATURES:
        val = raw.get(col, np.nan)
        val = _cap(col, val)
        if val is None or not np.isfinite(val):
            continue
        if col in POS_SHRINK_STATS:
            dmean = div_means.get((div_code, col))
            if dmean is None or not np.isfinite(dmean):
                dmean = POS_DEFAULTS.get(col, val)
            val = (n * val + POS_SHRINK_K * dmean) / (n + POS_SHRINK_K)
        out[col] = val
    return out


# --------------------------------------------------------------------------- #
#  "Current" attach: join shrunk positional features onto the fighters table.
#  cutoff=None -> as of today.  Used by ufc_model.merge_sources.
# --------------------------------------------------------------------------- #
def attach_positional_features(fighters, cutoff=None, pos_long=None, min_fights=3):
    """
    Return a COPY of `fighters` with the POSITIONAL_FEATURES columns added/overwritten,
    computed via the same as-of aggregation (cutoff=None == all bouts), then capped and
    shrunk toward the (as-of) division mean.  Fighters with no positional bouts are left
    at NaN (the model's get_stats fills the POS_DEFAULTS).
    """
    fighters = fighters.copy()
    if pos_long is None:
        pos_long = load_positional_long()

    div_of = dict(zip(fighters["fighter"].astype(str).str.strip().str.lower(),
                      fighters["division_code"]))
    div_means = compute_asof_pos_division_means(pos_long, div_of, cutoff=cutoff,
                                                min_fights=min_fights)

    # init columns
    for col in POSITIONAL_FEATURES:
        fighters[col] = np.nan

    name_to_idx = {str(nm).strip().lower(): i for i, nm in enumerate(fighters["fighter"])}
    for name_l, sub in pos_long.groupby("fighter_l"):
        idx = name_to_idx.get(name_l)
        if idx is None:
            continue
        if cutoff is not None:
            sub = sub[sub["DATE"] < cutoff]
            if len(sub) == 0:
                continue
        raw = asof_positional_stats_from_group(sub)
        if raw is None:
            continue
        dc = div_of.get(name_l)
        shrunk = shrink_positional(raw, raw["_n_pos"], div_means, dc)
        for col, v in shrunk.items():
            fighters.iat[idx, fighters.columns.get_loc(col)] = v
    return fighters


if __name__ == "__main__":
    pos = load_positional_long()
    print(f"positional long rows (fighter-bouts): {len(pos)}")
    print(f"unique fighters with positional data: {pos['fighter_l'].nunique()}")
    # population means (used to sanity-check POS_DEFAULTS)
    allf = asof_positional_stats_from_group(pos)
    print("\nUFC-wide positional means:")
    for col in POSITIONAL_FEATURES:
        v = allf.get(col)
        print(f"  {col:24s}: {v:.3f}" if v is not None and np.isfinite(v) else f"  {col:24s}: n/a")
