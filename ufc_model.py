"""
ufc_model.py
============
UFC rating + prediction model.

Pipeline
--------
1. merge_sources()            -> ufc_fighters.csv, ufc_fight_log.csv
2. compute_elo(log, db)       -> performance-adjusted, dominance-scaled, opponent-quality Elo
                                 (robbery-aware: a dominant fighter who eats a bad decision keeps a high rating)
3. matchup(a, b, ...)         -> style interaction + GRAPPLER PREMIUM
4. win_probability(a, b, ...) -> Elo + stat profile + situational (NO rankings)
5. method_round(a, b, ...)    -> P(distance) / KO-Sub-Dec split / round distribution

Run as a script to (re)build ufc_fighters.csv, ufc_fight_log.csv and ufc_ratings.csv.

The per-division source files were produced by several different builders, so column
names and the free-text `dominance` column are inconsistent.  Everything here is driven
off a normalization layer (canonical stat names) plus dominance RE-DERIVED from the
consistent fields (method, decision_type, score_margin) -- we never trust the raw
`dominance` string for anything load-bearing.
"""

import os
import glob
import math
import re
import datetime as dt

import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))

# --------------------------------------------------------------------------- #
#  Division metadata
# --------------------------------------------------------------------------- #
DIVISION_NAMES = {
    "bw":   "Bantamweight",
    "flw":  "Flyweight",
    "fw":   "Featherweight",
    "hw":   "Heavyweight",
    "lhw":  "Light Heavyweight",
    "lw":   "Lightweight",
    "mw":   "Middleweight",
    "wbw":  "Women's Bantamweight",
    "wflw": "Women's Flyweight",
    "wsw":  "Women's Strawweight",
    "ww":   "Welterweight",
}

# canonical stat name -> list of aliases that may appear in a source file
STAT_ALIASES = {
    "td_def":       ["td_def", "TD_def", "td_defense"],
    "td_acc":       ["td_acc", "TD_acc", "td_accuracy"],
    "td_per15":     ["td_per15", "TD_per15"],
    "str_def":      ["str_def", "sig_str_def", "str_defense"],
    "str_acc":      ["str_acc", "sig_str_acc", "str_accuracy"],
    "kd_per15":     ["kd_per15", "KD_per15"],
    "sub_att_per15": ["sub_att_per15", "subatt_per15"],
    "pct_distance": ["pct_distance", "pct_go_distance", "distance_rate",
                     "distance_pct", "pct_going_distance"],
    "SLpM":         ["SLpM", "slpm"],
    "SApM":         ["SApM", "sapm"],
    "dmg_per_round": ["dmg_per_round"],
    "finish_rate":  ["finish_rate"],
    "decision_rate": ["decision_rate"],
    "gets_finished_rate": ["gets_finished_rate", "got_finished_rate"],
    "times_kod":    ["times_kod", "times_KOd"],
    "ko_tko_wins":  ["ko_tko_wins", "KO_TKO_wins"],
    "times_submitted": ["times_submitted", "times_subbed"],
    "td_att_per_round": ["td_att_per_round", "TD_att_per_round"],
    "age":          ["age"],
    "reach_in":     ["reach_in"],
    "height_in":    ["height_in"],
    "stance":       ["stance"],
    "style":        ["style"],
    "layoff_days":  ["layoff_days"],
    "streak":       ["streak"],
    "country":      ["country"],
    "hometown":     ["hometown"],
    "rank":         ["rank"],
}
# control time: special-cased because of minute vs second units
CTRL_ALIASES = ["ctrl_per_round_sec", "ctrl_sec_per_round", "ctrl_min_per_round",
                "ctrl_per_round_min", "ctrl_min_per_round_proxy"]

# ---- DATA-HYGIENE constants (added to fix corrupted method/round + situational outputs) ----
# Hard physical caps on per-15 / per-minute / control rate stats.  Small-sample fighters
# produced impossible values (td_per15=11.11, SLpM=10.66, ctrl=209.5 s/rd); cap them.
RATE_CAPS = {
    "td_per15": 8.0,
    "SLpM": 12.0,
    "SApM": 12.0,
    "sub_att_per15": 5.0,
    "ctrl_sec_per_round": 300.0,
    "kd_per15": 3.0,           # also reasonable to bound power proxy
    "td_att_per_round": 8.0,
}
# Rate stats that additionally get small-sample SHRINKAGE toward the division mean.
SHRINK_STATS = ["td_per15", "SLpM", "SApM", "sub_att_per15",
                "ctrl_sec_per_round", "kd_per15", "td_att_per_round"]
SHRINK_K = 4.0          # pseudo-count of "division-average" fights

# default values used when a stat is missing for a fighter (rough divisional medians)
STAT_DEFAULTS = {
    "td_def": 0.55, "td_acc": 0.40, "td_per15": 1.0, "str_def": 0.53,
    "str_acc": 0.45, "kd_per15": 0.25, "sub_att_per15": 0.5, "pct_distance": 0.45,
    "SLpM": 3.1, "SApM": 3.1, "dmg_per_round": 14.0, "finish_rate": 0.45,
    "decision_rate": 0.45, "gets_finished_rate": 0.30, "times_kod": 1.0,
    "ko_tko_wins": 1.0, "times_submitted": 1.0, "td_att_per_round": 1.0,
    "age": 30.0, "reach_in": 72.0, "height_in": 70.0, "ctrl_sec_per_round": 30.0,
    "layoff_days": 250.0,
}


# --------------------------------------------------------------------------- #
#  Method classifier (single source of truth for KO/Sub/Dec from the fight log)
# --------------------------------------------------------------------------- #
def classify_method(method):
    """
    Map a free-text `method` value from the fight log onto KO / SUB / DEC / OTHER.

    The per-division source files used several spellings ('KO/TKO', 'TKO - Doctor's
    Stoppage', 'Submission', 'SUB', 'Decision - Unanimous', 'DEC', ...).  This is the
    ONE place we normalize them so the finish-tendency stats are reliable.
    """
    m = str(method or "").upper()
    if not m or m == "NAN":
        return "OTHER"
    # decision FIRST is unsafe ('DEC' substring is unique enough; check finishes first)
    if ("KO" in m or "TKO" in m or "DOCTOR" in m or "CNC" in m
            or "COULD NOT CONTINUE" in m):
        return "KO"
    if "SUB" in m:
        return "SUB"
    if "DEC" in m or m.startswith("DECISION"):
        return "DEC"
    return "OTHER"          # DQ, NC, Overturned, Other -> low information


def recompute_finish_stats(log):
    """
    RE-DERIVE finish-tendency stats directly from the fight log (reliable METHOD column),
    instead of trusting the inconsistent per-division `finish_rate`/`gets_finished_rate`
    columns -- which had a denominator bug that made elite hard-to-finish fighters show
    gets_finished_rate=1.0.

    Per fighter (over all logged fights with a usable result):
      finish_rate        = wins by KO/TKO or Submission / total_fights
      gets_finished_rate = losses by KO/TKO or Submission / total_fights
      decision_rate      = fights ending in any Decision / total_fights
      pct_distance       = fights that reached the scorecards (a Decision) / total_fights

    Returns a DataFrame indexed by fighter with those four columns + n_fights.
    """
    df = log.copy()
    df["fighter"] = df["fighter"].astype(str).str.strip()
    df["_cls"] = df["method"].map(classify_method)
    df["_res"] = df["result"].astype(str).str.upper().str[0]   # W / L / D / N

    rows = []
    for name, g in df.groupby("fighter"):
        n = len(g)
        if n == 0:
            continue
        is_fin = g["_cls"].isin(["KO", "SUB"])
        fin_win = int((is_fin & (g["_res"] == "W")).sum())
        fin_loss = int((is_fin & (g["_res"] == "L")).sum())
        n_dec = int((g["_cls"] == "DEC").sum())
        rows.append({
            "fighter": name,
            "finish_rate": fin_win / n,
            "gets_finished_rate": fin_loss / n,
            "decision_rate": n_dec / n,
            "pct_distance": n_dec / n,       # reaching the scorecards == went the distance
            "_log_nfights": n,
        })
    return pd.DataFrame(rows).set_index("fighter")


# --------------------------------------------------------------------------- #
#  1. MERGE
# --------------------------------------------------------------------------- #
_TEXT_CANON = {"stance", "style", "country", "hometown"}


def _coalesce(df, canon, aliases):
    """Return a Series for `canon`, taking the first alias present in df."""
    for a in aliases:
        if a in df.columns:
            if canon in _TEXT_CANON:
                return df[a]
            return pd.to_numeric(df[a], errors="coerce")
    return pd.Series([np.nan] * len(df))


def _normalize_db(path, code):
    raw = pd.read_csv(path)
    out = pd.DataFrame()
    out["fighter"] = raw["fighter"].astype(str).str.strip()
    out["division"] = DIVISION_NAMES.get(code, code)
    out["division_code"] = code

    for canon, aliases in STAT_ALIASES.items():
        out[canon] = _coalesce(raw, canon, aliases)

    # control time -> seconds per round
    ctrl_col = next((c for c in CTRL_ALIASES if c in raw.columns), None)
    if ctrl_col is not None:
        ctrl = pd.to_numeric(raw[ctrl_col], errors="coerce")
        if "min" in ctrl_col:          # minutes -> seconds
            ctrl = ctrl * 60.0
        out["ctrl_sec_per_round"] = ctrl
    else:
        out["ctrl_sec_per_round"] = np.nan

    # win/loss record (several different schemas)
    def pick(*names):
        for n in names:
            if n in raw.columns:
                return pd.to_numeric(raw[n], errors="coerce")
        return pd.Series([np.nan] * len(raw))

    out["wins"] = pick("ufc_wins", "wins", "record_W")
    out["losses"] = pick("ufc_losses", "losses", "record_L")
    out["last_fight_date"] = raw["last_fight_date"] if "last_fight_date" in raw.columns else np.nan
    return out


def merge_sources(write=True):
    """Merge all fighter_db_*.csv and fighter_log_*.csv into the two master files."""
    # --- fighters ---
    db_parts = []
    for path in sorted(glob.glob(os.path.join(BASE, "fighter_db_*.csv"))):
        code = os.path.basename(path)[len("fighter_db_"):-len(".csv")]
        db_parts.append(_normalize_db(path, code))
    fighters = pd.concat(db_parts, ignore_index=True)
    # a fighter can appear in more than one division file (moved up/down); keep the
    # row with the most recent last_fight_date, else the first.
    fighters["_lfd"] = pd.to_datetime(fighters["last_fight_date"], errors="coerce")
    fighters = (fighters.sort_values("_lfd", ascending=False, na_position="last")
                        .drop_duplicates("fighter", keep="first")
                        .drop(columns="_lfd")
                        .reset_index(drop=True))

    # --- fight log ---
    log_parts = []
    keep = ["fighter", "date", "event", "opponent", "result", "method",
            "round", "decision_type", "weightclass", "score_margin"]
    for path in sorted(glob.glob(os.path.join(BASE, "fighter_log_*.csv"))):
        code = os.path.basename(path)[len("fighter_log_"):-len(".csv")]
        raw = pd.read_csv(path)
        part = pd.DataFrame()
        for c in keep:
            part[c] = raw[c] if c in raw.columns else np.nan
        part["division_code"] = code
        log_parts.append(part)
    log = pd.concat(log_parts, ignore_index=True)
    log["fighter"] = log["fighter"].astype(str).str.strip()
    log["opponent"] = log["opponent"].astype(str).str.strip()
    log["date"] = pd.to_datetime(log["date"], errors="coerce")
    log = log.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)

    # ----------------------------------------------------------------------- #
    #  DEDUP: the same bout appears multiple times because a fighter can be
    #  listed in more than one division source file (e.g. Oleksiejczuk in both
    #  the lhw and mw logs), which double-counted records and inflated Elo.
    #  A bout is uniquely identified by (fighter, opponent, date); we keep the
    #  single RICHEST copy (most non-null fields + most informative method /
    #  decision_type strings, e.g. "Decision - Unanimous" over a bare "Decision").
    # ----------------------------------------------------------------------- #
    log = _dedup_log(log)

    # ----------------------------------------------------------------------- #
    #  DATA HYGIENE (the fixes)
    # ----------------------------------------------------------------------- #
    fighters = _apply_data_hygiene(fighters, log)

    if write:
        fighters.to_csv(os.path.join(BASE, "ufc_fighters.csv"), index=False)
        log.to_csv(os.path.join(BASE, "ufc_fight_log.csv"), index=False)
    return fighters, log


def _dedup_log(log):
    """
    Collapse duplicate bouts (same fighter+opponent+date) down to the single most
    informative copy.  Duplicates arise because a fighter listed in two division
    source files gets both files' rows; keeping both double-counted wins/losses and
    inflated Elo.  Richness = number of populated fields + a small bonus for longer
    (more specific) method / decision_type strings and a present event name.
    """
    df = log.copy()

    def _richness(r):
        score = float(r.notna().sum())
        score += 0.5 * len(str(r.get("method", "") or ""))
        score += 0.3 * len(str(r.get("decision_type", "") or ""))
        if str(r.get("event", "")) not in ("", "nan", "None"):
            score += 2.0
        return score

    df["_rich"] = df.apply(_richness, axis=1)
    df = (df.sort_values("_rich", ascending=False)
            .drop_duplicates(subset=["fighter", "opponent", "date"], keep="first")
            .drop(columns="_rich")
            .sort_values("date")
            .reset_index(drop=True))
    return df


def _apply_data_hygiene(fighters, log):
    """
    Apply the three data-hygiene fixes to the merged fighter table:

      (1) RE-DERIVE finish-tendency stats from the fight log (overrides the buggy
          per-division finish_rate / gets_finished_rate / decision_rate / pct_distance).
      (2) Hard physical CAPS + small-sample SHRINKAGE toward the division mean for the
          per-15 / per-minute / control rate stats.
      (3) Drop / impute placeholder all-zero rows.
    """
    fighters = fighters.copy()

    # --- log-based fight counts (true exposure for shrinkage & finish stats) ---
    log_n = log.groupby(log["fighter"].astype(str).str.strip()).size()
    fighters["_log_nfights"] = fighters["fighter"].map(log_n).fillna(0).astype(int)

    # --- (3a) identify placeholder all-zero rows BEFORE caps/shrinkage ---
    profile_cols = ["td_def", "td_acc", "str_def", "str_acc", "SLpM", "SApM",
                    "finish_rate", "decision_rate"]
    present = [c for c in profile_cols if c in fighters.columns]
    zero_mask = (fighters[present].fillna(0) == 0).all(axis=1)

    # --- (1) recompute finish-tendency stats from the log ---
    fs = recompute_finish_stats(log)
    for col in ["finish_rate", "gets_finished_rate", "decision_rate", "pct_distance"]:
        mapped = fighters["fighter"].map(fs[col])
        # use the log-derived value wherever the fighter appears in the log;
        # fall back to whatever was in the source file (then defaults later) otherwise.
        fighters[col] = mapped.where(mapped.notna(), fighters.get(col))

    # --- (2) caps + small-sample shrinkage toward the DIVISION mean ---
    # hard caps first (clip impossible values)
    for col, cap in RATE_CAPS.items():
        if col in fighters.columns:
            fighters[col] = pd.to_numeric(fighters[col], errors="coerce").clip(upper=cap, lower=0.0)

    # division means computed on the CAPPED, reasonably-sampled fighters so a couple of
    # small-sample outliers don't poison the mean.
    for col in SHRINK_STATS:
        if col not in fighters.columns:
            continue
        vals = pd.to_numeric(fighters[col], errors="coerce")
        # robust division mean: only fighters with >=3 logged fights and a real value
        for div_code, idx in fighters.groupby("division_code").groups.items():
            sub = fighters.loc[idx]
            reliable = sub["_log_nfights"] >= 3
            base = vals.loc[idx][reliable & vals.loc[idx].notna()]
            div_mean = base.mean()
            if not np.isfinite(div_mean):
                div_mean = vals[vals.notna()].mean()      # global fallback
            if not np.isfinite(div_mean):
                div_mean = STAT_DEFAULTS.get(col, 0.0)
            n = sub["_log_nfights"].astype(float)
            raw = vals.loc[idx]
            raw = raw.where(raw.notna(), div_mean)
            shrunk = (n * raw + SHRINK_K * div_mean) / (n + SHRINK_K)
            vals.loc[idx] = shrunk
        fighters[col] = vals

    # --- (3b) placeholder rows: impute finish stats from log if available, else drop ---
    # A row is a fabricated placeholder if every profile stat was zero in the source.
    # If the fighter has log fights we keep them (finish stats already recomputed above,
    # rate stats now sit at the division mean via shrinkage with raw=mean) -- a sane
    # division-average fighter rather than a fake all-zero one.  If they have NO log data
    # at all, drop them so we never display a fabricated 0 fighter.
    has_log = fighters["_log_nfights"] > 0
    drop_mask = zero_mask & (~has_log)
    global DROPPED_ZERO_ROWS, IMPUTED_ZERO_ROWS
    DROPPED_ZERO_ROWS = list(fighters.loc[drop_mask, "fighter"])        # for reporting
    IMPUTED_ZERO_ROWS = list(fighters.loc[zero_mask & has_log, "fighter"])
    fighters = fighters[~drop_mask].drop(columns=["_log_nfights"]).reset_index(drop=True)
    return fighters


# module-level reporting hooks populated by _apply_data_hygiene
DROPPED_ZERO_ROWS = []
IMPUTED_ZERO_ROWS = []


# --------------------------------------------------------------------------- #
#  Dominance scoring (re-derived from consistent fields)
# --------------------------------------------------------------------------- #
def dominance_score(row):
    """
    Return a signed dominance score in roughly [-1, +1] FROM THE WINNER'S PERSPECTIVE,
    plus a controversy flag.

      +1.0  -> utterly dominant (early finish / wide one-sided decision)
       0.0  -> a coin-flip result (split decision, 1-pt margin)
      -1.0  -> "you got robbed": you almost certainly won but the decision went against you

    The function is called on a single fighter's log row (so `result` is W/L for THAT
    fighter).  We return:
        dom        : how decisively THIS fighter performed (signed, perf not result)
        controversy: 0..1, how close/controversial the decision was (0 = clean)
    """
    method = str(row.get("method", "") or "")
    dtype = str(row.get("decision_type", "") or "")
    rnd = row.get("round", np.nan)
    margin = row.get("score_margin", np.nan)
    result = str(row.get("result", ""))

    is_finish = ("KO" in method) or ("TKO" in method) or ("Submission" in method) \
        or ("Could Not Continue" in method) or ("Doctor" in method)
    is_decision = method.startswith("Decision") or dtype in ("Unanimous", "Split", "Majority")

    controversy = 0.0
    if is_finish:
        # finishes carry their own decisiveness; earlier = more dominant
        try:
            r = float(rnd)
        except Exception:
            r = 2.0
        perf = 1.0 - 0.12 * (r - 1)          # R1 finish ~1.0, R5 ~0.52
        perf = max(0.45, min(1.0, perf))
        dom = perf
    elif is_decision:
        # how wide was the decision?  prefer real scorecard margin when present.
        if pd.notna(margin):
            # margin_pts ~ avg points across 3 judges; 1 = razor thin, 5+ = shutout
            width = float(margin)
            perf = max(0.05, min(0.95, (width - 1.0) / 5.0 + 0.25))
        else:
            if dtype == "Split":
                perf, width = 0.12, 1.0
            elif dtype == "Majority":
                perf, width = 0.30, 2.0
            else:  # unanimous, unknown width
                perf, width = 0.55, 3.0
        # controversy: split / razor-thin decisions are where robberies live
        if dtype == "Split":
            controversy = 0.9
        elif dtype == "Majority":
            controversy = 0.5
        elif pd.notna(margin) and float(margin) <= 1.7:
            controversy = 0.7
        else:
            controversy = max(0.0, 0.4 - 0.08 * width)
        dom = perf
    else:
        # DQ / NC / overturned -> treat as low-information, near coin-flip
        dom = 0.2
        controversy = 0.5

    return dom, controversy


# --------------------------------------------------------------------------- #
#  2. PERFORMANCE-ADJUSTED ELO
# --------------------------------------------------------------------------- #
START_ELO = 1500.0
BASE_K = 40.0           # raised from 32 so the mid-tier separates instead of bunching
PROVISIONAL_K = 90.0   # chess-style fast-start K for a fighter's first few UFC fights
PROVISIONAL_FIGHTS = 8 # K decays from PROVISIONAL_K -> BASE_K over this many fights


def _expected(ra, rb):
    return 1.0 / (1.0 + 10 ** ((rb - ra) / 400.0))


def compute_elo(log, fighters=None, write_back=False):
    """
    Process the fight log chronologically and return {fighter: elo}.

    K is scaled by:
      * DOMINANCE of the performance (strike diff/control/KD proxied via finish & margin,
        scorecard width, split-vs-unanimous from raw_scorecards / score_margin)
      * RECENCY (older fights move Elo less on a decayed schedule)
      * OPPONENT QUALITY (beating an elite fighter is worth far more than beating a can;
        this falls out of the Elo expectation term, and we additionally amplify upsets)

    ROBBERY HANDLING (the critical requirement):
      A close/controversial loss (split decision, <=1-2 pt margin) costs almost no Elo.
      A fighter who STATISTICALLY DOMINATED but lost a bad decision actually has his
      rating *protected* (and can even rise), because we blend the nominal result with a
      "performance result": if you out-performed your opponent but the judges robbed you,
      the performance result is treated as a near-win.  This keeps a dominant fighter who
      ate a robbery (e.g. Chimaev) at the top of his division.
    """
    elo = {}
    nfights = {}

    # build per (fighter,date,opponent) lookup so we can read BOTH corners' rows and
    # process each bout once.
    log = log.sort_values("date").reset_index(drop=True)
    today = log["date"].max()

    # index every row by (sorted pair, date) so that when we process a bout we can read
    # the OTHER corner -- needed to tell whether the loser of a controversial decision
    # was actually out-performing (a robbery) rather than just losing close.
    bout_rows = {}
    for _, rr in log.iterrows():
        o = rr["opponent"]
        if not isinstance(o, str) or o in ("", "nan"):
            continue
        bk = tuple(sorted([rr["fighter"], o])) + (rr["date"].toordinal(),)
        bout_rows.setdefault(bk, {})[rr["fighter"]] = rr

    def is_genuine_robbery(my_row, opp_row, my_result):
        """
        A loss counts as a GENUINE robbery (eligible for rating protection) ONLY when:
          * the loss came on the scorecards via a SPLIT or MAJORITY decision, AND
          * the fighter plausibly WON it -- i.e. their own scorecard margin was
            favorable (>=0) or, when margins are absent, the winner did NOT win it
            decisively (the bout was a coin-flip on the cards).
        A clean loss (KO/TKO/Submission, or a WIDE unanimous decision) is NEVER a
        robbery and must fully drop Elo.  This keeps Chimaev's protection vs a true
        48-47 split while letting clear losses (KO, wide UD) sink faded fighters.
        """
        if my_result != "L":
            return 0.0
        dtype = str(my_row.get("decision_type", "") or "")
        method = str(my_row.get("method", "") or "")
        is_close_dec = (dtype in ("Split", "Majority")) or ("Split" in method) or ("Majority" in method)
        if not is_close_dec:
            return 0.0                       # KO/Sub/wide UD -> not a robbery, ever
        my_margin = my_row.get("score_margin", np.nan)
        # If we have the loser's own margin and it is clearly negative (they lost the
        # cards by >1pt avg), it was a legitimate close loss, not a robbery.
        if pd.notna(my_margin) and float(my_margin) < -1.0:
            return 0.0
        # strength of robbery: split is stronger evidence than majority
        strength = 0.9 if (dtype == "Split" or "Split" in method) else 0.5
        return strength

    def effective_score(my_row, opp_row, my_result, dom, controversy, my_elo, opp_elo):
        """
        Map a nominal W/L/D into an 'effective score' in [0,1] that bakes in dominance
        and NARROW robbery protection.  `opp_row` is the other corner (may be None).

        Robbery protection is gated by is_genuine_robbery() -- only a split/majority
        decision the fighter plausibly won is softened.  Every other loss drops Elo, but
        a clean loss to a much HIGHER-rated opponent is cushioned (you "belong" near an
        elite, so a clear loss to one costs less than a clear loss to a peer/lesser).
        This lifts a fighter whose losses all came against elites above a fighter who
        merely beat mid-tier opponents.
        """
        sa = 0.5 if my_result == "D" else (1.0 if my_result == "W" else 0.0)
        robbery = is_genuine_robbery(my_row, opp_row, my_result) if my_result == "L" else 0.0
        # opponent-quality factor: how far above this fighter the opponent is rated
        # (0 vs a peer/lesser, 1 vs a fighter +150 or more).  Drives the elite-loss cushion.
        oq = max(0.0, min(1.0, (opp_elo - my_elo) / 150.0))
        if my_result == "W":
            perf = 0.5 + 0.5 * dom
            blend = 0.25
        elif my_result == "L":
            if robbery > 0:
                # genuine robbery -> treat the result as a near-win that was denied
                perf = 0.5 + 0.35 * robbery
                blend = 0.25 + 0.65 * robbery        # heavy performance weighting
            else:
                # clean / clear loss: base performance is a loss (0) plus a small credit
                # for a competitive showing (opponent not dominant), THEN an opponent-
                # quality cushion: a clear loss to a much higher-rated fighter is partly
                # excused.  Crucially the BLEND also rises with opp quality, so the cushion
                # actually counts -- a clean loss to a true elite barely dents Elo, while a
                # clean loss to a peer/lesser drops it fully.  This lifts a fighter whose
                # losses are all to elites (e.g. Fiziev) above one who only beat mid-tier.
                opp_dom = dominance_score(opp_row)[0] if opp_row is not None else 0.55
                comp = 0.12 * (1.0 - opp_dom)
                cushion = 0.55 * oq
                perf = comp + cushion
                blend = 0.25 + 0.55 * oq             # up to ~0.80 vs a clear elite
        else:  # draw
            perf = 0.5
            blend = 0.25
        return (1 - blend) * sa + blend * perf

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

        ra = elo.get(f, START_ELO)
        rb = elo.get(opp, START_ELO)

        # both corners' rows (opp row may be absent if only one side logged the bout)
        opp_row = bout_rows.get(key, {}).get(opp)
        opp_result = str(opp_row["result"]) if opp_row is not None else (
            "L" if result == "W" else ("W" if result == "L" else "D"))

        dom_f, con_f = dominance_score(row)
        dom_o, con_o = (dominance_score(opp_row) if opp_row is not None else (dom_f, con_f))
        controversy = max(con_f, con_o)

        eff_f = effective_score(row, opp_row, result, dom_f, controversy, ra, rb)
        eff_o = effective_score(opp_row, row, opp_result, dom_o, controversy, rb, ra) if opp_row is not None else (1 - eff_f)

        ea = _expected(ra, rb)
        eb = 1 - ea

        # ---- K scaling (shared bout factors) ----
        age_days = (today - date).days
        k_rec = 0.65 + 0.35 * math.exp(-age_days / (365.0 * 3.5))   # gentler recency decay

        # DOMINANCE multiplier.  How decisively the bout was won amplifies the move.
        # NOTE: the prior "opponent-quality gating" (sched_quality) that DAMPENED the
        # dominance bump vs non-elite opponents has been REMOVED -- it over-corrected
        # and flattened everyone toward 1500.  Opponent quality now enters ONLY where it
        # belongs: through the Elo expectation term (ea/eb) and the upset amplifier
        # below, so a dominant win over a solid opponent moves Elo properly while a
        # dominant win over a true elite (high rb) moves it even more (big surprise).
        dom_night = max(dom_f, dom_o)
        # opponent-quality factor for the DOMINANCE add-on only.  This is a MILD version
        # of the old gate (which used 0.35+0.65*q and over-corrected/flattened everyone):
        # here a dominant finish over a weak opponent still keeps ~55% of its dominance
        # bonus, while dominating a strong opponent gets the full bonus.  The BASE move is
        # untouched, so wins still register; this just stops a riser from vaulting elites
        # by smashing cans (e.g. Torres KO'ing 1370-rated opposition).
        def sched_q(opp_rating):
            return max(0.0, min(1.0, (opp_rating - 1420.0) / 230.0))   # 0 at 1420, 1 at 1650
        q_f = sched_q(rb)      # quality of f's opponent
        q_o = sched_q(ra)      # quality of opp's opponent
        k_dom_f = 0.75 + 0.85 * dom_night * (0.55 + 0.45 * q_f)
        k_dom_o = 0.75 + 0.85 * dom_night * (0.55 + 0.45 * q_o)

        def k_provisional(name):
            """High K for a fighter's first few UFC bouts, decaying to BASE_K, so a
            riser climbs quickly off the 1500 start instead of being anchored there."""
            n = nfights.get(name, 0)
            if n >= PROVISIONAL_FIGHTS:
                return BASE_K
            frac = n / float(PROVISIONAL_FIGHTS)          # 0 at debut -> 1
            return PROVISIONAL_K + (BASE_K - PROVISIONAL_K) * frac

        def k_for(name, k_dom):
            return k_provisional(name) * k_dom * k_rec

        # OPPONENT-QUALITY / UPSET amplification.  (eff - expectation) already rewards
        # beating a higher-rated fighter (low ea -> large positive surprise) and
        # punishes losing to a lower-rated one; we amplify the surprise more aggressively
        # than before so opponent quality DOMINATES the spread.
        delta_f = k_for(f, k_dom_f) * (1.0 + 0.9 * abs(eff_f - ea)) * (eff_f - ea)
        delta_o = k_for(opp, k_dom_o) * (1.0 + 0.9 * abs(eff_o - eb)) * (eff_o - eb)

        elo[f] = ra + delta_f
        elo[opp] = rb + delta_o

        nfights[f] = nfights.get(f, 0) + 1
        nfights[opp] = nfights.get(opp, 0) + 1

    if write_back and fighters is not None:
        fighters["elo"] = fighters["fighter"].map(elo).fillna(START_ELO)
    return elo


# --------------------------------------------------------------------------- #
#  Fighter-stat accessor
# --------------------------------------------------------------------------- #
def get_stats(fighters, name):
    """Return a dict of canonical, default-filled stats for a fighter (case-insensitive)."""
    m = fighters[fighters["fighter"].str.lower() == str(name).lower()]
    if len(m) == 0:
        # fuzzy: last-name contains
        m = fighters[fighters["fighter"].str.lower().str.contains(
            re.escape(str(name).lower()), na=False)]
    if len(m) == 0:
        return None
    row = m.iloc[0].to_dict()
    out = {}
    for k, default in STAT_DEFAULTS.items():
        v = row.get(k, np.nan)
        out[k] = default if (v is None or (isinstance(v, float) and math.isnan(v))) else v
    out["fighter"] = row["fighter"]
    out["division"] = row.get("division")
    out["division_code"] = row.get("division_code")
    out["style"] = str(row.get("style", "balanced") or "balanced")
    out["stance"] = str(row.get("stance", "Orthodox") or "Orthodox")
    out["country"] = row.get("country")
    out["pct_distance"] = row.get("pct_distance") if pd.notna(row.get("pct_distance", np.nan)) else 0.45
    return out


# --------------------------------------------------------------------------- #
#  3. STYLE MATCHUP + GRAPPLER PREMIUM
# --------------------------------------------------------------------------- #
def grappler_index(s):
    """0..1 how much of an effective grappler/wrestler this fighter is."""
    td_off = 0.5 * min(1.0, s["td_per15"] / 3.5) + 0.5 * min(1.0, s["td_acc"] / 0.55)
    ctrl = min(1.0, s["ctrl_sec_per_round"] / 90.0)
    subs = min(1.0, s["sub_att_per15"] / 1.5)
    style_bonus = {"wrestler": 0.20, "grappler": 0.22, "balanced": 0.05, "striker": 0.0}.get(s["style"], 0.05)
    gi = 0.40 * td_off + 0.35 * ctrl + 0.15 * subs + style_bonus
    return max(0.0, min(1.0, gi))


def striker_index(s):
    vol = min(1.0, s["SLpM"] / 5.5)
    acc = min(1.0, s["str_acc"] / 0.55)
    pwr = 0.5 * min(1.0, s["kd_per15"] / 0.6) + 0.5 * min(1.0, s["ko_tko_wins"] / 6.0)
    style_bonus = {"striker": 0.18, "balanced": 0.05}.get(s["style"], 0.0)
    si = 0.45 * vol + 0.25 * acc + 0.20 * pwr + style_bonus
    return max(0.0, min(1.0, si))


def td_leakiness(s):
    """0..1 how easily this fighter is taken down (leaky TD defense)."""
    return max(0.0, min(1.0, 1.0 - s["td_def"]))


def matchup(a, b):
    """
    Style-interaction edge for fighter A vs fighter B.

    Returns a dict with a signed `edge` in roughly [-1, +1] (positive favors A) plus the
    components, INCLUDING the GRAPPLER PREMIUM.

    Grappler premium logic (the market underprices wrestling, so we over-weight it):
      * Compute each fighter's grappler_index and the opponent's TD leakiness.
      * An elite grappler facing a striker with leaky TD defense gets a large boost,
        because the fight will likely hit the mat where the striker can't operate.
      * The boost scales with (grappler quality) x (opponent leakiness) x (likelihood the
        fight hits the mat).
    """
    ga, gb = grappler_index(a), grappler_index(b)
    sa, sb = striker_index(a), striker_index(b)
    la, lb = td_leakiness(a), td_leakiness(b)   # how leaky A/B are

    # baseline skill-ish edges from raw stat profiles (z-ish, capped)
    strike_edge = 0.5 * ((a["SLpM"] - b["SApM"]) - (b["SLpM"] - a["SApM"])) / 6.0
    strike_edge += 0.4 * (a["str_acc"] - b["str_acc"]) + 0.3 * (a["str_def"] - b["str_def"])

    # --- GRAPPLER PREMIUM ---
    # likelihood the fight reaches the mat = the better grappler's takedown threat
    # against the opponent's leakiness, PLUS an elite-wrestler floor: a truly elite
    # grappler (high grappler_index) imposes mat time even on a good-defense striker
    # (wrestling is systematically underpriced, so a 0.74-TD-def striker is NOT immune
    # to an elite wrestler).  The floor scales with the grappler's own quality only.
    def elite_floor(g):
        return 0.11 * max(0.0, (g - 0.72) / 0.28)      # 0 below gi 0.72, up to ~0.11 at gi 1.0
    mat_a = ga * min(1.0, lb + elite_floor(ga))         # A drags B down
    mat_b = gb * min(1.0, la + elite_floor(gb))         # B drags A down
    # premium: elite grappler vs leaky striker.  Multiplied up because wrestling is
    # systematically underpriced.
    PREM = 1.70
    grap_edge_a = PREM * mat_a * (0.5 + 0.5 * sb)   # bigger if opp is a pure striker
    grap_edge_b = PREM * mat_b * (0.5 + 0.5 * sa)
    grappler_premium = grap_edge_a - grap_edge_b    # signed, favors A if positive

    # control-time differential
    ctrl_edge = (a["ctrl_sec_per_round"] - b["ctrl_sec_per_round"]) / 120.0

    edge = 0.45 * strike_edge + 0.85 * grappler_premium + 0.25 * ctrl_edge
    edge = max(-1.2, min(1.2, edge))

    return {
        "edge": edge,
        "grappler_premium": grappler_premium,
        "strike_edge": strike_edge,
        "ctrl_edge": ctrl_edge,
        "grappler_index_a": ga, "grappler_index_b": gb,
        "striker_index_a": sa, "striker_index_b": sb,
        "mat_prob_a": mat_a, "mat_prob_b": mat_b,
    }


# --------------------------------------------------------------------------- #
#  4. WIN PROBABILITY  (NO rankings used anywhere)
# --------------------------------------------------------------------------- #
def _situational_edge(a, b):
    """Signed situational edge favoring A (age, layoff, cardio, stance, size)."""
    e = 0.0
    # age: prime ~27-31; older fades
    def age_pen(age):
        return -0.012 * max(0.0, age - 32.0) + 0.006 * max(0.0, 28.0 - abs(age - 29.0))
    e += (age_pen(a["age"]) - age_pen(b["age"]))
    # ring rust: long layoff hurts
    def rust(days):
        return -0.10 * min(1.0, max(0.0, days - 365.0) / 730.0)
    e += (rust(a["layoff_days"]) - rust(b["layoff_days"]))
    # cardio proxy: high pct_distance + low gets_finished implies durable late
    cardio = lambda s: 0.5 * s["pct_distance"] + 0.5 * (1 - s["gets_finished_rate"])
    e += 0.10 * (cardio(a) - cardio(b))
    # stance: southpaw vs orthodox slight edge to southpaw
    if a["stance"] == "Southpaw" and b["stance"] == "Orthodox":
        e += 0.03
    elif b["stance"] == "Southpaw" and a["stance"] == "Orthodox":
        e -= 0.03
    # size: reach + height
    e += 0.04 * ((a["reach_in"] - b["reach_in"]) / 6.0)
    e += 0.02 * ((a["height_in"] - b["height_in"]) / 6.0)
    return e


def win_probability(a, b, elo, hometown_for=None):
    """
    P(A beats B).  Combines:
      * Elo difference
      * stat-profile style matchup (incl. grappler premium)
      * situational factors (age, layoff, cardio, stance, size, hometown)
    NO ranking input whatsoever.

    `a`,`b` are stat dicts from get_stats(); `elo` is the {fighter:elo} map.
    `hometown_for` may be 'A' or 'B' to apply a small home-cage edge.
    """
    ra = elo.get(a["fighter"], START_ELO)
    rb = elo.get(b["fighter"], START_ELO)
    p_elo = _expected(ra, rb)
    logit_elo = math.log(p_elo / (1 - p_elo)) if 0 < p_elo < 1 else 0.0

    m = matchup(a, b)
    sit = _situational_edge(a, b)
    if hometown_for == "A":
        sit += 0.04
    elif hometown_for == "B":
        sit -= 0.04

    # combine in logit space.  Elo weight RAISED (0.85 -> 1.15) so the now-properly-
    # spread skill ratings actually drive the line; the stat-matchup (grappler premium)
    # and situational terms remain meaningful but no longer swamp a real Elo gap.
    logit = 1.33 * logit_elo + 1.12 * m["edge"] + 1.38 * sit
    p = 1.0 / (1.0 + math.exp(-logit))
    return {
        "p_a": p, "p_b": 1 - p,
        "elo_a": ra, "elo_b": rb, "p_elo": p_elo,
        "matchup_edge": m["edge"], "grappler_premium": m["grappler_premium"],
        "situational": sit,
    }


# --------------------------------------------------------------------------- #
#  5. METHOD + ROUND
# --------------------------------------------------------------------------- #
def method_round(a, b, wp=None, scheduled_rounds=3):
    """
    Predict method and round distribution for an A-vs-B fight.

    Returns:
      p_distance                : P(fight goes to the scorecards)
      method probs (overall)    : KO/TKO, Submission, Decision
      method | winner           : conditional on A (or B) winning
      round_dist                : P(finish in round r) over 1..scheduled_rounds

    Built from both fighters' finish/decision tendencies, control time, style
    interaction, and how much of the fight is standup vs on the mat.
    """
    m = matchup(a, b)

    # P(goes the distance): high when both fighters tend to go to decision, durable,
    # and neither has a strong finishing path.  Lower when there's a big grappler premium
    # (mat finishes) or heavy striking power.
    base_dist = 0.5 * (a["pct_distance"] + b["pct_distance"])
    finish_threat = 0.5 * (a["finish_rate"] + b["finish_rate"])
    vuln = 0.5 * (a["gets_finished_rate"] + b["gets_finished_rate"])
    mat_action = max(m["mat_prob_a"], m["mat_prob_b"])
    p_distance = base_dist - 0.45 * finish_threat - 0.30 * vuln - 0.25 * mat_action + 0.20
    if scheduled_rounds == 5:
        p_distance -= 0.06          # more time to finish in 5-rounders
    p_distance = max(0.08, min(0.92, p_distance))
    p_finish = 1 - p_distance

    # within a finish, KO/TKO vs Submission share.
    # KO weight from striking power & opp vulnerability; Sub weight from grappling.
    ko_w = (0.5 * (a["kd_per15"] + b["kd_per15"]) / 0.5
            + 0.5 * (a["ko_tko_wins"] + b["ko_tko_wins"]) / 6.0
            + vuln)
    sub_w = (a["sub_att_per15"] + b["sub_att_per15"]) / 1.5 + mat_action * 1.5
    if ko_w + sub_w == 0:
        ko_w, sub_w = 1.0, 1.0
    ko_share = ko_w / (ko_w + sub_w)
    p_ko = p_finish * ko_share
    p_sub = p_finish * (1 - ko_share)

    methods = {"KO/TKO": p_ko, "Submission": p_sub, "Decision": p_distance}

    # method conditional on each winner
    def cond(win_stats, lose_stats):
        ko = 0.45 * min(1.0, win_stats["kd_per15"] / 0.5) + 0.55 * lose_stats["gets_finished_rate"]
        sub = 0.5 * min(1.0, win_stats["sub_att_per15"] / 1.0) + 0.5 * min(1.0, win_stats["ctrl_sec_per_round"] / 90.0)
        dec = win_stats["pct_distance"]
        tot = ko + sub + dec
        if tot == 0:
            return {"KO/TKO": 0.33, "Submission": 0.33, "Decision": 0.34}
        # blend with the overall finish/distance split so it stays calibrated
        raw = {"KO/TKO": ko / tot, "Submission": sub / tot, "Decision": dec / tot}
        ko2 = 0.5 * raw["KO/TKO"] + 0.5 * (p_ko / max(1e-6, p_finish)) * p_finish
        sub2 = 0.5 * raw["Submission"] + 0.5 * (p_sub / max(1e-6, p_finish)) * p_finish
        dec2 = 0.5 * raw["Decision"] + 0.5 * p_distance
        t2 = ko2 + sub2 + dec2
        return {"KO/TKO": ko2 / t2, "Submission": sub2 / t2, "Decision": dec2 / t2}

    method_if_a = cond(a, b)
    method_if_b = cond(b, a)

    # round distribution for finishes (only over scheduled rounds)
    # finishes are front-loaded; later rounds get a fatigue bump.
    rounds = list(range(1, scheduled_rounds + 1))
    weights = []
    for r in rounds:
        w = math.exp(-0.45 * (r - 1))          # decay
        if r >= 3:
            w *= 1.15                           # late fatigue/finish bump
        weights.append(w)
    tot = sum(weights)
    round_dist = {f"R{r}": p_finish * w / tot for r, w in zip(rounds, weights)}
    round_dist["Decision"] = p_distance

    return {
        "p_distance": p_distance,
        "p_finish": p_finish,
        "methods": methods,
        "method_if_a_wins": method_if_a,
        "method_if_b_wins": method_if_b,
        "round_dist": round_dist,
        "scheduled_rounds": scheduled_rounds,
    }


# --------------------------------------------------------------------------- #
#  OVERALL RATING + RATINGS TABLE
# --------------------------------------------------------------------------- #
def overall_rating(s, elo):
    """0-100 composite blending Elo with stat quality (for display)."""
    e = elo.get(s["fighter"], START_ELO)
    elo_component = max(0.0, min(1.0, (e - 1300.0) / 500.0))   # ~1300..1800 -> 0..1
    skill = (0.30 * min(1.0, s["SLpM"] / 5.5)
             + 0.20 * s["str_def"]
             + 0.20 * grappler_index(s)
             + 0.15 * s["finish_rate"]
             + 0.15 * (1 - s["gets_finished_rate"]))
    return round(100 * (0.70 * elo_component + 0.30 * skill), 1)


def build_ratings(write=True):
    fighters, log = merge_sources(write=write)
    elo = compute_elo(log, fighters, write_back=True)

    rows = []
    for _, fr in fighters.iterrows():
        s = get_stats(fighters, fr["fighter"])
        if s is None:
            continue
        e = elo.get(fr["fighter"], START_ELO)
        rows.append({
            "fighter": fr["fighter"],
            "division": fr["division"],
            "elo": round(e, 1),
            "overall_rating": overall_rating(s, elo),
            "style": s["style"],
            "grappler_index": round(grappler_index(s), 3),
            "striker_index": round(striker_index(s), 3),
            "SLpM": round(s["SLpM"], 2),
            "str_def": round(s["str_def"], 3),
            "td_per15": round(s["td_per15"], 2),
            "td_def": round(s["td_def"], 3),
            "ctrl_sec_per_round": round(s["ctrl_sec_per_round"], 1),
            "sub_att_per15": round(s["sub_att_per15"], 2),
            "finish_rate": round(s["finish_rate"], 3),
            "gets_finished_rate": round(s["gets_finished_rate"], 3),
            "age": s["age"],
            "stance": s["stance"],
            "n_fights": int((log["fighter"] == fr["fighter"]).sum()),
        })
    ratings = pd.DataFrame(rows)
    # sort by Elo within division
    ratings = ratings.sort_values(["division", "elo"], ascending=[True, False]).reset_index(drop=True)
    if write:
        ratings.to_csv(os.path.join(BASE, "ufc_ratings.csv"), index=False)
    return ratings, elo, fighters, log


def predict(fighter_a, fighter_b, scheduled_rounds=3, hometown_for=None):
    """Convenience: full prediction for a single matchup using current ratings."""
    fighters = pd.read_csv(os.path.join(BASE, "ufc_fighters.csv"))
    log = pd.read_csv(os.path.join(BASE, "ufc_fight_log.csv"), parse_dates=["date"])
    elo = compute_elo(log, fighters)
    a, b = get_stats(fighters, fighter_a), get_stats(fighters, fighter_b)
    if a is None or b is None:
        raise ValueError(f"fighter not found: {fighter_a if a is None else fighter_b}")
    wp = win_probability(a, b, elo, hometown_for=hometown_for)
    mr = method_round(a, b, wp, scheduled_rounds=scheduled_rounds)
    return {"win": wp, "method_round": mr,
            "matchup": matchup(a, b),
            "a": a["fighter"], "b": b["fighter"]}


if __name__ == "__main__":
    ratings, elo, fighters, log = build_ratings(write=True)
    print(f"fightersInModel = {fighters['fighter'].nunique()}")
    print(f"fightLogRows    = {len(log)}")
    print("wrote: ufc_fighters.csv, ufc_fight_log.csv, ufc_ratings.csv")
    print("\nTop 3 by Elo per division:")
    for div, g in ratings.groupby("division"):
        top = g.head(3)
        names = ", ".join(f"{r.fighter} ({r.elo:.0f})" for r in top.itertuples())
        print(f"  {div:22s}: {names}")
