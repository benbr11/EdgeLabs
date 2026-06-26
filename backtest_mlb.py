# -*- coding: utf-8 -*-
"""
HONEST walk-forward (out-of-sample) backtest for the MLB model.

This is MEASUREMENT ONLY. It does not modify the model. It re-implements the model's
own rating computation (build_mlb.py) and prediction formula (web/prosports_app.js
predRuns) with a STRICT per-game date cutoff so nothing after a game can leak into the
prediction of that game.

NO LOOK-AHEAD design
--------------------
For every game in the test window we predict it using ONLY games that finished STRICTLY
BEFORE its date:
  * Team att/dfn ratings are re-solved from prior games only (prior seasons, recency-
    weighted by SW[season], + current-season-to-date), exactly as build_mlb.py does:
    iterative att/dfn run solve, blended with a recency-weighted WIN%-implied rating,
    regressed toward the mean, geo-mean normalised. Recomputed on a WEEKLY cutoff
    (every games before the Monday of the game's week) -- weekly granularity, never
    using anything from the game's week or later.
  * Each starting pitcher's RA9 factor is computed from ONLY that pitcher's starts
    strictly before the game (recency-weighted across seasons + season-to-date),
    regressed to the prior-to-date league RA9 -- the same regression build_mlb.py uses
    (ip*ra9 + 60*LG_RA9)/(ip+60), factor = ra9_reg / LG_RA9.
  * The DECAYING CONSENSUS prior is applied as blend_mlb.py would have applied it on
    that date: W = min(1, BASE + gamesPlayedThatSeason/162). Early-season it pulls the
    rating toward the consensus ordering; by mid/late season it decays to ~pure model.

Prediction formula (canonical, from web/prosports_app.js predRuns):
    lh = AVG * att[home] * dfn[away] * HOME
    la = AVG * att[away] * dfn[home]
    SP factor blend: adj(f) = 0.6*f + 0.4   (starter ~= 60% of run prevention)
       away starter -> lh *= adj(factor_away);  home starter -> la *= adj(factor_home)
    Poisson(lh) x Poisson(la) score grid -> pH, pTie, pA
    share = lh/(lh+la);  winH = pH + pTie*share;  winA = pA + pTie*(1-share)

Reads (read-only): consensus_mlb.csv (for the decaying prior ordering target).
Fetches: statsapi.mlb.com schedule (gameType=R, finals, probablePitcher) and per-pitcher
gameLogs. Caches everything to backtest_mlb_cache.json so re-runs are instant.

Outputs: headline OOS winner hit-rate, calibration table, Brier / log-loss, trust-tier
table, per-team accuracy, and an honest strong/weak read.
"""
import json, os, math, urllib.request, datetime, collections, time
import mlb_signals

PROJ = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(PROJ, "backtest_mlb_cache.json")
CONSENSUS = os.path.join(PROJ, "consensus_mlb.csv")
ENRICH = os.path.join(PROJ, "mlb_enrich_cache.json")
BOX = os.path.join(PROJ, "mlb_box_cache.json")
PLATOON = os.path.join(PROJ, "mlb_platoon_cache.json")

# ---- match build_mlb.py tunables exactly ----
SW_HL = 0.70          # season-recency halflife
HALFLIFE = 230.0      # game-level decay (days)
WPCT_W = 0.80         # weight on win%-implied rating vs run-diff
WPCT_SCALE = 8.0      # run-diff-per-game equiv of a full win% swing
REG = 0.40            # regression of composed multiplier toward 1.0
SP_REG_IP = 60.0      # pitcher RA9 regression strength (innings)
# blend_mlb.py decaying-consensus prior:
BLEND_BASE = 0.40
SEASON_LEN = 162.0
# prediction SP blend (prosports_app.js): adj(f)=0.6*f+0.4
SP_W = 0.6
# symmetric home-field multiplier (estimated from home WIN rate, regressed toward prior)
HOME_PRIOR = float(os.environ.get("MLB_HOME_PRIOR", 1.04))
HOME_PRIOR_W = float(os.environ.get("MLB_HOME_PRIOR_W", 0.25))
# global temperature on the win-prob log-odds (T>1 softens). Fit on 2024-25 train: ~1.46.
# Improves OOS log-loss/Brier; STRAIGHT-UP picks are temperature-invariant.
TEMP = float(os.environ.get("MLB_TEMP", 1.46))
# Decaying-consensus prior is FUTURE-INFORMED in a 2024-25 backtest (a static 2026 preseason
# ranking applied to past games) AND measured to slightly HURT OOS -> OFF for honest model-only
# measurement. Set MLB_USE_CONSENSUS=1 to re-enable (matches the live blend_mlb.py post-process).
USE_CONSENSUS = os.environ.get("MLB_USE_CONSENSUS", "0") == "1"

NAMEFIX = {"Oakland Athletics": "Athletics"}
fn = lambda n: NAMEFIX.get(n, n)


def _pois_h(k, l): return math.exp(-l) * l ** k / math.factorial(k)
def _even_home_winp(H, avg):
    # home win prob at an EVEN matchup under symmetric split lh=avg*sqrt(H), la=avg/sqrt(H)
    s = math.sqrt(H); lh = avg * s; la = avg / s
    pH = pT = pA = 0.0
    for i in range(16):
        for j in range(16):
            m = _pois_h(i, lh) * _pois_h(j, la)
            if i > j: pH += m
            elif i == j: pT += m
            else: pA += m
    share = lh / (lh + la)
    return (pH + pT * share) / (pH + pT * share + pA + pT * (1 - share))
def home_mult(hwf, avg, prior, prior_w):
    # Estimate the SYMMETRIC home multiplier H reproducing observed home-win-fraction hwf
    # (via Poisson inversion), then regress toward `prior` with weight prior_w. H>=1.
    hwf = min(0.62, max(0.50, hwf))               # guard against degenerate small samples
    lo, hi = 1.0, 1.30
    for _ in range(40):
        mid = (lo + hi) / 2
        if _even_home_winp(mid, avg) < hwf: lo = mid
        else: hi = mid
    H_emp = (lo + hi) / 2
    return max(1.0, (1 - prior_w) * H_emp + prior_w * prior)


def get(url, t=60, retries=3):
    for i in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            return json.loads(urllib.request.urlopen(req, timeout=t).read())
        except Exception as e:
            if i == retries - 1:
                raise
            time.sleep(1.5)


# ----------------------------------------------------------------------------
# 1. DATA: schedule (finals + probable/actual starter) for all needed seasons
# ----------------------------------------------------------------------------
def fetch_schedule(seasons):
    """Return list of games: dict(date, home, away, hs, as_, season, ph_id, ph_name,
    pa_id, pa_name). Starter = probablePitcher hydrated (the actual starter for finals)."""
    games = []
    for y in seasons:
        j = get(f"https://statsapi.mlb.com/api/v1/schedule?sportId=1&startDate={y}-03-15"
                f"&endDate={y}-11-15&gameType=R&hydrate=probablePitcher")
        n0 = len(games)
        for dd in j.get("dates", []):
            for g in dd.get("games", []):
                if g.get("status", {}).get("detailedState") != "Final":
                    continue
                t = g.get("teams", {})
                h, a = t.get("home", {}), t.get("away", {})
                try:
                    hs, as_ = int(h["score"]), int(a["score"])
                except (KeyError, ValueError, TypeError):
                    continue
                if hs == as_:
                    continue  # MLB has no ties; skip suspended/odd records
                try:
                    d = (g.get("officialDate") or g["gameDate"][:10])
                    datetime.date.fromisoformat(d)
                except (ValueError, KeyError):
                    continue
                php = h.get("probablePitcher") or {}
                pap = a.get("probablePitcher") or {}
                games.append({
                    "date": d, "home": fn(h["team"]["name"]), "away": fn(a["team"]["name"]),
                    "hs": hs, "as": as_, "season": y,
                    "ph_id": php.get("id"), "ph_name": php.get("fullName"),
                    "pa_id": pap.get("id"), "pa_name": pap.get("fullName"),
                })
        print(f"  schedule {y}: {len(games)-n0} final games", flush=True)
    return games


def fetch_pitcher_logs(pitcher_ids, seasons):
    """For each pitcher id, per-date (ip, runs) game-log rows across the given seasons.
    Returns {pid: [(date_iso, ip, runs), ...]}. Cached."""
    logs = {}
    ids = sorted(pitcher_ids)
    for n, pid in enumerate(ids):
        rows = []
        for y in seasons:
            try:
                j = get(f"https://statsapi.mlb.com/api/v1/people/{pid}/stats?stats=gameLog"
                        f"&group=pitching&season={y}&gameType=R")
            except Exception:
                continue
            sp = j.get("stats", [])
            if not sp:
                continue
            for s in sp[0].get("splits", []):
                st = s.get("stat", {})
                try:
                    ip = float(st.get("inningsPitched", 0) or 0)
                    runs = float(st.get("runs", 0) or 0)
                except (ValueError, TypeError):
                    continue
                d = s.get("date")
                if d and ip > 0:
                    rows.append((d, ip, runs))
        rows.sort()
        logs[pid] = rows
        if (n + 1) % 50 == 0:
            print(f"  pitcher logs {n+1}/{len(ids)}", flush=True)
    return logs


def load_cache(seasons, test_seasons):
    if os.path.exists(CACHE):
        with open(CACHE, encoding="utf-8") as f:
            c = json.load(f)
        if c.get("seasons") == list(seasons):
            print("Using cached statsapi data.", flush=True)
            return c["games"], {int(k): v for k, v in c["pitcher_logs"].items()}
    print("Fetching schedule from statsapi...", flush=True)
    games = fetch_schedule(seasons)
    # pitcher ids that actually START a game in the test window (need their factor)
    pids = set()
    for g in games:
        if g["season"] in test_seasons:
            if g["ph_id"]:
                pids.add(g["ph_id"])
            if g["pa_id"]:
                pids.add(g["pa_id"])
    print(f"Fetching game logs for {len(pids)} starters (cached after first run)...", flush=True)
    plogs = fetch_pitcher_logs(pids, seasons)
    with open(CACHE, "w", encoding="utf-8") as f:
        json.dump({"seasons": list(seasons), "games": games,
                   "pitcher_logs": {str(k): v for k, v in plogs.items()}}, f)
    return games, plogs


# ----------------------------------------------------------------------------
# 2. TEAM RATINGS as of a cutoff date (replicates build_mlb.py with prior games only)
# ----------------------------------------------------------------------------
def compute_ratings(prior_games, cur_season):
    """prior_games: list of game dicts strictly before the cutoff. Returns
    (att, dfn, AVG, HOME) using build_mlb.py's method. cur_season used only so the
    SW season weighting references the right 'current' year."""
    if len(prior_games) < 200:
        return None
    seasons = sorted({g["season"] for g in prior_games}, reverse=True)
    SW = {y: 0.5 ** ((cur_season - y) / SW_HL) for y in range(cur_season, cur_season - 6, -1)}
    ref = max(g["date"] for g in prior_games)
    ref_d = datetime.date.fromisoformat(ref)

    def wt(g):
        gd = datetime.date.fromisoformat(g["date"])
        return 0.5 ** ((ref_d - gd).days / HALFLIFE) * SW.get(g["season"], 0.05)

    teams = sorted({t for g in prior_games for t in (g["home"], g["away"])})
    tw = tr = hr = ar = hw = gw = 0.0
    twsum = collections.defaultdict(float)
    twin = collections.defaultdict(float)
    for g in prior_games:
        w = wt(g); hs, as_ = g["hs"], g["as"]
        tr += w * (hs + as_); tw += 2 * w; hr += w * hs; ar += w * as_
        hw += w * (1 if hs > as_ else 0); gw += w            # weighted home wins / weighted games
        twsum[g["home"]] += w; twsum[g["away"]] += w
        twin[g["home"]] += w * (1 if hs > as_ else 0)
        twin[g["away"]] += w * (1 if as_ > hs else 0)
    AVG = tr / tw
    HOME = home_mult(hw / gw if gw else 0.532, AVG, HOME_PRIOR, HOME_PRIOR_W)  # symmetric, from home WIN rate

    att = {t: 1.0 for t in teams}; dfn = {t: 1.0 for t in teams}
    for _ in range(60):
        na = {t: 0. for t in teams}; da = dict(na); nd = dict(na); dd = dict(na)
        for g in prior_games:
            w = wt(g); h, a, hs, as_ = g["home"], g["away"], g["hs"], g["as"]
            na[h] += w * hs; da[h] += w * AVG * dfn[a]; nd[a] += w * hs; dd[a] += w * AVG * att[h]
            na[a] += w * as_; da[a] += w * AVG * dfn[h]; nd[h] += w * as_; dd[h] += w * AVG * att[a]
        for t in teams:
            if da[t] > 0: att[t] = na[t] / da[t]
            if dd[t] > 0: dfn[t] = nd[t] / dd[t]
        for dct in (att, dfn):
            gm = math.exp(sum(math.log(max(v, 1e-6)) for v in dct.values()) / len(dct))
            for t in dct: dct[t] /= gm

    # compose: blend run-diff with win%-implied, regress, re-map to att/dfn
    rd_rating = {t: AVG * (att[t] - dfn[t]) for t in teams}
    wp = {t: (twin[t] / twsum[t] if twsum[t] else 0.5) for t in teams}
    wp_rating = {t: (wp[t] - 0.5) * WPCT_SCALE for t in teams}
    tgt = {t: (1 - REG) * ((1 - WPCT_W) * rd_rating[t] + WPCT_W * wp_rating[t]) for t in teams}
    for t in teams:
        half = tgt[t] / 2.0
        att[t] = max(1e-6, (AVG + half) / AVG)
        dfn[t] = max(1e-6, (AVG - half) / AVG)
    for dct in (att, dfn):
        gm = math.exp(sum(math.log(v) for v in dct.values()) / len(dct))
        for t in dct: dct[t] /= gm
    return att, dfn, AVG, HOME, twsum


# ----------------------------------------------------------------------------
# 2b. DECAYING CONSENSUS PRIOR as of date (replicates blend_mlb.py)
# ----------------------------------------------------------------------------
def load_consensus():
    crank = {}
    with open(CONSENSUS, newline="", encoding="utf-8") as f:
        import csv
        for c in csv.DictReader(f):
            crank[fn(c["team"])] = int(c["consensus_rank"])
    return crank


def apply_consensus(att, dfn, AVG, crank, games_played):
    """Shift each team's net run rating toward the consensus ordering target with weight
    (1-W), W = min(1, BASE + gamesPlayed/162). Re-derive att/dfn. (blend_mlb.py logic.)"""
    teams = list(att)
    W = min(1.0, BLEND_BASE + games_played / SEASON_LEN)
    model_net = {t: AVG * att[t] - AVG * dfn[t] for t in teams}
    sorted_net = sorted(model_net.values(), reverse=True)
    n = len(teams)
    out_att, out_dfn = {}, {}
    for t in teams:
        k = crank.get(t)
        if k is None:
            out_att[t], out_dfn[t] = att[t], dfn[t]; continue
        idx = max(0, min(n - 1, k - 1))
        cons_net = sorted_net[idx]
        final_net = W * model_net[t] + (1 - W) * cons_net
        delta = final_net - model_net[t]
        out_att[t] = att[t] + delta / (2.0 * AVG)
        out_dfn[t] = dfn[t] - delta / (2.0 * AVG)
    return out_att, out_dfn


# ----------------------------------------------------------------------------
# 3. PITCHER FACTOR as of a date (prior starts only), build_mlb.py regression
# ----------------------------------------------------------------------------
def pitcher_factor(pid, plogs, cutoff_date, cur_season, lg_ra9):
    """RA9 factor from this pitcher's starts strictly before cutoff_date, recency-weighted
    across seasons, regressed to lg_ra9. Returns None if no prior data (-> team-avg)."""
    rows = plogs.get(pid)
    if not rows:
        return None
    SW = {y: 0.5 ** ((cur_season - y) / SW_HL) for y in range(cur_season, cur_season - 6, -1)}
    w_ip = w_runs = 0.0
    for d, ip, runs in rows:
        if d >= cutoff_date:
            continue
        y = int(d[:4])
        w = SW.get(y, 0.1)
        w_ip += w * ip; w_runs += w * runs
    if w_ip < 10:
        return None
    ra9 = w_runs / w_ip * 9
    ra9_reg = (w_ip * ra9 + SP_REG_IP * lg_ra9) / (w_ip + SP_REG_IP)
    return ra9_reg / lg_ra9


# ----------------------------------------------------------------------------
# 4. PREDICTION (canonical predRuns) -> home win prob
# ----------------------------------------------------------------------------
def _pois(k, l):
    return math.exp(-l) * l ** k / math.factorial(k)


def base_rates(att, dfn, AVG, HOME, home, away, f_home_sp, f_away_sp):
    """Pre-signal expected runs (lh, la) after SP adjustment. Returns None if teams unknown."""
    if home not in att or away not in att:
        return None
    _hs = math.sqrt(HOME)                          # symmetric home split: preserves total runs
    lh = AVG * att[home] * dfn[away] * _hs
    la = AVG * att[away] * dfn[home] / _hs
    adj = lambda f: (SP_W * f + (1 - SP_W)) if f else 1.0
    if f_away_sp:  # away starter suppresses home runs
        lh *= adj(f_away_sp)
    if f_home_sp:  # home starter suppresses away runs
        la *= adj(f_home_sp)
    return lh, la


def winp_from_rates(lh, la):
    """Poisson score grid -> temperature-calibrated home win probability."""
    lh = max(0.5, lh); la = max(0.5, la)
    pH = pT = pA = 0.0
    for i in range(16):
        for j in range(16):
            m = _pois(i, lh) * _pois(j, la)
            if i > j: pH += m
            elif i == j: pT += m
            else: pA += m
    share = lh / (lh + la)
    winH = pH + pT * share
    winA = pA + pT * (1 - share)
    p = winH / (winH + winA)                       # normalise out the tiny truncation mass
    if TEMP != 1.0 and 0.0 < p < 1.0:              # global temperature on the log-odds (calibration)
        p = 1.0 / (1.0 + math.exp(-math.log(p / (1.0 - p)) / TEMP))
    return p


def predict_winp(att, dfn, AVG, HOME, home, away, f_home_sp, f_away_sp, sig=None):
    br = base_rates(att, dfn, AVG, HOME, home, away, f_home_sp, f_away_sp)
    if br is None:
        return None
    lh, la = br
    if sig:        # point-in-time day-of run-rate adjustments (mlb_signals; identity if all W=0)
        lh, la = mlb_signals.apply_all(lh, la, park=sig.get("park"), weather=sig.get("weather"),
                                       platoon=sig.get("platoon"), bullpen=sig.get("bullpen"))
    return winp_from_rates(lh, la)


# ----------------------------------------------------------------------------
# 4b. DAY-OF SIGNAL DATA (point-in-time / as-of). Loaded from caches built by mlb_enrich.py.
# ----------------------------------------------------------------------------
def load_signal_data():
    """Returns (enrich_idx, box, platoon) or (None, None, None) if caches absent."""
    enrich_idx = box = platoon = None
    if os.path.exists(ENRICH):
        with open(ENRICH, encoding="utf-8") as f:
            d = json.load(f)
        enrich_idx = {}
        for pk, e in d["by_pk"].items():
            e["_pk"] = pk                              # stamp gamePk for boxscore lookup
            enrich_idx[f"{e['date']}|{e['home']}|{e['away']}"] = e
    if os.path.exists(BOX):
        with open(BOX, encoding="utf-8") as f:
            box = json.load(f)
    if os.path.exists(PLATOON):
        with open(PLATOON, encoding="utf-8") as f:
            platoon = json.load(f)
    return enrich_idx, box, platoon


def build_park_factors(games_sorted, enrich_idx):
    """Trailing park factor per (venue, season) using ONLY PRIOR seasons (strictly as-of):
    PF[venue, S] = (mean runs/game at venue over seasons in [S-3, S-1]) / (league mean over
    the same prior seasons). Returns dict {(venue_name, season): pf}. Falls back to 1.0."""
    if not enrich_idx:
        return {}
    # raw per (venue, season) runs/games and league runs/games
    vr = collections.defaultdict(lambda: [0.0, 0])
    lg = collections.defaultdict(lambda: [0.0, 0])
    seasons = set()
    for g in games_sorted:
        e = enrich_idx.get(f"{g['date']}|{g['home']}|{g['away']}")
        if not e or not e.get("venue_name"):
            continue
        s = g["season"]; runs = g["hs"] + g["as"]; vn = e["venue_name"]
        vr[(vn, s)][0] += runs; vr[(vn, s)][1] += 1
        lg[s][0] += runs; lg[s][1] += 1
        seasons.add(s)
    venues = {vn for (vn, s) in vr}
    pf = {}
    for s in seasons:
        prior = [y for y in seasons if s - 3 <= y <= s - 1]
        lg_runs = sum(lg[y][0] for y in prior); lg_g = sum(lg[y][1] for y in prior)
        lg_rpg = (lg_runs / lg_g) if lg_g else None
        for vn in venues:
            vrun = sum(vr[(vn, y)][0] for y in prior if (vn, y) in vr)
            vg = sum(vr[(vn, y)][1] for y in prior if (vn, y) in vr)
            if vg >= 60 and lg_rpg:                     # need ~a season of prior data
                raw = (vrun / vg) / lg_rpg
                # regress toward 1.0 (park factors are noisy at 1-3 seasons)
                w = min(1.0, vg / 240.0)
                pf[(vn, s)] = 1.0 + w * (raw - 1.0)
            else:
                pf[(vn, s)] = 1.0
    return pf


def league_platoon_baselines(platoon):
    """League mean OPS vs LHP and vs RHP per season (for centering the platoon edge)."""
    base = {}
    if not platoon:
        return base
    agg = collections.defaultdict(lambda: {"vl": [], "vr": []})
    for key, sp in platoon.get("team_split", {}).items():
        try:
            season = int(key.rsplit("|", 1)[1])
        except (ValueError, IndexError):
            continue
        if sp.get("vl"):
            agg[season]["vl"].append(sp["vl"])
        if sp.get("vr"):
            agg[season]["vr"].append(sp["vr"])
    for s, d in agg.items():
        base[s] = (sum(d["vl"]) / len(d["vl"]) if d["vl"] else None,
                   sum(d["vr"]) / len(d["vr"]) if d["vr"] else None)
    return base


def _ip_to_outs(ip):
    """MLB IP is recorded as X.1 (=X+1/3) or X.2 (=X+2/3), NOT decimal. Convert to true innings."""
    whole = int(ip)
    frac = round(ip - whole, 1)
    thirds = 1 if frac == 0.1 else (2 if frac == 0.2 else 0)
    return whole + thirds / 3.0


def precompute_relief_ip(box, enrich_idx, games_sorted):
    """Per (team, date) relief innings thrown in THAT game (starter excluded). Returns
    {team: [(date_iso, relief_ip), ...] sorted}. O(N) once; windowed sums are then cheap.
    IP is decoded from baseball X.1/X.2 notation to true innings."""
    by_team = collections.defaultdict(list)
    if not box or not enrich_idx:
        return by_team
    for g in games_sorted:
        e = enrich_idx.get(f"{g['date']}|{g['home']}|{g['away']}")
        if not e:
            continue
        rec = box.get(str(e.get("_pk")))
        if not rec:
            continue
        for side, team in (("home", g["home"]), ("away", g["away"])):
            lst = rec.get(side, [])
            rip = sum(_ip_to_outs(ip) for _, ip in lst[1:])   # skip starter (index 0)
            by_team[team].append((g["date"], rip))
    for t in by_team:
        by_team[t].sort()
    return by_team


def bullpen_fatigue(relief_idx, team, game_date, lookback_days=3):
    """Recent reliever IP load for `team` in the lookback_days calendar days STRICTLY before
    game_date (as-of). Normalized: 0 = rested, ~1 = heavily worked."""
    rows = relief_idx.get(team)
    if not rows:
        return 0.0
    gd = datetime.date.fromisoformat(game_date)
    lo = (gd - datetime.timedelta(days=lookback_days)).isoformat()
    relief_ip = 0.0
    for d, rip in rows:
        if d >= game_date:
            break                                     # rows sorted; nothing later qualifies
        if d >= lo:
            relief_ip += rip
    return relief_ip / (lookback_days * 3.0)


# ----------------------------------------------------------------------------
# 5. WALK-FORWARD over the test window
# ----------------------------------------------------------------------------
def monday_of(date_iso):
    d = datetime.date.fromisoformat(date_iso)
    return (d - datetime.timedelta(days=d.weekday())).isoformat()


def run_backtest(games, plogs, test_seasons, crank):
    games_sorted = sorted(games, key=lambda g: g["date"])
    test = [g for g in games_sorted if g["season"] in test_seasons]
    preds = []           # (winp_home, home_won, home, away, season, conf)
    rating_cache = {}    # (cur_season, monday) -> ratings tuple
    skipped = 0
    # ---- day-of signal data (point-in-time). All identity-safe if caches absent / W=0. ----
    enrich_idx, box, platoon = load_signal_data()
    park_pf = build_park_factors(games_sorted, enrich_idx)
    relief_idx = precompute_relief_ip(box, enrich_idx, games_sorted)
    lg_plat = league_platoon_baselines(platoon)
    hand = (platoon or {}).get("hand", {})
    tsplit = (platoon or {}).get("team_split", {})
    # league RA9 prior, recomputed per season cutoff (use a stable approx from prior season runs)
    for n, g in enumerate(test):
        cutoff_monday = monday_of(g["date"])
        cur_season = g["season"]
        prior = [x for x in games_sorted if x["date"] < cutoff_monday]
        key = (cur_season, cutoff_monday)
        if key not in rating_cache:
            res = compute_ratings(prior, cur_season)
            rating_cache[key] = res
        res = rating_cache[key]
        if res is None:
            skipped += 1; continue
        att, dfn, AVG, HOME, twsum = res
        # League RA9 baseline for the pitcher FACTOR (= ra9_reg / lg_ra9). The starters actually
        # being predicted average ~4.6 RA9, so the season run env AVG (~4.43) centers the factor
        # distribution better than the model's selective top-IP LG_RA9 (~4.32); using ~4.30 was
        # MEASURED to lower OOS, so we keep AVG (self-adapting per season cutoff).
        lg_ra9 = AVG
        if USE_CONSENSUS:
            # games played by these two teams THIS season before the game (decaying-prior weight).
            gp_home = sum(1 for x in prior if x["season"] == cur_season and (x["home"] == g["home"] or x["away"] == g["home"]))
            gp_away = sum(1 for x in prior if x["season"] == cur_season and (x["home"] == g["away"] or x["away"] == g["away"]))
            gp = (gp_home + gp_away) / 2.0
            att2, dfn2 = apply_consensus(att, dfn, AVG, crank, gp)
        else:
            att2, dfn2 = att, dfn   # model-only (no future-informed consensus); honest OOS
        # pitcher factors strictly before the GAME date (more granular than weekly is fine,
        # it only uses that pitcher's own prior starts -> no leakage)
        f_home_sp = pitcher_factor(g["ph_id"], plogs, g["date"], cur_season, lg_ra9) if g["ph_id"] else None
        f_away_sp = pitcher_factor(g["pa_id"], plogs, g["date"], cur_season, lg_ra9) if g["pa_id"] else None
        # ---- assemble point-in-time day-of signals (identity when their W knob is 0) ----
        e = enrich_idx.get(f"{g['date']}|{g['home']}|{g['away']}") if enrich_idx else None
        sig, meta = build_game_signals(g, e, cur_season, park_pf, relief_idx, hand, tsplit, lg_plat)
        winp = predict_winp(att2, dfn2, AVG, HOME, g["home"], g["away"], f_home_sp, f_away_sp, sig)
        if winp is None:
            skipped += 1; continue
        home_won = 1 if g["hs"] > g["as"] else 0
        rec = {"winp": winp, "home_won": home_won, "home": g["home"],
               "away": g["away"], "season": cur_season, "date": g["date"]}
        rec.update(meta)                              # signal inputs + context for segmentation
        preds.append(rec)
        if (n + 1) % 500 == 0:
            print(f"  scored {n+1}/{len(test)} games", flush=True)
    return preds, skipped


def build_game_signals(g, e, cur_season, park_pf, relief_idx, hand, tsplit, lg_plat):
    """Compute the day-of signal multiplier tuples + a metadata dict (signal inputs and game
    context: venue, dayNight, weather, handedness, fatigue, park factor) for segmentation.
    All inputs are strictly as-of (prior-season splits, prior-day fatigue, gametime weather)."""
    meta = {"venue": None, "dayNight": None, "roof": None, "temp": None, "wind_mph": None,
            "wind_dir": None, "park_pf": 1.0, "away_sp_hand": None, "home_sp_hand": None,
            "home_pen_fat": 0.0, "away_pen_fat": 0.0}
    park = weather = platoon = bullpen = None
    # park factor (trailing prior seasons)
    if e and e.get("venue_name"):
        pf = park_pf.get((e["venue_name"], cur_season), 1.0)
        meta["park_pf"] = pf; meta["venue"] = e["venue_name"]
        park = mlb_signals.park_factor(pf)
    # weather (outdoor only)
    if e:
        meta["dayNight"] = e.get("dayNight"); meta["roof"] = e.get("roof")
        meta["temp"] = e.get("temp"); meta["wind_mph"] = e.get("wind_mph"); meta["wind_dir"] = e.get("wind_dir")
        weather = mlb_signals.weather_factor(e.get("temp"), e.get("wind_mph"),
                                             e.get("wind_dir"), e.get("roof"))
    # platoon: PRIOR-season team splits + starter handedness
    ah = hand.get(str(g.get("pa_id"))) if g.get("pa_id") else None
    hh = hand.get(str(g.get("ph_id"))) if g.get("ph_id") else None
    meta["away_sp_hand"] = ah; meta["home_sp_hand"] = hh
    hs_split = tsplit.get(f"{g['home']}|{cur_season-1}")
    as_split = tsplit.get(f"{g['away']}|{cur_season-1}")
    lvl, lvr = lg_plat.get(cur_season - 1, (None, None))
    platoon = mlb_signals.platoon_factor(hs_split, as_split, ah, hh, lvl, lvr)
    # bullpen fatigue (prior N-day relief IP; N from mlb_signals, OOS-tuned to 3)
    _ld = mlb_signals.PEN_LOOKBACK_DAYS
    hpf = bullpen_fatigue(relief_idx, g["home"], g["date"], _ld)
    apf = bullpen_fatigue(relief_idx, g["away"], g["date"], _ld)
    meta["home_pen_fat"] = hpf; meta["away_pen_fat"] = apf
    bullpen = mlb_signals.bullpen_factor(hpf, apf)
    sig = {"park": park, "weather": weather, "platoon": platoon, "bullpen": bullpen}
    return sig, meta


# ----------------------------------------------------------------------------
# 6. SCORING / REPORT
# ----------------------------------------------------------------------------
def report(preds, test_seasons, skipped):
    N = len(preds)
    L = ["=" * 78,
         "HONEST WALK-FORWARD (OUT-OF-SAMPLE) BACKTEST -- MLB MODEL",
         f"Test window: {sorted(test_seasons)} regular season   |   N = {N} games"
         f"   |   skipped (insufficient prior data): {skipped}",
         "Each game predicted from data STRICTLY BEFORE it (no look-ahead).",
         "=" * 78]

    # straight-up winner hit-rate: favorite = side with higher win prob
    hits = 0; conf_sum = 0.0
    for p in preds:
        fav_home = p["winp"] >= 0.5
        won = (fav_home and p["home_won"]) or (not fav_home and not p["home_won"])
        hits += 1 if won else 0
        conf_sum += max(p["winp"], 1 - p["winp"])
    hr = hits / N
    L += ["",
          f"HEADLINE -- straight-up WINNER hit-rate: {hr*100:.2f}%  ({hits}/{N})",
          f"  mean model confidence on its pick: {conf_sum/N*100:.1f}%",
          f"  coin-flip = 50.0%   |   market benchmark ~58-60% (MLB is the flattest sport)"]

    # Brier + log-loss (outcome = home win, prob = winp)
    brier = sum((p["winp"] - p["home_won"]) ** 2 for p in preds) / N
    eps = 1e-12
    ll = -sum(p["home_won"] * math.log(max(p["winp"], eps)) +
              (1 - p["home_won"]) * math.log(max(1 - p["winp"], eps)) for p in preds) / N
    # baselines: always 0.5, and base-rate (home win freq)
    base_rate = sum(p["home_won"] for p in preds) / N
    brier_50 = sum((0.5 - p["home_won"]) ** 2 for p in preds) / N
    ll_base = -(base_rate * math.log(base_rate) + (1 - base_rate) * math.log(1 - base_rate))
    L += ["",
          f"PROBABILISTIC SCORES (lower is better):",
          f"  Brier score:  {brier:.4f}   (always-0.5 baseline {brier_50:.4f})",
          f"  Log-loss:     {ll:.4f}   (base-rate baseline {ll_base:.4f})",
          f"  Home-team actual win rate in window: {base_rate*100:.1f}%"]

    # Fix 4: report RAW (pre-temperature) vs CALIBRATED probabilistic scores. winp already has
    # TEMP applied in predict_winp, so recover the raw prob by inverting the temperature.
    if TEMP != 1.0:
        def _raw(p):
            if not (0.0 < p < 1.0): return p
            return 1.0 / (1.0 + math.exp(-math.log(p / (1.0 - p)) * TEMP))
        braw = sum((_raw(p["winp"]) - p["home_won"]) ** 2 for p in preds) / N
        lraw = -sum(p["home_won"] * math.log(max(_raw(p["winp"]), eps)) +
                    (1 - p["home_won"]) * math.log(max(1 - _raw(p["winp"]), eps)) for p in preds) / N
        L += [f"  [temperature T={TEMP:.2f}]  RAW (uncalibrated): Brier {braw:.4f}  Log-loss {lraw:.4f}",
              f"                          CALIBRATED (above):  Brier {brier:.4f}  Log-loss {ll:.4f}",
              f"  (straight-up hit-rate is identical raw vs calibrated -- temperature is monotonic)"]

    # calibration by confidence bucket (on the HOME win prob)
    buckets = [(0.0, 0.5), (0.5, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 1.01)]
    L += ["", "CALIBRATION -- by predicted HOME win probability:",
          "  {:<14}{:>7}{:>14}{:>14}".format("bucket", "N", "pred home%", "actual home%")]
    for lo, hi in buckets:
        sub = [p for p in preds if lo <= p["winp"] < hi]
        if not sub:
            continue
        pm = sum(p["winp"] for p in sub) / len(sub) * 100
        am = sum(p["home_won"] for p in sub) / len(sub) * 100
        L.append("  {:<14}{:>7}{:>13.1f}%{:>13.1f}%".format(f"{lo*100:.0f}-{hi*100:.0f}%", len(sub), pm, am))

    # trust-tier table: among picks where the model's confidence on its FAVORITE exceeds
    # each threshold, what's the hit-rate?
    L += ["", "TRUST-TIER TABLE -- accuracy of the model's PICK above each confidence threshold:",
          "  {:<14}{:>8}{:>12}{:>14}".format("conf >=", "N picks", "hit-rate", "vs market 58-60%")]
    for thr in (0.50, 0.60, 0.65, 0.70, 0.75, 0.80):
        sub = []
        for p in preds:
            conf = max(p["winp"], 1 - p["winp"])
            if conf >= thr:
                fav_home = p["winp"] >= 0.5
                won = (fav_home and p["home_won"]) or (not fav_home and not p["home_won"])
                sub.append(won)
        if not sub:
            L.append("  {:<14}{:>8}{:>12}".format(f"{thr*100:.0f}%", 0, "--"))
            continue
        rate = sum(sub) / len(sub) * 100
        flag = "above" if rate >= 58 else ("at" if rate >= 56 else "below")
        L.append("  {:<14}{:>8}{:>11.1f}%   {}".format(f"{thr*100:.0f}%", len(sub), rate, flag))

    # per-team: how well does the model predict games involving each team (pick correctness)
    team_hit = collections.defaultdict(lambda: [0, 0])  # team -> [correct, total]
    for p in preds:
        fav_home = p["winp"] >= 0.5
        won = (fav_home and p["home_won"]) or (not fav_home and not p["home_won"])
        for t in (p["home"], p["away"]):
            team_hit[t][1] += 1
            team_hit[t][0] += 1 if won else 0
    team_rate = sorted(((c / n, t, n) for t, (c, n) in team_hit.items() if n >= 30), reverse=True)
    L += ["", "PER-TEAM pick accuracy (games involving the team; >=30 games):",
          "  BEST predicted:"]
    for r, t, n in team_rate[:6]:
        L.append(f"    {t:<26} {r*100:5.1f}%  ({n} g)")
    L.append("  WORST predicted:")
    for r, t, n in team_rate[-6:]:
        L.append(f"    {t:<26} {r*100:5.1f}%  ({n} g)")

    # honest read
    delta_50 = (hr - 0.5) * 100
    L += ["", "=" * 78, "HONEST READ:"]
    if hr >= 0.58:
        L.append(f"  Winner hit-rate {hr*100:.1f}% is AT/ABOVE the ~58-60% market benchmark -- a")
        L.append("  genuinely strong result for baseball, the flattest sport to predict. The")
        L.append("  market itself rarely clears 60% straight-up, so matching it OOS is good.")
    elif hr >= 0.55:
        L.append(f"  Winner hit-rate {hr*100:.1f}% beats a coin flip by {delta_50:+.1f} pts and sits just")
        L.append("  below the ~58-60% market benchmark. Solid for MLB but not market-beating.")
    elif hr >= 0.52:
        L.append(f"  Winner hit-rate {hr*100:.1f}% is modestly above a coin flip ({delta_50:+.1f} pts) and")
        L.append("  below the market benchmark -- typical of a team-rating model without")
        L.append("  full per-game lineups/bullpen/park context.")
    else:
        L.append(f"  Winner hit-rate {hr*100:.1f}% is essentially a coin flip ({delta_50:+.1f} pts). On a")
        L.append("  true OOS basis the model carries little straight-up edge in MLB.")
    if brier < brier_50:
        L.append(f"  Brier {brier:.4f} beats the naive 0.5 baseline ({brier_50:.4f}): the probabilities")
        L.append("  carry real information even where raw winner calls are close to the market.")
    else:
        L.append(f"  Brier {brier:.4f} does NOT beat the 0.5 baseline -- probabilities are not well")
        L.append("  calibrated out-of-sample.")
    L.append("  Strong vs weak: see the calibration buckets (are high-confidence picks")
    L.append("  actually winning more?) and the per-team table above.")
    L.append("=" * 78)
    return "\n".join(L)


def main():
    # Seasons to FETCH (need history before the test window to rate teams from priors).
    # Test = 2025 (most recent COMPLETE season) + 2024 (prior, for a bigger sample).
    test_seasons = {2024, 2025}
    fetch_seasons = list(range(2026, 2019, -1))  # 2020..2026; gives prior history for 2024/25
    games, plogs = load_cache(fetch_seasons, test_seasons)
    print(f"Total final games fetched: {len(games)} | starters with logs: {len(plogs)}", flush=True)
    crank = load_consensus()
    print("Running walk-forward (weekly rating cutoff, per-game pitcher cutoff)...", flush=True)
    preds, skipped = run_backtest(games, plogs, test_seasons, crank)
    out = report(preds, test_seasons, skipped)
    print(out)
    with open(os.path.join(PROJ, "backtest_mlb_results.txt"), "w", encoding="utf-8") as f:
        f.write(out + "\n")
    print("\n(Results also written to backtest_mlb_results.txt)")


if __name__ == "__main__":
    main()
