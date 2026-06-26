# -*- coding: utf-8 -*-
"""
backtest_nhl.py  --  HONEST walk-forward (out-of-sample) backtest of the NHL model.

WHAT THIS MEASURES
  How well the NHL team-rating + Poisson/Dixon-Coles win-prob engine predicts games
  it has NOT seen. For every game in the test window we rebuild the team ratings from
  ONLY the games that finished strictly BEFORE that game (prior seasons + current
  season to date), then predict. Nothing on or after the game date can leak in.

NO LOOK-AHEAD (the whole point)
  * Ratings are recomputed at a weekly cutoff. A game played on date D is predicted
    with the rating snapshot built from all games with gameDate < (the Monday of D's
    week). So no game in the same week or later is used. Recompute cadence = weekly
    (a compromise: per-game would be identical in spirit but ~25x slower; within a
    week ratings barely move, and we still never use the game itself or any future
    game).
  * The rating math is a faithful replica of build_nhl.py's PREDICTION-RELEVANT path:
      - recency-weighted (HALFLIFE=70d) Poisson attack/defense (60 iters)
      - MOV-weighted Elo
      - z-blend: attZ = GA*zAtt + XF*zXGF + PP*zPP + EA*zElo ;
                 defZ = GD*zDef + XA*zXGA + PK*zPK + GS*zGSAx + ED*zElo
      - mapped back to the goals log-scale, goal-level calibration k
    xG (XF/XA=0.30) is RE-ENABLED from a WINDOWED, point-in-time MoneyPuck shot snapshot
    (only shots before the cutoff) -- a validated OOS win; GSAx (GS) is kept 0.0 (it hurt
    OOS). PP/PK use prior-COMPLETED-season team summary stats (no look-ahead). See the
    TUNED FLAGS block for each change's KEEP/REVERT verdict and measured delta.
  * The decaying expert-consensus prior (blend_nhl.py) is applied exactly as it would
    have at the cutoff: W = min(1, BASE + gamesPlayed/82). gamesPlayed = current-season
    games each team has played at the cutoff, so W decays to 1.0 (pure on-ice model) as
    the season fills in. CAVEAT: consensus_nhl.csv is the *current* (2026-27) preseason
    consensus; no point-in-time historical consensus exists. To avoid leaking a future
    ranking into past seasons, the HEADLINE backtest is MODEL-ONLY (no consensus blend);
    we additionally report a consensus-blended variant for reference and flag it.

PREDICTION ENGINE  -- identical to nhl_predict.py:
  lambda_home = AVG * att[H] * dfn[A] * home_adv ; lambda_away = AVG * att[A] * dfn[H]
  Poisson score matrix (0..13 goals); regulation pH/pT/pA; ties resolved to OT/SO with a
  slight favourite edge (winH = pH + pT*(0.5 + (fav-0.5)*0.35)). The near-pick'em ABSTAIN
  band (|lambda margin| < 0.20 -> shrink to 0.50, graded=False) is applied; the headline is
  reported BOTH over all games and graded-only. The point-in-time starting-goalie adjustment
  is available (NHL_GOALIE=1) but OFF by default -- it did not improve OOS (see TUNED FLAGS).

OUTPUT: console report + _backtest_nhl_games.csv (every test game with its OOS prob).
READ-ONLY on all model files. Nothing here writes nhl_ratings.csv.
"""
import json, os, csv, math, datetime, collections, itertools, urllib.request, urllib.parse, hashlib

PROJ = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(PROJ, "_nhl_api_cache")
os.makedirs(CACHE, exist_ok=True)

# ---------------------------------------------------------------- HTTP (cached) ----
def get(url, t=30):
    fp = os.path.join(CACHE, hashlib.md5(url.encode()).hexdigest() + ".json")
    if os.path.exists(fp):
        try:
            return json.load(open(fp, encoding="utf-8"))
        except Exception:
            pass
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    data = json.loads(urllib.request.urlopen(req, timeout=t).read())
    json.dump(data, open(fp, "w", encoding="utf-8"))
    return data

# ---------------------------------------------------------------- model constants --
# (mirror build_nhl.py defaults exactly)
HALFLIFE = 70.0
SEASON_WEIGHTS = [1.0, 0.10, 0.03, 0.01, 0.005]   # newest current season -> older
EA, ED = 0.75, 0.74          # Elo weight (dominant)
GA, GD = 0.20, 0.18          # recency goals model (off/def split)
PP_W, PK_W = 0.16, 0.16      # special teams
POISSON_ITERS = 60
RELOCATE = {"ARI": "UTA"}
def fix(ab): return RELOCATE.get(ab, ab)

# =====================================================================================
# TUNED FLAGS -- each was implemented, measured walk-forward, and KEPT or REVERTED on its
# OUT-OF-SAMPLE result (tune on EY2025, validate on held-out EY2026). Defaults below are the
# FINAL kept configuration so a bare `python backtest_nhl.py` reproduces the shipped model.
# Every flag is overridable by env var for A/B re-measurement.
# =====================================================================================

# ---- [KEPT] windowed point-in-time xG z-blend weights (mirror build_nhl.py _XF/_XA/_GS) ----
# Production zeroed these because the STALE 5-season-flat aggregate lagged. A WINDOWED,
# point-in-time MoneyPuck xGF/xGA snapshot (only shots before the cutoff, same wt()/HALFLIFE
# as the rating build) instead ADDS real OOS discrimination: AUC 0.5835->0.5881, log-loss
# 0.6798->0.6790 at weight 0.30, and the held-out season SU rises 52.4%->53.6%. 0.30 is the
# OOS Brier/log-loss optimum. GSAx (GS) stays 0: adding it MEASURABLY HURT (AUC 0.5869->0.5853).
XF_W = float(os.environ.get("NHL_XF", "0.30"))
XA_W = float(os.environ.get("NHL_XA", "0.30"))
GS_W = float(os.environ.get("NHL_GS", "0.0"))

# ---- [REVERTED] point-in-time starting-goalie adjustment (mirror nhl_predict.py SOG/gadj) ----
# Off by default. The point-in-time starting goalie (max-TOI from each boxscore, prior GSAx
# fed through gadj()) did NOT transfer out of sample: raw it HURT (55.48->55.30, AUC down);
# heavily damped it was a +0.14% wash on all games but gave ZERO lift on the held-out season
# (the apparent gain lived entirely in the tuning season). Team strength already prices in
# team-average goaltending; single-game starter variance is noise at the win/loss level.
SOG = 29.0
USE_GOALIE = os.environ.get("NHL_GOALIE", "0") == "1"
GOALIE_SHRINK_K = float(os.environ.get("NHL_GOALIE_K", "0.0"))
GOALIE_MIN_SH = float(os.environ.get("NHL_GOALIE_MINSH", "8.0"))
GOALIE_DAMP = float(os.environ.get("NHL_GOALIE_DAMP", "1.0"))

# ---- [KEPT] abstain / regress near-pick'em band (change #3) ----
# Games with projected |lambda margin| < 0.20 hit only ~42-46% OOS (worse than a coin flip);
# shrinking them to 0.50 and reporting graded-only lifts the headline 55.5%->57.0% (graded)
# and improves AUC. Robust on both the tuning and held-out seasons. Default ON at 0.20.
ABSTAIN_MARGIN = float(os.environ.get("NHL_ABSTAIN", "0.20"))  # |lambda_home-lambda_away| below this -> shrink
ABSTAIN_SHRINK = float(os.environ.get("NHL_SHRINK", "1.0"))    # 1.0 = full shrink to 0.50; 0 = no shrink

# ---- [REVERTED] Dixon-Coles low-score tie correction (change #4) ----
# Off (rho=0). Measured ZERO effect on win predictions at rho in [-0.10,-0.05]: DC only
# perturbs the 4 lowest-score cells, which barely move winH in high-scoring hockey, and after
# renormalisation the favourite prob is unchanged (55.48/0.2434/0.6798/0.5835 either way).
DC_RHO = float(os.environ.get("NHL_DC_RHO", "0.0"))

# ---- [REVERTED] regularization (change #5) ----
# Both off. Pseudo-count shrinkage of att/dfn toward 1.0 was a null (AUC ~0.5836 at every
# kappa); between-season Elo reversion shifted a few coin-flips (+0.3% SU) but with FLAT AUC
# and slightly WORSE log-loss/Brier -> not a real discrimination gain, reverted.
POIS_KAPPA = float(os.environ.get("NHL_KAPPA", "0.0"))
ELO_CARRY = float(os.environ.get("NHL_ELO_CARRY", "1.0"))

# Win-prob recalibration (mirrors nhl_predict.py / web/nhl_app.js). The raw Poisson
# favourite probabilities are systematically overconfident OOS (see the calibration table
# below); a logit temperature T>1 softens the SCALE toward 0.5 without changing ordering
# (so the straight-up hit-rate is unchanged) -- T=2.0 minimises Brier + log-loss here.
WINPROB_TEMP = 2.0
# Optional fitted calibration (change #4): when CALIB_MODEL is set to ("platt", a, b) the
# raw favourite logit is mapped by a + b*logit instead of the fixed temperature. The model
# is FIT ON EARLIER SEASONS and applied to the held-out test seasons (see fit_calibration),
# so it is never tuned on the games it scores. None -> fixed-temperature production path.
CALIB_MODEL = None
def recalibrate(p, T=WINPROB_TEMP):
    p = min(max(p, 1e-9), 1 - 1e-9)
    z = math.log(p / (1 - p))
    if CALIB_MODEL is not None and CALIB_MODEL[0] == "platt":
        _, a, b = CALIB_MODEL
        return 1.0 / (1.0 + math.exp(-(a + b * z)))
    if T == 1.0:
        return p
    return 1.0 / (1.0 + math.exp(-z / T))

def fit_calibration(raw_probs, labels, l2=1.0):
    """Fit a Platt logistic a + b*logit(p) -> P(win) by Newton-Raphson on the calibration
    block. l2 ridge keeps b sane on small samples. Returns (a, b)."""
    import math as _m
    xs = [_m.log(min(max(p, 1e-9), 1 - 1e-9) / (1 - min(max(p, 1e-9), 1 - 1e-9))) for p in raw_probs]
    ys = labels
    a, b = 0.0, 1.0
    for _ in range(50):
        g0 = g1 = h00 = h01 = h11 = 0.0
        for x, y in zip(xs, ys):
            z = a + b * x
            q = 1.0 / (1.0 + _m.exp(-z))
            g0 += (q - y); g1 += (q - y) * x
            w = q * (1 - q)
            h00 += w; h01 += w * x; h11 += w * x * x
        g1 += l2 * b; h11 += l2          # ridge on slope only
        det = h00 * h11 - h01 * h01
        if abs(det) < 1e-12:
            break
        da = (h11 * g0 - h01 * g1) / det
        db = (h00 * g1 - h01 * g0) / det
        a -= da; b -= db
        if abs(da) < 1e-9 and abs(db) < 1e-9:
            break
    return a, b

# consensus blend params (mirror blend_nhl.py)
BLEND_BASE = 0.25
SEASON_LEN = 82.0

def season_start_year(d):
    return d.year if d.month >= 9 else d.year - 1

# ---------------------------------------------------------------- data load --------
def load_all_games(season_end_years):
    """season_end_years: list of NHL season END years to pull (e.g. [2023,2024,2025,2026]).
    Returns list of dicts: {date(datetime.date), h, a, hg, ag, gtype, season_int}."""
    # team list from standings
    stand = get("https://api-web.nhle.com/v1/standings/now")["standings"]
    abbrevs = sorted({fix(s["teamAbbrev"]["default"]) for s in stand})
    seasons = [f"{y-1}{y}" for y in season_end_years]   # 20242025 etc
    games = {}
    for ab in abbrevs:
        for season in seasons:
            try:
                data = get(f"https://api-web.nhle.com/v1/club-schedule-season/{ab}/{season}")
            except Exception:
                continue
            for g in data.get("games", []):
                if g.get("gameState") not in ("OFF", "FINAL"):
                    continue
                if g.get("gameType") not in (2, 3):     # 2=regular, 3=playoff
                    continue
                gid = g["id"]
                if gid in games:
                    continue
                h, a = fix(g["homeTeam"]["abbrev"]), fix(g["awayTeam"]["abbrev"])
                hg, ag = g["homeTeam"].get("score"), g["awayTeam"].get("score")
                if hg is None or ag is None:
                    continue
                games[gid] = {
                    "gid": gid,
                    "date": datetime.date.fromisoformat(g["gameDate"][:10]),
                    "h": h, "a": a, "hg": int(hg), "ag": int(ag),
                    "gtype": g["gameType"], "season": int(season),
                }
    return sorted(games.values(), key=lambda x: x["date"])

# ---------------------------------------------------------------- MoneyPuck shot data (windowed xG/GSAx + goalie) --
# We download the SAME MoneyPuck shot files build_nhl_xg.py uses, but instead of one stale
# all-seasons aggregate we keep each (game, goalie/team) raw tallies WITH the game's real
# date, so a point-in-time snapshot at any cutoff uses ONLY shots from games strictly before
# the cutoff (recency-weighted with the same wt()/HALFLIFE as the rating build). No future
# shot ever enters a prior prediction.
import zipfile, io
_MP_SEASONS_DONE = set()                       # which season START years have been folded in
_MP_GOALIE = collections.defaultdict(list)   # goalieId(int) -> list[(date, xg_faced, goals_against, shots_on_goal)]
_MP_TEAM = collections.defaultdict(list)      # teamAbbrev   -> list[(date, xgf, xga, n_marker)] aggregated per game
_UNBLOCKED = {"SHOT", "GOAL", "MISS"}; _ONGOAL = {"SHOT", "GOAL"}

def _mp_seasons_for(season_end_years):
    # MoneyPuck files are keyed by season START year. We need every season that could be a
    # 'prior' season for any cutoff: the test seasons themselves plus a few seasons back.
    start_years = set()
    for ey in season_end_years:
        start_years.add(ey - 1)
    return start_years

def load_moneypuck(season_start_years):
    """Populate _MP_GOALIE / _MP_TEAM from MoneyPuck shot zips for the given season START
    years. Aggregates per (game, goalieId) and per (game, team) with the game's date so a
    cutoff snapshot can sum only games strictly before it. Loads each season at most once
    (tracked in _MP_SEASONS_DONE) but WILL fold in any newly-requested season -> safe to call
    with different season ranges in one process."""
    todo = sorted(set(season_start_years) - _MP_SEASONS_DONE)
    if not todo:
        return
    for season in todo:
        cache = os.path.join(PROJ, f"mp_shots_{season}.zip")
        try:
            if os.path.exists(cache):
                data = open(cache, "rb").read()
            else:
                url = f"https://moneypuck.com/moneypuck/playerData/shots/shots_{season}.zip"
                print(f"  downloading MoneyPuck shots {season} ...", flush=True)
                req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
                data = urllib.request.urlopen(req, timeout=300).read()
                open(cache, "wb").write(data)
        except Exception as e:
            print(f"  skip MoneyPuck {season}: {e}", flush=True)
            continue
        z = zipfile.ZipFile(io.BytesIO(data))
        # per-game accumulators for this season
        gg = collections.defaultdict(lambda: collections.defaultdict(float))  # (gid,gid_goalie)->{xg,ga,sh}
        gt = collections.defaultdict(lambda: collections.defaultdict(float))  # (gid,team)->{xgf,xga}
        with z.open(z.namelist()[0]) as f:
            rd = csv.DictReader(io.TextIOWrapper(f, encoding="utf-8", errors="replace"))
            for r in rd:
                ev = r.get("event")
                if ev not in _UNBLOCKED:
                    continue
                try:
                    xg = float(r.get("xGoal") or 0)
                except ValueError:
                    xg = 0.0
                goal = 1.0 if ev == "GOAL" else 0.0
                mp_gid = r.get("game_id")
                if not mp_gid:
                    continue
                full_gid = int(f"{season}{int(mp_gid):06d}")   # reconstruct NHL gameId
                shoot = fix(r.get("teamCode")); home = fix(r.get("homeTeamCode")); away = fix(r.get("awayTeamCode"))
                isHome = (r.get("isHomeTeam") in ("1", "1.0"))
                defend = away if isHome else home
                # team xG for/against
                if shoot:
                    gt[(full_gid, shoot)]["xgf"] += xg
                if defend:
                    gt[(full_gid, defend)]["xga"] += xg
                # goalie GSAx components (xG faced over all unblocked shots; SOG = shots on goal)
                gidg = r.get("goalieIdForShot")
                if gidg not in (None, "", "NA"):
                    try:
                        gk = int(float(gidg))
                    except ValueError:
                        gk = None
                    if gk is not None:
                        d = gg[(full_gid, gk)]
                        d["xg"] += xg; d["ga"] += goal
                        if ev in _ONGOAL:
                            d["sh"] += 1.0
                    # team-level goaltending (GSAx) accrues to the DEFENDING team
                    if defend:
                        td = gt[(full_gid, defend)]
                        td["gxg"] += xg; td["gga"] += goal
                        if ev in _ONGOAL:
                            td["gsh"] += 1.0
        # fold per-game tallies into the date-tagged lists (date filled in main via _MP_DATE)
        for (full_gid, gk), d in gg.items():
            _MP_GOALIE[gk].append([full_gid, d["xg"], d["ga"], d["sh"]])
        for (full_gid, team), d in gt.items():
            _MP_TEAM[team].append([full_gid, d.get("xgf", 0.0), d.get("xga", 0.0),
                                   d.get("gxg", 0.0), d.get("gga", 0.0), d.get("gsh", 0.0)])
        print(f"  MoneyPuck {season}: {len(gg)} goalie-games, {len(gt)} team-games", flush=True)
        _MP_SEASONS_DONE.add(season)

def attach_dates(date_by_gid):
    """Replace the gameId placeholder in each MP record with the real date (from schedule).
    Records whose gameId has no known date (e.g. preseason) are dropped. IDEMPOTENT: records
    already carrying a datetime.date (from a prior call) are left untouched, so calling this
    once per backtest pass never corrupts the data."""
    def conv(rec, taillen):
        # rec[0] is either a gameId (int) not yet resolved, or an already-resolved date.
        if isinstance(rec[0], datetime.date):
            return rec
        d = date_by_gid.get(rec[0])
        return ([d] + rec[1:]) if d is not None else None
    for gk, recs in list(_MP_GOALIE.items()):
        out = [r for r in (conv(rec, 3) for rec in recs) if r is not None]
        _MP_GOALIE[gk] = out
    for team, recs in list(_MP_TEAM.items()):
        out = [r for r in (conv(rec, 5) for rec in recs) if r is not None]
        _MP_TEAM[team] = out

def team_xg_at_cutoff(cutoff_date, ref_date, teams):
    """Point-in-time per-team xGF/game, xGA/game, and team GSAx-per-shot, halflife-weighted
    over games strictly before cutoff_date (same wt()/HALFLIFE as the rating build). Returns
    {team: {'xgf':, 'xga':, 'gsax':}} for teams with any prior MoneyPuck data."""
    out = {}
    for t in teams:
        recs = _MP_TEAM.get(t)
        if not recs:
            continue
        wsum = xgf = xga = gxg = gga = gsh = 0.0
        for d, rxgf, rxga, rgxg, rgga, rgsh in recs:
            if d >= cutoff_date:
                continue
            w = 0.5 ** ((ref_date - d).days / HALFLIFE)
            wsum += w; xgf += w * rxgf; xga += w * rxga
            gxg += w * rgxg; gga += w * rgga; gsh += w * rgsh
        if wsum <= 0:
            continue
        out[t] = {"xgf": xgf / wsum, "xga": xga / wsum,
                  "gsax": ((gxg - gga) / gsh) if gsh > 0 else 0.0}
    return out

# ---- point-in-time starting goalie from the boxscore (max-TOI goalie per side) ----
def _toi_to_sec(toi):
    try:
        mm, ss = str(toi).split(":"); return int(mm) * 60 + int(ss)
    except Exception:
        return 0

_BOX_CACHE = {}
def starting_goalies(gid, game_date):
    """Return (homeGoalieId, awayGoalieId) = the max-TOI goalie on each side from the game's
    boxscore. ASSERTION: the boxscore's own gameDate must be < the day AFTER game_date, i.e.
    we never read a boxscore dated after the game we are predicting (no look-ahead)."""
    if gid in _BOX_CACHE:
        return _BOX_CACHE[gid]
    try:
        box = get(f"https://api-web.nhle.com/v1/gamecenter/{gid}/boxscore")
    except Exception:
        _BOX_CACHE[gid] = (None, None); return (None, None)
    # NO-LOOK-AHEAD ASSERTION: the boxscore must be for THIS game's date, never later.
    bd = box.get("gameDate")
    if bd:
        assert datetime.date.fromisoformat(bd[:10]) <= game_date, \
            f"LOOK-AHEAD: boxscore {gid} dated {bd} > game date {game_date}"
    pbg = box.get("playerByGameStats", {})
    def pick(side):
        gl = pbg.get(side, {}).get("goalies", [])
        if not gl:
            return None
        best = max(gl, key=lambda g: _toi_to_sec(g.get("toi", "0:00")))
        if _toi_to_sec(best.get("toi", "0:00")) <= 0:
            return None
        try:
            return int(best.get("playerId"))
        except (TypeError, ValueError):
            return None
    res = (pick("homeTeam"), pick("awayTeam"))
    _BOX_CACHE[gid] = res
    return res

def goalie_gsax_ps(goalie_id, cutoff_date, ref_date):
    """Point-in-time GSAx PER SHOT for a goalie: halflife-weighted (wt() w.r.t. ref_date,
    matching the rating build) over that goalie's games STRICTLY BEFORE cutoff_date.
    Returns (gsax_per_shot, weighted_shots) or None if below the min-sample floor."""
    recs = _MP_GOALIE.get(goalie_id)
    if not recs:
        return None
    xg = ga = sh = 0.0
    for d, gxg, gga, gsh in recs:
        if d >= cutoff_date:
            continue
        w = 0.5 ** ((ref_date - d).days / HALFLIFE)
        xg += w * gxg; ga += w * gga; sh += w * gsh
    if sh < GOALIE_MIN_SH:
        return None
    return (xg - ga) / sh, sh    # GSAx per shot on goal, weighted shot count

# ---------------------------------------------------------------- team season stats (PP/PK) --
# Prior-completed seasons only. Known before games are played -> no look-ahead.
_STAT_CACHE = {}
def team_summary(season_int):
    """Return {abbrev: {'pp':float,'pk':float,'gp':int}} for a completed season summary."""
    if season_int in _STAT_CACHE:
        return _STAT_CACHE[season_int]
    exp = urllib.parse.quote(f"seasonId={season_int} and gameTypeId=2")
    out = {}
    try:
        data = get(f"https://api.nhle.com/stats/rest/en/team/summary?limit=-1&cayenneExp={exp}").get("data", [])
    except Exception:
        data = []
    # need fullName -> abbrev; build from standings name map
    for r in data:
        # the summary endpoint carries teamId; map via abbrevs in standings is unreliable,
        # but teamFullName is stable. We reuse a name->abbrev map built lazily.
        out[r.get("teamFullName")] = {
            "pp": (r.get("powerPlayPct") or 0.0),
            "pk": (r.get("penaltyKillPct") or 0.0),
            "gp": (r.get("gamesPlayed") or 0),
        }
    _STAT_CACHE[season_int] = out
    return out

def build_name2ab():
    stand = get("https://api-web.nhle.com/v1/standings/now")["standings"]
    m = {s["teamName"]["default"]: fix(s["teamAbbrev"]["default"]) for s in stand}
    m["Arizona Coyotes"] = "UTA"
    return m
NAME2AB = build_name2ab()

def pppk_at_cutoff(cutoff_date):
    """Recency-weighted PP/PK per team using ONLY seasons completed before cutoff.
    Uses season summary stats of seasons strictly prior to the cutoff's season."""
    cur_start = season_start_year(cutoff_date)
    acc = collections.defaultdict(lambda: collections.defaultdict(float))
    accw = collections.defaultdict(float)
    # weight prior seasons by SEASON_WEIGHTS, newest completed first
    for i in range(1, 5):
        sy = cur_start - i
        season_int = int(f"{sy}{sy+1}")
        w = SEASON_WEIGHTS[min(i, len(SEASON_WEIGHTS) - 1)]
        for fullname, rec in team_summary(season_int).items():
            ab = NAME2AB.get(fullname)
            if not ab:
                continue
            gp = rec["gp"] or 1
            accw[ab] += w * gp
            acc[ab]["pp"] += w * gp * rec["pp"]
            acc[ab]["pk"] += w * gp * rec["pk"]
    return {ab: {"pp": acc[ab]["pp"] / accw[ab], "pk": acc[ab]["pk"] / accw[ab]}
            for ab in acc if accw[ab] > 0}

# ---------------------------------------------------------------- rating build -----
def zdict(d):
    v = list(d.values()); m = sum(v) / len(v)
    sd = (sum((x - m) ** 2 for x in v) / len(v)) ** 0.5 or 1.0
    return {t: (x - m) / sd for t, x in d.items()}

def build_ratings(games_prior, cutoff_date, pppk, txg=None):
    """Replicate build_nhl.py's prediction-relevant rating path from games strictly
    before cutoff_date. Returns (att, dfn, AVG, HOME_ADV, teams).
    txg: optional point-in-time {team:{'xgf','xga','gsax'}} (change #2). When the XF/XA/GS
    weights are 0 this is unused, so the build is byte-identical to the baseline."""
    G = [g for g in games_prior if g["date"] < cutoff_date]
    if not G:
        return None
    teams = sorted({t for g in G for t in (g["h"], g["a"])})
    if len(teams) < 20:
        return None
    ref = max(g["date"] for g in G)
    def wt(d):
        return 0.5 ** ((ref - d).days / HALFLIFE)
    # league averages + home adv (recency weighted)
    tg = tw = hg_ = ag_ = 0.0
    for g in G:
        w = wt(g["date"]); tg += w * (g["hg"] + g["ag"]); tw += 2 * w
        hg_ += w * g["hg"]; ag_ += w * g["ag"]
    AVG = tg / tw
    HOME_ADV = min(1.12, max(1.0, hg_ / ag_))
    # Poisson att/def. Pseudo-count shrinkage (change #5): adding POIS_KAPPA weighted
    # pseudo-observations to BOTH numerator and the (AVG-scaled) denominator pulls a team's
    # multiplier toward 1.0 in proportion to how little real data it has -> low-sample teams
    # regress to league mean. KAPPA=0 -> na/da exactly -> baseline.
    att = {t: 1.0 for t in teams}; dfn = {t: 1.0 for t in teams}
    for _ in range(POISSON_ITERS):
        na = {t: 0. for t in teams}; da = dict(na); nd = dict(na); dd = dict(na)
        for g in G:
            h, a, hgl, agl = g["h"], g["a"], g["hg"], g["ag"]; w = wt(g["date"])
            na[h] += w * hgl; da[h] += w * AVG * dfn[a]; nd[a] += w * hgl; dd[a] += w * AVG * att[h]
            na[a] += w * agl; da[a] += w * AVG * dfn[h]; nd[h] += w * agl; dd[h] += w * AVG * att[a]
        for t in teams:
            if da[t] + POIS_KAPPA > 0: att[t] = (na[t] + POIS_KAPPA * AVG) / (da[t] + POIS_KAPPA * AVG)
            if dd[t] + POIS_KAPPA > 0: dfn[t] = (nd[t] + POIS_KAPPA * AVG) / (dd[t] + POIS_KAPPA * AVG)
        for dct in (att, dfn):
            gmn = math.exp(sum(math.log(max(v, 1e-6)) for v in dct.values()) / len(dct))
            for t in dct: dct[t] /= gmn
    # Elo (MOV weighted). Between-season reversion (change #5): at each season boundary pull
    # toward 1500 with carry ELO_CARRY (1.0 = no reversion -> baseline continuous Elo).
    elo = {t: 1500.0 for t in teams}
    prev_season = None
    for g in sorted(G, key=lambda x: x["date"]):
        if ELO_CARRY != 1.0:
            s = season_start_year(g["date"])
            if prev_season is not None and s != prev_season:
                for t in elo:
                    elo[t] = 1500.0 + ELO_CARRY * (elo[t] - 1500.0)
            prev_season = s
        h, a, hgl, agl = g["h"], g["a"], g["hg"], g["ag"]
        eh, ea = elo[h], elo[a]
        exp = 1 / (1 + 10 ** ((ea - (eh + 50)) / 400))
        res = 1.0 if hgl > agl else 0.0 if hgl < agl else 0.5
        gg = 1 + 0.5 * abs(hgl - agl); dl = 6 * gg * (res - exp)
        elo[h] = eh + dl; elo[a] = ea - dl
    # z-blend. xG/GSAx terms (change #2): re-enabled at the tuned XF/XA/GS weights using a
    # WINDOWED, point-in-time MoneyPuck snapshot (txg) built only from shots before cutoff.
    # When XF=XA=GS=0 the next three z-dicts are multiplied by 0 -> identical to baseline.
    zA = zdict({t: math.log(att[t]) for t in teams})
    zD = zdict({t: -math.log(dfn[t]) for t in teams})
    zElo = zdict(elo)
    zPP = zdict({t: pppk.get(t, {}).get("pp", 0.0) for t in teams})
    zPK = zdict({t: pppk.get(t, {}).get("pk", 0.0) for t in teams})
    txg = txg or {}
    _xgfm = (sum(v["xgf"] for v in txg.values()) / len(txg)) if txg else 3.0
    _xgam = (sum(v["xga"] for v in txg.values()) / len(txg)) if txg else 3.0
    zXGF = zdict({t: txg.get(t, {}).get("xgf", _xgfm) for t in teams})
    zXGA = zdict({t: -txg.get(t, {}).get("xga", _xgam) for t in teams})
    zGSAx = zdict({t: txg.get(t, {}).get("gsax", 0.0) for t in teams})
    attZ = {t: GA * zA[t] + XF_W * zXGF[t] + PP_W * zPP[t] + EA * zElo[t] for t in teams}
    defZ = {t: GD * zD[t] + XA_W * zXGA[t] + PK_W * zPK[t] + GS_W * zGSAx[t] + ED * zElo[t] for t in teams}
    lA = [math.log(att[t]) for t in teams]; mA = sum(lA) / len(lA)
    sA = (sum((x - mA) ** 2 for x in lA) / len(lA)) ** 0.5
    lD = [-math.log(dfn[t]) for t in teams]; mD = sum(lD) / len(lD)
    sD = (sum((x - mD) ** 2 for x in lD) / len(lD)) ** 0.5
    att = {t: math.exp(mA + sA * attZ[t]) for t in teams}
    dfn = {t: math.exp(-(mD + sD * defZ[t])) for t in teams}
    # goal-level calibration k
    infl = sum(att[a] * dfn[b] for a, b in itertools.permutations(teams, 2)) / (len(teams) * (len(teams) - 1))
    k = infl ** 0.5
    for t in teams:
        att[t] /= k; dfn[t] /= k
    return {"att": att, "dfn": dfn, "AVG": AVG, "HOME": HOME_ADV, "teams": set(teams),
            "elo": elo, "ref": ref}

def apply_consensus_blend(rat, cons_rank, games_played):
    """Replicate blend_nhl.py in net space. Mutates a COPY of att/dfn. Returns new dicts."""
    teams = list(rat["teams"])
    model_net = {t: rat["att"][t] - rat["dfn"][t] for t in teams}
    net_desc = sorted(model_net.values(), reverse=True)
    att = dict(rat["att"]); dfn = dict(rat["dfn"])
    for t in teams:
        gp = games_played.get(t, 0)
        W = min(1.0, BLEND_BASE + gp / SEASON_LEN)
        rk = cons_rank.get(t)
        if rk is None or rk > len(net_desc):
            continue
        cnet = net_desc[rk - 1]
        fnet = W * model_net[t] + (1.0 - W) * cnet
        delta = fnet - model_net[t]
        att[t] = rat["att"][t] + delta / 2.0
        dfn[t] = rat["dfn"][t] - delta / 2.0
    return att, dfn

# ---------------------------------------------------------------- prediction engine (== nhl_predict.py) --
def _dc_tau(i, j, lh, la, rho):
    """Dixon-Coles low-score dependence correction (change #4). Inflates the 0-0/1-1 and
    deflates 1-0/0-1 cells so the model produces MORE 1-goal / tie finishes (independent
    Poisson under-produces the OT/SO tie rate). rho=0 -> tau==1 (pure independent Poisson)."""
    if rho == 0.0:
        return 1.0
    if i == 0 and j == 0:
        return 1.0 - lh * la * rho
    if i == 0 and j == 1:
        return 1.0 + lh * rho
    if i == 1 and j == 0:
        return 1.0 + la * rho
    if i == 1 and j == 1:
        return 1.0 - rho
    return 1.0

def predict(att, dfn, AVG, HOME, home, away, gadj_h=0.0, gadj_a=0.0):
    """gadj_h / gadj_a (change #1): point-in-time starting-goalie GOAL adjustments already
    computed via gadj() in the loop. gadj_h adjusts AWAY goals (home goalie), gadj_a adjusts
    HOME goals. Both 0.0 -> byte-identical to the team-average-goalie baseline."""
    lh = AVG * att[home] * dfn[away] * HOME
    la = AVG * att[away] * dfn[home]
    lh += gadj_a        # away goalie affects home scoring
    la += gadj_h        # home goalie affects away scoring
    lh = max(0.5, lh); la = max(0.5, la)
    P = lambda kk, l: math.exp(-l) * l ** kk / math.factorial(kk)
    pH = pT = pA = 0.0
    ph = [P(i, lh) for i in range(14)]
    pa = [P(j, la) for j in range(14)]
    for i in range(14):
        for j in range(14):
            m = ph[i] * pa[j] * _dc_tau(i, j, lh, la, DC_RHO)
            if i > j: pH += m
            elif i == j: pT += m
            else: pA += m
    tot = pH + pT + pA      # DC tau breaks normalisation slightly -> renormalise
    if tot > 0:
        pH /= tot; pT /= tot; pA /= tot
    fav = pH / (pH + pA) if pH + pA else 0.5
    winH_raw = pH + pT * (0.5 + (fav - 0.5) * 0.35)   # pre-recalibration favourite prob
    winH = recalibrate(winH_raw)    # production win-prob path: soften overconfident scale
    # ABSTAIN / regress the near-pick'em band (change #3): when the projected goal margin is
    # tiny the game is a coin flip the model cannot call -> shrink winH toward 0.50. graded
    # flags whether the game survives the abstain filter (reported separately).
    graded = True
    if ABSTAIN_MARGIN > 0.0 and abs(lh - la) < ABSTAIN_MARGIN:
        winH = 0.5 + (winH - 0.5) * (1.0 - ABSTAIN_SHRINK)
        graded = False
    return winH, lh, la, graded, winH_raw

# ---------------------------------------------------------------- backtest loop ----
def monday_of(d):
    return d - datetime.timedelta(days=d.weekday())

def run_backtest(test_season_end_years, history_seasons_back=3, use_consensus=False, label=""):
    # pull enough history: oldest test season needs `history_seasons_back` prior seasons
    earliest_test = min(test_season_end_years)
    pull_years = list(range(earliest_test - history_seasons_back, max(test_season_end_years) + 1))
    print(f"Loading games for season-end years {pull_years} ...", flush=True)
    all_games = load_all_games(pull_years)
    print(f"  loaded {len(all_games)} unique completed games "
          f"({all_games[0]['date']} .. {all_games[-1]['date']})", flush=True)

    # test games = regular+playoff games whose season-end-year is in test window
    def end_year(g):
        # season int like 20242025 -> end year 2025
        return int(str(g["season"])[4:])
    test_games = [g for g in all_games if end_year(g) in test_season_end_years]
    print(f"  test games (before min-data filter): {len(test_games)}", flush=True)

    cons_rank = {}
    if use_consensus:
        with open(os.path.join(PROJ, "consensus_nhl.csv"), encoding="utf-8") as f:
            for r in csv.DictReader(f):
                cons_rank[r["team"]] = int(r["consensus_rank"])

    # MoneyPuck shot data (windowed xG/GSAx + point-in-time goalie). Only loaded when a
    # change that needs it is active, so the pure baseline never touches the network for it.
    need_mp = USE_GOALIE or XF_W or XA_W or GS_W
    if need_mp:
        load_moneypuck(_mp_seasons_for(pull_years))
        attach_dates({g["gid"]: g["date"] for g in all_games})

    # weekly rating snapshots
    snap_cache = {}
    pppk_cache = {}
    txg_cache = {}
    results = []
    skipped = 0
    games_by_date = sorted(all_games, key=lambda x: x["date"])

    for g in test_games:
        cutoff = monday_of(g["date"])
        if cutoff not in snap_cache:
            cstart = season_start_year(cutoff)
            if cstart not in pppk_cache:
                pppk_cache[cstart] = pppk_at_cutoff(cutoff)
            base = build_ratings(all_games, cutoff, pppk_cache[cstart])
            txg = None
            if need_mp and base is not None:
                txg = team_xg_at_cutoff(cutoff, base["ref"], base["teams"])
                # Only rebuild ratings if the xG z-blend weights are active (change #2). The
                # goalie path (change #1) reads team_base GSAx straight from txg, so it does
                # NOT need a rebuild -> avoids a costly second 60-iter Poisson solve per cutoff.
                if (XF_W or XA_W or GS_W):
                    snap_cache[cutoff] = build_ratings(all_games, cutoff, pppk_cache[cstart], txg=txg)
                else:
                    snap_cache[cutoff] = base
            else:
                snap_cache[cutoff] = base
            txg_cache[cutoff] = txg
        rat = snap_cache[cutoff]
        if rat is None or g["h"] not in rat["teams"] or g["a"] not in rat["teams"]:
            skipped += 1
            continue
        # both teams need enough prior games this build to be rated (>=10 prior games each)
        # count current-season games played by each team before cutoff
        cstart = season_start_year(cutoff)
        if use_consensus:
            gp = collections.Counter()
            for pg in games_by_date:
                if pg["date"] >= cutoff:
                    break
                if season_start_year(pg["date"]) == cstart:
                    gp[pg["h"]] += 1; gp[pg["a"]] += 1
            att, dfn = apply_consensus_blend(rat, cons_rank, gp)
        else:
            att, dfn = rat["att"], rat["dfn"]

        # ---- point-in-time STARTING GOALIE adjustment (change #1) ----
        gadj_h = gadj_a = 0.0
        if USE_GOALIE:
            txg = txg_cache.get(cutoff) or {}
            gH, gA = starting_goalies(g["gid"], g["date"])
            ref = rat["ref"]
            # gadj() == nhl_predict.py: (team baseline GSAx/shot - goalie GSAx/shot) * SOG.
            # Positive => goalie worse than the team baseline already in the rating => MORE
            # goals to the opponent. Unknown goalie or too-thin history => 0 (no adjustment).
            def gadj(goalie_id, team):
                if goalie_id is None:
                    return 0.0
                res = goalie_gsax_ps(goalie_id, cutoff, ref)
                if res is None:
                    return 0.0
                gps, sh = res
                team_base = txg.get(team, {}).get("gsax", 0.0)
                # sample-size shrinkage toward no-nudge + optional global damping
                shrink = sh / (sh + GOALIE_SHRINK_K) if GOALIE_SHRINK_K > 0 else 1.0
                return (team_base - gps) * SOG * shrink * GOALIE_DAMP
            gadj_h = gadj(gH, g["h"])     # home goalie -> affects away goals
            gadj_a = gadj(gA, g["a"])     # away goalie -> affects home goals

        winH, lh, la, graded, winH_raw = predict(att, dfn, rat["AVG"], rat["HOME"], g["h"], g["a"],
                                                  gadj_h=gadj_h, gadj_a=gadj_a)
        home_won = 1 if g["hg"] > g["ag"] else 0
        # model favourite + whether it won
        if winH >= 0.5:
            fav_prob = winH; fav_won = home_won
        else:
            fav_prob = 1 - winH; fav_won = 1 - home_won
        results.append({
            "date": g["date"].isoformat(), "h": g["h"], "a": g["a"],
            "hg": g["hg"], "ag": g["ag"], "gtype": g["gtype"],
            "winH": winH, "home_won": home_won,
            "fav_prob": fav_prob, "fav_won": fav_won,
            "lh": lh, "la": la, "graded": graded, "winH_raw": winH_raw,
        })

    print(f"  scored {len(results)} games | skipped {skipped} (insufficient data / team not rated)", flush=True)
    return results

# ---------------------------------------------------------------- scoring / report -
def auc_score(probs, labels):
    """Area under ROC for binary labels via the Mann-Whitney rank statistic
    (handles ties at 0.5). O(n log n)."""
    import bisect
    pos = [p for p, y in zip(probs, labels) if y == 1]
    neg = sorted(p for p, y in zip(probs, labels) if y == 0)
    if not pos or not neg:
        return float("nan")
    tot = 0.0
    for p in pos:
        lo = bisect.bisect_left(neg, p); hi = bisect.bisect_right(neg, p)
        tot += lo + 0.5 * (hi - lo)
    return tot / (len(pos) * len(neg))

def report(results, label):
    n = len(results)
    if n == 0:
        print("NO GAMES SCORED"); return
    print("\n" + "=" * 74)
    print(f" OOS WALK-FORWARD BACKTEST  {label}   (n = {n} games)")
    print("=" * 74)

    # headline winner hit-rate
    hits = sum(r["fav_won"] for r in results)
    hitrate = hits / n
    # coin flip baseline & home-pick baseline
    home_wins = sum(r["home_won"] for r in results)
    print(f"\nHEADLINE  straight-up WINNER hit-rate (model favourite wins): "
          f"{hitrate*100:.2f}%  ({hits}/{n})")
    print(f"  baselines: coin-flip 50.0%  |  always-pick-home {home_wins/n*100:.2f}%  "
          f"|  market benchmark ~57-59%")

    # Brier + log-loss (on home win prob vs home_won)
    brier = sum((r["winH"] - r["home_won"]) ** 2 for r in results) / n
    eps = 1e-12
    logloss = -sum(r["home_won"] * math.log(max(r["winH"], eps)) +
                   (1 - r["home_won"]) * math.log(max(1 - r["winH"], eps)) for r in results) / n
    auc = auc_score([r["winH"] for r in results], [r["home_won"] for r in results])
    print(f"\nBrier score   = {brier:.4f}   (lower better; 0.25 = coin flip, market ~0.235-0.245)")
    print(f"Log-loss      = {logloss:.4f}   (lower better; 0.6931 = coin flip)")
    print(f"AUC           = {auc:.4f}   (discrimination; 0.5 = coin flip)")

    # GRADED-ONLY headline: if the abstain band is active, some games are flagged
    # graded=False (projected |lambda margin| in the near-pick'em dead zone where the
    # model is no better than a coin flip). Report the winner hit-rate on the GRADED
    # subset too -- this is the headline you would actually bet.
    graded = [r for r in results if r.get("graded", True)]
    n_ab = n - len(graded)
    if n_ab > 0 and graded:
        g_hits = sum(r["fav_won"] for r in graded)
        print(f"\nGRADED-ONLY  straight-up hit-rate (excl. {n_ab} abstained pick'em games): "
              f"{g_hits/len(graded)*100:.2f}%  ({g_hits}/{len(graded)})")

    # calibration buckets on the FAVOURITE probability
    print("\nCALIBRATION by favourite-confidence bucket")
    print(f"  {'bucket':>9s} {'N':>5s} {'pred win%':>10s} {'actual win%':>12s} {'gap':>7s}")
    buckets = [(0.50, 0.60), (0.60, 0.70), (0.70, 0.80), (0.80, 1.0001)]
    for lo, hi in buckets:
        sub = [r for r in results if lo <= r["fav_prob"] < hi]
        if not sub:
            print(f"  {int(lo*100)}-{int(hi*100):>3d}%   {0:>5d}        --           --      --")
            continue
        pred = sum(r["fav_prob"] for r in sub) / len(sub)
        act = sum(r["fav_won"] for r in sub) / len(sub)
        print(f"  {int(lo*100)}-{min(100,int(hi*100)):>3d}%   {len(sub):>5d} {pred*100:>9.1f}% "
              f"{act*100:>11.1f}% {(act-pred)*100:>+6.1f}%")

    # trust-tier table: among picks with fav_prob >= threshold
    print("\nTRUST-TIER table (picks at/above each confidence threshold)")
    print(f"  {'threshold':>10s} {'N picks':>8s} {'hit-rate':>9s}  {'vs market 57-59%':>18s}")
    for thr in (0.50, 0.60, 0.65, 0.70, 0.75, 0.80):
        sub = [r for r in results if r["fav_prob"] >= thr]
        if not sub:
            print(f"  >= {int(thr*100)}%      {0:>6d}       --")
            continue
        hr = sum(r["fav_won"] for r in sub) / len(sub)
        flag = "above" if hr >= 0.59 else ("in-range" if hr >= 0.57 else "below")
        print(f"  >= {int(thr*100)}%      {len(sub):>6d}   {hr*100:>7.2f}%        {flag:>12s}")

    # margin / goals diagnostics (MAE of projected goal margin vs actual)
    mae = sum(abs((r["lh"] - r["la"]) - (r["hg"] - r["ag"])) for r in results) / n
    print(f"\nProjected goal-margin MAE vs actual = {mae:.2f} goals "
          f"(hockey margins are very noisy; reference only)")
    print("ATS / closing-spread hit-rate: SKIPPED -- no cheap historical NHL puck-line "
          "closing odds source available.")

    # per-team strength/weakness: hit-rate when the team is the model favourite
    fav_team = collections.defaultdict(lambda: [0, 0])  # team -> [hits, n]
    for r in results:
        fav = r["h"] if r["winH"] >= 0.5 else r["a"]
        fav_team[fav][1] += 1
        fav_team[fav][0] += r["fav_won"]
    rated = [(t, h, c) for t, (h, c) in fav_team.items() if c >= 15]
    best = sorted(rated, key=lambda x: -x[1] / x[2])[:6]
    worst = sorted(rated, key=lambda x: x[1] / x[2])[:6]
    print("\nWHERE THE MODEL IS STRONG (best favourite-pick hit-rate, >=15 picks):")
    for t, h, c in best:
        print(f"  {t:4s} {h/c*100:5.1f}%  ({h}/{c})")
    print("WHERE THE MODEL IS WEAK (worst favourite-pick hit-rate, >=15 picks):")
    for t, h, c in worst:
        print(f"  {t:4s} {h/c*100:5.1f}%  ({h}/{c})")
    graded_hr = (sum(r["fav_won"] for r in graded) / len(graded)) if graded else float("nan")
    return {"n": n, "hitrate": hitrate, "brier": brier, "logloss": logloss, "auc": auc,
            "graded_hitrate": graded_hr, "n_graded": len(graded)}

# ---------------------------------------------------------------- main -------------
if __name__ == "__main__":
    # Most recent COMPLETE season is 2025-26 (end year 2026); the prior complete season
    # is 2024-25 (end 2025). Use both for a bigger sample.
    TEST = [2025, 2026]   # season END years -> 2024-25 and 2025-26 seasons
    # (We test seasons that have >=3 prior seasons of data so ratings are well-formed.)

    # CALIBRATION REFIT (change #4): if NHL_CALIB=1, fit the win-prob recalibration (Platt
    # a+b*logit) on an EARLIER block of OOS seasons and apply it to the held-out TEST window.
    # This replaces the in-sample fixed temperature with an out-of-sample-fit map, so the
    # calibration is never tuned on the games it scores. The straight-up hit-rate is
    # unaffected by a monotone map; only Brier/log-loss move.
    CALIB_TRAIN = [2023, 2024]   # earlier seasons used ONLY to fit calibration
    if os.environ.get("NHL_CALIB") == "1":
        print("### CALIBRATION FIT on earlier OOS seasons", CALIB_TRAIN, "###", flush=True)
        cal = run_backtest(CALIB_TRAIN, history_seasons_back=3, use_consensus=False,
                           label="CALIB-FIT")
        a, b = fit_calibration([r["winH_raw"] for r in cal], [r["home_won"] for r in cal])
        CALIB_MODEL = ("platt", a, b)
        print(f"  fitted Platt calibration on n={len(cal)} earlier OOS games: "
              f"a={a:.4f} b={b:.4f}  (b<1 => softens like a temperature)", flush=True)

    print("\n### NHL HONEST WALK-FORWARD BACKTEST (model-only, no consensus leak) ###\n")
    res_model = run_backtest(TEST, history_seasons_back=3, use_consensus=False,
                             label="MODEL-ONLY")
    summ_model = report(res_model, "[MODEL-ONLY, no consensus prior]")

    # reference: consensus-blended variant (uses CURRENT preseason consensus -> flagged)
    print("\n\n### REFERENCE: consensus-blended variant (current-season consensus; "
          "decays out as games are played) ###")
    res_blend = run_backtest(TEST, history_seasons_back=3, use_consensus=True,
                             label="CONSENSUS-BLEND")
    summ_blend = report(res_blend, "[CONSENSUS-BLEND -- ref only, see caveat]")

    # write per-game detail of the headline (model-only) run
    out = os.path.join(PROJ, "_backtest_nhl_games.csv")
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["date", "home", "away", "hg", "ag", "gtype", "p_home_win",
                    "home_won", "fav_prob", "fav_won", "lambda_home", "lambda_away",
                    "graded", "p_home_win_raw"])
        for r in res_model:
            w.writerow([r["date"], r["h"], r["a"], r["hg"], r["ag"], r["gtype"],
                        round(r["winH"], 4), r["home_won"], round(r["fav_prob"], 4),
                        r["fav_won"], round(r["lh"], 3), round(r["la"], 3),
                        int(r.get("graded", True)), round(r.get("winH_raw", r["winH"]), 4)])
    print(f"\nWrote {out}")
