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
      - recency-weighted (HALFLIFE=70d + season weights) Poisson attack/defense (60 iters)
      - MOV-weighted Elo
      - z-blend: attZ = GA*zAtt + EA*zElo ; defZ = GD*zDef + ED*zElo  (+PP/PK)
      - mapped back to the goals log-scale, goal-level calibration k
    The production model down-weights the xG/GSAx aggregate terms to 0.0 (XF=XA=GS=0),
    so omitting them here is faithful, NOT a simplification. PP/PK use prior-COMPLETED-
    season team summary stats (known before the season) so they carry no look-ahead.
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
  slight favourite edge (winH = pH + pT*(0.5 + (fav-0.5)*0.35)). Rest/goalie/availability
  knobs are NOT applied (we have no point-in-time starting-goalie/injury data; the
  ratings carry team strength, which is what we are measuring).

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

# Win-prob recalibration (mirrors nhl_predict.py / web/nhl_app.js). The raw Poisson
# favourite probabilities are systematically overconfident OOS (see the calibration table
# below); a logit temperature T>1 softens the SCALE toward 0.5 without changing ordering
# (so the straight-up hit-rate is unchanged) -- T=2.0 minimises Brier + log-loss here.
WINPROB_TEMP = 2.0
def recalibrate(p, T=WINPROB_TEMP):
    if T == 1.0:
        return p
    p = min(max(p, 1e-9), 1 - 1e-9)
    return 1.0 / (1.0 + math.exp(-math.log(p / (1 - p)) / T))

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
                    "date": datetime.date.fromisoformat(g["gameDate"][:10]),
                    "h": h, "a": a, "hg": int(hg), "ag": int(ag),
                    "gtype": g["gameType"], "season": int(season),
                }
    return sorted(games.values(), key=lambda x: x["date"])

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

def build_ratings(games_prior, cutoff_date, pppk):
    """Replicate build_nhl.py's prediction-relevant rating path from games strictly
    before cutoff_date. Returns (att, dfn, AVG, HOME_ADV, teams)."""
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
    # Poisson att/def
    att = {t: 1.0 for t in teams}; dfn = {t: 1.0 for t in teams}
    for _ in range(POISSON_ITERS):
        na = {t: 0. for t in teams}; da = dict(na); nd = dict(na); dd = dict(na)
        for g in G:
            h, a, hgl, agl = g["h"], g["a"], g["hg"], g["ag"]; w = wt(g["date"])
            na[h] += w * hgl; da[h] += w * AVG * dfn[a]; nd[a] += w * hgl; dd[a] += w * AVG * att[h]
            na[a] += w * agl; da[a] += w * AVG * dfn[h]; nd[h] += w * agl; dd[h] += w * AVG * att[a]
        for t in teams:
            if da[t] > 0: att[t] = na[t] / da[t]
            if dd[t] > 0: dfn[t] = nd[t] / dd[t]
        for dct in (att, dfn):
            gmn = math.exp(sum(math.log(max(v, 1e-6)) for v in dct.values()) / len(dct))
            for t in dct: dct[t] /= gmn
    # Elo (MOV weighted)
    elo = {t: 1500.0 for t in teams}
    for g in sorted(G, key=lambda x: x["date"]):
        h, a, hgl, agl = g["h"], g["a"], g["hg"], g["ag"]
        eh, ea = elo[h], elo[a]
        exp = 1 / (1 + 10 ** ((ea - (eh + 50)) / 400))
        res = 1.0 if hgl > agl else 0.0 if hgl < agl else 0.5
        gg = 1 + 0.5 * abs(hgl - agl); dl = 6 * gg * (res - exp)
        elo[h] = eh + dl; elo[a] = ea - dl
    # z-blend (xG/GSAx omitted -> weights are 0.0 in production)
    zA = zdict({t: math.log(att[t]) for t in teams})
    zD = zdict({t: -math.log(dfn[t]) for t in teams})
    zElo = zdict(elo)
    zPP = zdict({t: pppk.get(t, {}).get("pp", 0.0) for t in teams})
    zPK = zdict({t: pppk.get(t, {}).get("pk", 0.0) for t in teams})
    attZ = {t: GA * zA[t] + PP_W * zPP[t] + EA * zElo[t] for t in teams}
    defZ = {t: GD * zD[t] + PK_W * zPK[t] + ED * zElo[t] for t in teams}
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
    return {"att": att, "dfn": dfn, "AVG": AVG, "HOME": HOME_ADV, "teams": set(teams), "elo": elo}

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
def predict(att, dfn, AVG, HOME, home, away):
    lh = AVG * att[home] * dfn[away] * HOME
    la = AVG * att[away] * dfn[home]
    lh = max(0.5, lh); la = max(0.5, la)
    P = lambda kk, l: math.exp(-l) * l ** kk / math.factorial(kk)
    pH = pT = pA = 0.0
    ph = [P(i, lh) for i in range(14)]
    pa = [P(j, la) for j in range(14)]
    for i in range(14):
        for j in range(14):
            m = ph[i] * pa[j]
            if i > j: pH += m
            elif i == j: pT += m
            else: pA += m
    fav = pH / (pH + pA) if pH + pA else 0.5
    winH = pH + pT * (0.5 + (fav - 0.5) * 0.35)
    winH = recalibrate(winH)        # production win-prob path: soften overconfident scale
    return winH, lh, la

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

    # weekly rating snapshots
    snap_cache = {}
    pppk_cache = {}
    results = []
    skipped = 0
    games_by_date = sorted(all_games, key=lambda x: x["date"])

    for g in test_games:
        cutoff = monday_of(g["date"])
        if cutoff not in snap_cache:
            cstart = season_start_year(cutoff)
            if cstart not in pppk_cache:
                pppk_cache[cstart] = pppk_at_cutoff(cutoff)
            snap_cache[cutoff] = build_ratings(all_games, cutoff, pppk_cache[cstart])
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

        winH, lh, la = predict(att, dfn, rat["AVG"], rat["HOME"], g["h"], g["a"])
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
            "lh": lh, "la": la,
        })

    print(f"  scored {len(results)} games | skipped {skipped} (insufficient data / team not rated)", flush=True)
    return results

# ---------------------------------------------------------------- scoring / report -
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
    print(f"\nBrier score   = {brier:.4f}   (lower better; 0.25 = coin flip, market ~0.235-0.245)")
    print(f"Log-loss      = {logloss:.4f}   (lower better; 0.6931 = coin flip)")

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
    return {"n": n, "hitrate": hitrate, "brier": brier, "logloss": logloss}

# ---------------------------------------------------------------- main -------------
if __name__ == "__main__":
    # Most recent COMPLETE season is 2025-26 (end year 2026); the prior complete season
    # is 2024-25 (end 2025). Use both for a bigger sample.
    TEST = [2025, 2026]   # season END years -> 2024-25 and 2025-26 seasons
    # (We test seasons that have >=3 prior seasons of data so ratings are well-formed.)

    print("### NHL HONEST WALK-FORWARD BACKTEST (model-only, no consensus leak) ###\n")
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
                    "home_won", "fav_prob", "fav_won", "lambda_home", "lambda_away"])
        for r in res_model:
            w.writerow([r["date"], r["h"], r["a"], r["hg"], r["ag"], r["gtype"],
                        round(r["winH"], 4), r["home_won"], round(r["fav_prob"], 4),
                        r["fav_won"], round(r["lh"], 3), round(r["la"], 3)])
    print(f"\nWrote {out}")
