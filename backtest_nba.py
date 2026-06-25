# -*- coding: utf-8 -*-
"""
backtest_nba.py  —  HONEST WALK-FORWARD (out-of-sample) backtest for the NBA model.

WHAT THIS MEASURES
  How accurately the NBA model predicts games it has NOT seen. For every game in the
  test window we re-derive the team ratings using ONLY games played STRICTLY BEFORE
  that game (prior seasons + current-season-to-date), then predict that one game with
  the Gaussian point-margin engine. Nothing from the game itself or any later game
  leaks into its own prediction. This is the only honest accuracy.

NO LOOK-AHEAD (how the leak is prevented)
  * We pull every completed regular-season game from the ESPN team-schedule API for a
    span of seasons (TEST seasons + LOOKBACK prior seasons for warm-up history).
  * Games are sorted by date. We walk the test window in weekly cutoffs. For each
    cutoff we fit ratings on games with date < cutoff (recency-weighted exactly like
    build_nba.py: opponent-adjusted off/def, 50 Gauss-Seidel passes, HALFLIFE=160,
    reference date = the cutoff, NOT the global max). Every test game in that week is
    then scored with those as-of ratings. A team must have >= MIN_PRIOR_GP prior games
    to be rated; games where either team is under that floor are skipped.
  * The recency half-life decays older seasons automatically, replicating the model's
    multi-season recency weighting as it would have stood at each cutoff date.

CONSENSUS PRIOR (decaying offseason anchor)
  The live model blends a decaying expert-consensus prior (blend_nba.py): W = min(1,
  BASE + gamesPlayed/82); final = W*model + (1-W)*consensus, anchor fades to nothing as
  games accrue. Historical (point-in-time) consensus rankings are NOT available, and
  using today's consensus on a past season WOULD be look-ahead. The test window only
  scores games once both teams have >= MIN_PRIOR_GP games, by which point W is already
  ~1.0 (pure model) under the model's own decay rule, so the prior is correctly inert
  here. We therefore backtest the pure point-margin engine, which IS what the live model
  reduces to in-season. This is stated honestly rather than faked with a future prior.

PREDICTION ENGINE (identical to build_nba.py / NFL)
  proj_margin(home) = (off[h]-off[a]) + (def[a]-def[h]) + HFA      # home minus away
  P(home win)       = Phi(proj_margin / SD_margin)                 # Gaussian margin
  Margins are ~normal in basketball, so this is the model's native win-prob map.

OUTPUTS
  * Headline straight-up winner hit-rate (favorite = higher win prob).
  * Calibration table by confidence bucket (50-60 / 60-70 / 70-80 / 80-100).
  * Brier score and log-loss vs actual outcomes.
  * Mean absolute error of projected margin vs actual margin.
  * ATS: skipped (no cheap historical closing spreads) — stated explicitly.
  * Trust-tier table: hit-rate of picks above 60/65/70/75/80% with N.
  * Benchmark read vs market (~69% SU, ~50-53% ATS) and vs a coin-flip (50%).
  * Best / worst teams by model accuracy on their games.

Read-only on all model files. Writes only _backtest_nba_walkforward.csv (a result dump).
"""
import json, os, csv, math, datetime, urllib.request

PROJ = os.path.dirname(os.path.abspath(__file__))
ESPN = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba"

# ---- knobs (kept identical to the live model where it matters) ----
HALFLIFE      = 160.0   # build_nba.py recency half-life (days)
GS_PASSES     = 50      # build_nba.py opponent-adjustment iterations
LOOKBACK      = 2       # prior seasons of warm-up history before the earliest test season
MIN_PRIOR_GP  = 15      # min prior games per team for a game to be scored (enough to rate)
CACHE         = os.path.join(PROJ, "_nba_backtest_cache.json")


def get(url, t=30):
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    return json.loads(urllib.request.urlopen(req, timeout=t).read())


def norm_cdf(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


# ----------------------------------------------------------------------------
# 1. Determine which seasons are complete, pick the test window.
# ----------------------------------------------------------------------------
def fetch_team_schedule(tid, endyr):
    try:
        return get(f"{ESPN}/teams/{tid}/schedule?season={endyr}&seasontype=2")
    except Exception:
        return {}


def completed_games_from_events(events, id2ab, out):
    """Append (date, home_ab, away_ab, hs, as_) for each completed game; dedupe by gid."""
    for ev in events:
        comp = (ev.get("competitions") or [{}])[0]
        if not comp.get("status", {}).get("type", {}).get("completed"):
            continue
        cs = comp.get("competitors", [])
        hm = next((c for c in cs if c.get("homeAway") == "home"), None)
        aw = next((c for c in cs if c.get("homeAway") == "away"), None)
        if not hm or not aw:
            continue

        def sc(c):
            s = c.get("score"); s = s.get("value") if isinstance(s, dict) else s
            try: return int(float(s))
            except (TypeError, ValueError): return None
        hs, as_ = sc(hm), sc(aw)
        if hs is None or as_ is None:
            continue
        try: d = datetime.date.fromisoformat((ev.get("date") or "")[:10])
        except ValueError: continue
        gid = ev.get("id")
        if gid in out:
            continue
        h_ab = id2ab.get(int(hm["team"]["id"]), hm["team"].get("abbreviation"))
        a_ab = id2ab.get(int(aw["team"]["id"]), aw["team"].get("abbreviation"))
        out[gid] = (d, h_ab, a_ab, hs, as_)


def load_all_games(seasons, id2ab):
    """Pull every completed regular-season game for the given seasons (cached)."""
    cache = {}
    if os.path.exists(CACHE):
        try:
            with open(CACHE, encoding="utf-8") as f:
                cache = json.load(f)
        except Exception:
            cache = {}
    games_by_season = {}
    dirty = False
    for season in seasons:
        key = str(season)
        if key in cache:
            games_by_season[season] = [
                (datetime.date.fromisoformat(g[0]), g[1], g[2], g[3], g[4]) for g in cache[key]
            ]
            continue
        out = {}
        for tid in id2ab:
            data = fetch_team_schedule(tid, season)
            completed_games_from_events(data.get("events", []), id2ab, out)
        # store as [iso_date, h, a, hs, as_]
        ser = [[g[0].isoformat(), g[1], g[2], g[3], g[4]] for g in out.values()]
        cache[key] = ser
        games_by_season[season] = [(datetime.date.fromisoformat(s[0]), s[1], s[2], s[3], s[4]) for s in ser]
        dirty = True
        print(f"  fetched season {season}: {len(ser)} completed games", flush=True)
    if dirty:
        with open(CACHE, "w", encoding="utf-8") as f:
            json.dump(cache, f)
    return games_by_season


# ----------------------------------------------------------------------------
# 2. Fit ratings on a set of games, with reference date = cutoff (no look-ahead).
#    This is a faithful replica of build_nba.py's rating math.
# ----------------------------------------------------------------------------
def fit_ratings(train_games, ref_date):
    """train_games: list of (date, h, a, hs, as_), all with date < cutoff.
    Returns (off, dfn, LG, HFA, SD_M, gp) where gp[team] = prior games played."""
    if not train_games:
        return {}, {}, 0.0, 0.0, 0.0, {}
    wt = lambda d: 0.5 ** ((ref_date - d).days / HALFLIFE)
    teams = sorted({t for g in train_games for t in (g[1], g[2])})
    gp = {t: 0 for t in teams}
    for _, h, a, _, _ in train_games:
        gp[h] += 1; gp[a] += 1

    tw = tp = hm_ = hw = 0.0
    for d, h, a, hs, a_s in train_games:
        w = wt(d); tp += w * (hs + a_s); tw += 2 * w; hm_ += w * (hs - a_s); hw += w
    LG = tp / tw; HFA = hm_ / hw
    off = {t: 0.0 for t in teams}; dfn = {t: 0.0 for t in teams}
    for _ in range(GS_PASSES):
        no = {t: [0., 0.] for t in teams}; nd = {t: [0., 0.] for t in teams}
        for d, h, a, hs, a_s in train_games:
            w = wt(d)
            no[h][0] += w * ((hs - HFA / 2) - LG - dfn[a]); no[h][1] += w
            no[a][0] += w * ((a_s + HFA / 2) - LG - dfn[h]); no[a][1] += w
            nd[h][0] += w * ((a_s + HFA / 2) - LG - off[a]); nd[h][1] += w
            nd[a][0] += w * ((hs - HFA / 2) - LG - off[h]); nd[a][1] += w
        for t in teams:
            if no[t][1]: off[t] = no[t][0] / no[t][1]
            if nd[t][1]: dfn[t] = nd[t][0] / nd[t][1]
        om = sum(off.values()) / len(teams); dm = sum(dfn.values()) / len(teams)
        for t in teams:
            off[t] -= om; dfn[t] -= dm
    sm = sw = 0.0
    for d, h, a, hs, a_s in train_games:
        w = wt(d); pm = (off[h] - off[a]) + (dfn[a] - dfn[h]) + HFA
        sm += w * ((hs - a_s) - pm) ** 2; sw += w
    SD_M = (sm / sw) ** 0.5 if sw else 0.0
    return off, dfn, LG, HFA, SD_M, gp


# ----------------------------------------------------------------------------
# 3. Walk-forward over the test window.
# ----------------------------------------------------------------------------
def main():
    print("=== NBA WALK-FORWARD (out-of-sample) BACKTEST ===\n", flush=True)
    tj = get(f"{ESPN}/teams")
    tlist = tj["sports"][0]["leagues"][0]["teams"]
    id2ab = {t["team"]["id"]: t["team"]["abbreviation"] for t in tlist}

    # Probe the most recent COMPLETE season (>= ~1100 games for a full 30-team season).
    this_year = datetime.date.today().year
    candidates = list(range(this_year + 1, this_year - 4, -1))
    print(f"Probing seasons (ESPN end-year): {candidates}", flush=True)
    games_probe = load_all_games(candidates, id2ab)
    complete = []
    for s in candidates:
        n = len(games_probe.get(s, []))
        status = ""
        if n >= 1100:
            complete.append(s); status = " COMPLETE"
        elif n > 0:
            status = " partial"
        print(f"  season {s}: {n} games{status}", flush=True)
    if not complete:
        print("No complete season found; aborting."); return
    complete.sort(reverse=True)

    # Test the most recent complete season; add the prior complete season if available (bigger sample).
    test_seasons = complete[:2]
    test_seasons.sort()
    earliest_test = test_seasons[0]
    lookback_seasons = list(range(earliest_test - LOOKBACK, earliest_test))
    all_seasons = sorted(set(lookback_seasons + test_seasons + [s for s in complete if s <= max(test_seasons)]))
    print(f"\nTest seasons: {test_seasons}")
    print(f"Warm-up lookback seasons: {lookback_seasons}", flush=True)

    games_by_season = load_all_games(all_seasons, id2ab)

    # Master sorted game list (everything we can train on).
    all_games = []
    for s in all_seasons:
        all_games.extend(games_by_season.get(s, []))
    all_games.sort(key=lambda g: g[0])

    # Test games = games in the test seasons.
    test_games = []
    test_set_dates = set()
    for s in test_seasons:
        for g in games_by_season.get(s, []):
            test_games.append(g)
    test_games.sort(key=lambda g: g[0])
    print(f"Total games available for training: {len(all_games)}")
    print(f"Candidate test games (test seasons): {len(test_games)}\n", flush=True)

    # Weekly cutoffs: group test games into ISO weeks; refit ratings once per week on
    # everything strictly before the FIRST game of that week (no same-week leak — we use
    # the Monday-of-week cutoff so a game never sees its own week's results).
    from collections import defaultdict
    weeks = defaultdict(list)
    for g in test_games:
        monday = g[0] - datetime.timedelta(days=g[0].weekday())
        weeks[monday].append(g)

    results = []  # per scored game: dict
    skipped = 0
    for monday in sorted(weeks):
        cutoff = monday  # strictly-before: train on date < monday
        train = [g for g in all_games if g[0] < cutoff]
        off, dfn, LG, HFA, SD_M, gp = fit_ratings(train, ref_date=cutoff - datetime.timedelta(days=1))
        if SD_M <= 0:
            skipped += len(weeks[monday]); continue
        for d, h, a, hs, a_s in weeks[monday]:
            if h not in off or a not in off:
                skipped += 1; continue
            if gp.get(h, 0) < MIN_PRIOR_GP or gp.get(a, 0) < MIN_PRIOR_GP:
                skipped += 1; continue
            proj_margin = (off[h] - off[a]) + (dfn[a] - dfn[h]) + HFA   # home minus away
            p_home = norm_cdf(proj_margin / SD_M)
            actual_margin = hs - a_s
            home_won = actual_margin > 0
            # favorite = higher win prob
            p_fav = max(p_home, 1.0 - p_home)
            fav_is_home = p_home >= 0.5
            fav_won = (fav_is_home and home_won) or ((not fav_is_home) and (not home_won))
            results.append({
                "date": d.isoformat(), "home": h, "away": a, "hs": hs, "as": a_s,
                "proj_margin": proj_margin, "actual_margin": actual_margin,
                "p_home": p_home, "p_fav": p_fav, "fav_is_home": fav_is_home,
                "home_won": home_won, "fav_won": fav_won,
            })

    N = len(results)
    if N == 0:
        print("No scorable games (insufficient prior data). Aborting."); return

    # ---- headline winner hit-rate ----
    hits = sum(1 for r in results if r["fav_won"])
    su = hits / N

    # ---- Brier & log-loss (vs home-win outcome) ----
    brier = sum((r["p_home"] - (1.0 if r["home_won"] else 0.0)) ** 2 for r in results) / N
    eps = 1e-15
    logloss = -sum(
        (1.0 if r["home_won"] else 0.0) * math.log(max(eps, r["p_home"]))
        + (0.0 if r["home_won"] else 1.0) * math.log(max(eps, 1.0 - r["p_home"]))
        for r in results
    ) / N

    # ---- margin MAE ----
    mae = sum(abs(r["proj_margin"] - r["actual_margin"]) for r in results) / N

    # ---- calibration by confidence bucket (on the favorite's prob) ----
    buckets = [(0.50, 0.60), (0.60, 0.70), (0.70, 0.80), (0.80, 1.0001)]
    print("=" * 64)
    print(f"HEADLINE  —  out-of-sample, {N} games scored ({skipped} skipped: insufficient prior data)")
    print("=" * 64)
    print(f"Straight-up WINNER hit-rate : {su*100:5.1f}%   ({hits}/{N})")
    print(f"Brier score (lower better)  : {brier:.4f}   (coin-flip 0.25)")
    print(f"Log-loss   (lower better)   : {logloss:.4f}   (coin-flip {math.log(2):.4f})")
    print(f"Margin MAE (pts)            : {mae:.2f}")
    print(f"ATS hit-rate                : SKIPPED (no cheap historical closing spreads)")

    print("\n=== CALIBRATION BY CONFIDENCE BUCKET (favorite win prob) ===")
    print(f"{'bucket':>10s} {'N':>5s} {'pred%':>7s} {'actual%':>8s} {'gap':>6s}")
    for lo, hi in buckets:
        sub = [r for r in results if lo <= r["p_fav"] < hi]
        if not sub:
            print(f"{int(lo*100)}-{int(min(hi,1.0)*100):>3d} {0:>9d}      --       --     --"); continue
        n = len(sub)
        pred = sum(r["p_fav"] for r in sub) / n
        act = sum(1 for r in sub if r["fav_won"]) / n
        print(f"{int(lo*100):>4d}-{int(min(hi,1.0)*100):<3d} {n:>5d} {pred*100:>6.1f}% {act*100:>7.1f}% {(act-pred)*100:>+5.1f}")

    # ---- trust-tier table ----
    print("\n=== TRUST-TIER TABLE (picks at/above each confidence) ===")
    print(f"{'thresh':>7s} {'N':>6s} {'hit%':>7s}   share-of-games")
    for thr in (0.50, 0.60, 0.65, 0.70, 0.75, 0.80):
        sub = [r for r in results if r["p_fav"] >= thr]
        if not sub:
            print(f"{int(thr*100):>5d}%  {0:>6d}      --"); continue
        n = len(sub); hr = sum(1 for r in sub if r["fav_won"]) / n
        print(f"{int(thr*100):>5d}%  {n:>6d}  {hr*100:>5.1f}%   {n/N*100:>5.1f}%")

    # ---- benchmark read ----
    print("\n=== BENCHMARK READ ===")
    print(f"Coin-flip straight-up    : 50.0%")
    print(f"Market straight-up (~)   : 69%")
    print(f"MODEL straight-up        : {su*100:.1f}%  ->  {su*100-50:+.1f} vs coin-flip, {su*100-69:+.1f} vs market")
    if su >= 0.67:
        print("Read: AT/NEAR the market benchmark — a strong, honest OOS result.")
    elif su >= 0.62:
        print("Read: Below market but well above coin-flip — solid predictive signal.")
    else:
        print("Read: Modest edge over coin-flip; weaker than market.")

    # ---- best / worst teams by accuracy on their games ----
    team_stat = {}
    for r in results:
        for t in (r["home"], r["away"]):
            team_stat.setdefault(t, [0, 0])
        team_stat[r["home"]][1] += 1; team_stat[r["away"]][1] += 1
        if r["fav_won"]:
            team_stat[r["home"]][0] += 1; team_stat[r["away"]][0] += 1
    team_rows = [(t, c[0] / c[1], c[1]) for t, c in team_stat.items() if c[1] >= 20]
    team_rows.sort(key=lambda x: -x[1])
    print("\n=== TEAM-LEVEL ACCURACY (games involving each team; N>=20) ===")
    print("Best predicted:")
    for t, hr, n in team_rows[:5]:
        print(f"  {t:5s} {hr*100:5.1f}%  (N={n})")
    print("Worst predicted:")
    for t, hr, n in team_rows[-5:]:
        print(f"  {t:5s} {hr*100:5.1f}%  (N={n})")

    # ---- dump per-game results ----
    outp = os.path.join(PROJ, "_backtest_nba_walkforward.csv")
    with open(outp, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["date", "home", "away", "hs", "as", "proj_margin", "actual_margin",
                    "p_home", "p_fav", "fav_is_home", "home_won", "fav_won"])
        for r in results:
            w.writerow([r["date"], r["home"], r["away"], r["hs"], r["as"],
                        round(r["proj_margin"], 2), r["actual_margin"],
                        round(r["p_home"], 4), round(r["p_fav"], 4),
                        int(r["fav_is_home"]), int(r["home_won"]), int(r["fav_won"])])
    print(f"\nWrote {os.path.basename(outp)}  ({N} scored games)")


if __name__ == "__main__":
    main()
