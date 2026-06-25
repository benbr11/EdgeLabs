# -*- coding: utf-8 -*-
"""
HONEST walk-forward (out-of-sample) backtest for the club-league model
(EPL / La Liga / Serie A / Bundesliga / Ligue 1).

Measures how accurately the model predicts games it has NOT seen. For every game in
the test window we rebuild the team ratings using ONLY games that finished STRICTLY
BEFORE that game's date (prior seasons + current-season-to-date), exactly replicating
build_leagues.py's rating math (recency-weighted iterative Poisson att/def + MOV Elo +
the 0.8/0.2 goals/Elo consensus z-blend + the goal-level sqrt calibration), then apply
the SAME Dixon-Coles 1X2 prediction as web/leagues_app.js. No look-ahead, no leakage.

This is MEASUREMENT ONLY — it does not read or modify the model's ratings file; it
recomputes ratings from raw football-data.co.uk results with a per-date cutoff so the
backtest is honest. Read-only w.r.t. all model files.

Scores: winner hit-rate (favorite = higher of pH/pA vs the OTHER side; draws handled),
calibration by confidence bucket, Brier score, log-loss, and a trust-tier table.

Efficiency: rating fits are cached per (league, cutoff-date) so all games on the same
matchday reuse one fit. ~30-40 distinct matchdays/season -> fast.
"""
import csv, io, os, math, datetime, itertools, urllib.request

PROJ = os.path.dirname(os.path.abspath(__file__))
LEAGUES = [("epl", "Premier League", "E0"), ("laliga", "La Liga", "SP1"),
           ("seriea", "Serie A", "I1"), ("bundesliga", "Bundesliga", "D1"),
           ("ligue1", "Ligue 1", "F1")]

# --- engine constants: identical to build_leagues.py ---
HALFLIFE = 400.0
RHO = -0.12

# Test the two most recent COMPLETE seasons; ratings may use everything before each game.
TEST_SEASONS = ["2425", "2526"]            # 2024-25 and 2025-26 (both complete as of Jun 2026)
HISTORY_SEASONS = ["2021", "2122", "2223", "2324", "2425", "2526"]  # all available for the fit
MIN_PRIOR_GAMES = 6                        # both teams must have >= this many prior games to rate


def get(url, t=30):
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    return urllib.request.urlopen(req, timeout=t).read().decode("utf-8", "replace")


def pdate(s):
    for f in ("%d/%m/%Y", "%d/%m/%y"):
        try:
            return datetime.datetime.strptime(s, f).date()
        except (ValueError, TypeError):
            pass
    return None


def season_codes(n=6, today=None):
    d = today or datetime.date.today()
    sy = d.year if d.month >= 7 else d.year - 1
    return [f"{str(y)[2:]}{str(y+1)[2:]}" for y in range(sy, sy - n, -1)]


def load_league(fdc):
    """Return all games for one league across HISTORY_SEASONS as (date,h,a,hg,ag), sorted by date."""
    games = []
    for season in HISTORY_SEASONS:
        try:
            rows = list(csv.DictReader(io.StringIO(get(
                f"https://www.football-data.co.uk/mmz4281/{season}/{fdc}.csv"))))
        except Exception as e:
            print(f"    {fdc} {season}: skip ({e})", flush=True)
            continue
        for r in rows:
            h = (r.get("HomeTeam") or "").strip()
            a = (r.get("AwayTeam") or "").strip()
            d = pdate(r.get("Date", ""))
            try:
                hg, ag = int(r["FTHG"]), int(r["FTAG"])
            except (ValueError, KeyError, TypeError):
                continue
            if h and a and d:
                games.append((d, h, a, hg, ag))
    games.sort(key=lambda x: x[0])
    return games


def fit_ratings(games):
    """Replicate build_leagues.py's rating computation on the GIVEN games only.
    Returns dict: team -> {att, dfn} plus league params (avg, home_adv). No look-ahead:
    caller passes only games that finished strictly before the target date."""
    teams = sorted({t for g in games for t in (g[1], g[2])})
    ref = max(g[0] for g in games)
    wt = lambda d: 0.5 ** ((ref - d).days / HALFLIFE)

    tg = tw = hgs = ags = 0.0
    for d, h, a, hg, ag in games:
        w = wt(d); tg += w * (hg + ag); tw += 2 * w; hgs += w * hg; ags += w * ag
    AVG = tg / tw
    HOME = min(1.45, max(1.05, hgs / ags))

    att = {t: 1.0 for t in teams}; dfn = {t: 1.0 for t in teams}
    for _ in range(60):
        na = {t: 0. for t in teams}; da = dict(na); nd = dict(na); dd = dict(na)
        for d, h, a, hg, ag in games:
            w = wt(d)
            na[h] += w * hg; da[h] += w * AVG * dfn[a]; nd[a] += w * hg; dd[a] += w * AVG * att[h]
            na[a] += w * ag; da[a] += w * AVG * dfn[h]; nd[h] += w * ag; dd[h] += w * AVG * att[a]
        for t in teams:
            if da[t] > 0: att[t] = na[t] / da[t]
            if dd[t] > 0: dfn[t] = nd[t] / dd[t]
        for dct in (att, dfn):
            for t in dct:
                if dct[t] <= 0: dct[t] = 1e-3   # floor degenerate sub-samples (early-window guard)
            gm = math.exp(sum(math.log(max(v, 1e-6)) for v in dct.values()) / len(dct))
            for t in dct: dct[t] /= gm

    elo = {t: 1500.0 for t in teams}
    for d, h, a, hg, ag in sorted(games, key=lambda x: x[0]):
        eh, ea = elo[h], elo[a]; exp = 1 / (1 + 10 ** ((ea - (eh + 60)) / 400))
        res = 1.0 if hg > ag else 0.0 if hg < ag else 0.5
        g = 1 + 0.4 * abs(hg - ag); dl = 8 * g * (res - exp); elo[h] = eh + dl; elo[a] = ea - dl

    def z(d):
        v = list(d.values()); m = sum(v) / len(v)
        sd = (sum((x - m) ** 2 for x in v) / len(v)) ** .5 or 1
        return {t: (x - m) / sd for t, x in d.items()}
    zAg = z({t: math.log(att[t]) for t in teams})
    zDg = z({t: -math.log(dfn[t]) for t in teams})
    zE = z(elo)
    attZ = {t: 0.8 * zAg[t] + 0.2 * zE[t] for t in teams}
    defZ = {t: 0.8 * zDg[t] + 0.2 * zE[t] for t in teams}
    lA = [math.log(att[t]) for t in teams]; mA = sum(lA) / len(lA)
    sA = (sum((x - mA) ** 2 for x in lA) / len(lA)) ** .5
    lD = [-math.log(dfn[t]) for t in teams]; mD = sum(lD) / len(lD)
    sD = (sum((x - mD) ** 2 for x in lD) / len(lD)) ** .5
    att = {t: math.exp(mA + sA * attZ[t]) for t in teams}
    dfn = {t: math.exp(-(mD + sD * defZ[t])) for t in teams}

    if len(teams) > 1:
        infl = sum(att[a] * dfn[b] for a, b in itertools.permutations(teams, 2)) / (len(teams) * (len(teams) - 1))
        k = infl ** .5
        for t in teams:
            att[t] /= k; dfn[t] /= k

    return {"teams": {t: {"att": att[t], "dfn": dfn[t]} for t in teams},
            "avg": AVG, "home_adv": HOME}


def predict_1x2(rt, home, away):
    """Dixon-Coles 1X2, identical math to web/leagues_app.js predict()."""
    T = rt["teams"]
    lh = rt["avg"] * T[home]["att"] * T[away]["dfn"] * rt["home_adv"]
    la = rt["avg"] * T[away]["att"] * T[home]["dfn"]
    fac = [1.0] * 10
    for i in range(2, 10):
        fac[i] = fac[i - 1] * i
    Po = lambda k, l: math.exp(-l) * (l ** k) / fac[k]
    pH = pD = pA = 0.0
    for i in range(10):
        for j in range(10):
            if i == 0 and j == 0: tau = 1 - lh * la * RHO
            elif i == 0 and j == 1: tau = 1 + lh * RHO
            elif i == 1 and j == 0: tau = 1 + la * RHO
            elif i == 1 and j == 1: tau = 1 - RHO
            else: tau = 1.0
            m = Po(i, lh) * Po(j, la) * tau
            if i > j: pH += m
            elif i == j: pD += m
            else: pA += m
    s = pH + pD + pA or 1.0
    return pH / s, pD / s, pA / s


def run():
    print("=" * 72)
    print("HONEST WALK-FORWARD (OUT-OF-SAMPLE) 1X2 BACKTEST — club leagues")
    print("Test window:", ", ".join(TEST_SEASONS), "| ratings use only games BEFORE each match")
    print("=" * 72)

    test_start = {}  # earliest test date per season code -> derived from data below
    # all per-game prediction records across all leagues
    REC = []  # (league_name, season, date, home, away, pH, pD, pA, outcome 'H'/'D'/'A', fav_correct, fav_prob)
    per_league = {}

    for code, name, fdc in LEAGUES:
        print(f"\n--- {name} ({fdc}) ---", flush=True)
        games = load_league(fdc)
        if not games:
            print("    no data, skipping"); continue

        # Identify the test games (those within the TEST_SEASONS date ranges).
        # Derive each test season's date span from football-data directly.
        test_games = []
        for season in TEST_SEASONS:
            try:
                rows = list(csv.DictReader(io.StringIO(get(
                    f"https://www.football-data.co.uk/mmz4281/{season}/{fdc}.csv"))))
            except Exception:
                continue
            ds = [pdate(r.get("Date", "")) for r in rows]
            ds = [d for d in ds if d]
            if not ds: continue
            lo, hi = min(ds), max(ds)
            for g in games:
                if lo <= g[0] <= hi:
                    test_games.append((season, g))

        # Cache one rating fit per cutoff date (matchday). All games on date D are
        # predicted from a fit on games strictly before D.
        fit_cache = {}
        lg_rec = []
        for season, (d, h, a, hg, ag) in test_games:
            prior = [g for g in games if g[0] < d]
            if d not in fit_cache:
                fit_cache[d] = fit_ratings(prior) if len(prior) >= 50 else None
            rt = fit_cache[d]
            if rt is None or h not in rt["teams"] or a not in rt["teams"]:
                continue
            # require enough prior games for BOTH teams to be ratable
            ph_games = sum(1 for g in prior if g[1] == h or g[2] == h)
            pa_games = sum(1 for g in prior if g[1] == a or g[2] == a)
            if ph_games < MIN_PRIOR_GAMES or pa_games < MIN_PRIOR_GAMES:
                continue

            pH, pD, pA = predict_1x2(rt, h, a)
            outcome = "H" if hg > ag else "A" if ag > hg else "D"
            # model's pick = most likely of the three
            probs = {"H": pH, "D": pD, "A": pA}
            pick = max(probs, key=probs.get)
            fav_correct = int(pick == outcome)
            # "favorite" two-way (home vs away, ignoring draw) for the winner read
            fav_prob = max(pH, pA)
            rec = (name, season, d, h, a, pH, pD, pA, outcome, pick, fav_correct, fav_prob, hg, ag)
            REC.append(rec); lg_rec.append(rec)

        per_league[name] = lg_rec
        n = len(lg_rec)
        if n:
            hit = sum(r[10] for r in lg_rec) / n
            print(f"    test games rated: {n} | favorite (most-likely-of-3) hit-rate: {hit*100:.1f}%", flush=True)

    if not REC:
        print("\nNo rated test games — aborting."); return

    report(REC, per_league)


def report(REC, per_league):
    N = len(REC)
    print("\n" + "=" * 72)
    print(f"RESULTS — {N} out-of-sample games across {len(per_league)} leagues, seasons {TEST_SEASONS}")
    print("=" * 72)

    # --- Headline winner hit-rate (model's most-likely-of-3 pick) ---
    hit = sum(r[10] for r in REC) / N
    # outcome base rates in the test set
    nH = sum(1 for r in REC if r[8] == "H")
    nD = sum(1 for r in REC if r[8] == "D")
    nA = sum(1 for r in REC if r[8] == "A")
    # naive baselines
    always_home = nH / N
    # "best fixed pick" = pick the most common outcome class every time
    best_fixed = max(nH, nD, nA) / N
    print(f"\n[1] HEADLINE WINNER HIT-RATE (model picks most likely of H/D/A)")
    print(f"    Model hit-rate ......... {hit*100:5.1f}%   ({sum(r[10] for r in REC)}/{N})")
    print(f"    Always-pick-home ....... {always_home*100:5.1f}%   (home win base rate)")
    print(f"    Best fixed single pick . {best_fixed*100:5.1f}%   (always pick the commonest class)")
    print(f"    Random 3-way ........... 33.3%")
    print(f"    Outcome mix in test set: Home {nH/N*100:.1f}% / Draw {nD/N*100:.1f}% / Away {nA/N*100:.1f}%")

    # How often did the model pick a draw, and was it ever right to?
    npick = {"H": 0, "D": 0, "A": 0}
    for r in REC: npick[r[9]] += 1
    print(f"    Model's picks: Home {npick['H']} / Draw {npick['D']} / Away {npick['A']}")

    # --- Calibration by confidence bucket (on the model's PICK probability) ---
    print(f"\n[2] CALIBRATION — by the probability the model assigned to its PICK")
    buckets = [(0.33, 0.40), (0.40, 0.50), (0.50, 0.60), (0.60, 0.70), (0.70, 0.80), (0.80, 1.01)]
    print(f"    {'bucket':>12} {'N':>5} {'pred%':>7} {'actual%':>8}  {'note'}")
    for lo, hi in buckets:
        sub = [r for r in REC if lo <= probs_of_pick(r) < hi]
        if not sub: continue
        pred = sum(probs_of_pick(r) for r in sub) / len(sub)
        act = sum(r[10] for r in sub) / len(sub)
        flag = "over-confident" if pred - act > 0.05 else "under-confident" if act - pred > 0.05 else "well-calibrated"
        print(f"    {int(lo*100):>4}-{int(hi*100):<3}%   {len(sub):>5} {pred*100:>6.1f}% {act*100:>7.1f}%  {flag}")

    # --- Brier score & log-loss (full 3-way, vs actual outcome) ---
    brier = 0.0; logloss = 0.0
    # baselines: a fixed climatological prior using the test-set base rates
    base = {"H": nH / N, "D": nD / N, "A": nA / N}
    brier_base = 0.0; logloss_base = 0.0
    for r in REC:
        p = {"H": r[5], "D": r[6], "A": r[7]}
        y = r[8]
        brier += sum((p[k] - (1.0 if k == y else 0.0)) ** 2 for k in p)
        logloss += -math.log(max(p[y], 1e-12))
        brier_base += sum((base[k] - (1.0 if k == y else 0.0)) ** 2 for k in base)
        logloss_base += -math.log(max(base[y], 1e-12))
    brier /= N; logloss /= N; brier_base /= N; logloss_base /= N
    print(f"\n[3] PROPER SCORES (3-way, lower is better)")
    print(f"    Brier  — model {brier:.4f}  vs  base-rate prior {brier_base:.4f}")
    print(f"    LogLoss— model {logloss:.4f}  vs  base-rate prior {logloss_base:.4f}")
    print(f"    (base-rate prior = always predict the test-set's average H/D/A split)")

    # --- Trust-tier table: hit-rate of the two-way FAVORITE above each threshold ---
    # The two-way favorite = the side (home or away) with the higher win prob; "correct"
    # means that side actually won the match outright (draws count as a miss).
    print(f"\n[4] TRUST TIERS — when the model's FAVORITE side prob >= threshold,")
    print(f"    how often did that side WIN outright? (draws = miss). N = qualifying games.")
    print(f"    {'threshold':>10} {'N':>5} {'fav-win%':>9} {'covers':>8}")
    for thr in (0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80):
        sub = [r for r in REC if r[11] >= thr]
        if not sub: continue
        # did the favored side (the one with prob r[11]) win?
        wins = 0
        for r in sub:
            fav = "H" if r[5] >= r[7] else "A"
            wins += int(r[8] == fav)
        print(f"    >= {int(thr*100):>3}%   {len(sub):>5} {wins/len(sub)*100:>8.1f}% {len(sub)/N*100:>7.1f}%")

    # --- Per-league breakdown ---
    print(f"\n[5] PER-LEAGUE (most-likely-of-3 hit-rate / Brier / LogLoss / N)")
    print(f"    {'league':<18}{'N':>5}{'hit%':>8}{'Brier':>8}{'LogLoss':>9}")
    for name, recs in per_league.items():
        if not recs: continue
        n = len(recs); h = sum(r[10] for r in recs) / n
        b = sum(sum(({'H': r[5], 'D': r[6], 'A': r[7]}[k] - (1.0 if k == r[8] else 0.0)) ** 2 for k in 'HDA') for r in recs) / n
        ll = sum(-math.log(max({'H': r[5], 'D': r[6], 'A': r[7]}[r[8]], 1e-12)) for r in recs) / n
        print(f"    {name:<18}{n:>5}{h*100:>7.1f}%{b:>8.4f}{ll:>9.4f}")

    # --- Where it's strong / weak: by favorite strength and by outcome type ---
    print(f"\n[6] STRENGTH/WEAKNESS DIAGNOSTICS")
    # accuracy when the home side is favored vs away side favored
    home_fav = [r for r in REC if r[5] >= r[7]]
    away_fav = [r for r in REC if r[7] > r[5]]
    def winrate(sub):
        if not sub: return (0, 0.0)
        w = 0
        for r in sub:
            fav = "H" if r[5] >= r[7] else "A"
            w += int(r[8] == fav)
        return (len(sub), w / len(sub) * 100)
    nhf, whf = winrate(home_fav); naf, waf = winrate(away_fav)
    print(f"    Home favorites: {nhf} games, fav won {whf:.1f}%")
    print(f"    Away favorites: {naf} games, fav won {waf:.1f}%")
    # draw recall: of actual draws, how often did model even rank draw 2nd or higher?
    actual_draws = [r for r in REC if r[8] == "D"]
    draw_top = sum(1 for r in actual_draws if r[9] == "D")
    if actual_draws:
        print(f"    Draws are the hardest class: {len(actual_draws)} actual draws, "
              f"model picked draw outright {draw_top} times ({draw_top/len(actual_draws)*100:.1f}%).")
        print(f"    (Draws ~{nD/N*100:.0f}% of games but rarely the single most-likely outcome — expected.)")

    # biggest per-team prediction errors (teams the model reads best/worst), by avg log-loss
    team_ll = {}
    for r in REC:
        p = {"H": r[5], "D": r[6], "A": r[7]}; y = r[8]
        ll = -math.log(max(p[y], 1e-12))
        for t in (r[3], r[4]):
            team_ll.setdefault(t, []).append(ll)
    team_avg = {t: sum(v) / len(v) for t, v in team_ll.items() if len(v) >= 20}
    if team_avg:
        best = sorted(team_avg.items(), key=lambda x: x[1])[:6]
        worst = sorted(team_avg.items(), key=lambda x: -x[1])[:6]
        print(f"    Teams the model predicts BEST (lowest avg log-loss, >=20 games):")
        print("      " + ", ".join(f"{t} {v:.2f}" for t, v in best))
        print(f"    Teams the model predicts WORST (highest avg log-loss):")
        print("      " + ", ".join(f"{t} {v:.2f}" for t, v in worst))

    # --- Benchmark read ---
    print(f"\n[7] BENCHMARK READ")
    print(f"    1X2 is a 3-way market (home ~45% / draw ~25% / away ~30% league-wide).")
    print(f"    Picking the most-likely of three correctly ~50-55% is a SOLID, market-grade result.")
    band = "AT/ABOVE the strong 50-55% benchmark band" if hit >= 0.50 else \
           "below the 50-55% benchmark band" if hit >= 0.45 else "near the random-ish floor"
    print(f"    Model winner hit-rate = {hit*100:.1f}% -> {band}.")
    print(f"    vs coin-flip (50% two-way) the favorite-win trust tiers above show the edge by confidence.")
    print("=" * 72)


def probs_of_pick(r):
    return {"H": r[5], "D": r[6], "A": r[7]}[r[9]]


if __name__ == "__main__":
    run()
