# -*- coding: utf-8 -*-
"""
backtest_nfl.py -- HONEST walk-forward (out-of-sample) backtest of the NFL model.

WHAT THIS MEASURES
  How accurately the NFL model predicts games it has NOT seen. For every game in
  the test window we rebuild the model's ratings using ONLY data strictly BEFORE
  that game (prior seasons + current-season-to-date), then predict the game with
  the exact production engine (Gaussian point-margin). Nothing from the game --
  or any later game -- can leak into its own prediction.

NO LOOK-AHEAD (how leakage is prevented)
  * Ratings are rebuilt PER WEEK. For test (season, week) we feed in every play
    from a game whose gameday is strictly < the first gameday of that week. So a
    Week-7 game is predicted from data through Week 6 only.
  * Recency weights wt(dt) = 0.5 ** ((cutoff - dt).days / HALFLIFE) are referenced
    to the CUTOFF date (the week's start), exactly as the live model references
    "today" -- it never sees a negative age (future game).
  * EPA->points calibration (LG, HFA, KP, KT, SD_margin, SD_total) is refit each
    week on prior games only -- never on the test season's later games.
  * The decaying expert-consensus prior (consensus_nfl.csv) is applied as it would
    have applied at that date: W = min(1, BASE + gamesPlayedThisSeason/17). Early
    in a season the prior leans on consensus; by mid/late season W->1 (pure data).
    NOTE: consensus_nfl.csv is the CURRENT (2026) preseason consensus, so for past
    seasons it is an imperfect stand-in for that year's preseason ranks. We report
    results WITH and WITHOUT the prior so its effect is transparent.

  This is measurement only. It does NOT modify the model or any model file; it
  re-implements build_nfl.py's rating math + blend_nfl.py's blend with a cutoff.

DATA
  * pbp_<yr>.csv (already cached locally; same files build_nfl.py uses)
  * games.csv from nflverse (season, week, gameday, scores, spread_line, total_line)
"""
import csv, io, os, math, urllib.request, datetime
from collections import defaultdict

PROJ = os.path.dirname(os.path.abspath(__file__)); csv.field_size_limit(10**7)
HALFLIFE = float(os.environ.get("NFL_HALFLIFE", "230.0"))   # same default as build_nfl.py
BLEND_BASE = float(os.environ.get("NFL_BLEND_BASE", "0.15"))  # same default as blend_nfl.py
GAMES_PER_TEAM = 17.0
GAMES_URL = "https://raw.githubusercontent.com/nflverse/nfldata/master/data/games.csv"

# pbp seasons available locally -> training pool. Test the most recent complete
# season(s). 2025 is the most recent complete season; 2024 included for sample size.
PBP_SEASONS = [2022, 2023, 2024, 2025]
TEST_SEASONS = [2024, 2025]
ITERS = 60  # opponent-adjustment iterations (matches build_nfl.py)

RELO = {"OAK": "LV", "SD": "LAC", "STL": "LA", "LAR": "LA"}
def fix(t): return RELO.get(t, t)
def fl(x):
    try: return float(x)
    except (TypeError, ValueError): return None
def cdf(x): return 0.5 * (1 + math.erf(x / 2 ** 0.5))


# ----------------------------------------------------------------------------
# Load play-by-play once. Keep only the fields the rating math needs, plus a date.
# ----------------------------------------------------------------------------
def load_pbp():
    plays = []  # list of dicts (lightweight)
    for yr in PBP_SEASONS:
        p = os.path.join(PROJ, f"pbp_{yr}.csv")
        if not os.path.exists(p):
            print(f"  WARNING: pbp_{yr}.csv missing -- skipping", flush=True)
            continue
        n = 0
        with open(p, encoding="utf-8") as f:
            for r in csv.DictReader(f):
                o = fix(r.get("posteam") or ""); d = fix(r.get("defteam") or "")
                gid = r.get("game_id")
                gd = r.get("game_date") or ""
                try: dt = datetime.date.fromisoformat(gd)
                except ValueError:
                    dt = datetime.date(yr, 11, 1)
                rec = {
                    "dt": dt, "o": o, "d": d, "gid": gid,
                    "pt": r.get("play_type") or "",
                    "epa": fl(r.get("epa")), "succ": fl(r.get("success")),
                    "dr": r.get("fixed_drive"),
                    "dres": r.get("fixed_drive_result") or "",
                    "syl": fl(r.get("drive_start_yard_line_100")) or fl(r.get("yardline_100")),
                }
                plays.append(rec)
                n += 1
        print(f"  loaded pbp {yr}: {n} plays", flush=True)
    plays.sort(key=lambda r: r["dt"])
    return plays


def load_games():
    req = urllib.request.Request(GAMES_URL, headers={"User-Agent": "Mozilla/5.0"})
    data = urllib.request.urlopen(req, timeout=180).read().decode("utf-8", "replace")
    rows = list(csv.DictReader(io.StringIO(data)))
    games = []
    for r in rows:
        try: yr = int(r["season"])
        except (ValueError, KeyError): continue
        if yr not in PBP_SEASONS: continue
        try: hs, as_ = int(r["home_score"]), int(r["away_score"])
        except (ValueError, KeyError, TypeError): continue  # not played
        h, a = fix(r["home_team"]), fix(r["away_team"])
        try: dt = datetime.date.fromisoformat(r["gameday"])
        except (ValueError, KeyError, TypeError): dt = datetime.date(yr, 11, 1)
        try: wk = int(r["week"])
        except (ValueError, KeyError, TypeError): wk = 0
        games.append({
            "yr": yr, "week": wk, "gtype": r.get("game_type") or "", "dt": dt,
            "h": h, "a": a, "hs": hs, "as": as_,
            "spread": fl(r.get("spread_line")), "total_line": fl(r.get("total_line")),
        })
    games.sort(key=lambda g: (g["dt"], g["h"]))
    return games


def load_consensus():
    crank = {}
    p = os.path.join(PROJ, "consensus_nfl.csv")
    with open(p, newline="", encoding="utf-8") as f:
        for x in csv.DictReader(f):
            if x.get("team"): crank[fix(x["team"])] = int(x["consensus_rank"])
    return crank


# ----------------------------------------------------------------------------
# Rebuild ratings + calibration from plays/games strictly BEFORE `cutoff`.
# Mirrors build_nfl.py: weighted opponent-adjusted off/def EPA, then KP/HFA/etc.
# wt() referenced to `cutoff` (the live model references "today").
# Returns dict of params + per-team off/def, or None if too little data.
# ----------------------------------------------------------------------------
def build_ratings(plays, games, cutoff):
    def wt(dt): return 0.5 ** ((cutoff - dt).days / HALFLIFE)

    Mepa = defaultdict(lambda: defaultdict(float)); Mn = defaultdict(lambda: defaultdict(float))
    teams = set()
    for r in plays:
        if r["dt"] >= cutoff: break  # plays are date-sorted; nothing at/after cutoff
        o, d, pt, epa = r["o"], r["d"], r["pt"], r["epa"]
        if not o or not d: continue
        teams.add(o); teams.add(d)
        if pt in ("pass", "run") and epa is not None:
            w = wt(r["dt"])
            Mepa[o][d] += w * epa; Mn[o][d] += w
    teams = sorted(t for t in teams if t)
    if len(teams) < 24:
        return None

    off = {t: 0.0 for t in teams}; dfn = {t: 0.0 for t in teams}
    for _ in range(ITERS):
        no = {t: [0.0, 0.0] for t in teams}; nd = {t: [0.0, 0.0] for t in teams}
        for o in teams:
            for d, n in Mn[o].items():
                if n <= 0: continue
                avg = Mepa[o][d] / n
                no[o][0] += n * (avg - dfn[d]); no[o][1] += n
                nd[d][0] += n * (avg - off[o]); nd[d][1] += n
        for t in teams:
            if no[t][1]: off[t] = no[t][0] / no[t][1]
            if nd[t][1]: dfn[t] = nd[t][0] / nd[t][1]
        om = sum(off.values()) / len(teams); dm = sum(dfn.values()) / len(teams)
        for t in teams: off[t] -= om; dfn[t] -= dm

    # calibrate EPA->points on games strictly before cutoff (both teams rated)
    gms = [g for g in games if g["dt"] < cutoff and g["h"] in off and g["a"] in off]
    if len(gms) < 80:
        return None
    tw = tp = hw = hm = 0.0
    for g in gms:
        w = wt(g["dt"]); tp += w * (g["hs"] + g["as"]); tw += 2 * w
        hm += w * (g["hs"] - g["as"]); hw += w
    LG = tp / tw; HFA = hm / hw
    sxy = sxx = sxyt = sxxt = 0.0
    for g in gms:
        w = wt(g["dt"]); h, a = g["h"], g["a"]
        dE = (off[h] + dfn[a]) - (off[a] + dfn[h]); m = (g["hs"] - g["as"]) - HFA
        sxy += w * dE * m; sxx += w * dE * dE
        sE = (off[h] + dfn[a]) + (off[a] + dfn[h]); tt = (g["hs"] + g["as"]) - 2 * LG
        sxyt += w * sE * tt; sxxt += w * sE * sE
    KP = sxy / sxx if sxx else 60.0
    KT = sxyt / sxxt if sxxt else KP
    sm = st = sw = 0.0
    for g in gms:
        w = wt(g["dt"]); h, a = g["h"], g["a"]
        pm = KP * ((off[h] + dfn[a]) - (off[a] + dfn[h])) + HFA
        ptot = 2 * LG + KT * ((off[h] + dfn[a]) + (off[a] + dfn[h]))
        sm += w * ((g["hs"] - g["as"]) - pm) ** 2; st += w * ((g["hs"] + g["as"]) - ptot) ** 2; sw += w
    SD_M = (sm / sw) ** 0.5; SD_T = (st / sw) ** 0.5
    return {"off": off, "dfn": dfn, "teams": teams,
            "LG": LG, "HFA": HFA, "KP": KP, "KT": KT, "SD_M": SD_M, "SD_T": SD_T}


def apply_consensus_blend(R, crank, games_played):
    """Replicate blend_nfl.py's decaying-consensus prior at this point in time.
    Returns a copy of off/dfn shifted toward consensus by weight (1-W)."""
    off, dfn, teams = R["off"], R["dfn"], R["teams"]
    n = len(teams)
    W = min(1.0, BLEND_BASE + games_played / GAMES_PER_TEAM)
    model_net = {t: off[t] - dfn[t] for t in teams}
    ladder = sorted(model_net.values(), reverse=True)
    noff, ndfn = dict(off), dict(dfn)
    for t in teams:
        mnet = model_net[t]
        rank = crank.get(t)
        cons_net = ladder[min(max(rank, 1), n) - 1] if rank else mnet
        final_net = W * mnet + (1.0 - W) * cons_net
        delta = final_net - mnet
        noff[t] = off[t] + delta / 2.0
        ndfn[t] = dfn[t] - delta / 2.0
    return noff, ndfn, W


def predict(R, off, dfn, h, a):
    """Production engine (prosports_app.js / build_nfl.py): Gaussian point-margin."""
    LG, HFA, KP, KT, SD_M, SD_T = R["LG"], R["HFA"], R["KP"], R["KT"], R["SD_M"], R["SD_T"]
    eH = LG + KP * (off[h] + dfn[a]) + HFA / 2.0
    eA = LG + KP * (off[a] + dfn[h]) - HFA / 2.0
    m = eH - eA            # projected home margin
    tot = 2 * LG + KT * ((off[h] + dfn[a]) + (off[a] + dfn[h]))
    winH = cdf(m / SD_M)
    return {"m": m, "tot": tot, "winH": winH}


# ----------------------------------------------------------------------------
# Walk-forward
# ----------------------------------------------------------------------------
def run():
    print("Loading data ...", flush=True)
    plays = load_pbp()
    games = load_games()
    crank = load_consensus()
    print(f"  {len(games)} completed games across {PBP_SEASONS}", flush=True)

    # results per game: (use_blend) -> list of dicts
    out = {True: [], False: []}

    for tyr in TEST_SEASONS:
        # group test games by week; cutoff = first gameday of the week
        wk_games = defaultdict(list)
        for g in games:
            if g["yr"] == tyr and g["gtype"] == "REG":
                wk_games[g["week"]].append(g)
        for wk in sorted(wk_games):
            wgames = wk_games[wk]
            cutoff = min(g["dt"] for g in wgames)
            R = build_ratings(plays, games, cutoff)
            if R is None:
                continue
            # games-played-this-season for the decay (REG games before cutoff, /32)
            gp_team = defaultdict(int)
            for g in games:
                if g["yr"] == tyr and g["gtype"] == "REG" and g["dt"] < cutoff:
                    gp_team[g["h"]] += 1; gp_team[g["a"]] += 1
            games_played = (sum(gp_team.values()) / 32.0) if gp_team else 0.0
            off_b, dfn_b, W = apply_consensus_blend(R, crank, games_played)

            for g in wgames:
                h, a = g["h"], g["a"]
                if h not in R["off"] or a not in R["off"]:
                    continue
                if g["hs"] == g["as"]:
                    continue  # tie: no winner to score (rare)
                home_won = g["hs"] > g["as"]
                actual_margin = g["hs"] - g["as"]
                for use_blend in (True, False):
                    of, df = (off_b, dfn_b) if use_blend else (R["off"], R["dfn"])
                    pr = predict(R, of, df, h, a)
                    out[use_blend].append({
                        "yr": tyr, "week": wk, "h": h, "a": a,
                        "winH": pr["winH"], "pm": pr["m"], "ptot": pr["tot"],
                        "home_won": home_won, "actual_margin": actual_margin,
                        "actual_total": g["hs"] + g["as"],
                        "spread": g["spread"], "total_line": g["total_line"],
                    })
            print(f"  {tyr} wk{wk:2d}: cutoff {cutoff} | {len(wgames)} games | W={W:.3f} | KP={R['KP']:.1f} HFA={R['HFA']:.1f} SDm={R['SD_M']:.1f}", flush=True)

    report(out)


# ----------------------------------------------------------------------------
# Scoring / reporting
# ----------------------------------------------------------------------------
def report(out):
    for use_blend in (True, False):
        rows = out[use_blend]
        label = "WITH decaying consensus prior" if use_blend else "PURE on-field model (no consensus)"
        print("\n" + "=" * 78)
        print(f"  OUT-OF-SAMPLE BACKTEST -- {label}")
        print(f"  Test seasons {TEST_SEASONS} (regular season), N = {len(rows)} games")
        print("=" * 78)
        if not rows:
            print("  no games"); continue
        score_block(rows)


def score_block(rows):
    N = len(rows)
    # straight-up: model favorite = side with higher win prob
    su_hits = 0; brier = 0.0; logloss = 0.0; mae_margin = 0.0
    for r in rows:
        fav_home = r["winH"] >= 0.5
        p_fav = r["winH"] if fav_home else 1 - r["winH"]
        fav_won = (fav_home and r["home_won"]) or ((not fav_home) and (not r["home_won"]))
        if fav_won: su_hits += 1
        # brier/logloss on P(home win) vs home_won
        y = 1.0 if r["home_won"] else 0.0
        p = min(max(r["winH"], 1e-9), 1 - 1e-9)
        brier += (p - y) ** 2
        logloss += -(y * math.log(p) + (1 - y) * math.log(1 - p))
        mae_margin += abs(r["pm"] - r["actual_margin"])
    print(f"\n  STRAIGHT-UP WINNER hit-rate : {su_hits}/{N} = {100*su_hits/N:.1f}%")
    print(f"  Brier score (P[home win])   : {brier/N:.4f}   (lower better; 0.25 = coin-flip)")
    print(f"  Log-loss                    : {logloss/N:.4f}   (lower better; 0.693 = coin-flip)")
    print(f"  Mean abs error of MARGIN     : {mae_margin/N:.2f} pts")

    # calibration by confidence bucket (on the FAVORITE's win prob)
    print("\n  CALIBRATION by predicted-favorite confidence bucket:")
    print("    bucket      N    pred-win%   actual-win%   gap")
    buckets = [(0.50, 0.60), (0.60, 0.70), (0.70, 0.80), (0.80, 1.01)]
    for lo, hi in buckets:
        sel = []
        for r in rows:
            p_fav = r["winH"] if r["winH"] >= 0.5 else 1 - r["winH"]
            if lo <= p_fav < hi:
                fav_home = r["winH"] >= 0.5
                fav_won = (fav_home and r["home_won"]) or ((not fav_home) and (not r["home_won"]))
                sel.append((p_fav, fav_won))
        if not sel:
            print(f"    {int(lo*100)}-{int(hi*100 if hi<=1 else 100)}      0       --          --         --"); continue
        n = len(sel); pred = sum(p for p, _ in sel) / n; act = sum(1 for _, w in sel if w) / n
        print(f"    {int(lo*100)}-{min(int(hi*100),100):>3d}  {n:4d}    {100*pred:6.1f}%      {100*act:6.1f}%     {100*(act-pred):+5.1f}")

    # trust-tier table: hit-rate of picks above each confidence threshold
    print("\n  TRUST-TIER (picks at/above each favorite-confidence threshold):")
    print("    threshold    N    hit-rate")
    for thr in (0.50, 0.60, 0.65, 0.70, 0.75, 0.80):
        sel = []
        for r in rows:
            p_fav = r["winH"] if r["winH"] >= 0.5 else 1 - r["winH"]
            if p_fav >= thr:
                fav_home = r["winH"] >= 0.5
                fav_won = (fav_home and r["home_won"]) or ((not fav_home) and (not r["home_won"]))
                sel.append(fav_won)
        if sel:
            print(f"    >= {int(thr*100)}%     {len(sel):4d}    {100*sum(sel)/len(sel):.1f}%")
        else:
            print(f"    >= {int(thr*100)}%        0       --")

    # ATS + totals (where lines exist)
    ats_rows = [r for r in rows if r["spread"] is not None]
    if ats_rows:
        ats_hit = ats_push = 0
        for r in ats_rows:
            # spread is the home line: home favored by `spread` pts (positive => home favored)
            model_pick_home = r["pm"] > r["spread"]
            home_cover_margin = r["actual_margin"] - r["spread"]
            if abs(home_cover_margin) < 1e-9:
                ats_push += 1; continue
            home_covered = home_cover_margin > 0
            if model_pick_home == home_covered:
                ats_hit += 1
        decided = len(ats_rows) - ats_push
        print(f"\n  AGAINST-THE-SPREAD (vs closing line, {len(ats_rows)} games, {ats_push} pushes):")
        print(f"    ATS hit-rate (decided) : {ats_hit}/{decided} = {100*ats_hit/decided:.1f}%   (breakeven 52.4%)")
    tot_rows = [r for r in rows if r["total_line"] is not None]
    if tot_rows:
        ou_hit = ou_push = 0
        for r in tot_rows:
            model_over = r["ptot"] > r["total_line"]
            diff = r["actual_total"] - r["total_line"]
            if abs(diff) < 1e-9:
                ou_push += 1; continue
            actual_over = diff > 0
            if model_over == actual_over:
                ou_hit += 1
        dec = len(tot_rows) - ou_push
        print(f"  OVER/UNDER hit-rate     : {ou_hit}/{dec} = {100*ou_hit/dec:.1f}%   (breakeven 52.4%)")

    # team-level read: SU accuracy when each team is involved
    team_rec = defaultdict(lambda: [0, 0])  # team -> [correct, total]
    for r in rows:
        fav_home = r["winH"] >= 0.5
        fav_won = (fav_home and r["home_won"]) or ((not fav_home) and (not r["home_won"]))
        for t in (r["h"], r["a"]):
            team_rec[t][1] += 1
            if fav_won: team_rec[t][0] += 1
    ranked = sorted(team_rec.items(), key=lambda kv: kv[1][0] / kv[1][1])
    worst = ranked[:5]; best = ranked[-5:][::-1]
    print("\n  Hardest teams to predict (lowest SU hit-rate in their games):")
    for t, (c, n) in worst:
        print(f"    {t}: {c}/{n} = {100*c/n:.0f}%")
    print("  Easiest teams to predict (highest SU hit-rate in their games):")
    for t, (c, n) in best:
        print(f"    {t}: {c}/{n} = {100*c/n:.0f}%")


if __name__ == "__main__":
    run()
