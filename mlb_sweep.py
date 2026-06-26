# -*- coding: utf-8 -*-
"""
mlb_sweep.py -- fast OOS strength sweep for the day-of signals (Goal 1).

Runs the walk-forward ONCE to capture, per game, the pre-signal base run rates (lh, la) and
all raw signal INPUTS (from meta). Then re-scores cheaply for any (PARK_W, WEATHER_W,
PLATOON_W, PEN_W) combination by recomputing the signal multipliers in pure Python and
re-running only the Poisson grid. This makes a full strength sweep take seconds instead of
re-solving ratings each time.

Discipline: report SU / Brier / log-loss OVERALL and PER SEASON (2024 vs 2025) so a signal is
only kept if it helps OOS and isn't a one-season fluke.
"""
import math, importlib
import backtest_mlb as B
import mlb_signals as S


def capture_base():
    """Walk-forward once (signals OFF). Returns list of per-game dicts with base lh/la + meta."""
    # force signals off during capture
    for k in ("PARK_W", "WEATHER_W", "PLATOON_W", "PEN_W"):
        setattr(S, k, 0.0)
    test_seasons = {2024, 2025}
    fetch_seasons = list(range(2026, 2019, -1))
    games, plogs = B.load_cache(fetch_seasons, test_seasons)
    crank = B.load_consensus()
    games_sorted = sorted(games, key=lambda g: g["date"])
    test = [g for g in games_sorted if g["season"] in test_seasons]
    enrich_idx, box, platoon = B.load_signal_data()
    park_pf = B.build_park_factors(games_sorted, enrich_idx)
    relief_idx = B.precompute_relief_ip(box, enrich_idx, games_sorted)
    lg_plat = B.league_platoon_baselines(platoon)
    hand = (platoon or {}).get("hand", {})
    tsplit = (platoon or {}).get("team_split", {})
    rating_cache = {}
    out = []
    for n, g in enumerate(test):
        cm = B.monday_of(g["date"]); cs = g["season"]
        key = (cs, cm)
        if key not in rating_cache:
            prior = [x for x in games_sorted if x["date"] < cm]
            rating_cache[key] = B.compute_ratings(prior, cs)
        res = rating_cache[key]
        if res is None:
            continue
        att, dfn, AVG, HOME, twsum = res
        lg_ra9 = AVG
        f_h = B.pitcher_factor(g["ph_id"], plogs, g["date"], cs, lg_ra9) if g["ph_id"] else None
        f_a = B.pitcher_factor(g["pa_id"], plogs, g["date"], cs, lg_ra9) if g["pa_id"] else None
        br = B.base_rates(att, dfn, AVG, HOME, g["home"], g["away"], f_h, f_a)
        if br is None:
            continue
        e = enrich_idx.get(f"{g['date']}|{g['home']}|{g['away']}") if enrich_idx else None
        _, meta = B.build_game_signals(g, e, cs, park_pf, relief_idx, hand, tsplit, lg_plat)
        # capture prior-season splits + league baseline for platoon recompute
        meta["_hs_split"] = tsplit.get(f"{g['home']}|{cs-1}")
        meta["_as_split"] = tsplit.get(f"{g['away']}|{cs-1}")
        meta["_lvl"], meta["_lvr"] = lg_plat.get(cs - 1, (None, None))
        out.append({"lh": br[0], "la": br[1], "home_won": 1 if g["hs"] > g["as"] else 0,
                    "season": cs, "meta": meta})
        if (n + 1) % 1000 == 0:
            print(f"  captured {n+1}/{len(test)}", flush=True)
    print(f"Captured {len(out)} games.", flush=True)
    return out


def score(cap, park_w=0.0, weather_w=0.0, platoon_w=0.0, pen_w=0.0):
    """Re-apply signals at given weights and score SU/Brier/LL overall + per season."""
    S.PARK_W, S.WEATHER_W, S.PLATOON_W, S.PEN_W = park_w, weather_w, platoon_w, pen_w
    agg = {"all": [0, 0, 0.0, 0.0], 2024: [0, 0, 0.0, 0.0], 2025: [0, 0, 0.0, 0.0]}
    for c in cap:
        m = c["meta"]; lh, la = c["lh"], c["la"]
        park = S.park_factor(m["park_pf"])
        weather = S.weather_factor(m["temp"], m["wind_mph"], m["wind_dir"], m["roof"])
        platoon = S.platoon_factor(m["_hs_split"], m["_as_split"], m["away_sp_hand"],
                                   m["home_sp_hand"], m["_lvl"], m["_lvr"])
        bullpen = S.bullpen_factor(m["home_pen_fat"], m["away_pen_fat"])
        alh, ala = S.apply_all(lh, la, park=park, weather=weather, platoon=platoon, bullpen=bullpen)
        p = B.winp_from_rates(alh, ala)
        hw = c["home_won"]
        for bucket in ("all", c["season"]):
            a = agg[bucket]
            a[0] += 1
            a[1] += 1 if (p >= 0.5) == bool(hw) else 0
            a[2] += (p - hw) ** 2
            a[3] += -(hw * math.log(max(p, 1e-12)) + (1 - hw) * math.log(max(1 - p, 1e-12)))
    res = {}
    for k, a in agg.items():
        if a[0]:
            res[k] = (a[1] / a[0] * 100, a[2] / a[0], a[3] / a[0], a[0])
    return res


def line(tag, r):
    a = r["all"]; s4 = r.get(2024); s5 = r.get(2025)
    return (f"  {tag:<26} SU {a[0]:5.2f}%  Brier {a[1]:.4f}  LL {a[2]:.4f}  (N={a[3]}) "
            f"| 2024 SU {s4[0]:.2f}/LL {s4[2]:.4f}  2025 SU {s5[0]:.2f}/LL {s5[2]:.4f}")


if __name__ == "__main__":
    cap = capture_base()
    base = score(cap)
    print("\n=== BASELINE (all signals OFF) ===")
    print(line("baseline", base))

    print("\n=== PARK factor sweep (PARK_W) ===")
    for w in (0.25, 0.5, 0.75, 1.0):
        print(line(f"PARK_W={w}", score(cap, park_w=w)))

    print("\n=== WEATHER sweep (WEATHER_W) ===")
    for w in (0.5, 1.0, 1.5, 2.0):
        print(line(f"WEATHER_W={w}", score(cap, weather_w=w)))

    print("\n=== PLATOON sweep (PLATOON_W) ===")
    for w in (0.25, 0.5, 0.75, 1.0, 1.5):
        print(line(f"PLATOON_W={w}", score(cap, platoon_w=w)))

    print("\n=== BULLPEN sweep (PEN_W) [full box cache, IP-corrected] ===")
    for w in (0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0):
        print(line(f"PEN_W={w}", score(cap, pen_w=w)))

    print("\n=== COMBO tests (best individual settings together) ===")
    print(line("PEN0.5", score(cap, pen_w=0.5)))
    print(line("PEN0.5+PARK0.5", score(cap, pen_w=0.5, park_w=0.5)))
    print(line("PEN0.5+PARK1.0", score(cap, pen_w=0.5, park_w=1.0)))
    print(line("PARK1.0+WEATHER1.0", score(cap, park_w=1.0, weather_w=1.0)))
