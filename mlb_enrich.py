# -*- coding: utf-8 -*-
"""
mlb_enrich.py -- builds a point-in-time ENRICHMENT cache for the MLB backtest day-of signals.

Two caches (both keyed so build_mlb.py and backtest_mlb.py read them identically):

1) mlb_enrich_cache.json  -- per-game schedule-level context (ONE schedule call/season):
     key = gamePk (str) -> {date, home, away, venue_id, venue_name, dayNight,
                            temp (int F or None), wind_mph (float or None),
                            wind_dir (str raw or None), condition, roof (bool: dome/closed)}
   Plus a date+home+away -> gamePk index for joining to the rating cache's game dicts.
   Weather here is the GAMETIME weather statsapi records, which equals the pregame conditions
   posted before first pitch -> legitimately as-of (no leakage of the result).

2) mlb_box_cache.json  -- per-game pitcher USAGE from the boxscore (for bullpen fatigue):
     key = gamePk (str) -> {home: [[pid, ip], ...], away: [[pid, ip], ...]}
   The FIRST entry in each list is the starter; the rest are relievers. IP is that game's IP.
   Used strictly from games BEFORE the prediction date -> as-of.

Run standalone to (re)build:  python mlb_enrich.py
Idempotent / incremental: keeps existing entries, only fetches missing gamePks.
"""
import json, os, urllib.request, time, datetime, collections

PROJ = os.path.dirname(os.path.abspath(__file__))
ENRICH = os.path.join(PROJ, "mlb_enrich_cache.json")
BOX = os.path.join(PROJ, "mlb_box_cache.json")
PLATOON = os.path.join(PROJ, "mlb_platoon_cache.json")

# statsapi team ids -> our canonical team names (after NAMEFIX).
TEAM_IDS = {
    109: "Arizona Diamondbacks", 144: "Atlanta Braves", 110: "Baltimore Orioles",
    111: "Boston Red Sox", 112: "Chicago Cubs", 145: "Chicago White Sox",
    113: "Cincinnati Reds", 114: "Cleveland Guardians", 115: "Colorado Rockies",
    116: "Detroit Tigers", 117: "Houston Astros", 118: "Kansas City Royals",
    108: "Los Angeles Angels", 119: "Los Angeles Dodgers", 146: "Miami Marlins",
    158: "Milwaukee Brewers", 142: "Minnesota Twins", 121: "New York Mets",
    147: "New York Yankees", 133: "Athletics", 143: "Philadelphia Phillies",
    134: "Pittsburgh Pirates", 135: "San Diego Padres", 137: "San Francisco Giants",
    136: "Seattle Mariners", 138: "St. Louis Cardinals", 139: "Tampa Bay Rays",
    140: "Texas Rangers", 141: "Toronto Blue Jays", 120: "Washington Nationals",
}

NAMEFIX = {"Oakland Athletics": "Athletics"}
fn = lambda n: NAMEFIX.get(n, n)

# A game is "roofed" (wind irrelevant) when statsapi's GAMETIME weather CONDITION says the
# dome/roof is shut. This is per-GAME and point-in-time (known before first pitch) -- far more
# accurate than a static venue list, since retractable roofs open/close game-by-game. The
# condition string also lets us neutralize indoor games for the weather signal cleanly.
ROOF_CONDITIONS = {"Dome", "Roof Closed"}


def get(url, t=45, retries=4):
    for i in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            return json.loads(urllib.request.urlopen(req, timeout=t).read())
        except Exception:
            if i == retries - 1:
                raise
            time.sleep(1.5)


def parse_wind(wind_str):
    """'3 mph, Out To LF' -> (3.0, 'Out To LF'). '0 mph, None' -> (0.0, 'None').
    Returns (mph or None, dir or None)."""
    if not wind_str:
        return None, None
    try:
        parts = wind_str.split(",", 1)
        mph = float(parts[0].strip().split()[0])
        direction = parts[1].strip() if len(parts) > 1 else None
        return mph, direction
    except (ValueError, IndexError):
        return None, None


def is_roof(condition):
    """Per-game roof state from the gametime condition string (point-in-time, no leakage)."""
    return (condition or "").strip() in ROOF_CONDITIONS


def build_schedule_enrich(seasons):
    """One schedule call per season w/ venue+weather hydration -> per-gamePk context."""
    enrich = {}
    if os.path.exists(ENRICH):
        with open(ENRICH, encoding="utf-8") as f:
            enrich = json.load(f).get("by_pk", {})
    for y in seasons:
        j = get(f"https://statsapi.mlb.com/api/v1/schedule?sportId=1&startDate={y}-03-01"
                f"&endDate={y}-11-15&gameType=R&hydrate=venue,weather")
        n0 = len(enrich)
        for dd in j.get("dates", []):
            for g in dd.get("games", []):
                if g.get("status", {}).get("detailedState") != "Final":
                    continue
                pk = str(g.get("gamePk"))
                t = g.get("teams", {})
                h, a = t.get("home", {}), t.get("away", {})
                try:
                    home = fn(h["team"]["name"]); away = fn(a["team"]["name"])
                except KeyError:
                    continue
                d = g.get("officialDate") or g.get("gameDate", "")[:10]
                ven = g.get("venue", {}) or {}
                vid, vname = ven.get("id"), ven.get("name")
                w = g.get("weather", {}) or {}
                temp = None
                try:
                    temp = int(w.get("temp")) if w.get("temp") not in (None, "") else None
                except (ValueError, TypeError):
                    temp = None
                wind_mph, wind_dir = parse_wind(w.get("wind"))
                roof = is_roof(w.get("condition"))
                enrich[pk] = {
                    "date": d, "home": home, "away": away,
                    "venue_id": vid, "venue_name": vname,
                    "dayNight": g.get("dayNight"),
                    "temp": temp, "wind_mph": wind_mph, "wind_dir": wind_dir,
                    "condition": w.get("condition"), "roof": roof,
                }
        print(f"  enrich schedule {y}: +{len(enrich)-n0} games (total {len(enrich)})", flush=True)
    # date+home+away -> pk index
    idx = {}
    for pk, e in enrich.items():
        idx[f"{e['date']}|{e['home']}|{e['away']}"] = pk
    with open(ENRICH, "w", encoding="utf-8") as f:
        json.dump({"by_pk": enrich, "index": idx}, f)
    return enrich, idx


def build_box_cache(seasons, enrich):
    """Per-game pitcher usage (starter + relievers, with IP) for bullpen fatigue.
    Only games in the given seasons; fetched from boxscore. Incremental."""
    box = {}
    if os.path.exists(BOX):
        with open(BOX, encoding="utf-8") as f:
            box = json.load(f)
    want = [pk for pk, e in enrich.items()
            if e["date"][:4].isdigit() and int(e["date"][:4]) in seasons and pk not in box]
    print(f"  boxscores to fetch: {len(want)} (already cached {len(box)})", flush=True)
    for n, pk in enumerate(want):
        try:
            j = get(f"https://statsapi.mlb.com/api/v1/game/{pk}/boxscore")
        except Exception:
            continue
        tm = j.get("teams", {})
        rec = {}
        for side in ("home", "away"):
            s = tm.get(side, {})
            players = s.get("players", {})
            lst = []
            for pid in (s.get("pitchers") or []):
                p = players.get(f"ID{pid}", {})
                st = p.get("stats", {}).get("pitching", {})
                try:
                    ip = float(st.get("inningsPitched", 0) or 0)
                except (ValueError, TypeError):
                    ip = 0.0
                lst.append([pid, ip])
            rec[side] = lst
        box[pk] = rec
        if (n + 1) % 200 == 0:
            print(f"    boxscore {n+1}/{len(want)}", flush=True)
            with open(BOX, "w", encoding="utf-8") as f:
                json.dump(box, f)
    with open(BOX, "w", encoding="utf-8") as f:
        json.dump(box, f)
    return box


def build_platoon_cache(starter_ids, seasons):
    """pitcher hand by id + team OPS vs LHP/RHP by (team, season). Used by the
    starting-pitcher-handedness x opponent-platoon-split signal. PRIOR-season splits only
    (a game in year Y uses the team's year Y-1 split) -> strictly as-of, no leakage."""
    cache = {"hand": {}, "team_split": {}}
    if os.path.exists(PLATOON):
        with open(PLATOON, encoding="utf-8") as f:
            cache = json.load(f)
    # 1) pitcher handedness (one /people call per ~100 ids via comma list)
    have = set(cache["hand"].keys())
    need = [str(p) for p in starter_ids if str(p) not in have and p]
    print(f"  pitcher hands to fetch: {len(need)}", flush=True)
    for i in range(0, len(need), 100):
        chunk = need[i:i + 100]
        try:
            j = get("https://statsapi.mlb.com/api/v1/people?personIds=" + ",".join(chunk))
        except Exception:
            continue
        for p in j.get("people", []):
            cache["hand"][str(p["id"])] = (p.get("pitchHand", {}) or {}).get("code")
    # 2) team OPS vs LHP / RHP per season
    for y in seasons:
        for tid, tname in TEAM_IDS.items():
            key = f"{tname}|{y}"
            if key in cache["team_split"]:
                continue
            try:
                j = get(f"https://statsapi.mlb.com/api/v1/teams/{tid}/stats?stats=statSplits"
                        f"&group=hitting&season={y}&sitCodes=vl,vr&gameType=R")
            except Exception:
                continue
            vl = vr = None
            for s in j.get("stats", []):
                for x in s.get("splits", []):
                    code = (x.get("split", {}) or {}).get("code")
                    try:
                        ops = float(x.get("stat", {}).get("ops"))
                    except (ValueError, TypeError):
                        ops = None
                    if code == "vl":
                        vl = ops
                    elif code == "vr":
                        vr = ops
            cache["team_split"][key] = {"vl": vl, "vr": vr}
        print(f"  team splits {y}: done", flush=True)
    with open(PLATOON, "w", encoding="utf-8") as f:
        json.dump(cache, f)
    return cache


def main():
    # Fetch enrichment for the seasons used by the backtest (2024-25 test + a little prior is
    # unnecessary for schedule context; box only needs test seasons since fatigue looks back days).
    test_seasons = {2024, 2025}
    sched_seasons = list(range(2020, 2027))   # cheap (1 call/season); covers anything we join to
    print("Building schedule enrichment (venue/dayNight/weather)...", flush=True)
    enrich, idx = build_schedule_enrich(sched_seasons)
    print(f"Total enriched games: {len(enrich)}", flush=True)
    print("Building boxscore (bullpen usage) cache for test seasons...", flush=True)
    build_box_cache(test_seasons, enrich)
    print("Done.", flush=True)


if __name__ == "__main__":
    main()
