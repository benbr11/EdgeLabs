# -*- coding: utf-8 -*-
"""Bake the live model into web/data.js for the static web app.
Run after build_ratings.py (and --refresh) so the site shows current numbers:
    python build_ratings.py --refresh
    python export_web.py
"""
import csv, json
import groups as G

import os
PROJ = os.path.dirname(os.path.abspath(__file__))
RHO = -0.12
MODS = {
    "avail_floor": 0.60, "fatigue_per_day": 0.025,
    "stakes": {"clinched": 0.93, "eliminated": 1.0, "must-win": 1.04, "normal": 1.0},
    "alt_pen_per_km": 0.05, "alt_buffer": 500, "heat_pen_per_c": 0.005, "heat_buffer": 8,
    "weather": {"clear": 1.0, "rain": 0.90, "cold": 0.95, "heat": 0.93},
}

# --- ratings + params --------------------------------------------------------
R = {}; avg = home_adv = None
for r in csv.DictReader(open(PROJ + r"\ratings.csv", encoding="utf-8")):
    R[r["team"]] = {"att_mult": float(r["attack_mult"]), "dfn_mult": float(r["defense_mult"]),
                    "att100": float(r["attack_100"]), "def100": float(r["defense_100"]),
                    "elo": round(float(r["elo"])), "fifa": round(float(r["fifa_points"]))}
    avg = float(r["league_avg_goals"]); home_adv = float(r["home_adv_mult"])

for r in csv.DictReader(open(PROJ + r"\context.csv", encoding="utf-8")):
    if r["team"] in R:
        R[r["team"]].update({"home_temp": float(r["home_temp_c"]), "home_alt": float(r["home_alt_m"]),
                             "pop": float(r["population_m"]), "climate": r["climate"]})

# --- groups, standings, situations ------------------------------------------
groups, played, sched = G.get_groups()
group_complete = sum(len(G.group_state(g, played, sched)[4]) for g in groups) == 0
groups_sorted = sorted(groups, key=lambda grp: max(R[t]["att100"] + R[t]["def100"] for t in grp if t in R),
                       reverse=True)
group_of = {}; groups_out = []
for i, grp in enumerate(groups_sorted, 1):
    name = f"Group {i}"
    pts, gf, ga, pl, rem = G.group_state(grp, played, sched)
    table = sorted(grp, key=lambda t: (-pts[t], -(gf[t] - ga[t]), -gf[t]))
    for t in grp:
        group_of[t] = name
        if t in R:
            lab, stk = G.situation(t, grp, pts, rem)
            R[t]["stakes"] = stk; R[t]["stakes_label"] = lab
    groups_out.append({"name": name,
                       "table": [{"team": t, "P": pl[t], "pts": pts[t], "gf": gf[t],
                                  "ga": ga[t], "gd": gf[t] - ga[t]} for t in table],
                       "remaining": [[h, a] for h, a in rem]})
for t in R:
    R[t].setdefault("group", group_of.get(t, "?"))
    R[t].setdefault("stakes", "normal"); R[t].setdefault("stakes_label", "")

def stage_of(h, a):
    if group_of.get(h) and group_of.get(h) == group_of.get(a):
        return "group"
    return "knockout" if group_complete else "unknown"

# --- xG for played matches ---------------------------------------------------
XG_NAME = {"Cabo Verde": "Cape Verde", "Congo DR": "DR Congo", "Czechia": "Czech Republic",
           "Côte d'Ivoire": "Ivory Coast", "IR Iran": "Iran", "Türkiye": "Turkey", "USA": "United States"}
# 2026 host venues -> (altitude m, typical match-day temp C). Roofed/AC stadiums
# get a comfortable temp (no heat penalty); altitude always applies.
VENUE_COND = {"Mexico City":(2240,22),"Guadalajara":(1560,24),"Monterrey":(540,31),
    "Dallas":(180,21),"Houston":(15,21),"Atlanta":(320,21),"Los Angeles":(30,23),
    "Vancouver":(5,21),"Miami":(2,31),"Kansas City":(270,30),"New York":(10,26),
    "Philadelphia":(10,27),"Boston":(30,24),"San Francisco":(5,22),"Seattle":(50,22),"Toronto":(80,24)}
def venue_cond(stadium):
    for k, (alt, temp) in VENUE_COND.items():
        if stadium.startswith(k) or k in stadium: return {"alt": alt, "temp": temp, "name": k}
    return None

mxg = {}; mstage = {}; mvenue = {}
try:
    for r in csv.DictReader(open(PROJ + r"\wc2026_xg.csv", encoding="utf-8")):
        h = XG_NAME.get(r["home_team_name"], r["home_team_name"])
        a = XG_NAME.get(r["away_team_name"], r["away_team_name"])
        key = (r["date"], frozenset((h, a)))
        if r.get("stage_name"):
            mstage[key] = r["stage_name"]
        if r.get("stadium_name"):
            vc = venue_cond(r["stadium_name"])
            if vc: mvenue[key] = vc
        if r["status"] == "Completed" and r["home_xg"]:
            mxg[key] = {h: float(r["home_xg"]), a: float(r["away_xg"])}
except FileNotFoundError:
    pass

# --- fixtures ----------------------------------------------------------------
fixtures = []
def venfields(key):
    vc = mvenue.get(key)
    return {"valt": vc["alt"] if vc else None, "vtemp": vc["temp"] if vc else None,
            "venue": vc["name"] if vc else None}
for (d, h, a, hs, as_) in played:
    key = (d, frozenset((h, a)))
    xg = mxg.get(key, {})
    fixtures.append({"date": d, "home": h, "away": a, "status": "played",
                     "hs": hs, "as": as_, "hxg": xg.get(h), "axg": xg.get(a),
                     "stage": stage_of(h, a), "round": mstage.get(key), **venfields(key)})
for (d, h, a) in sched:
    key = (d, frozenset((h, a)))
    fixtures.append({"date": d, "home": h, "away": a, "status": "scheduled",
                     "stage": stage_of(h, a), "round": mstage.get(key), **venfields(key)})
fixtures.sort(key=lambda x: x["date"])
generated = max((f["date"] for f in fixtures if f["status"] == "played"), default="")

# --- player model: shot-level xG + set pieces (StatsBomb) blended with squad data --
import unicodedata, re
def _norm(s):
    s = "".join(c for c in unicodedata.normalize("NFKD", s.lower()) if not unicodedata.combining(c))
    return set(t for t in s.replace("-", " ").split() if len(t) >= 4)
def _clean_name(s):
    # Some scraped squads glue/duplicate name tokens: "Neymar Neymar Jr",
    # "MARTINELLIGabriel Gabriel". Split glued ALLCAPS-surname+FirstName, de-shout
    # ALLCAPS tokens, and drop consecutive duplicate tokens.
    s = re.sub(r"([A-Z]{2,})([A-Z][a-z])", r"\1 \2", s)
    out = []
    for tok in s.split():
        t = tok.capitalize() if tok.isupper() else tok
        if not out or out[-1].lower() != t.lower():
            out.append(t)
    return " ".join(out) or s
SB_TEAMFIX = {"Cape Verde Islands":"Cape Verde","Congo DR":"DR Congo","Korea Republic":"South Korea",
              "Côte d'Ivoire":"Ivory Coast","Czechia":"Czech Republic","Türkiye":"Turkey","United States of America":"United States"}
def _matchlist(lst, name):
    tk = _norm(name); best = None; bestn = 0
    for c in lst:
        n = len(tk & c["tok"])
        if n > bestn: bestn, best = n, c
    return best if bestn >= 1 else None

# (1) UNIVERSAL international scoring profile (martj42 goalscorers.csv) -- covers EVERY
# national team, recency-weighted, open-play vs penalty split. The backbone of the model.
intl_by_team = {}
try:
    for r in csv.DictReader(open(PROJ + r"\player_intl.csv", encoding="utf-8")):
        tm = XG_NAME.get(r["team"], r["team"])
        intl_by_team.setdefault(tm, []).append({"tok": _norm(r["scorer"]),
            "op_rate": float(r["op_rate"]), "pen_rate": float(r["pen_rate"]),
            "opg_w": float(r["op_goals_w"]), "peng_w": float(r["pen_goals_w"])})
except FileNotFoundError:
    pass

# (2) StatsBomb shot QUALITY (player_xg.csv) -- nudges the rate where we have shot data.
sb_by_team = {}
try:
    for r in csv.DictReader(open(PROJ + r"\player_xg.csv", encoding="utf-8")):
        tm = SB_TEAMFIX.get(r["team"], r["team"]); tm = XG_NAME.get(tm, tm)
        if float(r["apps"]) < 0.5: continue
        sb_by_team.setdefault(tm, []).append({"tok": _norm(r["player"]), "apps": float(r["apps"]),
            "npxg": float(r["npxg"]), "pen_sh": float(r["pen_sh"]), "pen_g": float(r["pen_g"]), "fk_xg": float(r["fk_xg"])})
except FileNotFoundError:
    pass

PLAYERS = {}
try:
    pj = json.load(open(PROJ + r"\players_raw.json", encoding="utf-8"))
    id2name = {p["team_id"]: p["team_name"] for p in pj}
    bucket = {}
    for r in csv.DictReader(open(PROJ + r"\squads.csv", encoding="utf-8")):
        team = XG_NAME.get(id2name.get(int(r["team_id"]), ""), id2name.get(int(r["team_id"]), ""))
        if team not in R or r["position"] == "GK":
            continue
        pos = r["position"]; caps = int(r["caps"] or 0); goals = int(r["goals"] or 0)
        if caps < 5: continue
        val = float(r["market_value_eur"] or 0) / 1e6
        career_op = goals/(caps+2) * 0.85                         # career open-play rate (stability prior)
        intl = _matchlist(intl_by_team.get(team, []), r["player_name"])
        sb = _matchlist(sb_by_team.get(team, []), r["player_name"])
        sb_apps = sb["apps"] if sb else 0.0
        # --- open-play goals/game: universal recency rate, nudged by StatsBomb shot quality,
        #     then shrunk toward the career prior for thin samples (Bayesian-ish) ---
        if intl:
            op_core = intl["op_rate"]
            if sb_apps >= 1.5:
                op_core = 0.6*intl["op_rate"] + 0.4*(sb["npxg"]/sb_apps)   # blend actual + shot xG
            n_eff = intl["opg_w"] + intl["peng_w"] + 0.5*sb_apps           # how much real signal
            peng = intl["pen_rate"]; pengw = intl["peng_w"]                # data-driven penalty rate
        elif sb_apps >= 1.5:
            op_core = 0.5*(sb["npxg"]/sb_apps) + 0.5*career_op
            n_eff = sb_apps; peng = (sb["pen_g"]/sb_apps); pengw = sb["pen_g"]
        else:
            op_core = career_op; n_eff = 0.0; peng = 0.0; pengw = 0.0
        K = 3.0
        op = (n_eff*op_core + K*career_op) / (n_eff + K)
        fk = round(sb["fk_xg"]/sb_apps, 3) if sb_apps >= 1.5 else 0.0
        # confidence 0..1: how much real data backs this player (drives the app's trust flag)
        conf = min(1.0, (intl["opg_w"]+intl["peng_w"] if intl else 0)/5.0 + min(sb_apps, 8)/16.0)
        if not intl and sb_apps < 1.5: conf = min(conf, 0.15)
        if op < 0.02 and pengw < 0.3: continue                   # not a goal threat
        bucket.setdefault(team, []).append({"n": _clean_name(r["player_name"]), "pos": pos, "val": round(val),
            "op": round(op, 3), "peng": round(peng, 3), "fk": fk, "pen": 0, "pengw": pengw, "conf": round(conf, 2)})
    for team, lst in bucket.items():
        # ONE designated penalty taker per team for the PK badge: most recent-weighted pen goals
        cand = max(lst, key=lambda p: p["pengw"], default=None)
        if cand and cand["pengw"] >= 0.8: cand["pen"] = 1
        for p in lst: p["thr"] = round(p["op"] + p["peng"] + p["fk"], 3)
        lst.sort(key=lambda p: -p["thr"])
        PLAYERS[team] = [{k: v for k, v in p.items() if k not in ("thr", "pengw")} for p in lst[:8]]
    print(f"Players: {sum(len(v) for v in PLAYERS.values())} contributors, {len(PLAYERS)} teams; "
          f"{sum(1 for t in PLAYERS for p in PLAYERS[t] if p['pen'])} penalty takers; "
          f"{sum(1 for t in PLAYERS for p in PLAYERS[t] if p['conf']>=0.35)} high/med-confidence")
    for tm, nm in [("Bosnia and Herzegovina","demirovic"),("Qatar","afif"),("France","mbappe"),
                   ("England","kane"),("Norway","haaland"),("Argentina","messi")]:
        for p in PLAYERS.get(tm, []):
            if nm in _clean_name(p["n"]).lower() or nm in p["n"].lower():
                print(f"  {tm}/{p['n']}: op={p['op']} peng={p['peng']} fk={p['fk']} pen={p['pen']} conf={p['conf']}")
except FileNotFoundError:
    print("squads.csv/players_raw.json missing -- skipping player odds")

played_max = max((row["P"] for g in groups_out for row in g["table"]), default=0)
if group_complete:
    _kos = sorted([f for f in fixtures if f["status"] == "scheduled" and f.get("round")], key=lambda x: x["date"])
    stage_label = _kos[0]["round"] if _kos else "Knockouts"
else:
    stage_label = f"Matchday {played_max + 1}"

DATA = {"params": {"avg": avg, "home_adv": home_adv, "rho": RHO, "hosts": sorted(G.HOSTS),
                   "group_complete": group_complete, "generated": generated, "stage_label": stage_label},
        "mods": MODS, "teams": R, "groups": groups_out, "fixtures": fixtures, "players": PLAYERS}

with open(PROJ + r"\web\data.js", "w", encoding="utf-8") as f:
    f.write("window.WC_DATA = " + json.dumps(DATA, ensure_ascii=False) + ";\n")
print(f"Wrote web/data.js: {len(R)} teams, {len(groups_out)} groups, "
      f"{len(fixtures)} fixtures, generated {generated}, group_complete={group_complete}")
