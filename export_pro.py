# -*- coding: utf-8 -*-
"""Bake NFL / NBA / MLB ratings into web/{nfl,nba,mlb}_data.js. Reads the CSVs (no re-fetch)."""
import csv, json, os, datetime
PROJ = os.path.dirname(os.path.abspath(__file__))
GEN = datetime.date.today().isoformat()
NFL_NAMES = {"ARI":"Arizona Cardinals","ATL":"Atlanta Falcons","BAL":"Baltimore Ravens","BUF":"Buffalo Bills",
"CAR":"Carolina Panthers","CHI":"Chicago Bears","CIN":"Cincinnati Bengals","CLE":"Cleveland Browns","DAL":"Dallas Cowboys",
"DEN":"Denver Broncos","DET":"Detroit Lions","GB":"Green Bay Packers","HOU":"Houston Texans","IND":"Indianapolis Colts",
"JAX":"Jacksonville Jaguars","KC":"Kansas City Chiefs","LA":"Los Angeles Rams","LAC":"Los Angeles Chargers","LV":"Las Vegas Raiders",
"MIA":"Miami Dolphins","MIN":"Minnesota Vikings","NE":"New England Patriots","NO":"New Orleans Saints","NYG":"New York Giants",
"NYJ":"New York Jets","PHI":"Philadelphia Eagles","PIT":"Pittsburgh Steelers","SEA":"Seattle Seahawks","SF":"San Francisco 49ers",
"TB":"Tampa Bay Buccaneers","TEN":"Tennessee Titans","WAS":"Washington Commanders"}

def write(fname, var, data):
    with open(PROJ + r"\web\\" + fname, "w", encoding="utf-8") as f:
        f.write(f"window.{var} = " + json.dumps(data, ensure_ascii=False) + ";\n")

# ---- NFL (play-by-play: independent off/def EPA + granular components + players + kickers) ----
def F(x):
    try: return float(x)
    except (TypeError, ValueError): return 0.0
rows = list(csv.DictReader(open(PROJ + r"\nfl_ratings.csv", encoding="utf-8")))
p = rows[0]
nfl_teams = {}
for r in rows:
    nfl_teams[r["team"]] = {
        "name": NFL_NAMES.get(r["team"], r["team"]),
        "off": F(r["off_epa"]), "dfn": F(r["def_epa"]), "net": F(r["net_epa"]),
        "or": int(r["off_rank"]), "dr": int(r["def_rank"]), "nr": int(r["net_rank"]),
        "ppf": F(r["ppf"]), "ppa": F(r["ppa"]),
        "osucc": F(r["off_succ"]), "dsucc": F(r["def_succ"]),
        "cmp": F(r["cmp_pct"]), "cmpA": F(r["cmp_pct_allowed"]),
        "ypa": F(r["ypa"]), "ypaA": F(r["ypa_allowed"]), "ypc": F(r["ypc"]), "ypcA": F(r["ypc_allowed"]),
        "cpoe": F(r["cpoe"]), "third": F(r["third"]), "thirdA": F(r["third_allowed"]),
        "dsc": F(r["drive_score"]), "dscA": F(r["drive_score_allowed"]),
        "gv": F(r["giveaways"]), "tk": F(r["takeaways"]), "sk": F(r["sacks"]),
        "kr": F(r["kr_avg"]), "pr": F(r["pr_avg"]), "sfp": F(r["start_fp"]), "pen": F(r["pen_pg"]), "fg": F(r["fg_pct"]),
    }
# players by team
nfl_players = {}
try:
    for r in csv.DictReader(open(PROJ + r"\nfl_players.csv", encoding="utf-8")):
        nfl_players.setdefault(r["team"], []).append(
            {"n": r["player"], "pos": r["pos"], "val": F(r["value"]),
             "py": int(F(r["pass_yds"])), "ry": int(F(r["rush_yds"])), "recy": int(F(r["rec_yds"])), "td": int(F(r["tds"]))})
except FileNotFoundError: pass
# kickers
nfl_kickers = []
try:
    for r in csv.DictReader(open(PROJ + r"\nfl_kickers.csv", encoding="utf-8")):
        nfl_kickers.append({"k": r["kicker"], "team": r["team"], "u30": r["u30"], "b39": r["30-39"],
                            "b49": r["40-49"], "b50": r["50+"], "att": int(F(r["att"]))})
except FileNotFoundError: pass
nfl = {"params": {"lg": F(p["lg_ppg"]), "hfa": F(p["hfa"]), "kp": F(p["kp"]), "kt": F(p["kt"]),
                  "sd_m": F(p["sd_margin"]), "sd_t": F(p["sd_total"])},
       "teams": nfl_teams, "players": nfl_players, "kickers": nfl_kickers, "generated": GEN}
write("nfl_data.js", "NFL_DATA", nfl); print(f"NFL: {len(nfl_teams)} teams, {sum(len(v) for v in nfl_players.values())} players, {len(nfl_kickers)} kickers")

# ---- NBA ----
rows = list(csv.DictReader(open(PROJ + r"\nba_ratings.csv", encoding="utf-8")))
p = rows[0]
nba_players = []
try:
    for r in csv.DictReader(open(PROJ + r"\nba_players.csv", encoding="utf-8")):
        nba_players.append({"n": r["player"], "team": r["team"], "pos": r["pos"], "val": float(r["value"]),
                            "ppg": float(r["ppg"]), "rpg": float(r["rpg"]), "apg": float(r["apg"]),
                            "spg": float(r["spg"]), "bpg": float(r["bpg"])})
except FileNotFoundError: pass
nba = {"params": {"lg": float(p["lg_ppg"]), "hfa": float(p["hfa"]), "sd_m": float(p["sd_margin"]), "sd_t": float(p["sd_total"])},
       "teams": {r["team"]: {"name": r["name"], "off": float(r["off"]), "dfn": float(r["def"]), "net": float(r["net"])} for r in rows},
       "players": nba_players, "generated": GEN}
write("nba_data.js", "NBA_DATA", nba); print(f"NBA: {len(nba['teams'])} teams, {len(nba_players)} players")

# ---- MLB ----
rows = list(csv.DictReader(open(PROJ + r"\mlb_ratings.csv", encoding="utf-8")))
p = rows[0]
prows = list(csv.DictReader(open(PROJ + r"\mlb_pitchers.csv", encoding="utf-8")))
# keep likely starters (per-season IP) for the selector, grouped by current team
pit = [{"n": r["pitcher"], "team": r["team"], "ra9": float(r["ra9"]), "factor": float(r["factor"]), "ip": int(r["ip"])}
       for r in prows if int(r["ip"]) >= 80]
hit = []
try:
    for r in csv.DictReader(open(PROJ + r"\mlb_hitters.csv", encoding="utf-8")):
        hit.append({"n": r["hitter"], "team": r["team"], "ops": float(r["ops"]), "avg": float(r["avg"]),
                    "hr": int(r["hr"]), "rbi": int(r["rbi"]), "r": int(r["r"]), "sb": int(r["sb"]), "pa": int(r["pa"])})
except FileNotFoundError: pass
mlb = {"params": {"avg": float(p["avg_runs"]), "home": float(p["home_adv"]), "lg_ra9": float(p["lg_ra9"])},
       "teams": {r["team"]: {"att": float(r["att"]), "dfn": float(r["dfn"])} for r in rows},
       "pitchers": pit, "hitters": hit, "generated": GEN}
write("mlb_data.js", "MLB_DATA", mlb); print(f"MLB: {len(mlb['teams'])} teams, {len(pit)} starters, {len(hit)} hitters")
print("Wrote web/nfl_data.js, nba_data.js, mlb_data.js")
