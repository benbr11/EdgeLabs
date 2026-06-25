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

# ---- NFL ----
rows = list(csv.DictReader(open(PROJ + r"\nfl_ratings.csv", encoding="utf-8")))
p = rows[0]
nfl = {"params": {"lg": float(p["lg_ppg"]), "hfa": float(p["hfa"]), "sd_m": float(p["sd_margin"]), "sd_t": float(p["sd_total"])},
       "teams": {r["team"]: {"name": NFL_NAMES.get(r["team"], r["team"]), "off": float(r["off"]), "dfn": float(r["def"]), "net": float(r["net"])} for r in rows},
       "generated": GEN}
write("nfl_data.js", "NFL_DATA", nfl); print(f"NFL: {len(nfl['teams'])} teams")

# ---- NBA ----
rows = list(csv.DictReader(open(PROJ + r"\nba_ratings.csv", encoding="utf-8")))
p = rows[0]
nba = {"params": {"lg": float(p["lg_ppg"]), "hfa": float(p["hfa"]), "sd_m": float(p["sd_margin"]), "sd_t": float(p["sd_total"])},
       "teams": {r["team"]: {"name": r["name"], "off": float(r["off"]), "dfn": float(r["def"]), "net": float(r["net"])} for r in rows},
       "generated": GEN}
write("nba_data.js", "NBA_DATA", nba); print(f"NBA: {len(nba['teams'])} teams")

# ---- MLB ----
rows = list(csv.DictReader(open(PROJ + r"\mlb_ratings.csv", encoding="utf-8")))
p = rows[0]
prows = list(csv.DictReader(open(PROJ + r"\mlb_pitchers.csv", encoding="utf-8")))
# only keep pitchers with a real workload, grouped by current team for the selector
pit = [{"n": r["pitcher"], "team": r["team"], "ra9": float(r["ra9"]), "factor": float(r["factor"]), "ip": int(r["ip"])}
       for r in prows if int(r["ip"]) >= 40]
mlb = {"params": {"avg": float(p["avg_runs"]), "home": float(p["home_adv"]), "lg_ra9": float(p["lg_ra9"])},
       "teams": {r["team"]: {"att": float(r["att"]), "dfn": float(r["dfn"])} for r in rows},
       "pitchers": pit, "generated": GEN}
write("mlb_data.js", "MLB_DATA", mlb); print(f"MLB: {len(mlb['teams'])} teams, {len(pit)} pitchers")
print("Wrote web/nfl_data.js, nba_data.js, mlb_data.js")
