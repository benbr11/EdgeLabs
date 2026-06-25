# -*- coding: utf-8 -*-
"""Validate the 5 club leagues: data completeness (all seasons, current dates),
team rosters (current vs relegated), and goal-level calibration."""
import csv, io, json, os, urllib.request, datetime
PROJ = os.path.dirname(os.path.abspath(__file__))
LEAGUES = [("epl","Premier League","E0",20),("laliga","La Liga","SP1",20),("seriea","Serie A","I1",20),
           ("bundesliga","Bundesliga","D1",18),("ligue1","Ligue 1","F1",18)]
def get(url,t=30):
    req=urllib.request.Request(url,headers={"User-Agent":"Mozilla/5.0"}); return urllib.request.urlopen(req,timeout=t).read().decode("utf-8","replace")
def es(n=6,today=None):
    d=today or datetime.date.today(); sy=d.year if d.month>=7 else d.year-1
    return [f"{str(y)[2:]}{str(y+1)[2:]}" for y in range(sy,sy-n,-1)]
def pdate(s):
    for f in("%d/%m/%Y","%d/%m/%y"):
        try:return datetime.datetime.strptime(s,f).date()
        except:pass
    return None
SEASONS=es(6); print(f"seasons: {SEASONS}\n")
built=json.load(open(PROJ+r"\web\leagues_data.js",encoding="utf-8").read().split("=",1)[1].rstrip().rstrip(";") and io.StringIO("")) if False else None
raw=open(PROJ+r"\web\leagues_data.js",encoding="utf-8").read(); LD=json.loads(raw[raw.index("{"):raw.rindex("}")+1])["leagues"]

for code,name,fdc,expect in LEAGUES:
    perseason={}; allteams=set(); curteams=set(); tot=0; ng=0; lastd=None
    for si,season in enumerate(SEASONS):
        try: rows=list(csv.DictReader(io.StringIO(get(f"https://www.football-data.co.uk/mmz4281/{season}/{fdc}.csv"))))
        except Exception as e: perseason[season]=f"ERR {e}"; continue
        g=0; dmin=dmax=None; teams=set()
        for r in rows:
            h,a=(r.get("HomeTeam")or"").strip(),(r.get("AwayTeam")or"").strip()
            d=pdate(r.get("Date",""));
            try:hg,ag=int(r["FTHG"]),int(r["FTAG"])
            except:continue
            if h and a and d:
                g+=1; teams|={h,a}; tot+=hg+ag; ng+=1
                if not dmin or d<dmin:dmin=d
                if not dmax or d>dmax:dmax=d
        perseason[season]=f"{g} games, {len(teams)} teams, {dmin}..{dmax}"
        allteams|=teams
        if si==0: curteams=teams; lastd=dmax
    bteams=set(LD[code]["teams"].keys()) if code in LD else set()
    relegated=sorted(allteams-curteams)
    print(f"=== {name} ===")
    for s in SEASONS: print(f"   {s}: {perseason.get(s)}")
    print(f"   current-season teams: {len(curteams)} (expected {expect})  | all-3-season: {len(allteams)} | in built file: {len(bteams)}")
    print(f"   built-file teams NOT in current season (stale/relegated shown in dropdown): {sorted(bteams-curteams)}")
    print(f"   actual avg total goals/match: {tot/ng:.2f}  | model avg total: {2*LD[code]['params']['avg']:.2f}")
    print()
