# -*- coding: utf-8 -*-
"""UFC rankings (INFO display) — scrape CURRENT official rankings from UFC.com (server-rendered,
so the GitHub Action's plain Python can refresh it). ESPN's rankings API is stale (~2021).
Outputs ufc_rankings.csv + web/ufc_rankings.js (window.UFC_RANKINGS)."""
import urllib.request, re, csv, os, sys, html, json, datetime
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass
PROJ = os.path.dirname(os.path.abspath(__file__))
def get(u, t=30):
    req = urllib.request.Request(u, headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"})
    return urllib.request.urlopen(req, timeout=t).read().decode("utf-8", "replace")
h = get("https://www.ufc.com/rankings")
segs = re.split(r'<div class="view-grouping">', h)
divisions = {}
for seg in segs[1:]:
    hm = re.search(r'<div class="view-grouping-header">\s*([^<]+?)\s*(?:<span|</div)', seg)
    if not hm: continue
    name = html.unescape(hm.group(1).strip())
    names = [html.unescape(n.strip()) for n in re.findall(r'href="/athlete/[^"]+"[^>]*>\s*([^<]+?)\s*</a>', seg)]
    seen=set(); ordered=[]
    for n in names:
        if n and n not in seen: seen.add(n); ordered.append(n)
    if not ordered: continue
    if name not in divisions or len(ordered) > len(divisions[name]): divisions[name] = ordered
order_pref = ["Men's Pound-for-Pound","Heavyweight","Light Heavyweight","Middleweight","Welterweight",
              "Lightweight","Featherweight","Bantamweight","Flyweight","Women's Pound-for-Pound",
              "Women's Bantamweight","Women's Flyweight","Women's Strawweight"]
ordered_divs = [d for d in order_pref if d in divisions] + [d for d in divisions if d not in order_pref]
rows=[]; web={}
for name in ordered_divs:
    fs=divisions[name]; web[name]={"champion":fs[0],"contenders":fs[1:16]}
    rows.append([name,"C",fs[0]])
    for i,f in enumerate(fs[1:16],1): rows.append([name,i,f])
with open(os.path.join(PROJ,"ufc_rankings.csv"),"w",newline="",encoding="utf-8") as f:
    w=csv.writer(f); w.writerow(["division","rank","fighter"]); w.writerows(rows)
with open(os.path.join(PROJ,r"web\ufc_rankings.js"),"w",encoding="utf-8") as f:
    f.write("window.UFC_RANKINGS = "+json.dumps({"divisions":ordered_divs,"rankings":web,
            "generated":datetime.date.today().isoformat(),"source":"UFC.com"},ensure_ascii=False)+";\n")
print(f"{len(divisions)} divisions -> ufc_rankings.csv + web/ufc_rankings.js")
