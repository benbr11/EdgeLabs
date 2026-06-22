# -*- coding: utf-8 -*-
"""
EMPIRICAL measurement of the "dead rubber" effect.

Reconstructs past group stages (World Cups 1998-2022 + Euros 1996-2012, all the
same 4-team / top-2-advance format), classifies each team going into its FINAL
group match as:
    CLINCHED    already qualified (6 pts after 2 games -> guaranteed top 2)
    ELIMINATED  cannot reach top 2 (0 pts after 2 games & >=2 rivals on >=4)
    LIVE        still has something to play for
then compares how those teams ACTUALLY performed in that final match against
their Elo-implied expected score. The gap is the empirical stakes/resting effect,
which we feed back into simulate.py instead of guessing.
"""
import csv, math, datetime, itertools, collections

PROJ = r"C:\Users\bbraudo\Desktop\Claude Output\World Cup Model"

WC_YEARS  = {1998,2002,2006,2010,2014,2018,2022}
EURO_YEARS = {1996,2000,2004,2008,2012}

rows = []
with open(PROJ + r"\results.csv", encoding="utf-8") as f:
    for r in csv.DictReader(f):
        try:
            d = datetime.date.fromisoformat(r["date"]); hs=int(r["home_score"]); a_=int(r["away_score"])
        except (ValueError, KeyError):
            continue
        rows.append((d, r["home_team"], r["away_team"], hs, a_,
                     r["neutral"].strip().upper()=="TRUE", r["tournament"]))
rows.sort(key=lambda x: x[0])

# ---- Elo over full history; we snapshot pre-match Elo for flagged matches ----
def build_elo_snapshots(flagged):
    elo = {}; snaps = {}
    for d,h,a,hs,a_,neu,trn in rows:
        eh=elo.get(h,1500.0); ea=elo.get(a,1500.0)
        key=(d,h,a)
        if key in flagged: snaps[key]=(eh,ea)
        adj = 0.0 if neu else 65.0
        exp_h = 1.0/(1.0+10**((ea-(eh+adj))/400.0))
        res_h = 1.0 if hs>a_ else (0.5 if hs==a_ else 0.0)
        gd=abs(hs-a_); g = 1.0 if gd<=1 else (1.5 if gd==2 else (11+gd)/8.0)
        delta = 30.0*g*(res_h-exp_h); elo[h]=eh+delta; elo[a]=ea-delta
    return snaps

# ---- reconstruct groups per tournament instance --------------------------------
def instances():
    by = collections.defaultdict(list)
    for m in rows:
        d,h,a,hs,a_,neu,trn = m; y=d.year
        if (trn=="FIFA World Cup" and y in WC_YEARS) or (trn=="UEFA Euro" and y in EURO_YEARS):
            by[(trn,y)].append(m)
    return by

def first3_opponents(matches):
    """Each team's first 3 opponents by date (= its group-stage games)."""
    games = collections.defaultdict(list)
    for m in sorted(matches, key=lambda x:x[0]):
        d,h,a,hs,a_,neu,trn = m
        games[h].append(m); games[a].append(m)
    opp = {}
    for t,gs in games.items():
        opp[t] = set()
        for m in gs[:3]:
            opp[t].add(m[2] if m[1]==t else m[1])
    return opp

def find_groups(matches):
    opp = first3_opponents(matches)
    teams = list(opp)
    adj = {t:{u for u in opp[t] if t in opp.get(u,set())} for t in teams}  # mutual
    groups = []; used=set()
    for t in teams:
        if t in used: continue
        nbrs = [u for u in adj[t] if u not in used]
        for combo in itertools.combinations(nbrs,3):
            four = (t,)+combo
            if all(b in adj[a] for a,b in itertools.combinations(four,2)):  # K4
                groups.append(set(four)); used.update(four); break
    return groups

PT = lambda gf,ga: 3 if gf>ga else (1 if gf==ga else 0)

flagged = {}          # (date,home,away) -> {team: status}
records = []          # (status, date, home, away)
for (trn,y), matches in instances().items():
    for grp in find_groups(matches):
        # the 6 group matches among these 4 teams (first meeting per pair)
        gm=[]; seen=set()
        for m in sorted(matches, key=lambda x:x[0]):
            d,h,a,hs,a_,neu,trn2 = m
            if h in grp and a in grp:
                key=frozenset((h,a))
                if key in seen: continue
                seen.add(key); gm.append(m)
        if len(gm)!=6: continue
        r12, r3 = gm[:4], gm[4:]
        pts = {t:0 for t in grp}; gf={t:0 for t in grp}; ga={t:0 for t in grp}
        for d,h,a,hs,a_,neu,trn2 in r12:
            pts[h]+=PT(hs,a_); pts[a]+=PT(a_,hs)
            gf[h]+=hs; ga[h]+=a_; gf[a]+=a_; ga[a]+=hs
        for d,h,a,hs,a_,neu,trn2 in r3:
            for team in (h,a):
                p=pts[team]
                others=[pts[o] for o in grp if o!=team]
                if p==6: status="CLINCHED"
                elif p==0 and sum(1 for o in others if o>=4)>=2: status="ELIMINATED"
                else: status="LIVE"
                flagged.setdefault((d,h,a),{})[team]=status
                records.append((status,(d,h,a),team,h))

snaps = build_elo_snapshots(set(flagged))

# ---- aggregate actual vs Elo-expected score ------------------------------------
agg = collections.defaultdict(lambda: {"n":0,"act":0.0,"exp":0.0,"gf":0,"ga":0,
                                        "w":0,"d":0,"l":0,"elo_adv":0.0})
res_by_key = {(d,h,a):(hs,a_,neu) for d,h,a,hs,a_,neu,trn in rows}
for status,key,team,home in records:
    if key not in snaps: continue
    d,h,a = key; hs,a_,neu = res_by_key[key]
    eh,ea = snaps[key]
    adj = 0.0 if neu else 65.0
    if team==h:
        exp = 1.0/(1.0+10**((ea-(eh+adj))/400.0)); my,opp=hs,a_; elo_adv=(eh+adj)-ea
    else:
        exp = 1.0/(1.0+10**(((eh+adj)-ea)/400.0)); my,opp=a_,hs; elo_adv=ea-(eh+adj)
    act = 1.0 if my>opp else (0.5 if my==opp else 0.0)
    s=agg[status]; s["n"]+=1; s["act"]+=act; s["exp"]+=exp
    s["gf"]+=my; s["ga"]+=opp; s["elo_adv"]+=elo_adv
    s["w"]+= my>opp; s["d"]+= my==opp; s["l"]+= my<opp

print("Empirical final-group-match performance (actual vs Elo-expected)\n")
print(f"{'status':<11}{'N':>4}{'eloAdv':>8}{'actPPM':>8}{'expPPM':>8}{'ratio':>7}"
      f"{'GF':>6}{'GA':>6}{'W%':>6}{'D%':>6}{'L%':>6}")
mult = {}
for st in ("CLINCHED","ELIMINATED","LIVE"):
    s=agg[st]; n=s["n"]
    if not n: continue
    act=s["act"]/n; exp=s["exp"]/n; ratio=act/exp if exp else 0
    print(f"{st:<11}{n:>4}{s['elo_adv']/n:>8.0f}{act:>8.3f}{exp:>8.3f}{ratio:>7.3f}"
          f"{s['gf']/n:>6.2f}{s['ga']/n:>6.2f}{100*s['w']/n:>6.0f}{100*s['d']/n:>6.0f}{100*s['l']/n:>6.0f}")
    mult[st]=ratio

print("\nInterpretation: 'ratio' = actual score / Elo-expected score "
      "(1.0 = performs to strength; <1 = underperforms).")
if "LIVE" in mult and mult["LIVE"]:
    for st in ("CLINCHED","ELIMINATED"):
        if st in mult:
            rel = mult[st]/mult["LIVE"]
            print(f"  {st}: performs at {rel:.2f}x of a motivated (LIVE) team of equal strength.")
