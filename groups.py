# -*- coding: utf-8 -*-
"""
Auto-detect each team's 2026 World Cup group situation from the LIVE standings,
so stakes don't have to be set by hand. Reconstructs the 12 groups, computes
standings from played matches, and for any team determines whether it is already
guaranteed a top-2 finish (CLINCHED -> rests starters -> 0.93x, per analyze_stakes.py).

Per the empirical study, CLINCHED is the only situation that changes a prediction;
ELIMINATED / MUST-WIN / LIVE all play to full strength. Those are still labelled
for clarity. Re-run after `build_ratings.py --refresh` and situations update.
"""
import csv, itertools, collections

import os
PROJ = os.path.dirname(os.path.abspath(__file__))
HOSTS = {"United States", "Canada", "Mexico"}
PT = lambda gf, ga: 3 if gf > ga else (1 if gf == ga else 0)

def _load_2026():
    played, sched = [], []
    with open(PROJ + r"\results.csv", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["tournament"] != "FIFA World Cup" or not r["date"].startswith("2026"):
                continue
            try:
                played.append((r["date"], r["home_team"], r["away_team"],
                               int(r["home_score"]), int(r["away_score"])))
            except ValueError:
                sched.append((r["date"], r["home_team"], r["away_team"]))
    return played, sched

def get_groups():
    played, sched = _load_2026()
    allm = [(d, h, a) for (d, h, a, hs, a_) in played] + sched
    games = collections.defaultdict(list)
    for d, h, a in sorted(allm):
        games[h].append((d, h, a)); games[a].append((d, h, a))
    opp = {}
    for t, gs in games.items():
        opp[t] = set()
        for d, h, a in gs[:3]:
            opp[t].add(a if h == t else h)
    teams = list(opp)
    adj = {t: {u for u in opp[t] if t in opp.get(u, set())} for t in teams}
    groups, used = [], set()
    for t in teams:
        if t in used: continue
        nb = [u for u in adj[t] if u not in used]
        for combo in itertools.combinations(nb, 3):
            four = (t,) + combo
            if all(b in adj[a] for a, b in itertools.combinations(four, 2)):
                groups.append(sorted(four)); used.update(four); break
    return groups, played, sched

def group_state(grp, played, sched):
    pts = {t: 0 for t in grp}; gf = {t: 0 for t in grp}
    ga = {t: 0 for t in grp}; pl = {t: 0 for t in grp}
    for d, h, a, hs, a_ in played:
        if h in grp and a in grp:
            pts[h] += PT(hs, a_); pts[a] += PT(a_, hs)
            gf[h] += hs; ga[h] += a_; gf[a] += a_; ga[a] += hs
            pl[h] += 1; pl[a] += 1
    remaining = [(h, a) for d, h, a in sched if h in grp and a in grp]
    return pts, gf, ga, pl, remaining

def _enumerate(pts, remaining):
    """Yield every possible final points dict over remaining W/D/L outcomes."""
    for outs in itertools.product((0, 1, 2), repeat=len(remaining)):
        p = dict(pts)
        for (h, a), o in zip(remaining, outs):
            if o == 0: p[h] += 3
            elif o == 1: p[h] += 1; p[a] += 1
            else: p[a] += 3
        yield p

def situation(team, grp, pts, remaining):
    """Return (label, stakes_flag). stakes_flag is what simulate.py consumes."""
    if not remaining:                              # group already finished
        rank = sorted(grp, key=lambda t: -pts[t])
        return ("FINISHED", "normal")
    top2_all = True; top2_any = False
    for p in _enumerate(pts, remaining):
        ge = sum(1 for o in grp if o != team and p[o] >= p[team])  # rivals >= team
        gt = sum(1 for o in grp if o != team and p[o] > p[team])   # rivals strictly >
        if ge > 1: top2_all = False                # not guaranteed top-2 (conservative)
        if gt <= 1: top2_any = True                # top-2 is still reachable
    if top2_all:
        return ("CLINCHED (guaranteed top 2 — likely rotates)", "clinched")
    if not top2_any:
        return ("ELIMINATED from top 2 (best-3rd may remain; plays to strength)", "eliminated")
    return ("LIVE / must-win (plays to strength)", "normal")

def match_stage(teamA, teamB):
    """Classify a pairing: 'group' (same group), 'knockout' (group stage done,
    so cross-group ties are knockouts), or 'unknown' (bracket not set yet)."""
    groups, played, sched = get_groups()
    gA = next((g for g in groups if teamA in g), None)
    if gA is not None and teamB in gA:
        return "group"
    remaining = sum(len(group_state(g, played, sched)[4]) for g in groups)
    return "knockout" if remaining == 0 else "unknown"

def auto_context(teamA, teamB):
    """For simulate.py --auto: returns (host_or_None, stakesA, stakesB, infoA, infoB)."""
    groups, played, sched = get_groups()
    host = teamA if teamA in HOSTS else (teamB if teamB in HOSTS else None)
    grpA = next((g for g in groups if teamA in g and teamB in g), None)
    if grpA is None:                               # not a current group pairing
        return host, "normal", "normal", "no current group (stakes=normal)", \
               "no current group (stakes=normal)"
    pts, gf, ga, pl, rem = group_state(grpA, played, sched)
    labA, sA = situation(teamA, grpA, pts, rem)
    labB, sB = situation(teamB, grpA, pts, rem)
    return host, sA, sB, labA, labB
