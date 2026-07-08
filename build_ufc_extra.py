#!/usr/bin/env python3
"""Build feature DB rows for UFC fighters who are NOT ranked (so the rankings-driven
per-division builders skip them) but DO have enough recent UFC history to model.

Why: the fighter DB (ufc_fighters.csv) is built only from ranked fighters + their
opponents, so established-but-unranked fighters (Terrance McKinney, Adrian Yanez, ...)
show up as "insufficient data" on cards. This adds them using the SAME feature
computation as build_ufc_lw.py (copied verbatim below), so they land on the same scale.

ISOLATION: extras are written to NEW per-division files (fighter_db_lwx.csv, etc.) with
their own division_code ("lwx", ...). The merge/shrinkage in ufc_model.py groups by
division_code, so the existing ranked divisions ("lw", ...) are untouched -> ranked
fighters' ratings are byte-identical. DIVISION_NAMES maps the x-codes back to the real
division for display (added in ufc_model.py).

True debutants (0 fights) are inherently excluded -- no data exists to model them.

Usage:  python build_ufc_extra.py    (then rebuild: import ufc_model; ufc_model.load(write=True))
"""
import csv, re, sys, os
from datetime import datetime
from collections import defaultdict

try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass

BASE = os.path.dirname(os.path.abspath(__file__))
def P(name): return os.path.join(BASE, name)

MIN_FIGHTS   = 3       # need at least this many completed UFC bouts to model
RECENT_YEARS = 6       # and a fight within this many years (active-ish)

# ---- weightclass string -> division code (order matters; specific first) ----
def wc_to_code(wc):
    w = (wc or "").lower()
    if "women" in w:
        if "strawweight" in w: return "wsw"
        if "flyweight" in w:   return "wflw"
        if "bantamweight" in w or "featherweight" in w: return "wbw"
        return None
    if "light heavyweight" in w: return "lhw"
    if "strawweight" in w:  return "wsw"
    if "flyweight" in w:    return "flw"
    if "bantamweight" in w: return "bw"
    if "featherweight" in w:return "fw"
    if "welterweight" in w: return "ww"
    if "middleweight" in w: return "mw"
    if "lightweight" in w:  return "lw"
    if "heavyweight" in w:  return "hw"
    return None            # catch weight / open weight -> ambiguous, skip

# =========================================================================== #
#  Parsing helpers + feature computation -- copied verbatim from build_ufc_lw.py
# =========================================================================== #
def norm_event(s): return re.sub(r"\s+", " ", s or "").strip()
def to_int(x):
    try: return int(x)
    except: return 0
def parse_x_of_y(s):
    if not s: return (0, 0)
    m = re.match(r"(\d+)\s+of\s+(\d+)", s.strip())
    return (int(m.group(1)), int(m.group(2))) if m else (0, 0)
def parse_time_to_sec(s):
    if not s or s.strip() in ("---", "--", ""): return 0
    m = re.match(r"(\d+):(\d+)", s.strip())
    return int(m.group(1))*60 + int(m.group(2)) if m else 0
def parse_date(s):
    s = (s or "").strip()
    for fmt in ("%B %d, %Y", "%b %d, %Y", "%b. %d, %Y"):
        try: return datetime.strptime(s, fmt)
        except: pass
    return None
def parse_height(s):
    m = re.match(r"(\d+)'\s*(\d+)", (s or "").strip())
    return int(m.group(1))*12 + int(m.group(2)) if m else None
def parse_reach(s):
    try: return float((s or "").strip().replace('"', ""))
    except: return None
def split_bout(bout):
    parts = re.split(r"\s+vs\.?\s+", bout.strip(), maxsplit=1)
    return (parts[0].strip(), parts[1].strip()) if len(parts) == 2 else (None, None)

# ---- load raw sources ----
event_date = {}
for r in csv.DictReader(open(P("raw_events.csv"), encoding="utf-8")):
    event_date[norm_event(r["EVENT"])] = parse_date(r["DATE"])

results = {}
for r in csv.DictReader(open(P("raw_fight_results.csv"), encoding="utf-8")):
    nev = norm_event(r["EVENT"]); A, B = split_bout(r["BOUT"])
    if not A: continue
    results[(nev, A, B)] = {"outcome": r["OUTCOME"].strip(), "method": r["METHOD"].strip(),
        "round": to_int(r["ROUND"]), "time": r["TIME"].strip(), "wc": r["WEIGHTCLASS"].strip(),
        "details": r["DETAILS"].strip(), "tf": r["TIME FORMAT"].strip()}

rb = defaultdict(list)
for r in csv.DictReader(open(P("raw_fight_stats.csv"), encoding="utf-8")):
    nev = norm_event(r["EVENT"]); bout = r["BOUT"].strip()
    sl, sa = parse_x_of_y(r["SIG.STR."]); tdl, tda = parse_x_of_y(r["TD"])
    rb[(nev, bout, r["FIGHTER"].strip())].append({"round": to_int(r["ROUND"].replace("Round","").strip()),
        "kd": int(float(r["KD"] or 0)), "sig_l": sl, "sig_a": sa, "td_l": tdl, "td_a": tda,
        "subatt": int(float(r["SUB.ATT"] or 0)), "ctrl": parse_time_to_sec(r["CTRL"])})

tott = {}
for r in csv.DictReader(open(P("raw_fighter_tott.csv"), encoding="utf-8")):
    tott[r["FIGHTER"].strip()] = {"height_in": parse_height(r["HEIGHT"]), "reach_in": parse_reach(r["REACH"]),
        "stance": (r["STANCE"] or "").strip(), "dob": parse_date(r["DOB"])}

hometown = {}
try:
    for r in csv.DictReader(open(P("raw_hometowns.csv"), encoding="utf-8")):
        hometown[r["fighter"].strip()] = (r["hometown"].strip(), r["country"].strip())
except FileNotFoundError: pass

sc_dom = {}
try:
    for r in csv.DictReader(open(P("raw_scorecards.csv"), encoding="utf-8")):
        sc_dom[frozenset([r["fighter_a"].strip().lower(), r["fighter_b"].strip().lower()])] = \
            (r["dominance_flag"].strip() == "True", r.get("margin_pts","").strip())
except FileNotFoundError: pass

# ---- index bouts by fighter ----
fighter_bouts = defaultdict(list)
for (nev, A, B), rec in results.items():
    d = event_date.get(nev)
    fighter_bouts[A].append((d, nev, f"{A} vs. {B}", B, True, rec))
    fighter_bouts[B].append((d, nev, f"{A} vs. {B}", A, False, rec))

latest_event = max([d for d in event_date.values() if d], default=datetime(2026,6,25))

# ---- who to build: established, active-ish, NOT already in ufc_fighters.csv ----
existing = set()
try:
    for r in csv.DictReader(open(P("ufc_fighters.csv"), encoding="utf-8")):
        existing.add(r["fighter"].strip().lower())
except FileNotFoundError: pass

def completed_count_and_lastdate_and_wc(name):
    n = 0; last = None; last_wc = None
    for (d, nev, bout, opp, isA, rec) in fighter_bouts.get(name, []):
        o = rec["outcome"]
        if o in ("W/L", "L/W"):           # decisive/valid completed bout
            n += 1
            if d and (last is None or d > last): last = d; last_wc = rec["wc"]
    return n, last, last_wc

targets = {}   # raw_name -> division_code
for name in fighter_bouts:
    if name.lower() in existing: continue
    n, last, last_wc = completed_count_and_lastdate_and_wc(name)
    if n < MIN_FIGHTS or last is None: continue
    if (latest_event - last).days > RECENT_YEARS * 366: continue
    code = wc_to_code(last_wc)
    if code is None: continue
    targets[name] = code

# build profiles for targets + their opponents (so rec_vs_<style> can be computed)
build_set = set(targets)
for name in list(targets):
    for (d, nev, bout, opp, isA, rec) in fighter_bouts.get(name, []):
        build_set.add(opp)

# =========================================================================== #
#  First pass: per-fighter profile  (verbatim logic from build_ufc_lw.py)
# =========================================================================== #
def did_win(rec, isA):
    o = rec["outcome"]
    if o == "W/L": return True if isA else False
    if o == "L/W": return False if isA else True
    return None
def fight_minutes(rec):
    secs = (rec["round"] - 1) * 5 * 60 + parse_time_to_sec(rec["time"])
    return secs / 60.0 if secs > 0 else (rec["round"] * 5.0)
def classify_style(slpm, sapm, td15, ctrl_pr, subatt15, sub_wins, ko_wins, n):
    if n == 0: return "unknown"
    grappling_score = td15 + subatt15 * 1.5 + sub_wins * 0.5 + ctrl_pr / 60.0
    if subatt15 >= 1.0 or (sub_wins >= 2 and sub_wins / max(n,1) >= 0.25): return "grappler"
    if td15 >= 2.0 or ctrl_pr >= 120: return "wrestler"
    if slpm >= 4.0 and td15 < 1.0: return "striker"
    if slpm >= 3.0 and grappling_score < 2.0: return "striker"
    return "balanced"

profile = {}
for name in build_set:
    bouts_sorted = sorted(fighter_bouts.get(name, []), key=lambda x: (x[0] is None, x[0] or datetime.min))
    tot_min = 0.0; sig_l = sig_a = sig_absorbed = td_l = td_a = td_against = td_against_att = 0
    kd_for = subatt = ctrl_sec = n_rounds = 0
    r1_sig = r1_rounds = late_sig = late_rounds = 0
    ko_wins = decision_wins = sub_wins = times_kod = times_subbed = 0
    wins = losses = draws_nc = finishes = decisions = got_finished = went_distance = 0
    log = []
    for (d, nev, bout, opp, isA, rec) in bouts_sorted:
        my = rb.get((nev, bout, name), []); opp_rounds = rb.get((nev, bout, opp), [])
        tot_min += fight_minutes(rec)
        for rd in my:
            n_rounds += 1; sig_l += rd["sig_l"]; sig_a += rd["sig_a"]; td_l += rd["td_l"]; td_a += rd["td_a"]
            kd_for += rd["kd"]; subatt += rd["subatt"]; ctrl_sec += rd["ctrl"]
            if rd["round"] == 1: r1_sig += rd["sig_l"]; r1_rounds += 1
            if rd["round"] >= 3: late_sig += rd["sig_l"]; late_rounds += 1
        for rd in opp_rounds:
            sig_absorbed += rd["sig_l"]; td_against += rd["td_l"]; td_against_att += rd["td_a"]
        won = did_win(rec, isA); ml = rec["method"].lower()
        dec_type = rec["method"].split("-")[-1].strip() if "decision" in ml else ""
        dom = sc_dom.get(frozenset([name.lower(), opp.lower()]), (None, None))[0]
        result_str = "W" if won else ("L" if won is False else "D/NC")
        if won is True:
            wins += 1
            if "ko/tko" in ml or "tko" in ml: ko_wins += 1; finishes += 1
            elif "submission" in ml: sub_wins += 1; finishes += 1
            elif "decision" in ml: decision_wins += 1; decisions += 1; went_distance += 1
            else: decisions += 1
        elif won is False:
            losses += 1
            if "ko/tko" in ml or "tko" in ml: times_kod += 1; got_finished += 1
            elif "submission" in ml: times_subbed += 1; got_finished += 1
            elif "decision" in ml: went_distance += 1
        else:
            draws_nc += 1
            if "decision" in ml: went_distance += 1
        log.append({"date": d.strftime("%Y-%m-%d") if d else "", "event": nev, "opponent_raw": opp,
            "result": result_str, "method": rec["method"], "round": rec["round"],
            "decision_type": dec_type, "dominance": "dominant" if dom else ("" if dom is None else "competitive")})
    n_fights = wins + losses + draws_nc
    mins = tot_min if tot_min > 0 else 1e-9; per15 = 15.0 / mins
    opp_sig_att = sum(rd["sig_a"] for (d,nev,bout,opp,isA,rec) in bouts_sorted for rd in rb.get((nev,bout,opp),[]))
    profile[name] = {
        "n_fights": n_fights, "wins": wins, "losses": losses, "draws_nc": draws_nc,
        "SLpM": sig_l/mins, "SApM": sig_absorbed/mins, "str_acc": (sig_l/sig_a) if sig_a else 0.0,
        "str_def": (1 - sig_absorbed/opp_sig_att) if opp_sig_att else 0.0,
        "kd_per15": kd_for*per15, "dmg_per_round": (sig_l/n_rounds) if n_rounds else 0.0,
        "ko_tko_wins": ko_wins, "times_kod": times_kod,
        "td_per15": td_l*per15, "td_acc": (td_l/td_a) if td_a else 0.0,
        "td_def": (1 - td_against/td_against_att) if td_against_att else 0.0,
        "td_att_per_round": (td_a/n_rounds) if n_rounds else 0.0,
        "total_td": td_l, "times_taken_down": td_against,
        "subatt_per15": subatt*per15, "sub_wins": sub_wins, "times_submitted": times_subbed,
        "ctrl_per_round": (ctrl_sec/n_rounds) if n_rounds else 0.0,
        "finish_rate": finishes/n_fights if n_fights else 0.0,
        "decision_rate": decisions/n_fights if n_fights else 0.0,
        "got_finished_rate": got_finished/n_fights if n_fights else 0.0,
        "distance_rate": went_distance/n_fights if n_fights else 0.0,
        "cardio_fade": ((late_sig/late_rounds)/(r1_sig/r1_rounds)) if (r1_rounds and late_rounds and r1_sig) else None,
        "_log": log, "_bouts": bouts_sorted,
    }
    profile[name]["style"] = classify_style(profile[name]["SLpM"], profile[name]["SApM"],
        profile[name]["td_per15"], profile[name]["ctrl_per_round"], profile[name]["subatt_per15"],
        sub_wins, ko_wins, n_fights)

def style_of(nm):
    p = profile.get(nm); return p["style"] if p else "unknown"

# =========================================================================== #
#  Second pass: emit db + log rows for targets, grouped by division code
# =========================================================================== #
db_by_code = defaultdict(list); log_by_code = defaultdict(list)
for name, code in targets.items():
    p = profile[name]; bouts_sorted = p["_bouts"]
    vs = defaultdict(lambda: [0,0,0])
    for (d, nev, bout, opp, isA, rec) in bouts_sorted:
        won = did_win(rec, isA); ost = style_of(opp)
        if won is True: vs[ost][0]+=1
        elif won is False: vs[ost][1]+=1
        else: vs[ost][2]+=1
    def vsrec(s): w,l,dn = vs.get(s,[0,0,0]); return f"{w}-{l}-{dn}"
    t = tott.get(name, {}); dob = t.get("dob")
    age = round((latest_event - dob).days/365.25, 1) if dob else None
    dated = [b[0] for b in bouts_sorted if b[0]]
    last_date = dated[-1] if dated else None
    layoff_days = (latest_event - last_date).days if last_date else None
    streak = 0; streak_type = ""
    for (d, nev, bout, opp, isA, rec) in reversed(bouts_sorted):
        won = did_win(rec, isA)
        if won is None: break
        cur = "W" if won else "L"
        if streak == 0: streak_type = cur; streak = 1
        elif cur == streak_type: streak += 1
        else: break
    streak_str = f"{streak_type}{streak}" if streak else "0"
    last3 = [b for b in bouts_sorted if b[0]][-3:]
    weights = [1,2,3][-len(last3):]; num = den = 0.0; rf_results = []
    for w,(d,nev,bout,opp,isA,rec) in zip(weights, last3):
        won = did_win(rec, isA); val = 1.0 if won else (0.0 if won is False else 0.5)
        num += w*val; den += w; rf_results.append("W" if won else ("L" if won is False else "D"))
    recent_form = round(num/den,3) if den else None
    home = hometown.get(name, ("",""))
    db_by_code[code].append({
        "rank": "", "fighter": name, "n_fights": p["n_fights"],
        "record": f"{p['wins']}-{p['losses']}-{p['draws_nc']}",
        "SLpM": round(p["SLpM"],3), "SApM": round(p["SApM"],3), "str_accuracy": round(p["str_acc"],3),
        "str_defense": round(p["str_def"],3), "kd_per15": round(p["kd_per15"],3),
        "dmg_per_round": round(p["dmg_per_round"],2), "ko_tko_wins": p["ko_tko_wins"], "times_kod": p["times_kod"],
        "td_per15": round(p["td_per15"],3), "td_accuracy": round(p["td_acc"],3), "td_defense": round(p["td_def"],3),
        "td_att_per_round": round(p["td_att_per_round"],3), "total_takedowns": p["total_td"],
        "times_taken_down": p["times_taken_down"], "subatt_per15": round(p["subatt_per15"],3),
        "sub_wins": p["sub_wins"], "times_submitted": p["times_submitted"],
        "ctrl_sec_per_round": round(p["ctrl_per_round"],1), "style": p["style"],
        "rec_vs_striker": vsrec("striker"), "rec_vs_wrestler": vsrec("wrestler"),
        "rec_vs_grappler": vsrec("grappler"), "rec_vs_balanced": vsrec("balanced"),
        "finish_rate": round(p["finish_rate"],3), "decision_rate": round(p["decision_rate"],3),
        "got_finished_rate": round(p["got_finished_rate"],3), "distance_rate": round(p["distance_rate"],3),
        "octagon_control_proxy": round(p["ctrl_per_round"]/300.0,3),
        "cardio_fade": round(p["cardio_fade"],3) if p["cardio_fade"] is not None else "",
        "age": age if age is not None else "", "dob": dob.strftime("%Y-%m-%d") if dob else "",
        "height_in": t.get("height_in") if t.get("height_in") is not None else "",
        "reach_in": t.get("reach_in") if t.get("reach_in") is not None else "",
        "stance": t.get("stance",""), "hometown": home[0], "country": home[1],
        "recent_form_w": recent_form if recent_form is not None else "",
        "recent_form_results": "/".join(rf_results),
        "layoff_days": layoff_days if layoff_days is not None else "",
        "last_fight_date": last_date.strftime("%Y-%m-%d") if last_date else "", "streak": streak_str,
    })
    for L in p["_log"]:
        log_by_code[code].append({"fighter": name, "date": L["date"], "event": L["event"],
            "opponent": L["opponent_raw"], "result": L["result"], "method": L["method"],
            "round": L["round"], "decision_type": L["decision_type"], "dominance": L["dominance"]})

# ---- write per-division extra files with new codes (<code>x) ----
log_cols = ["fighter","date","event","opponent","result","method","round","decision_type","dominance"]
total = 0
for code, rows in sorted(db_by_code.items()):
    rows.sort(key=lambda r: -r["n_fights"])
    dbp = P(f"fighter_db_{code}x.csv")
    with open(dbp, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    lg = sorted(log_by_code[code], key=lambda r: (r["fighter"], r["date"]))
    with open(P(f"fighter_log_{code}x.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=log_cols); w.writeheader(); w.writerows(lg)
    total += len(rows)
    print(f"  fighter_db_{code}x.csv: +{len(rows)} unranked fighters")
print(f"\nWrote {total} established-unranked fighters across {len(db_by_code)} divisions "
      f"(MIN_FIGHTS={MIN_FIGHTS}, within {RECENT_YEARS}y).")
