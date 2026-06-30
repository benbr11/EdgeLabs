# -*- coding: utf-8 -*-
"""
Monte Carlo World Cup match simulator (Poisson model) with a match-day modifier
layer for situational factors.

BASE matchup (always on, from ratings.csv):
    lambda_A = league_avg * attack_mult[A] * defense_mult[B] * home_A
    lambda_B = league_avg * attack_mult[B] * defense_mult[A] * home_B

MATCH-DAY MODIFIERS (optional flags; each is a transparent, tunable multiplier on
a team's attacking output `am` and/or defensive solidity `dm`; dm>1 = defends
better, which lowers the OPPONENT's goals):
    availability  injuries / rotation / fitness   --avail-a/-b  (0.5..1.0, def 1.0)
    fatigue       days rest before this match      --rest-a/-b   (days, def 4)
    stakes        must-win raises risk, dead lowers --stakes-a/-b (must-win|normal|dead)
    climate/alt   penalty if venue hotter/higher    --venue-temp / --venue-altitude
                  than a team is acclimatised to    (auto-set to host's if --home)
    weather       global goal suppression           --weather (clear|rain|cold|heat)

Examples:
    python simulate.py "Brazil" "Argentina"
    python simulate.py "Mexico" "Germany" --home Mexico --venue-altitude 2240
    python simulate.py France Senegal --avail-a 0.8 --rest-a 2 --stakes-b must-win --weather heat --venue-temp 36
"""
import csv, math, random, argparse, sys, collections, unicodedata
import groups as G

PROJ = r"C:\Users\bbraudo\Desktop\Claude Output\World Cup Model"

# ---- tunable modifier magnitudes (all documented; edit freely) --------------
AVAIL_FLOOR   = 0.60   # avail=1.0 -> factor 1.0 ; avail=0.5 -> 0.80
FATIGUE_PER_DAY = 0.025  # per day of rest below 4 (rest 0 -> -10%)
# Stakes/motivation multipliers (applied to a team's whole performance, atk AND def).
# clinched 0.93 is EMPIRICAL (analyze_stakes.py: already-qualified teams rest ~7%).
# must-win 1.04 reflects motivation/necessity -- a team that needs the result raises
# its level (the backtest shows this effect is small, so it's a modest, tunable bump).
# Stack with --avail for heavier-than-typical rotation.
STAKES_PERF = {"clinched":0.93, "eliminated":1.00, "must-win":1.04, "normal":1.00}
ALT_PEN_PER_KM = 0.05  # performance loss per 1000 m of UNaccustomed altitude (>500 m buffer)
ALT_BUFFER_M  = 500
HEAT_PEN_PER_C = 0.005 # loss per deg C the venue exceeds (home_temp + buffer)
HEAT_BUFFER_C = 8
WEATHER_GOALS = {"clear":1.0, "rain":0.90, "cold":0.95, "heat":0.93}
RHO = -0.12  # Dixon-Coles low-score correction (tuned via backtest.py); <0 lifts draws & 0-0/1-1
VAR_BASE, VAR_SLOPE = 6.0, 0.34  # variance-by-rating: NegBin dispersion grows with rating (weak teams = more volatile)
# MISMATCH COMPRESSION (tuned via totals_fix.py walk-forward). A multiplicative
# attack x defense model inflates expected goals in lopsided games (a strong team's
# lambda balloons to 3-4 when reality is a 2-0 coast), which over-predicts totals AND
# makes win probabilities overconfident. Raising each multiplier to COMPRESS<1 shrinks
# that spread; C2_RELEVEL (computed at runtime over the field) keeps the average goal
# level unchanged. Backtest: W/D/L log-loss 0.869->0.829 with picks unchanged, and
# totals calibration sharply improved. COMPRESS=1.0 recovers the old behaviour exactly.
COMPRESS = 0.60

ALIASES = {
    "usa":"United States","us":"United States","america":"United States",
    "korea":"South Korea","holland":"Netherlands","czechia":"Czech Republic",
    "turkiye":"Turkey","bosnia":"Bosnia and Herzegovina","drc":"DR Congo",
    "cabo verde":"Cape Verde","nz":"New Zealand",
}

def fold(s):
    return "".join(c for c in unicodedata.normalize("NFKD", s.lower())
                   if not unicodedata.combining(c))

def load_csv(name, key="team"):
    with open(PROJ + "\\" + name, encoding="utf-8") as f:
        return {r[key]: r for r in csv.DictReader(f)}

def resolve(name, teams):
    if name in teams: return name
    low = name.strip().lower()
    if low in ALIASES and ALIASES[low] in teams: return ALIASES[low]
    for t in teams:
        if t.lower() == low: return t
    fl = fold(name)
    for t in teams:
        if fold(t) == fl: return t
    hits = [t for t in teams if fl in fold(t)]
    if len(hits) == 1: return hits[0]
    if len(hits) > 1: sys.exit(f"'{name}' is ambiguous: {hits}")
    sys.exit(f"Team '{name}' not found. Available:\n  " + ", ".join(sorted(teams)))

def nb_pmf(mu, r, mg):
    """Negative-binomial goal distribution: mean mu, variance mu + mu^2/r.
    Large r -> Poisson (consistent); smaller r -> more volatile (weaker teams)."""
    return [math.exp(math.lgamma(k+r) - math.lgamma(r) - math.lgamma(k+1)
                     + r*math.log(r/(r+mu)) + k*math.log(mu/(r+mu))) for k in range(mg+1)]

def dc_matrix(lh, la, rh=50.0, ra=50.0, rho=RHO):
    """Dixon-Coles-corrected joint score distribution with per-team variance
    (rh/ra = dispersion; lower = more volatile)."""
    maxg = max(12, int(lh + la) + 8)
    ph = nb_pmf(lh, rh, maxg); pa = nb_pmf(la, ra, maxg)
    M = [[ph[i]*pa[j] for j in range(maxg+1)] for i in range(maxg+1)]
    M[0][0] *= max(0.0, 1 - lh*la*rho)   # DC correction on the four low scores
    M[0][1] *= max(0.0, 1 + lh*rho)
    M[1][0] *= max(0.0, 1 + la*rho)
    M[1][1] *= max(0.0, 1 - rho)
    s = sum(sum(r) for r in M)
    return [[v/s for v in r] for r in M], maxg

def team_modifiers(T, avail, rest, stakes, venue_temp, venue_alt, ctx):
    """Return (am, dm, notes) for one team."""
    am = dm = 1.0; notes = []
    if avail < 1.0:
        f = AVAIL_FLOOR + (1 - AVAIL_FLOOR) * avail
        am *= f; dm *= f; notes.append(f"availability {avail:.2f} (x{f:.3f})")
    if rest < 4:
        f = max(0.0, 1 - FATIGUE_PER_DAY * (4 - rest))
        am *= f; dm *= f; notes.append(f"fatigue {rest}d rest (x{f:.3f})")
    f = STAKES_PERF.get(stakes, 1.0)
    if f != 1.0:
        am *= f; dm *= f
        notes.append(f"stakes={stakes} (perf x{f:.2f}, empirical)")
    home_temp = float(ctx[T]["home_temp_c"]); home_alt = float(ctx[T]["home_alt_m"])
    if venue_alt is not None and venue_alt > home_alt + ALT_BUFFER_M:
        f = max(0.0, 1 - ALT_PEN_PER_KM * (venue_alt - home_alt - ALT_BUFFER_M) / 1000.0)
        am *= f; dm *= f; notes.append(f"altitude {venue_alt:.0f}m vs home {home_alt:.0f}m (x{f:.3f})")
    if venue_temp is not None and venue_temp > home_temp + HEAT_BUFFER_C:
        f = max(0.0, 1 - HEAT_PEN_PER_C * (venue_temp - home_temp - HEAT_BUFFER_C))
        am *= f; dm *= f; notes.append(f"heat {venue_temp:.0f}C vs home {home_temp:.0f}C (x{f:.3f})")
    return am, dm, notes

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("team_a"); ap.add_argument("team_b")
    ap.add_argument("--home", default=None)
    ap.add_argument("--sims", type=int, default=50000)
    ap.add_argument("--seed", type=int, default=None)
    for x in ("a","b"):
        ap.add_argument(f"--avail-{x}", type=float, default=1.0, dest=f"avail_{x}")
        ap.add_argument(f"--rest-{x}", type=int, default=4, dest=f"rest_{x}")
        ap.add_argument(f"--stakes-{x}", default="normal", dest=f"stakes_{x}",
                        choices=["clinched","eliminated","must-win","normal"])
    ap.add_argument("--venue-temp", type=float, default=None, dest="venue_temp")
    ap.add_argument("--venue-altitude", type=float, default=None, dest="venue_alt")
    ap.add_argument("--weather", default="clear", choices=list(WEATHER_GOALS))
    ap.add_argument("--auto", action="store_true",
                    help="auto-detect host + group stakes from the live standings")
    ap.add_argument("--odds", nargs=3, type=float, default=None, metavar=("A", "DRAW", "B"),
                    help="decimal odds (teamA-win draw teamB-win) to flag market edges")
    ap.add_argument("--knockout", action="store_true",
                    help="knockout tie: add extra time + penalties and report who advances")
    a = ap.parse_args()
    if a.seed is not None: random.seed(a.seed)

    R = load_csv("ratings.csv"); ctx = load_csv("context.csv")
    # re-level constant so COMPRESS leaves the average matchup goal level unchanged:
    # C2_RELEVEL = mean over all team pairings of (attack_mult^P * defense_mult^P).
    # At COMPRESS=1 this equals the build's mean(att*dfn)=1, so lambdas are unchanged.
    _C2s = 0.0; _C2n = 0
    for _ta in R:
        _apa = float(R[_ta]["attack_mult"]) ** COMPRESS
        for _tb in R:
            if _ta == _tb: continue
            _C2s += _apa * (float(R[_tb]["defense_mult"]) ** COMPRESS); _C2n += 1
    C2_RELEVEL = _C2s / _C2n if _C2n else 1.0
    A = resolve(a.team_a, R); B = resolve(a.team_b, R)
    host = resolve(a.home, R) if a.home else None
    stakesA, stakesB = a.stakes_a, a.stakes_b
    knockout = a.knockout
    if a.auto:
        host_d, dsa, dsb, ia, ib = G.auto_context(A, B)
        if host is None and host_d: host = host_d
        stage = G.match_stage(A, B)
        if stage == "knockout":
            knockout = True
            print("[auto] stage: KNOCKOUT (group stage complete) -> extra time + penalties; teams play to strength")
        elif stage == "group":
            if stakesA == "normal": stakesA = dsa
            if stakesB == "normal": stakesB = dsb
            print(f"[auto] stage: GROUP   {A}: {ia}   |   {B}: {ib}")
        else:
            print("[auto] stage: UNKNOWN -- knockout bracket not set yet, so this is not a")
            print("       confirmed fixture. Predicting a one-off 90-min match; add --knockout")
            print("       to force extra time + penalties.")
        if host: print(f"[auto] home advantage: {host}")
        print()
    avg = float(R[A]["league_avg_goals"]); hadv = float(R[A]["home_adv_mult"])

    # venue conditions default to the host's home conditions (visitors may be unacclimatised)
    venue_temp = a.venue_temp if a.venue_temp is not None else (
        float(ctx[host]["home_temp_c"]) if host else None)
    venue_alt = a.venue_alt if a.venue_alt is not None else (
        float(ctx[host]["home_alt_m"]) if host else None)

    amA, dmA, nA = team_modifiers(A, a.avail_a, a.rest_a, stakesA, venue_temp, venue_alt, ctx)
    amB, dmB, nB = team_modifiers(B, a.avail_b, a.rest_b, stakesB, venue_temp, venue_alt, ctx)
    wfac = WEATHER_GOALS[a.weather]

    hfA = hadv if host == A else 1.0
    hfB = hadv if host == B else 1.0
    base_aA = float(R[A]["attack_mult"])**COMPRESS; base_dA = float(R[A]["defense_mult"])**COMPRESS
    base_aB = float(R[B]["attack_mult"])**COMPRESS; base_dB = float(R[B]["defense_mult"])**COMPRESS
    lamA = avg * (base_aA*amA) * (base_dB/dmB) * hfA * wfac / C2_RELEVEL
    lamB = avg * (base_aB*amB) * (base_dA/dmA) * hfB * wfac / C2_RELEVEL

    dA = VAR_BASE + VAR_SLOPE*((float(R[A]["attack_100"])+float(R[A]["defense_100"]))/2)
    dB = VAR_BASE + VAR_SLOPE*((float(R[B]["attack_100"])+float(R[B]["defense_100"]))/2)
    M, maxg = dc_matrix(lamA, lamB, dA, dB)
    rng = range(maxg+1)
    pA = 100*sum(M[i][j] for i in rng for j in rng if i > j)
    pB = 100*sum(M[i][j] for i in rng for j in rng if j > i)
    pD = 100*sum(M[i][i] for i in rng)
    exA = sum(i*sum(M[i]) for i in rng)
    exB = sum(j*sum(M[i][j] for i in rng) for j in rng)
    flat = sorted(((M[i][j], (i,j)) for i in rng for j in rng), reverse=True)
    top = [((i,j), p) for p,(i,j) in flat[:3]]; likely = top[0][0]

    if knockout:
        fA, fD, fB = pA/100.0, pD/100.0, pB/100.0
        Met, mg2 = dc_matrix(lamA/3.0, lamB/3.0, dA, dB); r2 = range(mg2+1)   # extra time = 1/3 of 90'
        petA = sum(Met[i][j] for i in r2 for j in r2 if i > j)
        petB = sum(Met[i][j] for i in r2 for j in r2 if j > i)
        petD = sum(Met[i][i] for i in r2)
        share = fA/(fA+fB) if (fA+fB) > 0 else 0.5
        psA = min(0.55, max(0.45, 0.5 + (share-0.5)*0.2))   # shootout ~ coin flip, tiny favourite edge
        advA = 100*(fA + fD*(petA + petD*psA))
        advB = 100*(fB + fD*(petB + petD*(1-psA)))
        p_et = pD; p_pen = pD*petD
    winner = A if pA > pB else B
    venue = f"(host: {host}" if host else "(neutral"
    cond = []
    if venue_alt: cond.append(f"{venue_alt:.0f}m")
    if venue_temp is not None: cond.append(f"{venue_temp:.0f}C")
    if a.weather != "clear": cond.append(a.weather)
    venue += (", " + ", ".join(cond) if cond else "") + ")"

    print("=" * 64)
    print(f"  {A}  vs  {B}   {venue}{'   [KNOCKOUT]' if knockout else ''}")
    print(f"  exact Dixon-Coles{' + extra time + penalties' if knockout else ' probabilities'} (rho={RHO})")
    print("=" * 64)
    print(f"  Base ratings   {A}: ATK {R[A]['attack_100']} / DEF {R[A]['defense_100']}"
          f"  (pop {ctx[A]['population_m']}M)")
    print(f"                 {B}: ATK {R[B]['attack_100']} / DEF {R[B]['defense_100']}"
          f"  (pop {ctx[B]['population_m']}M)")
    if nA or nB or a.weather != "clear":
        print("-" * 64); print("  Match-day adjustments:")
        print(f"    {A}: " + ("; ".join(nA) if nA else "none"))
        print(f"    {B}: " + ("; ".join(nB) if nB else "none"))
        if a.weather != "clear":
            print(f"    weather={a.weather}: total goals x{wfac}")
    print("-" * 64)
    print(f"  Expected goals:  {A} {exA:.2f}   |   {B} {exB:.2f}")
    print(f"  Most likely scoreline:    {A} {likely[0]} - {likely[1]} {B}  ({top[0][1]*100:.1f}%)")
    print("-" * 64)
    if knockout:
        print(f"  TO ADVANCE     {A}:  {advA:5.1f}%")
        print(f"  TO ADVANCE     {B}:  {advB:5.1f}%")
        print(f"  (90 min: {A} {pA:.0f}% / draw {pD:.0f}% / {B} {pB:.0f}%"
              f"   ->   extra time {p_et:.0f}%, penalties {p_pen:.1f}%)")
        print(f"  ---> Predicted to advance: {A if advA>=advB else B}  ({max(advA,advB):.1f}%)")
    else:
        print(f"  Win probability   {A}:  {pA:5.1f}%")
        print(f"  Win probability   {B}:  {pB:5.1f}%")
        print(f"  Draw probability     :  {pD:5.1f}%")
        print(f"  ---> Predicted winner: {winner}  ({max(pA,pB):.1f}% in 90 min)")
        if pD >= max(pA, pB):
            print("       (a draw is the single most likely 90-min result)")
    print("-" * 64); print("  Top 3 most likely scorelines:")
    for (ga,gb), c in top:
        print(f"     {A} {ga} - {gb} {B}   {c*100:5.1f}%")
    if a.odds:
        oA, oD, oB = a.odds; raw = [1/oA, 1/oD, 1/oB]; ov = sum(raw)
        imp = [r/ov*100 for r in raw]; mp = [pA, pD, pB]; od = [oA, oD, oB]; lbl = [A, "Draw", B]
        print("-" * 64)
        print(f"  EDGE vs market  (book overround {(ov-1)*100:.1f}%; model is independent of odds)")
        print(f"    {'outcome':<16}{'model':>7}{'mkt':>7}{'odds':>7}{'EV':>8}{'1/4-Kelly':>10}")
        for i in range(3):
            p = mp[i]/100; b = od[i]-1; ev = p*od[i]-1
            kel = max(0.0, (b*p-(1-p))/b)/4 if b > 0 else 0.0
            tag = "  VALUE" if ev > 0 else ""
            print(f"    {lbl[i]:<16}{mp[i]:>6.1f}%{imp[i]:>6.1f}%{od[i]:>7.2f}{ev*100:>+7.1f}%{kel*100:>9.1f}%{tag}")
    print("=" * 64)

if __name__ == "__main__":
    main()
