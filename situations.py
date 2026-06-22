# -*- coding: utf-8 -*-
"""
Print the live 2026 World Cup group landscape: standings so far, and for every
remaining match each team's detected situation. Re-run after a data refresh and
it recomputes. Matches flagged CLINCHED get the empirical resting adjustment when
simulated with `--auto`.

Usage:  python situations.py
"""
import groups as G

grps, played, sched = G.get_groups()
print(f"2026 World Cup — {len(grps)} groups | {len(played)} played, {len(sched)} scheduled\n")

for i, grp in enumerate(sorted(grps), 1):
    pts, gf, ga, pl, rem = G.group_state(grp, played, sched)
    order = sorted(grp, key=lambda t: (-pts[t], -(gf[t]-ga[t]), -gf[t]))
    print(f"Group {i}")
    print(f"  {'team':<24}{'P':>3}{'Pts':>5}{'GF':>4}{'GA':>4}{'GD':>5}")
    for t in order:
        print(f"  {t:<24}{pl[t]:>3}{pts[t]:>5}{gf[t]:>4}{ga[t]:>4}{gf[t]-ga[t]:>+5}")
    if rem:
        print("  remaining:")
        for h, a in rem:
            lh, sh = G.situation(h, grp, pts, rem)
            la, sa = G.situation(a, grp, pts, rem)
            flag = "  *RESTING*" if "clinched" in (sh, sa) else ""
            print(f"    {h} vs {a}{flag}")
            print(f"        {h:<22} {lh}")
            print(f"        {a:<22} {la}")
    else:
        print("  group complete")
    print()
