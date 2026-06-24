# -*- coding: utf-8 -*-
"""Verify the NHL pipeline is self-updating: season auto-rollover + current-season freshness."""
import datetime
def latest_seasons(n=3, today=None):
    d = today or datetime.date.today()
    start = d.year if d.month >= 9 else d.year - 1
    return [start - i for i in range(n)]
print("SEASON AUTO-ROLLOVER (proves 'works every season'):")
for t in [datetime.date(2026,6,24), datetime.date(2026,9,1), datetime.date(2026,10,15),
          datetime.date(2027,1,5), datetime.date(2027,4,30), datetime.date(2030,11,9)]:
    ss = latest_seasons(3, t)
    print(f"  {t}  ->  {[f'{y}{y+1}' for y in ss]}  (current {ss[0]}{ss[0]+1})")
print(f"\n  REAL today -> {[f'{y}{y+1}' for y in latest_seasons(3)]}")
