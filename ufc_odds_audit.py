# -*- coding: utf-8 -*-
"""
WEEKLY UFC self-audit — the automated "compare our odds to the market and catch issues" loop.

Runs the walk-forward backtest (ufc_backtest.py), which prices every recent fight the model
had NOT seen and compares it to the de-vigged CLOSING line, then:
  * logs a dated row to ufc_audit_log.csv (model-vs-market accuracy, best-bet tier, log-loss)
    so the model's edge-vs-market is TRACKED over time as new fights + closing lines arrive;
  * surfaces any SYSTEMATIC fighter-TYPE bias (e.g. a grappler over-fire) that the backtest's
    style breakdown flags — the automated version of the manual Pimblett catch;
  * (optional) if card_odds.csv is present, flags the current card's biggest model-vs-market
    discrepancies for review.

IMPORTANT — this DETECTS and LOGS; it does NOT auto-change the model. Big model-vs-market gaps
are usually the model's EDGE (grappling is validated as market-beating), and auto-tuning toward
the market would ERASE that edge. Flagged items are candidates for a HUMAN-VALIDATED fix (tested
on the backtest before adoption, exactly how the submission-threat fix was added). That is the
honest "keep improving" loop: accumulate market-relative evidence -> principled fixes.

Usage:  python ufc_odds_audit.py     (run weekly by .github/workflows/audit.yml)
"""
import subprocess, sys, os, re, csv, datetime, json
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass
BASE = os.path.dirname(os.path.abspath(__file__))

print("Running walk-forward backtest for the weekly market audit ...", flush=True)
res = subprocess.run([sys.executable, "ufc_backtest.py"], cwd=BASE, capture_output=True, text=True)
out = res.stdout

def _grab(pat, default=""):
    m = re.search(pat, out)
    return m.group(1) if m else default

# tier-2 (FULL MODEL, AS-OF) straight-up hit-rate = the honest OOS accuracy
seg = out.split("(2) FULL MODEL, AS-OF")[-1]
m = re.search(r"straight-up hit-rate\s*:\s*[\d.]+\s*\(([\d.]+)%\)", seg)
straight_up = m.group(1) if m else ""
bb = re.search(r"HITRATE=([\d.]+)\s+N=(\d+)", out)
best_bet = f"{float(bb.group(1))*100:.1f}" if bb else ""
best_n   = bb.group(2) if bb else ""
model_hit = _grab(r"MODEL favorite hit-rate\s*:\s*[\d.]+\s*\(([\d.]+)%\)")
book_hit  = _grab(r"BOOK  favorite hit-rate\s*:\s*[\d.]+\s*\(([\d.]+)%\)")
model_ll  = _grab(r"model log-loss\s*:\s*([\d.]+)")
book_ll   = _grab(r"book  log-loss\s*:\s*([\d.]+)")
bias_lines = [ln.strip() for ln in out.splitlines()
              if re.search(r"(grappler|striker|balanced) favs", ln)]
flags = [ln.strip() for ln in out.splitlines() if "OVER-FIRE" in ln]

today = datetime.date.today().isoformat()
row = {"date": today, "straight_up_pct": straight_up, "best_bet_pct": best_bet, "best_bet_n": best_n,
       "model_favhit_pct": model_hit, "book_favhit_pct": book_hit,
       "model_logloss": model_ll, "book_logloss": book_ll,
       "flags": " | ".join(flags) if flags else "none"}
log_path = os.path.join(BASE, "ufc_audit_log.csv")
new = not os.path.exists(log_path)
with open(log_path, "a", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(row.keys()))
    if new: w.writeheader()
    w.writerow(row)

print("\n" + "=" * 66)
print(f"UFC WEEKLY MARKET AUDIT — {today}")
print("=" * 66)
print(f"  OOS straight-up accuracy : {straight_up}%   (market ~64-66%)")
print(f"  best-bet tier            : {best_bet}% on N={best_n}")
print(f"  vs closing line          : model {model_hit}% / book {book_hit}%   (log-loss "
      f"model {model_ll} vs book {book_ll})")
print("  fighter-type bias check:")
for ln in bias_lines: print("    " + ln)
if flags:
    print("\n  ⚠ FLAGGED FOR REVIEW (possible systematic bias — validate a fix on the backtest,")
    print("    do NOT auto-tune to the market):")
    for fl in flags: print("      - " + fl)
else:
    print("\n  ✅ No systematic fighter-type bias flagged this week.")

# ---- optional: current-card model-vs-market discrepancies (needs card_odds.csv) ----
card_odds = os.path.join(BASE, "card_odds.csv")
card_js = os.path.join(BASE, "web", "ufc_card.js")
if os.path.exists(card_odds) and os.path.exists(card_js):
    try:
        card = json.loads(open(card_js, encoding="utf-8").read().split("=", 1)[1].strip().rstrip(";"))
        def imp(a): a = float(a); return (-a)/((-a)+100) if a < 0 else 100/(a+100)
        odds = {}
        for r in csv.DictReader(open(card_odds, encoding="utf-8")):
            odds[(r["fighter_a"].strip().lower(), r["fighter_b"].strip().lower())] = (float(r["odds_a"]), float(r["odds_b"]))
        print("\n  CURRENT CARD — biggest model-vs-market gaps:")
        rows = []
        for b in card["bouts"]:
            if b.get("dataGap"): continue
            for (fa, fb), (oa, ob) in odds.items():
                if fa in b["a"].lower() and fb in b["b"].lower():
                    pa, pb = imp(oa), imp(ob); mkt = pa/(pa+pb); gap = (b["winA"] - mkt) * 100
                    rows.append((abs(gap), b["a"], b["b"], b["winA"]*100, mkt*100, gap))
        for _, a, bn, mo, mk, g in sorted(rows, reverse=True)[:6]:
            print(f"    {a} vs {bn}: model {mo:.0f}% / market {mk:.0f}%  ({g:+.0f})" + ("  <== review" if abs(g) >= 15 else ""))
    except Exception as e:
        print("  (current-card comparison skipped:", e, ")")
else:
    print("\n  (current-card comparison: drop a card_odds.csv [fighter_a,fighter_b,odds_a,odds_b] to enable)")

print(f"\nLogged to ufc_audit_log.csv  ({today}). Trend of these rows = the model's market-relative record.")
