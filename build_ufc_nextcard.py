# -*- coding: utf-8 -*-
"""
build_ufc_nextcard.py
=====================
Fetch the NEXT upcoming UFC card from the ESPN MMA API and write raw_nextcard.json
(the file export_ufc.py turns into web/ufc_card.js). This is the piece that was
missing: raw_nextcard.json used to be hand-authored, so the "next card" never
advanced. Now it self-updates to whatever the next scheduled event is.

Source: site.api.espn.com scoreboard (same endpoint probe_ufc.py validated).
Selection: the earliest event that is not completed and not in the past.
Ordering: ESPN lists bouts prelims-first; we reverse so the main event is bouts[0]
(export_ufc.py treats bouts[0] as the main event).

Usage:  python build_ufc_nextcard.py            # fetch + write raw_nextcard.json
        (export_ufc.py calls fetch_nextcard() itself, so normally you just run that)
"""
import urllib.request, json, datetime as dt, os, sys

try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass

BASE = os.path.dirname(os.path.abspath(__file__))
SCOREBOARD = "https://site.api.espn.com/apis/site/v2/sports/mma/ufc/scoreboard"
NEXTCARD_PATH = os.path.join(BASE, "raw_nextcard.json")

# ESPN abbreviates women's divisions as "W <div>"; expand for display.
WCLASS_MAP = {
    "W Strawweight": "Women's Strawweight", "W Flyweight": "Women's Flyweight",
    "W Bantamweight": "Women's Bantamweight", "W Featherweight": "Women's Featherweight",
}


def _get(url, timeout=25):
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def _pick_event(scoreboard, today=None):
    """Earliest non-completed, non-past event on the scoreboard."""
    today = today or dt.date.today()
    cands = []
    for e in scoreboard.get("events", []):
        state = (e.get("status", {}).get("type", {}) or {}).get("state")  # pre/in/post
        if state == "post":
            continue
        try:
            d = dt.date.fromisoformat(e["date"][:10])
        except (KeyError, ValueError):
            continue
        if d < today:
            continue
        cands.append((d, e))
    cands.sort(key=lambda x: x[0])
    return cands[0][1] if cands else None


def _build_bouts(event):
    # ESPN orders competitions prelims-first; reverse so the main event is first.
    bouts = []
    for c in reversed(event.get("competitions", [])):
        cs = sorted(c.get("competitors", []), key=lambda x: x.get("order", 99))
        if len(cs) < 2:
            continue
        a = (cs[0].get("athlete") or {}).get("displayName")
        b = (cs[1].get("athlete") or {}).get("displayName")
        if not a or not b:
            continue
        periods = (c.get("format", {}).get("regulation", {}) or {}).get("periods") or 3
        wc = c.get("type", {}).get("abbreviation", "") or ""
        wc = WCLASS_MAP.get(wc, wc)
        note = (c.get("note") or "").lower()
        is_title = ("title" in note) or ("championship" in note)
        bouts.append({
            "fighterA": a, "fighterB": b, "weightClass": wc,
            "rounds": int(periods), "isTitle": bool(is_title),
        })
    return bouts


def fetch_nextcard(write=True):
    """Fetch the next card and (optionally) write raw_nextcard.json. Returns the dict."""
    sb = _get(SCOREBOARD)
    e = _pick_event(sb)
    if e is None:
        raise RuntimeError("no upcoming UFC event found on the ESPN scoreboard")
    comp0 = (e.get("competitions") or [{}])[0]
    venue = comp0.get("venue", {}) or {}
    addr = venue.get("address") or {}
    location = ", ".join(x for x in [addr.get("city"), addr.get("state") or addr.get("country")] if x)
    card = {
        "event": e.get("name", ""),
        "date": e.get("date", "")[:10],
        "venue": venue.get("fullName", ""),
        "location": location,
        "bouts": _build_bouts(e),
    }
    if not card["bouts"]:
        raise RuntimeError("event found but no parseable bouts: " + card["event"])
    if write:
        with open(NEXTCARD_PATH, "w", encoding="utf-8") as fh:
            json.dump(card, fh, ensure_ascii=False, indent=2)
    return card


if __name__ == "__main__":
    c = fetch_nextcard(write=True)
    print(f"wrote {NEXTCARD_PATH}")
    print(f"  {c['event']}  ({c['date']})  @ {c['venue']}, {c['location']}")
    print(f"  {len(c['bouts'])} bouts; main event: {c['bouts'][0]['fighterA']} vs "
          f"{c['bouts'][0]['fighterB']} ({c['bouts'][0]['rounds']}R)")
