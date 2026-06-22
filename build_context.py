# -*- coding: utf-8 -*-
"""
Static per-team context attributes used by the match-day modifier layer.

  population_m   approx population in millions  (DISPLAY ONLY by default -- see README:
                 results-based ratings already capture realized talent, so weighting
                 population would penalise overperformers. Kept for context.)
  home_temp_c    typical match-day temperature the team is acclimatised to (deg C)
  home_alt_m     typical altitude of the team's home venues (metres)
  climate        coarse climate label (for readability)

These drive the climate/altitude *mismatch* penalty in simulate.py: a team is
penalised only when the venue is materially hotter / higher than it is used to.
Values are approximate and easily edited.
"""
import csv
import os
PROJ = os.path.dirname(os.path.abspath(__file__))

# team: (population_m, home_temp_c, home_alt_m, climate)
CTX = {
    "Canada":(40,18,100,"temperate"),       "Mexico":(129,20,2000,"hot/high-altitude"),
    "United States":(335,23,200,"temperate"),"Australia":(27,22,40,"hot"),
    "Iran":(89,26,1200,"hot-arid/altitude"),"Iraq":(44,32,40,"hot-arid"),
    "Japan":(124,21,40,"temperate"),        "Jordan":(11,27,800,"hot-arid/altitude"),
    "Qatar":(2.7,34,10,"very-hot-arid"),    "Saudi Arabia":(37,33,600,"hot-arid"),
    "South Korea":(52,19,40,"temperate"),   "Uzbekistan":(35,24,450,"continental"),
    "Algeria":(45,25,200,"hot"),            "Cape Verde":(0.6,25,50,"hot-humid"),
    "DR Congo":(102,27,300,"hot-humid"),    "Egypt":(112,29,50,"hot-arid"),
    "Ghana":(34,28,60,"hot-humid"),         "Ivory Coast":(28,28,50,"hot-humid"),
    "Morocco":(37,23,300,"warm-temperate"), "Senegal":(18,28,20,"hot"),
    "South Africa":(60,19,1400,"temperate/altitude"),"Tunisia":(12,26,20,"hot"),
    "Curaçao":(0.15,29,10,"hot-humid"),     "Haiti":(11.7,29,50,"hot-humid"),
    "Panama":(4.4,29,20,"hot-humid"),       "Argentina":(46,18,100,"temperate"),
    "Brazil":(216,26,400,"hot-humid"),      "Colombia":(52,17,2000,"high-altitude"),
    "Ecuador":(18,15,2500,"high-altitude"), "Paraguay":(6.9,25,100,"hot-humid"),
    "Uruguay":(3.4,17,40,"temperate"),      "New Zealand":(5.2,15,30,"temperate-cool"),
    "Austria":(9,16,200,"temperate-cool"),  "Belgium":(11.7,15,50,"temperate-cool"),
    "Bosnia and Herzegovina":(3.2,18,500,"temperate"),"Croatia":(3.9,21,120,"warm-temperate"),
    "Czech Republic":(10.7,16,250,"temperate-cool"),"England":(56,15,50,"temperate-cool"),
    "France":(68,18,150,"temperate"),       "Germany":(84,16,150,"temperate-cool"),
    "Netherlands":(17.8,15,5,"temperate-cool"),"Norway":(5.5,13,50,"cold"),
    "Portugal":(10.3,21,100,"warm-temperate"),"Scotland":(5.5,13,50,"cold"),
    "Spain":(48,23,600,"warm/altitude"),    "Sweden":(10.5,14,30,"cold"),
    "Switzerland":(8.8,15,450,"temperate-cool"),"Turkey":(85,21,100,"warm-temperate"),
}

with open(PROJ + r"\context.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["team","population_m","home_temp_c","home_alt_m","climate"])
    for t,(p,tc,al,cl) in CTX.items():
        w.writerow([t,p,tc,al,cl])
print(f"Wrote context.csv ({len(CTX)} teams)")
