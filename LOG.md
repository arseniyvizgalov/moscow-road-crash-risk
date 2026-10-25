# Project log

Raw notes for the README and the essay. Every number is copied from a notebook output.

## 2026-10-25: Data

- Source: dtp-stat.ru open data (from GIBDD records), Moscow, `moskva.geojson` (206 MB)
- 94,786 accidents where someone was injured or killed, from 2015-01-01 to Feb 2026, 22 columns
- Accidents per year: 7,641 (2022) to 10,299 (2015). 2026 has only 392 (Jan–Feb)
- Severity: severe 53,905 (56.9%), light 36,670 (38.7%), fatal 4,208 (4.4%), 3 rows "no data" (removed)
- Missing values: coordinates in 37 rows, address in 8,880 (address is not used)
- Weather: overcast 47,120, clear 42,172, rain 4,098, snowfall 1,948. Fog 35, blizzard 18, hurricane wind 1 (too rare to learn from)

## Decisions

- Target: `fatal` = 1 if severity is "С погибшими", otherwise 0. The fatal rate is 4.44% after removing the 3 rows
  - Why: death is unambiguous, while "severe vs light" depends on how injuries are classified
- Baseline "never fatal" scores 95.6% accuracy, so accuracy is useless here. I'll use ROC-AUC and recall
- Split by time: train on 2015–2023, validate on 2024, test on 2025 (once, at the end). 2026 is dropped
- Leakage: `severity`, `dead_count`, `injured_count` and `health_status` are never used as features
- Split sizes: train 77,181 (4.65% fatal), validation 8,859 (3.62%), test 8,352 (3.41%)
  - The fatal share falls over time, so a model trained on 2015–2023 may overestimate risk for 2025. That affects calibration and log loss, not ranking (ROC-AUC)

## First findings (fatality rate = share of crashes with a death)

- By hour: 2–5 am ≈ 10–11%, 8 am–6 pm ≈ 2.5–3.7%. Lowest at 8 am (2.5%)
- By type:
  - pedestrian hit 6.3% (32,792 crashes)
  - collision 2.5% (42,461)
  - hitting an obstacle 9.7% (5,019)
  - hitting a parked vehicle 8.8% (4,095)
  - cyclist hit 1.4% (3,884)
  - passenger fall 0.7% (4,412)
- Rates based on fewer than ~100 crashes are unreliable (e.g. 27.6% based on 76 crashes)

## My notes (I write these myself)

- What surprised me:
- Hardest bug or concept, and what made it click:
- First time the loss went down (from → to, at which step):
- Final test metrics:
