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

## 2026-10-25: Logistic regression from scratch (NumPy)

- 261 features (200 one-hot, 60 multi-hot, 1 standardized number)
- Unit tests passed: sigmoid(0) = 0.5, loss(0.5, 1) = 0.693, the gradient matches the hand-computed example
- Full-batch gradient descent, lr = 0.5, 1000 epochs, weights start at 0
- Train loss: 0.6931 (epoch 0) → 0.1776 (epoch 100) → 0.1691 (epoch 1000), still slowly falling
- Validation (2024) loss: 0.1414. Baseline ("every crash = 4.65%"): 0.1571, so the model is ~10% better
- Validation (2024) ROC-AUC: 0.739. The baseline gets 0.5 by definition, since it gives every crash the same number
- Top 10% riskiest crashes by model (2024): 12.7% were fatal (vs 3.6% overall) and they contain 34.9% of all fatal crashes
- Largest odds ratios (correlation, not causation):
  - raise: pedestrians involved x1.87, motorcyclists x1.74, hitting an obstacle x1.59, dark without street lighting x1.40, hours 0–6 x1.2–1.3
  - lower: collision x0.43, daylight x0.47, passenger fall x0.56
- Bug found: `participant_categories_Все участники` is 1 in every crash, so it acts as a second bias term. Its "x0.44" is meaningless
- Validation loss is lower than train loss because 2024 has fewer fatal crashes (3.6% vs 4.65%), which makes it easier to predict. Losses on sets with different fatal rates can't be compared directly

## My notes (I write these myself)

- What surprised me:
- Hardest bug or concept, and what made it click:
- First time the loss went down (from → to, at which step):
- Final test metrics:
