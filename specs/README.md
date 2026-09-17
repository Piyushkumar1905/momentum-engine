# Search result: what survived, and what it is worth

Six hypothesis families were explored in parallel over 25 factors, then every candidate
that cleared a naive out-of-sample bar was handed to a separate agent whose job was to
refute it. **All five were refuted** — on control-draw noise, silent window truncation,
overlap-inflated trade counts, or an inert ranking.

`RECOMMENDED.json` is what remained after re-testing against a zero-noise benchmark
(the full eligible universe) with overlap-aware inference.

## What it is

Rank by 12-month momentum (skipping the last month), but only inside a universe capped
at 35% annualised volatility and trading above its 200-day average. Hold up to 40
sessions with a 2.5 ATR stop and a 3.5 ATR trailing stop. No profit target — the old 2R
target truncated winners.

## What the evidence actually supports

| Window | Mean edge / 40d | Newey-West t |
|---|---|---|
| train 2021-2023 | +0.75% | 1.98 |
| validate 2024-2025 | +0.80% | 1.73 |
| holdout 2025-2026 | +0.52% | 0.76 |
| full period | +0.72% | **2.62** |

Positive in all six calendar years (+1.19 / +0.36 / +0.71 / +1.22 / +0.48 / +0.08 %),
62-68% of scan dates positive every year, 295 distinct stocks traded.

**It is NOT statistically significant in either out-of-sample window on its own.** Only
the full period clears t=2, and that is before adjusting for the ~46 specifications
tested. Treat it as a plausible weak tilt with an economic story, not a proven edge.

## Why it is more believable than the alternatives

The edge decomposes into two anti-correlated parts:

| Component | Train (bull) | Correction |
|---|---|---|
| Momentum ranking | +1.51% | +0.02% |
| Low-vol + 200DMA filter | -0.76% | +0.50% |

Momentum carried the melt-up; the low-risk constraint carried the correction. Neither is
stable alone. The combination is steadier precisely because the two fail at different
times — which is a mechanism, not a curve fit. Both are long-documented anomalies.

## What it is not

- Not an alpha guarantee. +0.72% per 40 sessions against the *eligible universe* is
  roughly 4% a year gross, before slippage and market impact.
- Not measured against the Nifty. The benchmark is the liquid universe, so this is
  selection skill, not a return forecast.
- Not free of survivorship bias. The universe is today's Nifty 500 list.

## Reproduce

```
python tools/final_validation.py --spec specs/RECOMMENDED.json
```
