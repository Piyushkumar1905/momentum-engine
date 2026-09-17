# The strategy, and exactly how much to trust it

Rank the liquid universe by **12-month momentum skipping the most recent month**, tilt
**25% toward low beta**, take the **top 20**, hold up to **60 sessions** with a 2.5 ATR
stop and a 3.5 ATR trailing stop. No profit target.

Four parameters. Every other filter that was tried - a volatility cap, a 200DMA
constraint, regime switching, concentration - was tested and **removed** because it did
not survive.

## Evidence

Edge is measured against **every eligible name on the same dates** under the same exit
policy, so it is selection skill, not market direction. Inference is Newey-West at the
overlap lag; overlapping holds are not independent observations.

| Period | 6-month blocks positive | Mean edge / 60d | NW t |
|---|---|---|---|
| 2009-2026 (16.7y) | 27/35 (77%) | +0.87% | **3.33** |
| 2009-2020 backward OOS | 15/21 (71%) | +0.64% | 2.59 |
| 2021-2023 | - | +1.84% | 2.54 |
| 2024-2025 | - | +0.66% | 1.01 |
| 2025-2026 | - | +1.31% | 1.78 |

Sign test on 6-month blocks: full period **p=0.0045**, backward OOS **p=0.0392**.
2009-2020 is genuine out-of-sample - the parameters were chosen on 2021-2026 and that
decade was only fetched afterwards.

## Why these parameters and not others

Each was chosen from the **middle of a plateau**, never a peak.

**Holding period (60).** The clean plateau: t = 2.32 / 3.02 / **3.37** / 3.29 at 20 /
40 / 60 / 90 sessions. The original 20-session cap was the single biggest constraint on
the strategy - it captured a fraction of an edge that needs two months to express.

**Low-beta weight (0.25).** t = 2.48 / 2.91 / **3.33** / 3.13 / 2.39 / 1.76 at weights
0.00 / 0.10 / 0.25 / 0.40 / 0.60 / 1.00. Smooth and single-peaked. Note what it does:
mean edge is FLAT across 0.00-0.40 while block dispersion halves (2.17% -> 1.29%). The
tilt does not add return; it removes variance. That is why it is here.

**top_n (20).** t = 3.60 / 3.30 / 3.02 / 2.37 / 2.41 at 10 / 15 / 20 / 30 / 40.
Concentration scores higher but its worst block is twice as deep. 20 is the
risk-adjusted middle.

**No volatility cap.** 0.35 looked excellent (t=3.02) but sat on a spike - 0.30 gave
1.16 and 0.45 gave 1.48. Removing it entirely scored BETTER (28/35 blocks vs 26/34).
It was fitted noise.

## Known failure mode

**Momentum crashes.** Without the low-beta tilt, 2024-01 to 2025-06 was -0.80% per 60
days (t=-0.70). The tilt turns that window positive (+0.66%) but does not eliminate the
risk. Expect 12-18 month stretches of no edge. The worst 6-month block in 16.7 years was
-2.66%.

## What this is not

- **Not significant in every individual window.** Only the full period and train clear
  t=2. Validate is t=1.01. The case rests on consistency across 35 blocks, not on any
  one window.
- **Not adjusted for the search.** Roughly 60 specifications were tested. A t of 3.33
  survives that far better than the earlier candidates did, but it is not untouched.
- **Survivorship biased.** The universe is today's Nifty 500, and the bias grows the
  further back you go. Reassuringly, the most recent window - where the bias is smallest
  - is also positive.
- **Not measured against the Nifty.** +0.87% per 60 sessions is against the liquid
  universe: roughly 3-4% a year gross, before slippage and impact.

## Reproduce

```
python tools/final_validation.py --spec specs/RECOMMENDED.json
python tools/walk_forward.py --config config_research.yaml --start 2009-01-01 \
       --spec specs/RECOMMENDED.json --blocks --months 6
```

The live site publishes this strategy's picks daily and tracks them forward. That
forward record - unfittable, accruing from 2026-09 - will eventually be worth more than
everything above.
