---
name: backtest-autopsy
description: Use when evaluating whether a trading strategy or screening rule actually works - reviewing backtest results, judging win rate or profit factor, deciding whether to change a threshold, explaining why a filter rejected a winner, or investigating an outlier trade. Also use when a backtest shows a suspiciously large single-trade loss or an unusually strong result.
---

# Backtest Autopsy

## Overview

A backtest reports how the trades it took performed. Four things it does not tell you
will each, on their own, produce a confident and wrong conclusion:

1. Whether the selection rule beat **picking at random** from the same universe
2. What the filter **rejected** (the winners never counted against it)
3. Whether an apparent edge **survives data it did not see**
4. Whether an extreme trade is a real loss or a **corporate-action artifact**

This skill is the order in which to check them. Run it before believing any backtest,
including your own.

## The four checks, in order

### 1. Baseline lift — always compute the denominator

A hit rate means nothing alone. Compute the same rate for the whole eligible universe
and divide.

```
lift = P(outcome | passed filter) / P(outcome | liquid universe)
```

- `lift > 1` — the filter selects better than chance
- `lift ≈ 1` — the filter is decoration; the ranking is doing the work
- `lift < 1` — **the filter is anti-predictive**; it selects worse than random

A screen can be profitable at `lift < 1` purely from ranking or exits. Knowing which
part carries the result determines what you are allowed to change.

**Never report a conditional rate without its baseline.** "10% of rejected stocks ran"
is meaningless until you know the universe rate was 6.7%.

### 2. Corporate-action artifacts — check the outliers before the statistics

Split and dividend adjustment is standard in vendor feeds. **Demergers are not.** When
value leaves for a separately-listed entity, the price series shows a collapse the
holder never took.

Detection: an overnight gap worse than about -30% that **never retraces**. A real crash
usually retraces part of the fall; a rebasing never does.

Do not "adjust" it — the demerged entity's value is not in that series, so any
adjustment is invented. **Exclude trades spanning the gap** and report the count.

One unadjusted demerger booked a -10R phantom loss and was the single worst trade in
a 1,362-trade sample.

### 3. Miss attribution — and its trap

For each scan date, take the eligible universe, compute forward returns, and attribute
each winner the screen refused to the **single rule that rejected it**. Only count
stocks failing exactly one rule; a stock failing four tells you nothing about any one.

**Then do not act on it.** Miss attribution counts the winners a filter blocked and is
structurally silent on the losers it blocked at the same moment. Relaxing admits both.

Check separately whether the rule is even **binding**: if relaxing it changes the final
pick count by 2 out of 613, it was never the constraint — the ranking already demoted
those names.

### 4. Out-of-sample gate — the only thing that authorises a change

Split the window. Fit on the first half, judge on the second, which never informed the
change. **A change must improve both halves to survive.**

| Result | Verdict |
|---|---|
| Better in both | Keep |
| Better in-sample only | **Curve fit — reject** |
| Better out-of-sample only | Noise — reject |
| Worse in both | Reject |

### 5. Three ways a harness lies about its own significance

Each of these inflated a result in this repo until it was found. Check all three before
believing any comparison.

**The benchmark is one draw.** A control of N random picks has sampling error. With
N=20 its SD was 55-65% of the edges being measured, so the "edge" was mostly which
control got drawn. Benchmark against the **entire eligible universe** — zero sampling
noise — or average many draws and report the spread.

**Long holds silently truncate the test window.** If the simulator refuses trades whose
holding window runs past the end of data, a 120-day hold discards the last ~6 months of
the holdout while a 20-day hold does not. The amount discarded scales with the very
parameter being selected, so long holds are structurally flattered. Require a full
window for every arm and report the dropped dates.

**Overlapping trades are not observations.** Weekly rebalancing with a 60-day hold means
~12 cohorts open at once and each "trade" shares ~92% of its window with others. 5,000
trades can be 11 independent bets. Use Newey-West at the overlap lag, and cross-check
with strictly non-overlapping cohorts. A naive t of 6 became 2.3 under this correction.

### 6. Separate ranking from universe

A screen does two different things: it **constrains** the universe (filters) and it
**ranks** within it. These can have opposite regime exposures, and reporting only the
combined number hides which one is working.

Test by inverting the ranking while holding filters fixed. If inverted ranking scores
the same, the ranking is inert and the result belongs to the filter. Measured here:
momentum ranking added +0.0151 in a bull window and +0.0002 in a correction, while a
low-volatility filter did the reverse. Neither was stable; the combination was, and only
because the two are anti-correlated.

### 7. Volatility-match before claiming skill

A ranking that quietly prefers volatile names earns a risk premium, not alpha, and that
premium reverses in a crash. Benchmarking against the whole universe cannot tell the
difference.

Compare each pick against eligible names in the **same ex-ante volatility decile** on
the same date. Report the ratio of pick volatility to universe volatility alongside
both numbers.

- Most of the edge survives matching → the ranking is finding something beyond risk
- The edge collapses → it is a leveraged beta bet; size it as one, or discard it

Measured here: picks at 1.16x universe volatility, raw edge +0.743% (t=2.48), matched
edge +0.728% (t=2.64) — 98% retained. An earlier rejected candidate carried 1.34-1.44x
and did not survive.

### 8. Parameter values: plateau, never peak

Sweep each parameter and look at the SHAPE, not the maximum.

| Shape | Meaning | Action |
|---|---|---|
| Smooth, wide, single-peaked | Real | Deploy from the middle |
| Sharp spike, neighbours collapse | Fitted | Remove the parameter |
| Monotone to an edge of the range | Untested | Extend the sweep |

Two parameters from the same study, both scoring t≈3.0 at their best value:

```
volatility cap:  0.30 -> 1.16   0.35 -> 3.02   0.45 -> 1.48     SPIKE -> removed
low-beta weight: 0.10 -> 2.91   0.25 -> 3.33   0.40 -> 3.13     PLATEAU -> kept
```

Identical headline t. Opposite verdicts. Removing the spiked parameter improved
out-of-sample results; the plateau parameter halved dispersion without changing mean
return, which is the signature of a genuine risk control rather than a fit.

## Red flags — stop and re-check

- Reporting a hit rate without the universe baseline
- Changing a threshold because the miss analysis flagged it
- A single trade worse than the stop distance allows (check for corporate actions)
- Tuning on the most recent months
- Concluding "the strategy works" from profit factor alone, with no lift computed
- An edge that appears only after you tried several variants
- Picks visibly more volatile than the universe, with no volatility-matched comparison
- A parameter whose neighbours score far worse than the chosen value

## Rationalizations

| Excuse | Reality |
|---|---|
| "The filter blocked 124 winners, so loosen it" | It blocked the losers too. Run the OOS gate. |
| "This outlier is just a bad trade" | Losses larger than the stop are usually data, not trading. |
| "Profit factor is 1.3, it works" | Against what? Compute lift before claiming skill. |
| "It improved in-sample, close enough" | That is the definition of a curve fit. |
| "Only the last few months are relevant" | The most recent data is the easiest to overfit. |
| "The sample is small but the edge is obvious" | Obvious small-sample edges are the ones that vanish. |
| "5,000 trades is plenty of evidence" | Overlapping holds. Count independent cohorts, not trades. |
| "It beat the random control" | One control draw has SD comparable to the edge. Benchmark the full universe. |
| "Longer holds performed better" | Check whether long holds silently dropped the end of the test window. |
| "The composite works, so the factors work" | Invert the ranking. If the score is unchanged, the filter did it. |
| "This value tested best, so use it" | Check its neighbours. Best-scoring and robust are different things. |
| "The picks beat the universe" | Were they riskier? Match on volatility before calling it skill. |

## Reference implementation

In this repository:

- `tools/analyze_period.py` — portfolio simulation, trade statistics, stop autopsy,
  miss attribution
- `tools/sweep_thresholds.py` — variant testing behind the out-of-sample gate
- `engine/data.py::detect_corporate_actions` — the gap detector
- `engine/rules.py::momentum_conditions` — per-rule masks, so attribution reads the
  same definitions the screen uses rather than a re-implementation that can drift

Position sizing must be stated with any rupee figure, and percentage statistics
reported alongside, because those hold at any capital.

## Real-world result

Applied to a Nifty 500 momentum screen over 2024-09 to 2026-09:

- Momentum filter lift **0.86x** — anti-predictive; reversal **1.35x**
- Eight corporate-action artifacts found; one worth -10R
- Miss attribution suggested relaxing two rules; **all six variants failed the OOS
  gate**, and trade counts moved by 2 of 613 — the rules were never binding

Without checks 1 and 4, the obvious conclusion was "loosen the filters". It was wrong.

A later six-family parallel search over 25 factors produced five candidates that cleared
a naive out-of-sample bar. Adversarial re-testing refuted **all five** - on control-draw
noise, window truncation, overlap-inflated trade counts, or an inert ranking. What
survived was much smaller than any of the original claims: a low-volatility universe
constraint with trend ranking inside it, +0.7% per 40 sessions against the eligible
universe, positive in all six calendar years but not individually significant in either
out-of-sample window. The search found the truth only because the refutation round was
built in from the start.
