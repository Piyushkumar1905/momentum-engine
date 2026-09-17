# Momentum & Reversal Discovery Engine — v1 (NSE, end-of-day)

Research tool that scans NSE stocks with two rule sets, lists the top 20 of each, and backtests
the rules point-in-time to check whether they actually work.

| Model | Hypothesis | Buys near |
|---|---|---|
| `momentum` | Strength continues | 52-week high, in an uptrend, with volume and RS confirmation |
| `reversal` | Weakness is exhausted | A *confirmed* turn from a 52-week low (higher low, reclaim of 20/50 DMA, RS divergence) |
| `control_naive_low` | Baseline, expected to lose | Any liquid stock within 5% of its 52-week low |

## Why the backtest does NOT reuse today's 20 stocks
Momentum rules select stocks *because* they already went up, so replaying their past gains is
circular. Instead, the engine replays the rules on every past scan date (weekly), using only data
available on that date, enters at the **next day's open**, and measures what happened **afterwards**.
Today's 20 are saved and can be forward-tracked with `python run.py track`.

## Setup (once)
1. Install Python 3.10+.
2. In this folder: `pip install -r requirements.txt`
3. Universe: download the **Nifty 500 constituents CSV** from niftyindices.com (index constituent
   download, file `ind_nifty500list.csv`) and put it in `universe/`.
   A small example file is in `universe/watchlist_example.csv`, but percentile ranks and sector
   strength need a broad universe to mean anything.
4. Check the install: `python run.py demo` (synthetic data; numbers are meaningless) and `python -m pytest -q`.

## Run
```
python run.py backtest     # rules tested 2021 -> today + today's lists   (first run downloads data)
python run.py scan         # today's lists only
python run.py track        # outcomes of every saved list so far
```
Tip: set `max_symbols: 60` in `config.yaml` for a quick first run, then set it back to `null`.

Outputs go to `output/`:
* `momentum_engine_<cmd>_<date>.xlsx`, with these sheets:
  * Read_me, Latest lists, Summary_forward, Summary_trades
  * Summary_by_regime, Summary_by_period, Summary_by_year
  * Trades, Regime_weekly
* `signals_*.csv`: every historical pick with its 5/10/20-day return, max upside (MFE), max drawdown (MAE), universe average and Nifty.
* `picks_<date>.csv`: today's lists with reasons, risk flags, stop and target references.
* `tracking.csv`: forward results of saved lists.

## Website (`site/`)
A static dashboard. There is **no backend**: a scheduled job runs the rules and commits JSON,
and the page is flat files that read it. Nothing serves requests, and no API key exists to leak.

```
python tools/export_site.py           # full: rebuilds backtest.json too
python tools/export_site.py --daily   # lists + tracking only (what the cron runs)
python -m http.server 8765 --directory site
```
Opening `site/index.html` directly will not work — `fetch` is blocked on `file://`. Serve it.

What refreshes, and when:

| Layer | Content | Clock |
|---|---|---|
| Tier 0 | method, limits, thresholds | on edit |
| Tier 1 | backtest evidence | **frozen**; rebuilds when `rules_hash` changes |
| Tier 2 | today's lists, regime, tracking | weekdays, 12:00 UTC (17:30 IST) |
| Tier 3 | verdicts, ages, market state | every page load, in-browser, no fetch |
| Tier 4 | universe constituents | manual, archived dated |

Tier 1 is frozen deliberately. One extra scan date out of ~283 moves nothing, and a backtest
that re-renders every morning turns fixed pass-marks into a live feed — inviting exactly the
tune-on-recent-data habit the discipline above forbids. It rebuilds when the rules change.

`site/data/*.json` carries **derived output only** — rankings, scores, stops, statistics. The
vendor's raw OHLCV is never published; it stays in the local cache, which `.gitignore` excludes.

**Deployment.** `.github/workflows/refresh.yml` runs the rules on a weekday cron, commits the
refreshed JSON and publishes `site/` to GitHub Pages. Enable Pages (Settings → Pages → source
*GitHub Actions*) once; after that it maintains itself. `workflow_dispatch` gives a manual
refresh, with a `full` toggle to rebuild the backtest.

The daily run only fetches bars it does not already have, so it stays inside the vendor's rate
limits. That relies on the price cache surviving between runs via `actions/cache`; if the cache
is ever evicted, the next run re-downloads full history once and then returns to tail fetches.

## How to judge the results (decide before looking)
| Question | Where | Pass mark (proposal) |
|---|---|---|
| Do picks beat *any* liquid stock? | Summary_forward → `avg_excess_vs_universe` | > 0 at 5, 10 and 20 days |
| Is there upside to capture? | `avg_max_upside_MFE`, `share_reaching_+5pct` | Clearly above `universe_avg_MFE` |
| Do rule-based trades make money after costs? | Summary_trades → `expectancy`, `profit_factor` | expectancy > 0, PF ≥ 1.2 |
| Does it hold out-of-sample? | Summary_by_period | Same sign in both periods |
| Does the reversal filter add value? | reversal vs control_naive_low | reversal clearly better |
| Regime dependence | Summary_by_regime | Know where it fails; size down there |
| Enough evidence? | `trades` / `signals` | ≥ 300 trades per model |

Tuning discipline:
* Change one threshold at a time, and tune on the in-sample period only.
* Accept a change only if the out-of-sample period also improves.
* Never tune on the last few weeks.

## Main rules (all editable in `config.yaml`)
**Universe hygiene (both models)**
* ≥ 252 sessions of history
* 20-day median traded value ≥ ₹10 cr
* price ≥ ₹50

**Momentum**
* close ≥ 85% of the 52W high
* close > 50DMA > 200DMA, with the 200DMA rising
* ≤ 3 ATR above the 20DMA and ≤ 20% above the 50DMA
* ATR% between 1.5% and 6%
* RS vs Nifty (63D) > 0
* Bear regime: ≥ 90% of the high and RS ≥ +5%

Score:
* price momentum 23.5%, relative strength 17.6%, volume 17.6%
* structure 17.6%, breakout quality 11.8%, sector 11.8%
* The 10% fundamentals and 5% catalyst weights are re-distributed in v1.

Trade plan: stop = entry − 1.5 ATR; target = 2R; exit after 20 sessions if neither is hit.

**Reversal**
* ≥ 30% fall before the low; the 52W low was made 20–60 sessions ago
* last 15 sessions ≥ 1 ATR above the low
* ≥ 3 earlier sessions traded in the support band
* close above the 20DMA and 50DMA, with the 20DMA rising
* RS vs Nifty made a higher low

Score: RS divergence, reclaim, accumulation, higher low, capitulation volume, sector.

Trade plan: stop = 15-session low − 0.5 ATR; target = 2.5R; 20-session time exit.

## Known limitations of v1 (read before trusting any number)
* **Survivorship bias.** The universe is today's index list, so delisted stocks and stocks that dropped out are missing. Results will look better than reality.
* **Missing data.** No delivery %, fundamentals, results dates, pledges or news in v1, and sector strength is derived from universe members, not official sector indices.
* **Execution assumptions.**
  * Entries fill at the next open. Real fills also face slippage (partly covered by `cost_pct`), circuit limits and gaps.
  * If a stop and target are hit on the same day, the stop is assumed first (conservative).
* **Data quality.** Yahoo EOD data is for personal research. Check its terms before any commercial or client use, and spot-check prices against NSE. NSE's website terms prohibit automated scraping; for production use licensed data (NSE Data & Analytics, a broker API or an authorised vendor) through `data_source: csv` or a new source class.
* **Not advice.** A research output, not a recommendation to buy or sell.

## Structure
```
config.yaml          thresholds, weights, dates (no hard-coded rules in code)
run.py               CLI: scan | backtest | track | demo
engine/data.py       universe, Yahoo / CSV / synthetic sources, caching, calendar alignment
engine/features.py   point-in-time indicators + market regime
engine/rules.py      eligibility filters, scores, setup labels, explanations, stops
engine/backtest.py   scans on past dates, forward outcomes, trade simulation, stats, tracking
engine/report.py     console summary, Excel + CSV outputs
tests/               look-ahead checks, trade logic, data parsers, pipeline smoke tests
```
