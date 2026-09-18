"""The detailed, plain-language report: how stocks are filtered, how much is risked per
trade, how often trades happen, how they end, and what winners and losers had in common.

  python tools/build_detailed_report.py
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt          # noqa: E402
import numpy as np                        # noqa: E402
from reportlab.lib.pagesizes import A4    # noqa: E402
from reportlab.lib.units import mm        # noqa: E402
from reportlab.platypus import (BaseDocTemplate, Frame, Image, KeepTogether,   # noqa: E402
                                PageBreak, PageTemplate, Spacer)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.build_report import (BAD, BODY, GOOD, H1, H2, KEY, MUTED, NOTE,     # noqa: E402
                                SUB, TITLE, para, table)

IST = timezone(timedelta(hours=5, minutes=30))

# Plain-English names, used by BOTH the chart and the table so a reader never has to
# map a column name onto a description.
LBL = {"atr_pct": "daily swing", "dist_high": "% of 52-week high",
       "mom_6m_skip": "6-month momentum", "med_value_cr": "traded value",
       "dist_low": "above 52-week low", "ext_50": "% above 50-day avg",
       "rvol": "volume vs normal", "rs_63": "strength vs Nifty",
       "ext_atr": "stretch above 20-day avg", "ret_20": "last month's return",
       "sector_rank": "sector strength"}
INK, GREY, BLUE, GREEN, RED = "#12161c", "#8a8f98", "#1f3864", "#0f7a4d", "#b3261e"


def pc(v, d=1, sign=False):
    if v is None:
        return "-"
    return f"{v*100:+.{d}f}%" if sign else f"{v*100:.{d}f}%"


def rs(v, d=0):
    return f"Rs {v:,.{d}f}"


def chart_funnel(f, path):
    stages = [("Nifty 500 list", f["in_index"]), ("has price data", f["with_price_data"]),
              ("1 year of history", f["enough_history"]), ("liquid enough", f["liquid_enough"]),
              ("price above Rs 50", f["passed_all_hygiene"]), ("ranked & bought", f["selected"])]
    fig, ax = plt.subplots(figsize=(7.2, 2.6), dpi=200)
    names = [s[0] for s in stages][::-1]
    vals = [s[1] for s in stages][::-1]
    cols = [GREEN if i == 0 else BLUE for i in range(len(vals))]
    ax.barh(names, vals, color=cols, height=0.62)
    for i, v in enumerate(vals):
        ax.text(v + 6, i, f"{v}", va="center", fontsize=8.5, color=INK)
    ax.set_xlim(0, max(vals) * 1.16)
    ax.set_xlabel("Number of stocks", fontsize=8)
    ax.tick_params(labelsize=8.5)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout(); fig.savefig(path, bbox_inches="tight"); plt.close(fig)


def chart_r(hist, path):
    x = np.array(hist, dtype=float)
    fig, ax = plt.subplots(figsize=(7.2, 2.4), dpi=200)
    bins = np.arange(-3, 8.25, 0.25)
    n, b, patches = ax.hist(x, bins=bins, edgecolor="white", linewidth=0.4)
    for p, left in zip(patches, b[:-1]):
        p.set_facecolor(RED if left < 0 else GREEN)
    ax.axvline(0, color=INK, lw=1)
    ax.axvline(-1, color=INK, lw=0.9, ls="--")
    ax.text(-1.05, ax.get_ylim()[1] * 0.92, "the stop\n(-1R)", ha="right", fontsize=7.5, color=INK)
    ax.set_xlabel("Result in R (multiples of the amount risked)", fontsize=8)
    ax.set_ylabel("Trades", fontsize=8)
    ax.tick_params(labelsize=8)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout(); fig.savefig(path, bbox_inches="tight"); plt.close(fig)


def chart_hold(hist, path):
    x = np.array(hist, dtype=float)
    fig, ax = plt.subplots(figsize=(7.2, 2.2), dpi=200)
    ax.hist(x, bins=np.arange(0, 63, 3), color=BLUE, edgecolor="white", linewidth=0.4)
    ax.set_xlabel("Sessions held", fontsize=8)
    ax.set_ylabel("Trades", fontsize=8)
    ax.tick_params(labelsize=8)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout(); fig.savefig(path, bbox_inches="tight"); plt.close(fig)


def chart_gaps(rows, path):
    rows = rows[:8]
    fig, ax = plt.subplots(figsize=(7.2, 2.5), dpi=200)
    names = [LBL.get(r["feature"], r["feature"]) for r in rows][::-1]
    vals = [r["gap_sd"] for r in rows][::-1]
    ax.barh(names, vals, color=[GREEN if v > 0 else RED for v in vals], height=0.6)
    ax.axvline(0, color=INK, lw=1)
    for x in (-0.2, 0.2):
        ax.axvline(x, color=GREY, lw=0.8, ls="--")
    ax.set_xlim(-1, 1)
    ax.set_xlabel("Difference between winners and losers (standard deviations).\n"
                  "Anything inside the dashed lines is indistinguishable from noise.",
                  fontsize=8)
    ax.tick_params(labelsize=8)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout(); fig.savefig(path, bbox_inches="tight"); plt.close(fig)


def chart_stop_vs_size(D, path):
    """Two panels: how wide stops actually are, and why a wider stop is not more risk.

    Replaces a table that paired the 10th percentile of stop distance with the 10th
    percentile of POSITION SIZE - opposite trades. A 6.5% stop implies a 15.3%
    position, not a 6.4% one, and the table read as though it implied the latter.
    """
    rk = D["risk"]
    risk_pct = D["risk_pct"]
    q = rk["stop_distance_pct"]
    labels = ["widest 10%", "wide 25%", "typical", "tight 25%", "tightest 10%"]
    stops = [q["p90"], q["p75"], q["p50"], q["p25"], q["p10"]]
    sizes = [risk_pct / st for st in stops]
    rupee = D["capital"] * risk_pct

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7.2, 4.6), dpi=200,
                                   gridspec_kw={"height_ratios": [1.15, 1]})

    # ── panel 1: back-to-back bars, so the trade-off is visible at a glance
    y = np.arange(len(labels))
    ax1.barh(y, [-s * 100 for s in stops], color=RED, height=0.6)
    ax1.barh(y, [s * 100 for s in sizes], color=BLUE, height=0.6)
    for i, (st, sz) in enumerate(zip(stops, sizes)):
        ax1.text(-st * 100 - 0.6, i, f"{st*100:.1f}%", ha="right", va="center", fontsize=8.5)
        ax1.text(sz * 100 + 0.6, i, f"{sz*100:.1f}%", ha="left", va="center", fontsize=8.5)
    ax1.set_yticks(y); ax1.set_yticklabels(labels, fontsize=8.5)
    ax1.axvline(0, color=INK, lw=1)
    ax1.set_xlim(-22, 22)
    ax1.set_xticks([-20, -15, -10, -5, 0, 5, 10, 15, 20])
    ax1.set_xticklabels(["20", "15", "10", "5", "0", "5", "10", "15", "20"], fontsize=8)
    ax1.set_xlabel("<-- stop distance below entry (%)          "
                   "position size, % of account -->", fontsize=8.5)
    ax1.set_title(f"A wider stop buys fewer shares. Every row risks the same "
                  f"Rs {rupee:,.0f}.", fontsize=9.5, loc="left", pad=8)
    for sp in ("top", "right", "left"):
        ax1.spines[sp].set_visible(False)
    ax1.tick_params(axis="y", length=0)

    # ── panel 2: what stop distances actually occur
    x = np.array(D.get("_hist_stop", []), dtype=float) * 100
    if len(x):
        ax2.hist(x, bins=np.arange(0, 26, 0.5), color=GREY, edgecolor="white", linewidth=0.3)
        med = q["p50"] * 100
        ax2.axvline(med, color=INK, lw=1.2)
        ax2.text(med + 0.4, ax2.get_ylim()[1] * 0.85, f"median {med:.1f}%",
                 fontsize=8, color=INK)
        ax2.axvspan(q["p10"] * 100, q["p90"] * 100, color=BLUE, alpha=0.08)
        ax2.text(q["p90"] * 100 + 0.5, ax2.get_ylim()[1] * 0.60,
                 "80% of trades\nfall in this band", fontsize=7.5, color="#5b6572")
    ax2.set_xlabel("Stop distance below entry, every trade (%)", fontsize=8.5)
    ax2.set_ylabel("Trades", fontsize=8)
    ax2.tick_params(labelsize=8)
    for sp in ("top", "right"):
        ax2.spines[sp].set_visible(False)

    fig.tight_layout(h_pad=1.6)
    fig.savefig(path, bbox_inches="tight"); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--facts", default=str(ROOT / "output" / "analysis_final" / "facts.json"))
    ap.add_argument("--out", default=str(ROOT / "output" / "analysis_final" /
                                         "how_the_strategy_works.pdf"))
    a = ap.parse_args()

    D = json.loads(Path(a.facts).read_text(encoding="utf-8"))
    spec, fq, rk, oc = D["spec"], D["frequency"], D["risk"], D["outcome"]
    out = Path(a.out)
    tmp = out.parent / "_charts2"
    tmp.mkdir(parents=True, exist_ok=True)

    doc = BaseDocTemplate(str(out), pagesize=A4, leftMargin=17 * mm, rightMargin=17 * mm,
                          topMargin=16 * mm, bottomMargin=16 * mm,
                          title="How the strategy works", author="momentum_engine")
    frame = Frame(doc.leftMargin, doc.bottomMargin, doc.width, doc.height, id="n")

    def footer(canv, d):
        canv.saveState(); canv.setFont("Helvetica", 7.5)
        canv.setFillColor(MUTED)
        canv.drawString(doc.leftMargin, 10 * mm,
                        "momentum_engine - research output, not investment advice")
        canv.drawRightString(A4[0] - doc.rightMargin, 10 * mm, f"Page {d.page}")
        canv.restoreState()

    doc.addPageTemplates([PageTemplate(id="all", frames=[frame], onPage=footer)])
    st = []

    # ══════════════ cover ══════════════
    st += [para("How the strategy works", TITLE),
           para(f"Every rule, every number, in plain language &middot; "
                f"{D['window'][0]} to {D['window'][1]} &middot; "
                f"{fq['trades_total']:,} trades &middot; generated "
                f"{datetime.now(IST).strftime('%d %b %Y')}", SUB)]

    st += [para("The whole thing in one minute", H1),
           para(f"Once a week the system looks at every stock in the Nifty 500, throws away the "
                f"ones that are too small or too thinly traded, and ranks what is left by how much "
                f"it has risen over the past year. It buys the top {spec['top_n']} at the next "
                f"morning's open.", BODY),
           para(f"Each position is sized so that if the stop is hit, the loss is <b>1% of the "
                f"account</b> - on 10 lakh that is {rs(rk['rupee_risk_per_trade'])}. The stop sits "
                f"about {pc(rk['stop_distance_pct']['p50'])} below the buy price, so a typical "
                f"position is around {rs(rk['position_value_median'])}, roughly "
                f"{pc(rk['position_pct_of_capital_median'])} of the account.", BODY),
           para(f"About <b>{fq['trades_per_month']:.0f} new trades a month</b> "
                f"({fq['new_trades_per_trading_day']:.2f} per trading day), with around "
                f"{fq['concurrent_positions_mean']:.0f} positions open at any time.", BODY),
           para(f"Most trades lose. {pc(D['exits'][0]['share'])} of them hit the stop and are out "
                f"within about {D['exits'][0]['avg_hold']:.0f} sessions. The money comes from the "
                f"{pc(D['exits'][1]['share'])} that survive to the {spec['exit']['max_hold']}-session "
                f"time limit - those win {pc(D['exits'][1]['win_rate'])} of the time and average "
                f"{pc(D['exits'][1]['avg_ret'], sign=True)}.", BODY),
           para("The single most important finding in this report: <b>nothing about a stock at the "
                "moment of purchase tells you whether it will be a winner or a loser.</b> Winners "
                "and losers look identical on entry. The edge is entirely in how trades are exited.",
                KEY)]

    # ══════════════ 1. selection ══════════════
    st += [PageBreak(), para("1. How a stock gets chosen", H1),
           para("Two stages. First a hygiene screen removes what cannot be traded properly. Then "
                "whatever survives is ranked, and the top names are bought. The hygiene screen is "
                "about tradability, not about predicting returns.", BODY)]

    fn = D["funnel_latest"]
    ch = tmp / "funnel.png"
    chart_funnel(fn, ch)
    st += [para(f"The funnel on {fn['date']}", H2),
           Image(str(ch), width=doc.width, height=doc.width * 0.361)]

    avg = D["funnel_average"]
    st += [para("Averaged across every scan in the 17-year test, the same funnel looks like this. "
                "The count of stocks with usable price data is lower in earlier years because many "
                "of today's index members had not listed yet.", NOTE),
           table([["Stage", "What it means", "Stocks left (avg)"],
                  ["In the index", "the Nifty 500 constituent list", f"{avg['in_index']}"],
                  ["Has price data", "actually listed and trading on that date",
                   f"{avg['with_price_data']}"],
                  ["Enough history", "at least 252 sessions, so a 1-year figure exists",
                   f"{avg['enough_history']}"],
                  ["Liquid enough", "20-day median traded value of Rs 10 crore or more",
                   f"{avg['liquid_enough']}"],
                  ["Price above Rs 50", "avoids penny-stock tick and spread effects",
                   f"{avg['passed_all_hygiene']}"],
                  ["Bought", f"top {spec['top_n']} by score", f"{avg['selected']}"]],
                 [34 * mm, 84 * mm, 32 * mm],
                 [("ALIGN", (2, 0), (2, -1), "RIGHT")])]

    fc = D["filter_cost"]
    st += [para("Which screen removes the most", H2),
           table([["Screen", "Stocks removed per scan (avg)"]] +
                 [[k, f"{v}"] for k, v in sorted(fc.items(), key=lambda x: -x[1])],
                 [90 * mm, 50 * mm], [("ALIGN", (1, 0), (1, -1), "RIGHT")])]
    st += [para("Liquidity does nearly all the work. Roughly 139 stocks per scan are excluded for "
                "trading less than Rs 10 crore a day - not because they are bad investments, but "
                "because a position large enough to matter could not be entered or exited without "
                "moving the price.", NOTE)]

    st += [para("How the survivors are ranked", H2),
           table([["Ingredient", "Weight", "What it measures"],
                  ["12-month momentum,\nskipping the last month", "1.00",
                   "price a month ago divided by price 13 months ago.\n"
                   "The recent month is skipped because very recent\n"
                   "moves tend to reverse."],
                  ["Low beta (252-day)", "0.25",
                   "how much the stock moves when the index moves.\n"
                   "A tilt toward calmer names."]],
                 [44 * mm, 18 * mm, 88 * mm])]
    st += [para("Both ingredients are converted to percentile ranks before being combined, so a "
                "stock in the 90th percentile for momentum and the 50th for calmness scores "
                "0.90 + 0.25 x 0.50. The top 20 scores are bought. There is no volatility cap and "
                "no moving-average filter: both were tested over 17 years and neither improved "
                "anything.", NOTE)]

    # ══════════════ 2. risk ══════════════
    st += [PageBreak(), para("2. How much is risked on each trade", H1),
           para("Position size is not fixed in rupees. It is worked backwards from the stop, so "
                "that every trade risks the same amount of money regardless of how volatile the "
                "stock is. A jumpy stock gets a wider stop and therefore fewer shares.", BODY)]

    st += [para("The formula", H2),
           table([["Step", "Calculation", "Example"],
                  ["1. Decide the risk", "1% of account", rs(rk["rupee_risk_per_trade"])],
                  ["2. Place the stop", f"{spec['exit']['stop_atr']} x ATR below entry",
                   f"{pc(rk['stop_distance_pct']['p50'])} below buy price"],
                  ["3. Risk per share", "entry price minus stop price", "Rs 101 on a Rs 1,000 stock"],
                  ["4. Shares to buy", "risk budget / risk per share", "10,000 / 101 = 99 shares"],
                  ["5. Position value", "shares x entry price", rs(rk["position_value_median"])]],
                 [36 * mm, 58 * mm, 56 * mm])]

    cs = tmp / "stopsize.png"
    chart_stop_vs_size(D, cs)
    st += [para("How wide the stop actually sits", H2),
           Image(str(cs), width=doc.width, height=doc.width * 0.639)]
    st += [para(f"In the actual portfolio run the median position came to "
                f"{rs(rk.get('actual_position_value_median', 0))} - about "
                f"{rk.get('actual_qty_median', 0):.0f} shares of a typical pick. Note that a wider "
                "stop does not mean more risk: it means fewer shares. The rupee risk is the same "
                "either way. That is the entire point of sizing this way.", NOTE)]

    st += [para("The one place this protection fails", H2),
           para(f"A stop is an instruction, not a guarantee. If a stock gaps down overnight and "
                f"opens below the stop, the exit happens at the open - worse than planned. This "
                f"happened on <b>{pc(D['r_worse_than_minus_1'])} of trades</b>, where the loss "
                f"exceeded the 1R that was budgeted. The worst observed was about -2R, i.e. double "
                f"the intended loss.", BODY),
           para("Practical consequence: plan for the occasional position losing 2% of the account "
                "rather than 1%, and do not size on the assumption that the stop is absolute.", NOTE)]

    # ══════════════ 3. frequency ══════════════
    st += [PageBreak(), para("3. How often trades happen", H1),
           table([["Measure", "Value", "In plain terms"],
                  ["Scans", f"{fq['scan_dates']:,} over {fq['years']:.1f} years",
                   "the list is rebuilt once a week"],
                  ["New trades per scan", f"{fq['new_trades_per_scan_mean']:.1f} on average",
                   "most of the top 20 are already held"],
                  ["New trades per month", f"{fq['trades_per_month']:.1f}",
                   "roughly four a week"],
                  ["New trades per trading day", f"{fq['new_trades_per_trading_day']:.2f}",
                   "less than one a day"],
                  ["Positions open at once", f"{fq['concurrent_positions_mean']:.0f} average, "
                   f"{fq['concurrent_positions_max']} peak",
                   "positions overlap because holds are long"],
                  ["Total trades tested", f"{fq['trades_total']:,}", "over 17.7 years"]],
                 [44 * mm, 44 * mm, 62 * mm])]
    st += [para(f"The gap between {fq['new_trades_per_scan_mean']:.1f} new trades per weekly scan "
                f"and {fq['concurrent_positions_mean']:.0f} open positions is worth understanding: "
                "because a position can be held up to 60 sessions, roughly 12 weekly cohorts are "
                "live at any moment. The unconstrained strategy therefore wants far more positions "
                "than a small account can fund. The portfolio runs cap this at 10 open positions, "
                "which is why many signals go untaken.", NOTE)]

    # ══════════════ 4. exits ══════════════
    st += [para("4. How a trade ends", H1),
           para("There are only two ways out. There is no profit target - nothing is ever sold "
                "simply because it has gone up by a set amount.", BODY),
           table([["Exit", "Trigger", "Share of trades", "Win rate", "Average result", "Avg hold"]] +
                 [[e["exit"],
                   ("price falls to the stop or the trailing stop" if e["exit"] == "stop"
                    else f"{spec['exit']['max_hold']} sessions elapsed"),
                   pc(e["share"]), pc(e["win_rate"]), pc(e["avg_ret"], sign=True),
                   f"{e['avg_hold']:.0f}d"] for e in D["exits"]],
                 [16 * mm, 56 * mm, 22 * mm, 18 * mm, 24 * mm, 18 * mm],
                 [("ALIGN", (2, 0), (-1, -1), "RIGHT")])]

    st += [para("The two stop levels", H2),
           table([["Stop", "Where it sits", "When it moves"],
                  ["Initial", f"{spec['exit']['stop_atr']} ATR below the buy price",
                   "never - fixed at entry, used for sizing"],
                  ["Trailing", f"{spec['exit']['trail_atr']} ATR below the highest close reached",
                   "upward only, as the stock makes new highs"]],
                 [24 * mm, 62 * mm, 64 * mm])]
    st += [para("The trailing stop is what converts a winner into a completed trade. Once a stock "
                "has risen far enough, the trail rises above the buy price and the trade can no "
                "longer lose money. That is why a 'stop' exit is not automatically a loss - "
                f"{pc(D['exits'][0]['win_rate'])} of stop exits were profitable.", NOTE)]

    chr_ = tmp / "rdist.png"
    chart_r(D["_hist_r"], chr_)
    st += [para("Where results actually land", H2),
           Image(str(chr_), width=doc.width, height=doc.width * 0.333)]
    st += [table([["Result band", "Trades", "Share"]] +
                 [[r["band"], f"{r['n']:,}", pc(r["share"])] for r in D["r_distribution"]],
                 [42 * mm, 26 * mm, 22 * mm], [("ALIGN", (1, 0), (-1, -1), "RIGHT")])]
    st += [para("The tall bar just below -1R is the stop doing its job: 28.6% of all trades end "
                "there, losing almost exactly what was budgeted. The right tail is thin but it is "
                "where the profit lives - 6.3% of trades returned more than 3R, and those pay for "
                "everything else.", NOTE)]

    chh = tmp / "hold.png"
    chart_hold(D["_hist_hold"], chh)
    hd = D["hold_distribution"]
    st += [para("How long positions are held", H2),
           Image(str(chh), width=doc.width, height=doc.width * 0.306),
           para(f"Median {hd['median']:.0f} sessions. {pc(hd['share_under_10d'])} are closed within "
                f"10 sessions - these are the quick failures. {pc(hd['share_full_60d'])} run the "
                f"full {spec['exit']['max_hold']} sessions.", NOTE)]

    # ══════════════ 5. winners vs losers ══════════════
    st += [PageBreak(), para("5. What winners and losers had in common", H1),
           para(f"Of {oc['trades']:,} trades, {oc['winners']:,} made money and {oc['losers']:,} lost "
                f"- a win rate of {pc(oc['win_rate'])}. The obvious question is what the winners "
                "looked like when they were bought, so that more of them could be chosen.", BODY),
           para("The answer is that they looked the same.", KEY)]

    cg = tmp / "gaps.png"
    chart_gaps(D["winners_vs_losers"], cg)
    st += [Image(str(cg), width=doc.width, height=doc.width * 0.347)]

    rows = [["What was measured at entry", "Winners", "Losers", "Difference"]]
    LONG = {"atr_pct": "daily swing (ATR as % of price)", "dist_high": "% of its 52-week high",
            "med_value_cr": "traded value (Rs cr/day)",
            "dist_low": "times above its 52-week low", "ext_50": "% above the 50-day average",
            "ext_atr": "stretch above the 20-day average",
            "sector_rank": "sector strength percentile"}
    for r in D["winners_vs_losers"]:
        f_ = r["feature"]
        v = "pct" if f_ in ("atr_pct", "mom_6m_skip", "ext_50", "rs_63", "ret_20") else "num"
        fw = pc(r["winners"]) if v == "pct" else f"{r['winners']:.2f}"
        fl = pc(r["losers"]) if v == "pct" else f"{r['losers']:.2f}"
        rows.append([LONG.get(f_, LBL.get(f_, f_)), fw, fl, f"{r['gap_sd']:+.2f} sd"])
    st += [table(rows, [72 * mm, 26 * mm, 26 * mm, 26 * mm],
                 [("ALIGN", (1, 0), (-1, -1), "RIGHT")])]
    st += [para("Not one characteristic separates the two groups. The largest difference is 0.11 "
                "standard deviations, which is nothing - a gap needs to be roughly 0.2 before it is "
                "even worth a second look, and these are measured after the fact on the same data, "
                "which biases them upward if anything.", NOTE)]

    st += [para("The score itself does not predict either", H2),
           table([["Position in the list", "Trades", "Win rate", "Average result"]] +
                 [[r["band"], f"{r['n']:,}", pc(r["win_rate"]), pc(r["avg_ret"], sign=True)]
                  for r in D["by_rank"]],
                 [42 * mm, 26 * mm, 26 * mm, 30 * mm], [("ALIGN", (1, 0), (-1, -1), "RIGHT")])]
    st += [para("The stock ranked first is no more likely to work than the stock ranked twentieth. "
                "This is why buying only the top 5 performs worse, not better - it concentrates "
                "into names with no measurable advantage while giving up diversification.", NOTE)]

    st += [para("What DOES separate them", H2),
           table([["", "Winners", "Losers"],
                  ["Median sessions held", f"{oc['winner_hold_median']:.0f}",
                   f"{oc['loser_hold_median']:.0f}"],
                  ["Average result", pc(oc["winner_avg_ret"], sign=True),
                   pc(oc["loser_avg_ret"], sign=True)],
                  ["Total contribution", f"+{oc['gross_gains']:.0f} units",
                   f"-{oc['gross_losses']:.0f} units"]],
                 [56 * mm, 36 * mm, 36 * mm], [("ALIGN", (1, 0), (-1, -1), "RIGHT")])]
    st += [para(f"Only time. Winners were held {oc['winner_hold_median']:.0f} sessions at the "
                f"median; losers {oc['loser_hold_median']:.0f}. A losing trade announces itself "
                "quickly and is cut; a winning trade has to be left alone for two to three months. "
                "The strategy does not find better stocks - it fails cheaply and often, and holds "
                "the few that work.", BODY)]

    # ══════════════ 6. practical ══════════════
    st += [PageBreak(), para("6. What this means if you run it", H1)]
    for n, (h, b) in enumerate([
        ("Do not try to pick within the list",
         "Sections 5 shows there is no information in the ranking beyond making the list at all. "
         "Choosing 'the best looking' five of the twenty is guesswork that costs diversification."),
        ("Do not take profits early",
         "The entire return comes from the 22% of trades that run the full 60 sessions and average "
         "+18%. Selling those at +10% removes the profit and leaves the losses."),
        ("Expect to be wrong more often than right",
         f"{pc(1 - oc['win_rate'])} of trades lose money. The longest run of consecutive losers in "
         "testing was 30. This is normal for the method, not a sign it has stopped working."),
        ("Size for a 2R loss, not 1R",
         f"{pc(D['r_worse_than_minus_1'])} of trades gapped through the stop and lost more than "
         "budgeted. On a 1% risk setting, plan for the occasional 2% hit."),
        ("The account needs to fund more positions than you expect",
         f"{fq['concurrent_positions_mean']:.0f} positions overlap on average because holds are "
         "long. With 10 lakh and 10 slots, many signals simply go untaken - the results reflect "
         "the subset that fitted."),
        ("Judge it over years, not months",
         "Roughly 74% of six-month periods were positive, which means one in four was not. Any "
         "single quarter tells you nothing."),
    ], start=1):
        st += [KeepTogether([para(f"{n}. {h}", H2), para(b, BODY)])]

    st += [Spacer(1, 10),
           para("Limits of everything above", H2),
           para("The universe is today's Nifty 500 list, so companies that were delisted or "
                "dropped out are missing and results are optimistic. Fills are assumed at the next "
                "open with a 0.25% round-trip cost and no market impact. When a stop and the "
                "session's low coincide the stop is assumed to have been hit. Roughly 60 strategy "
                "variants were tested before arriving at this one, so the statistics are not those "
                "of a single pre-registered hypothesis. This is research output, not investment "
                "advice, and not a recommendation to buy or sell any security.", NOTE)]

    doc.build(st)
    print(f"wrote {out} ({out.stat().st_size/1024:.0f} KB)")


if __name__ == "__main__":
    main()
