"""Summary PDF for the final strategy, in the same shape as the period report.

Pulls together the two portfolio runs, the six-month block distribution, the
parameter-stability maps and the volatility-matched check into one document.

  python tools/build_final_report.py
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
import pandas as pd                       # noqa: E402
from reportlab.lib import colors          # noqa: E402
from reportlab.lib.pagesizes import A4    # noqa: E402
from reportlab.lib.units import mm        # noqa: E402
from reportlab.platypus import (BaseDocTemplate, Frame, Image, KeepTogether,   # noqa: E402
                                PageBreak, PageTemplate, Spacer)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.build_report import (ACCENT, BAD, BODY, GOOD, H1, H2, KEY, MUTED,   # noqa: E402
                                NOTE, SUB, TITLE, pct, pct0, para, table)

IST = timezone(timedelta(hours=5, minutes=30))


def chart_blocks(blocks, path):
    df = pd.DataFrame(blocks)
    fig, ax = plt.subplots(figsize=(7.2, 2.4), dpi=200)
    cols = ["#0f7a4d" if v > 0 else "#b3261e" for v in df["mean_edge"]]
    ax.bar(range(len(df)), df["mean_edge"] * 100, color=cols, width=0.75)
    ax.axhline(0, color="#12161c", lw=0.9)
    ax.set_ylabel("Edge per 60 sessions (%)", fontsize=8)
    ax.set_xlabel("Consecutive six-month blocks, 2009 to 2026", fontsize=8)
    ax.set_xticks([])
    ax.tick_params(labelsize=8)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def chart_equity(curve, nifty, path, label):
    df = pd.DataFrame(curve)
    df["date"] = pd.to_datetime(df["date"])
    fig, ax = plt.subplots(figsize=(7.2, 2.4), dpi=200)
    ax.plot(df["date"], df["equity"] / df["equity"].iloc[0] * 100, lw=1.6,
            color="#1f3864", label=label)
    if nifty is not None and len(nifty):
        n = nifty.reindex(df["date"], method="ffill")
        ax.plot(df["date"], n.to_numpy() / n.iloc[0] * 100, lw=1.3, color="#8a8f98",
                ls="--", label="Nifty 50")
    ax.axhline(100, color="#c8ccd2", lw=0.8)
    ax.set_ylabel("Indexed to 100", fontsize=8)
    ax.legend(fontsize=8, frameon=False, loc="upper left")
    ax.tick_params(labelsize=8)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def portfolio_table(p, bm, label):
    return [["Metric", label, "Nifty 50"],
            ["Starting capital", f"{p['starting_capital']:,.0f}", "-"],
            ["Ending equity", f"{p['ending_equity']:,.0f}", "-"],
            ["Net P&L", f"{p['net_pnl']:,.0f}", "-"],
            ["Total return", pct(p["total_return_pct"]), pct(bm["nifty_return_pct"])],
            ["CAGR", pct(p["cagr_pct"]), pct(bm["nifty_cagr_pct"])],
            ["Alpha (total / CAGR)",
             f"{pct(p['alpha_vs_nifty_pct'])} / {pct(p['alpha_cagr_pct'])}", "-"],
            ["Max drawdown", pct(p["max_drawdown_pct"]), "-"],
            ["Capital deployed", f"{p['total_deployed']:,.0f}", f"{p['turnover_x']}x turnover"],
            ["Positions taken", f"{p['positions_taken']:,}", f"{p['skipped_no_cash']:,} skipped, no cash"]]


def stat_table(s):
    return [["Measure", "Value", "Measure", "Value"],
            ["Trades", f"{s['trades']:,}", "Average hold", f"{s['avg_hold_days']} days"],
            ["Win rate", pct0(s["win_rate"]), "Median hold", f"{s['median_hold_days']:.0f} days"],
            ["Average win", pct(s["avg_win_pct"]), "Hold, winners", f"{s['avg_hold_days_winners']} days"],
            ["Average loss", pct(s["avg_loss_pct"]), "Hold, losers", f"{s['avg_hold_days_losers']} days"],
            ["Expectancy", pct(s["expectancy_pct"], 2), "Average R", f"{s['avg_R']}"],
            ["Profit factor", f"{s['profit_factor']}", "Realised reward:risk",
             f"{s['realised_reward_risk']} : 1"],
            ["Longest win streak", f"{s['longest_win_streak']}", "Longest loss streak",
             f"{s['longest_loss_streak']}"]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "output" / "analysis_final" /
                                         "final_strategy_report.pdf"))
    a = ap.parse_args()

    R = ROOT / "output"
    recent = json.loads((R / "analysis_final" / "analysis.json").read_text(encoding="utf-8"))
    back = json.loads((R / "analysis_backward" / "analysis.json").read_text(encoding="utf-8"))
    wf = json.loads((R / "research" / "lb_h60_full.json").read_text(encoding="utf-8"))
    vm = json.loads((R / "research" / "vol_matched.json").read_text(encoding="utf-8"))
    sw = json.loads((R / "research" / "sweeps.json").read_text(encoding="utf-8"))
    spec = json.loads((ROOT / "specs" / "RECOMMENDED.json").read_text(encoding="utf-8"))

    out = Path(a.out)
    tmp = out.parent / "_charts"
    tmp.mkdir(parents=True, exist_ok=True)

    doc = BaseDocTemplate(str(out), pagesize=A4, leftMargin=17 * mm, rightMargin=17 * mm,
                          topMargin=16 * mm, bottomMargin=16 * mm,
                          title="Final strategy - summary", author="momentum_engine")
    frame = Frame(doc.leftMargin, doc.bottomMargin, doc.width, doc.height, id="n")

    def footer(canv, d):
        canv.saveState()
        canv.setFont("Helvetica", 7.5)
        canv.setFillColor(MUTED)
        canv.drawString(doc.leftMargin, 10 * mm,
                        "momentum_engine - research output, not investment advice")
        canv.drawRightString(A4[0] - doc.rightMargin, 10 * mm, f"Page {d.page}")
        canv.restoreState()

    doc.addPageTemplates([PageTemplate(id="all", frames=[frame], onPage=footer)])
    st = []

    st += [para("Final strategy", TITLE),
           para(f"12-month momentum with a low-beta tilt &middot; generated "
                f"{datetime.now(IST).strftime('%d %b %Y %H:%M IST')}", SUB)]

    st += [para("The rule", H1),
           table([["Component", "Setting"],
                  ["Rank by", "12-month momentum, skipping the most recent month"],
                  ["Tilt", "25% weight toward low beta (252-day)"],
                  ["Universe", "liquid screen only - no volatility cap, no 200DMA filter"],
                  ["Positions", f"top {spec['top_n']}, weekly rescan"],
                  ["Stop", f"{spec['exit']['stop_atr']} ATR below entry, fixed at entry"],
                  ["Trail", f"{spec['exit']['trail_atr']} ATR below the highest close"],
                  ["Target", "none - the trail or the clock decides"],
                  ["Time exit", f"{spec['exit']['max_hold']} sessions"]],
                 [42 * mm, 120 * mm])]

    # ── the two portfolio runs ───────────────────────────────────────────────
    st += [para("Did it make money?", H1),
           para("Two runs, both at 10 lakh starting capital, 1% equity risk per trade, at most "
                "10 open positions, entry at the next open, 0.25% round-trip costs. The first "
                "window is the one the previous report covered, so the two are directly "
                "comparable. The second is a decade the parameters were never fitted to.", BODY)]

    st += [para("A. Same window as the previous report (2024-09 to 2026-09)", H2),
           table(portfolio_table(recent["portfolio"], recent["benchmark"], "Strategy"),
                 [60 * mm, 54 * mm, 48 * mm],
                 [("ALIGN", (1, 0), (-1, -1), "RIGHT"),
                  ("TEXTCOLOR", (1, 4), (1, 6), GOOD)])]
    st += [para("The previous rules lost 6.8% over this window and beat the index only because "
                "the index fell further. This makes 4.4% while the Nifty loses 7.9%, with the "
                "maximum drawdown cut from 27% to 18.5%.", NOTE)]

    eq = tmp / "eq_recent.png"
    nf = R / "analysis" / "nifty.csv"
    nifty = (pd.read_csv(nf, parse_dates=["date"]).set_index("date")["close"]
             if nf.exists() else None)
    chart_equity(recent["portfolio"]["curve"], nifty, eq, "Strategy")
    st += [Image(str(eq), width=doc.width, height=doc.width * 0.333)]

    st += [PageBreak(), para("B. The decade the parameters never saw (2009-01 to 2020-03)", H2),
           table(portfolio_table(back["portfolio"], back["benchmark"], "Strategy"),
                 [60 * mm, 54 * mm, 48 * mm],
                 [("ALIGN", (1, 0), (-1, -1), "RIGHT")])]
    st += [para("This is the honest test: the parameters were chosen on 2021-2026 and this data "
                "was only fetched afterwards. Over 11.2 years it turns 10 lakh into 38.8 lakh, "
                "against 36.8 lakh for the index - a CAGR of 12.9% versus 12.4%. The edge over "
                "the index is real but <b>modest: about half a percent a year</b>, far smaller "
                "than window A suggests.", BODY)]

    # ── trade statistics ─────────────────────────────────────────────────────
    st += [para("Trade statistics", H1),
           para("Signal-level, independent of position size.", BODY),
           para("Window A - 2024-09 to 2026-09", H2),
           table(stat_table(recent["all_signals"]),
                 [42 * mm, 34 * mm, 44 * mm, 42 * mm],
                 [("ALIGN", (1, 0), (1, -1), "RIGHT"), ("ALIGN", (3, 0), (3, -1), "RIGHT")]),
           para("Window B - 2009 to 2020", H2),
           table(stat_table(back["all_signals"]),
                 [42 * mm, 34 * mm, 44 * mm, 42 * mm],
                 [("ALIGN", (1, 0), (1, -1), "RIGHT"), ("ALIGN", (3, 0), (3, -1), "RIGHT")])]

    st += [para("Best and worst single trades", H2),
           table([["Window", "", "Symbol", "Date", "Return", "R", "Held"],
                  ["A", "Best", recent["all_signals"]["best_trade"]["symbol"],
                   recent["all_signals"]["best_trade"]["date"],
                   pct(recent["all_signals"]["best_trade"]["return_pct"]),
                   f"{recent['all_signals']['best_trade']['R']}",
                   f"{recent['all_signals']['best_trade']['hold_days']}d"],
                  ["A", "Worst", recent["all_signals"]["worst_trade"]["symbol"],
                   recent["all_signals"]["worst_trade"]["date"],
                   pct(recent["all_signals"]["worst_trade"]["return_pct"]),
                   f"{recent['all_signals']['worst_trade']['R']}",
                   f"{recent['all_signals']['worst_trade']['hold_days']}d"],
                  ["B", "Best", back["all_signals"]["best_trade"]["symbol"],
                   back["all_signals"]["best_trade"]["date"],
                   pct(back["all_signals"]["best_trade"]["return_pct"]),
                   f"{back['all_signals']['best_trade']['R']}",
                   f"{back['all_signals']['best_trade']['hold_days']}d"],
                  ["B", "Worst", back["all_signals"]["worst_trade"]["symbol"],
                   back["all_signals"]["worst_trade"]["date"],
                   pct(back["all_signals"]["worst_trade"]["return_pct"]),
                   f"{back['all_signals']['worst_trade']['R']}",
                   f"{back['all_signals']['worst_trade']['hold_days']}d"]],
                 [18 * mm, 18 * mm, 28 * mm, 26 * mm, 24 * mm, 16 * mm, 16 * mm],
                 [("ALIGN", (4, 0), (-1, -1), "RIGHT"),
                  ("TEXTCOLOR", (4, 1), (4, 1), GOOD), ("TEXTCOLOR", (4, 2), (4, 2), BAD),
                  ("TEXTCOLOR", (4, 3), (4, 3), GOOD), ("TEXTCOLOR", (4, 4), (4, 4), BAD)])]
    st += [para("Note the asymmetry the trailing stop produces: the best trades run to +100% or "
                "more while the worst are capped near -25%. That is where a 35-43% win rate still "
                "produces a positive expectancy. It also means long losing runs are normal - the "
                "longest was 28 consecutive losers in window A and 30 in window B.", NOTE)]

    # ── consistency ──────────────────────────────────────────────────────────
    st += [PageBreak(), para("Consistency over 16.7 years", H1)]
    b = wf["blocks"]
    s = wf["summary"]
    ch = tmp / "blocks.png"
    chart_blocks(b, ch)
    st += [Image(str(ch), width=doc.width, height=doc.width * 0.333)]
    st += [table([["Measure", "Value"],
                  ["Six-month blocks", f"{s['blocks']}"],
                  ["Blocks positive", f"{s['blocks_positive']} ({pct0(s['share_blocks_positive'])})"],
                  ["Mean block edge", pct(s["mean_block_edge"], 2)],
                  ["Median block edge", pct(s["median_block_edge"], 2)],
                  ["Best block", pct(s["best_block"], 2)],
                  ["Worst block", pct(s["worst_block"], 2)],
                  ["Newey-West t (overlap adjusted)", f"{s['full_nw_t']}"]],
                 [70 * mm, 45 * mm], [("ALIGN", (1, 0), (1, -1), "RIGHT")])]
    st += [para("Edge here is measured against <b>every eligible name on the same dates</b> under "
                "the same exit policy, so market direction is removed. Overlapping holds are not "
                "independent observations, so the t-statistic is Newey-West at the overlap lag.", NOTE)]

    st += [para("Is it skill, or just holding riskier stocks?", H2),
           table([["Benchmark", "Mean edge", "Newey-West t"],
                  ["Whole eligible universe", pct(vm["edge_raw"], 2), f"{vm['raw_nw_t']}"],
                  ["Same volatility decile", pct(vm["edge_vol_matched"], 2), f"{vm['matched_nw_t']}"]],
                 [70 * mm, 40 * mm, 35 * mm], [("ALIGN", (1, 0), (-1, -1), "RIGHT")])]
    st += [para(f"The picks carry {vm['vol_ratio']}x the universe's volatility. Benchmarked against "
                f"stocks of similar volatility on the same dates, <b>{vm['share_retained']:.0%} of "
                "the edge survives</b> and the t-statistic improves. A volatility premium would "
                "have collapsed here.", NOTE)]

    # ── parameters ───────────────────────────────────────────────────────────
    st += [para("Why these parameter values", H1),
           para("Each was taken from the middle of a plateau. A value that scores well while its "
                "neighbours score badly has been fitted to the sample.", BODY)]
    for axis, title in [("max_hold", "Holding period (sessions)"), ("top_n", "Number of positions")]:
        rows = [["Value", "Mean edge", "NW t", "% blocks positive", "Worst block"]]
        for r in sw["sweeps"][axis]:
            if not r.get("scans"):
                continue
            rows.append([f"{r['value']}", pct(r["mean_edge"], 2), f"{r['nw_t']}",
                         pct0(r["pct_blocks_positive"]), pct(r["worst_block"], 2)])
        st += [para(title, H2),
               table(rows, [26 * mm, 32 * mm, 22 * mm, 38 * mm, 30 * mm],
                     [("ALIGN", (1, 0), (-1, -1), "RIGHT")])]
    st += [para("The volatility cap that an earlier draft recommended was removed for failing this "
                "test: it scored t=3.02 at 0.35 while 0.30 gave 1.16 and 0.45 gave 1.48. A spike, "
                "not a plateau. Removing it improved results.", NOTE)]

    # ── honest limits ────────────────────────────────────────────────────────
    st += [PageBreak(), para("What to expect, and what not to", H1)]
    for n, (h, t_) in enumerate([
        ("Roughly half a percent a year over the index, not twelve",
         "Window A's +12.3% alpha covers two years that overlap the data the parameters were "
         "chosen from. Window B - a clean decade - gives 12.9% CAGR against the index's 12.4%. "
         "Believe the second number."),
        ("Long losing runs are the normal operating state",
         "28 and 30 consecutive losers in the two windows, at a 35-43% win rate. The method "
         "depends on a handful of trades running to +100% while losers are capped near -25%. Any "
         "sizing or temperament that cannot survive 30 straight losses will not survive this."),
        ("Drawdowns of 20-30% are expected",
         "-18.5% in window A and -30.3% in window B, both at 1% risk per trade with at most ten "
         "open positions. Larger positions scale this proportionally."),
        ("Momentum crashes are the known failure mode",
         "2024-2025 was one: the untilted signal lost 0.80% per 60 days through it. The low-beta "
         "tilt turned that window positive and halved block dispersion, but it does not remove "
         "the risk."),
        ("The search itself is not free",
         "Roughly 60 specifications were tested to arrive here. The out-of-sample decade is the "
         "strongest defence against that, but a t-statistic chosen from 60 candidates is not the "
         "same as one computed from a single pre-registered hypothesis."),
        ("Survivorship bias is present and grows with history",
         "The universe is today's Nifty 500, so window B is more affected than window A. That "
         "cuts against the cleaner result, which is a reason to treat +0.5% CAGR as an upper "
         "bound rather than a floor."),
    ], start=1):
        st += [KeepTogether([para(f"{n}. {h}", H2), para(t_, BODY)])]

    st += [Spacer(1, 8), para(
        "The site publishes this strategy's picks daily and tracks them forward from September "
        "2026. That record cannot be refitted, and within a year it will be better evidence than "
        "anything in this document. Research output, not investment advice.", NOTE)]

    doc.build(st)
    print(f"wrote {out} ({out.stat().st_size/1024:.0f} KB)")


if __name__ == "__main__":
    main()
