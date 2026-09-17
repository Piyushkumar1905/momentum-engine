"""Build the period-analysis PDF from analysis.json + sweep.json.

  python tools/build_report.py
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
from reportlab.lib.enums import TA_LEFT   # noqa: E402
from reportlab.lib.pagesizes import A4    # noqa: E402
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet   # noqa: E402
from reportlab.lib.units import mm        # noqa: E402
from reportlab.platypus import (BaseDocTemplate, Frame, Image, KeepTogether,  # noqa: E402
                                PageBreak, PageTemplate, Paragraph, Spacer, Table,
                                TableStyle)

ROOT = Path(__file__).resolve().parents[1]
IST = timezone(timedelta(hours=5, minutes=30))

INK = colors.HexColor("#12161c")
MUTED = colors.HexColor("#5b6572")
LINE = colors.HexColor("#d8dde3")
BAND = colors.HexColor("#f2f4f7")
ACCENT = colors.HexColor("#1f3864")
GOOD = colors.HexColor("#0f7a4d")
BAD = colors.HexColor("#b3261e")
WARN = colors.HexColor("#8a5a00")

S = getSampleStyleSheet()


def style(name, **kw):
    opts = dict(fontName="Helvetica", textColor=INK, fontSize=9.5, leading=14, alignment=TA_LEFT)
    opts.update(kw)                      # callers may override any default, incl. fontName
    return ParagraphStyle(name, parent=S["Normal"], **opts)


TITLE = style("t", fontName="Helvetica-Bold", fontSize=20, leading=24, spaceAfter=2)
SUB = style("s", fontSize=10.5, textColor=MUTED, spaceAfter=14)
H1 = style("h1", fontName="Helvetica-Bold", fontSize=13.5, leading=17, spaceBefore=16, spaceAfter=7,
           textColor=ACCENT)
H2 = style("h2", fontName="Helvetica-Bold", fontSize=10.5, leading=14, spaceBefore=10, spaceAfter=4)
BODY = style("b", spaceAfter=7)
NOTE = style("n", fontSize=8.5, leading=12, textColor=MUTED, spaceAfter=6)
KEY = style("k", fontName="Helvetica-Bold", fontSize=10.5, leading=15, textColor=ACCENT, spaceAfter=6)


def pct(v, d=1):
    return "-" if v is None else f"{v*100:+.{d}f}%"


def pct0(v, d=1):
    return "-" if v is None else f"{v*100:.{d}f}%"


def rs(v):
    return "-" if v is None else f"RS {v:,.0f}"


def para(t, s=BODY):
    return Paragraph(t, s)


def table(rows, widths, align=None, head=True, size=8.5):
    t = Table(rows, colWidths=widths, repeatRows=1 if head else 0, hAlign="LEFT")
    cmds = [
        ("FONT", (0, 0), (-1, -1), "Helvetica", size),
        ("TEXTCOLOR", (0, 0), (-1, -1), INK),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("LINEBELOW", (0, 0), (-1, -2), 0.4, LINE),
    ]
    if head:
        cmds += [("FONT", (0, 0), (-1, 0), "Helvetica-Bold", size),
                 ("BACKGROUND", (0, 0), (-1, 0), BAND),
                 ("LINEBELOW", (0, 0), (-1, 0), 0.8, LINE)]
    for c in (align or []):
        cmds.append(c)
    t.setStyle(TableStyle(cmds))
    return t


def chart_equity(curve, nifty, path):
    df = pd.DataFrame(curve)
    df["date"] = pd.to_datetime(df["date"])
    fig, ax = plt.subplots(figsize=(7.2, 2.5), dpi=200)
    ax.plot(df["date"], df["equity"] / df["equity"].iloc[0] * 100, lw=1.6,
            color="#1f3864", label="Strategy")
    if nifty is not None and len(nifty):
        n = nifty.reindex(df["date"], method="ffill")
        ax.plot(df["date"], n.to_numpy() / n.iloc[0] * 100, lw=1.3, color="#8a8f98",
                ls="--", label="Nifty 50")
    ax.axhline(100, color="#c8ccd2", lw=0.8)
    ax.set_ylabel("Indexed to 100", fontsize=8)
    ax.legend(fontsize=8, frameon=False, loc="lower left")
    ax.tick_params(labelsize=8)
    for s_ in ("top", "right"):
        ax.spines[s_].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def chart_lift(baseline, rows, path):
    fig, ax = plt.subplots(figsize=(7.2, 2.2), dpi=200)
    labels = [r[0] for r in rows]
    vals = [r[1] / baseline for r in rows]
    cols = ["#0f7a4d" if v >= 1 else "#b3261e" for v in vals]
    ax.barh(labels, vals, color=cols, height=0.6)
    ax.axvline(1.0, color="#12161c", lw=1.0)
    ax.set_xlabel("Big-move rate relative to any liquid stock (1.0 = no edge)", fontsize=8)
    ax.tick_params(labelsize=8)
    for s_ in ("top", "right"):
        ax.spines[s_].set_visible(False)
    for i, v in enumerate(vals):
        ax.text(v + 0.02, i, f"{v:.2f}x", va="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def build(analysis: dict, sweep: dict | None, nifty: pd.Series | None, out: Path,
          lift: dict | None) -> Path:
    tmp = out.parent / "_charts"
    tmp.mkdir(parents=True, exist_ok=True)
    p, s, w = analysis["portfolio"], analysis["all_signals"], analysis["window"]
    bm, miss = analysis["benchmark"], analysis["missed"]
    stops = analysis["mistakes"]["stops"]

    doc = BaseDocTemplate(str(out), pagesize=A4, leftMargin=17 * mm, rightMargin=17 * mm,
                          topMargin=16 * mm, bottomMargin=16 * mm,
                          title="Momentum Engine - Period Analysis",
                          author="momentum_engine")
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
    st: list = []

    # ── cover ────────────────────────────────────────────────────────────────
    st += [para("Momentum &amp; Reversal Engine", TITLE),
           para(f"Period analysis &middot; {w['start']} to {w['end']} &middot; "
                f"{w['scan_dates']} weekly scans &middot; generated "
                f"{datetime.now(IST).strftime('%d %b %Y %H:%M IST')}", SUB)]

    verdict = ("The strategy lost money over these two years and beat the index only because "
               "the index fell further. Alpha of "
               f"<b>{pct(p['alpha_vs_nifty_pct'])}</b> over {bm['years']} years is inside noise, "
               "and it was earned with a 27% drawdown.")
    st += [para(verdict, KEY)]

    head = [["Metric", "Result", "Benchmark"],
            ["Starting capital", f"{p['starting_capital']:,.0f}", "-"],
            ["Ending equity", f"{p['ending_equity']:,.0f}", "-"],
            ["Net P&L", f"{p['net_pnl']:,.0f}", "-"],
            ["Total return", pct(p["total_return_pct"]), pct(bm["nifty_return_pct"])],
            ["CAGR", pct(p["cagr_pct"]), pct(bm["nifty_cagr_pct"])],
            ["Alpha (total / CAGR)", f"{pct(p['alpha_vs_nifty_pct'])} / {pct(p['alpha_cagr_pct'])}", "-"],
            ["Max drawdown", pct(p["max_drawdown_pct"]), "-"],
            ["Capital deployed", f"{p['total_deployed']:,.0f}", f"{p['turnover_x']}x turnover"],
            ["Positions taken", f"{p['positions_taken']:,}", f"{p['skipped_no_cash']:,} skipped, no cash"]]
    st += [para("Headline", H1),
           table(head, [62 * mm, 55 * mm, 45 * mm],
                 [("ALIGN", (1, 0), (-1, -1), "RIGHT"),
                  ("TEXTCOLOR", (1, 4), (1, 6), BAD)])]
    st += [para(f"Sizing: risk {analysis['assumptions']['risk_pct']:.0%} of equity per trade across the "
                f"entry-to-stop distance, at most {analysis['assumptions']['max_positions']} open "
                f"positions and {analysis['assumptions']['max_weight']:.0%} of equity in any one name; "
                f"entry at the next session's open; {analysis['assumptions']['cost_pct']:.2%} round-trip "
                "costs. Percentage statistics below are sizing-independent and hold at any capital.",
                NOTE)]

    eq = tmp / "equity.png"
    chart_equity(p["curve"], nifty, eq)
    st += [para("Equity curve", H2), Image(str(eq), width=doc.width, height=doc.width * 0.347)]

    st += [para(f"{p['skipped_no_cash']:,} signals were skipped for lack of cash against "
                f"{p['positions_taken']:,} taken. The screen generates roughly five times more "
                "positions than a 10 lakh account can fund, so results at this capital reflect "
                "the subset that fitted, not the full signal set.", NOTE)]

    # ── trade statistics ─────────────────────────────────────────────────────
    st += [PageBreak(), para("1. Trade statistics", H1),
           para(f"All {s['trades']:,} momentum and reversal trades in the window. These are "
                "signal-level and do not depend on position size.", BODY)]

    rows = [["Measure", "Value", "Measure", "Value"],
            ["Trades", f"{s['trades']:,}", "Average hold", f"{s['avg_hold_days']} days"],
            ["Win rate", pct0(s["win_rate"]), "Median hold", f"{s['median_hold_days']:.0f} days"],
            ["Average win", pct(s["avg_win_pct"]), "Hold, winners", f"{s['avg_hold_days_winners']} days"],
            ["Average loss", pct(s["avg_loss_pct"]), "Hold, losers", f"{s['avg_hold_days_losers']} days"],
            ["Expectancy", pct(s["expectancy_pct"], 2), "Average R", f"{s['avg_R']}"],
            ["Profit factor", f"{s['profit_factor']}", "Realised reward:risk", f"{s['realised_reward_risk']} : 1"],
            ["Longest win streak", f"{s['longest_win_streak']}", "Longest loss streak",
             f"{s['longest_loss_streak']}"]]
    st += [table(rows, [42 * mm, 34 * mm, 44 * mm, 42 * mm],
                 [("ALIGN", (1, 0), (1, -1), "RIGHT"), ("ALIGN", (3, 0), (3, -1), "RIGHT"),
                  ("TEXTCOLOR", (1, 6), (1, 6), BAD), ("TEXTCOLOR", (3, 7), (3, 7), BAD)])]

    st += [para("Best and worst", H2),
           table([["", "Symbol", "Date", "Return", "R", "Held", "Exit"],
                  ["Best", s["best_trade"]["symbol"], s["best_trade"]["date"],
                   pct(s["best_trade"]["return_pct"]), f"{s['best_trade']['R']}",
                   f"{s['best_trade']['hold_days']}d", s["best_trade"]["exit"]],
                  ["Worst", s["worst_trade"]["symbol"], s["worst_trade"]["date"],
                   pct(s["worst_trade"]["return_pct"]), f"{s['worst_trade']['R']}",
                   f"{s['worst_trade']['hold_days']}d", s["worst_trade"]["exit"]]],
                 [16 * mm, 28 * mm, 26 * mm, 24 * mm, 16 * mm, 16 * mm, 24 * mm],
                 [("ALIGN", (3, 0), (-1, -1), "RIGHT"),
                  ("TEXTCOLOR", (3, 1), (3, 1), GOOD), ("TEXTCOLOR", (3, 2), (3, 2), BAD)])]
    st += [para("The best trade exited on time at only 1.19R, not at target - the 2R target was never "
                "reached despite a 34% gain, because the move came after the 20-session clock ran out. "
                "The worst was a reversal trade whose stop sat 34% below entry: reversal stops anchor to "
                "the 15-session low, which can be far away. It lost exactly 1R as designed, but 1R was "
                "a third of the position.", NOTE)]

    st += [para("Exit mix", H2),
           table([["Exit reason", "Share of trades"]] +
                 [[k, pct0(v)] for k, v in sorted(s["exit_mix"].items(), key=lambda x: -x[1])],
                 [50 * mm, 40 * mm], [("ALIGN", (1, 0), (1, -1), "RIGHT")])]

    st += [para("By model", H2),
           table([["Model", "Trades", "Win rate", "Expectancy", "Profit factor", "Avg R"]] +
                 [[m, f"{v['trades']:,}", pct0(v["win_rate"]), pct(v["expectancy_pct"], 2),
                   f"{v['profit_factor']}", f"{v['avg_R']}"]
                  for m, v in sorted(analysis["by_model"].items())],
                 [40 * mm, 22 * mm, 24 * mm, 26 * mm, 28 * mm, 20 * mm],
                 [("ALIGN", (1, 0), (-1, -1), "RIGHT")])]
    st += [para("Momentum produced 1,219 trades at a profit factor of 0.93 - it lost money. Reversal "
                "produced 143 at 1.35 and made money. The engine spends almost all of its activity on "
                "the model that does not work.", NOTE)]

    # ── mistakes ─────────────────────────────────────────────────────────────
    st += [PageBreak(), para("2. Where the money was lost", H1)]

    st += [para("2.1 A data error, not a trading error", H2),
           para("The single worst trade in the first run was Vedanta at "
                "<b>-60.3%</b>, a -10R loss. It never happened. On 30 April 2026 VEDL gapped from "
                "773.60 to 289.50 overnight: the demerger moved value to separately-listed entities. "
                "Yahoo's adjustment handles splits and dividends, not demergers, so the price series "
                "shows a collapse the shareholder never took.", BODY),
           para("Eight such discontinuities were found across the universe (VEDL, ABFRL, TMPV, TRENT, "
                "GPIL, CGCL, MOTILALOFS, PATANJALI). The engine now detects them - an overnight gap "
                "worse than -30% that never retraces - and excludes any trade spanning one, rather "
                "than inventing adjusted prices for value that left the listing. Every figure in this "
                "report is computed after that fix.", BODY)]

    st += [para("2.2 Stops", H2),
           table([["Measure", "Value"],
                  ["Trades stopped out", f"{stops['stopped_trades']:,} ({pct0(s['exit_mix'].get('stop', 0))} of all)"],
                  ["Would have reached target without the stop", f"{stops['would_have_hit_target_without_stop']} ({pct0(stops['would_have_hit_target_pct'])})"],
                  ["Closed above entry by the 20-day horizon", f"{stops['closed_above_entry_by_horizon']} ({pct0(stops['closed_above_entry_pct'])})"],
                  ["Average loss when stopped", pct(stops["avg_loss_on_stops_pct"])]],
                 [96 * mm, 60 * mm], [("ALIGN", (1, 0), (1, -1), "RIGHT")])]
    st += [para("28% of stopped trades finished above entry within the horizon, and 13% would have hit "
                "target outright. That is the measurable cost of the stop distance. It is not proof the "
                "stop is wrong: it exists to cap the other 72%, whose average loss was -5.4%. Widening "
                "it would recover some of those 207 trades and deepen the rest.", NOTE)]

    st += [para("2.3 The loss streak", H2),
           para(f"The longest run of consecutive losers was <b>{s['longest_loss_streak']}</b>, against a "
                f"best winning run of {s['longest_win_streak']}. At 37% win rate a run of 18 is not "
                "anomalous - it is what a 37% process does regularly. Any position sizing that cannot "
                "survive 18 straight losses is mis-specified, independent of the edge.", BODY)]

    # ── misses ───────────────────────────────────────────────────────────────
    st += [PageBreak(), para("3. What the filters missed", H1),
           para(f"Across the window, {miss['universe_big_movers']:,} stock-dates in the liquid universe "
                f"produced a gain of {miss['big_move_threshold']:.0%} or more within "
                f"{miss['horizon_days']} sessions. The momentum screen surfaced "
                f"{miss['captured_by_screen']:,} of them - a capture rate of "
                f"<b>{pct0(miss['capture_rate'])}</b>.", BODY)]

    if lift:
        lf = tmp / "lift.png"
        chart_lift(lift["baseline"], [("Any liquid stock", lift["baseline"]),
                                      ("Passed momentum", lift["momentum"]),
                                      ("Passed reversal", lift["reversal"])], lf)
        st += [Image(str(lf), width=doc.width, height=doc.width * 0.305)]
        st += [para(f"This is the finding that matters most. A stock picked at random from the liquid "
                    f"universe had a {lift['baseline']:.2%} chance of a +15% move. One that passed every "
                    f"momentum rule had {lift['momentum']:.2%} - <b>a lift of "
                    f"{lift['momentum']/lift['baseline']:.2f}x, meaning the momentum filter selected "
                    f"stocks slightly less likely to run than chance</b>. Reversal reached "
                    f"{lift['reversal']:.2%}, a genuine {lift['reversal']/lift['baseline']:.2f}x edge.",
                    BODY)]

    rows = [["Rule", "Blocked alone", "Big movers blocked", "Hit rate of blocked", "vs baseline"]]
    bl = lift["baseline"] if lift else None
    for r in miss["by_rule"]:
        hr = r["hit_rate_of_blocked"]
        rows.append([r["rule"], f"{r['blocked_solely']:,}", f"{r['big_movers_blocked']:,}",
                     pct0(hr) if hr is not None else "-",
                     f"{hr/bl:.2f}x" if (hr and bl) else "-"])
    st += [para("Which rule rejected the winners", H2),
           table(rows, [44 * mm, 28 * mm, 34 * mm, 32 * mm, 24 * mm],
                 [("ALIGN", (1, 0), (-1, -1), "RIGHT")])]
    st += [para("Read the last column, not the third. <b>ma_aligned</b> blocked the most big movers in "
                "absolute terms, but the pool it rejected had a below-baseline hit rate - it is working. "
                "<b>not_extended_50dma</b> and <b>near_52w_high</b> rejected pools that beat the "
                "baseline, which is the signature of a filter throwing away good stocks.", NOTE)]

    st += [para("The largest individual misses", H2),
           table([["Date", "Symbol", "20-day gain", "Rules failed"]] +
                 [[m["date"], m["symbol"], pct(m["fwd_return"]), m["failed"].replace(",", ", ")]
                  for m in miss["top_misses"][:10]],
                 [24 * mm, 26 * mm, 24 * mm, 88 * mm],
                 [("ALIGN", (2, 0), (2, -1), "RIGHT"), ("FONT", (3, 1), (3, -1), "Helvetica", 7.5)])]
    st += [para("Almost every large miss failed <b>ma_aligned</b> and <b>near_52w_high</b> together: "
                "these were stocks not yet in an established uptrend and not near their highs - "
                "early-stage recoveries. A momentum screen is designed not to buy those. They are the "
                "reversal model's territory, and reversal is the model that works.", NOTE)]

    # ── what happens if you act on it ────────────────────────────────────────
    if sweep:
        st += [PageBreak(), para("4. Acting on the misses makes it worse", H1),
               para("The obvious response to section 3 is to relax the two filters that reject good "
                    "stocks. Each variant below was judged on the second year, which never informed it. "
                    "A change has to improve both halves to be worth keeping.", BODY)]
        rows = [["Variant", "IS trades", "IS profit factor", "OOS trades", "OOS profit factor", "Verdict"]]
        base = next(v for v in sweep["variants"] if v["variant"] == "baseline")
        for v in sweep["variants"]:
            i, o = v["in_sample"], v["out_of_sample"]
            if v["variant"] == "baseline":
                verdict = "reference"
            else:
                bi, bo = base["in_sample"]["expectancy"], base["out_of_sample"]["expectancy"]
                bet_i = i["expectancy"] is not None and i["expectancy"] > bi
                bet_o = o["expectancy"] is not None and o["expectancy"] > bo
                verdict = "keep" if bet_i and bet_o else "reject"
            rows.append([v["variant"], f"{i['trades']:,}", f"{i['profit_factor']}",
                         f"{o['trades']:,}", f"{o['profit_factor']}", verdict])
        st += [table(rows, [34 * mm, 22 * mm, 30 * mm, 24 * mm, 32 * mm, 20 * mm],
                     [("ALIGN", (1, 0), (-2, -1), "RIGHT")])]
        st += [para("<b>Every relaxation was rejected.</b> The miss analysis counts only the winners a "
                    "filter blocked; it is silent on the losers it blocked at the same time. Relaxing "
                    "admits both, and the losers dominate. Note also that trade counts barely move - "
                    "613 to 615 - because the score ranking already demotes those stocks, so the filter "
                    "rarely changes the final twenty. A rule can look expensive at universe level and "
                    "be nearly irrelevant to what actually gets bought.", BODY),
               para("No thresholds were changed as a result of this analysis. That is the correct "
                    "outcome, not a failure to act.", KEY)]

    # ── conclusions ──────────────────────────────────────────────────────────
    st += [PageBreak(), para("5. What to do", H1)]
    for n, (h, b) in enumerate([
        ("Stop trading the momentum model at full size",
         "1,219 trades at a 0.93 profit factor and a 0.86x selection lift. Two years is enough "
         "evidence to say it is not working as specified, and it consumes 90% of the activity."),
        ("Investigate widening the reversal model instead",
         "143 trades, 1.35 profit factor, 1.35x lift, and it held up in both halves of the full "
         "backtest. Its weakness is sample size, not quality. Loosening reversal's entry conditions "
         "is the one change with an evidential basis - and it must clear the same out-of-sample gate."),
        ("Cap reversal stop distance",
         "Reversal stops anchor to the 15-session low and reached 34% of entry. A 1R loss is survivable; "
         "a 1R loss that is a third of the position is a sizing problem. Cap risk per trade at a "
         "fixed fraction of price and skip signals that exceed it."),
        ("Keep the corporate-action guard and extend it",
         "One unadjusted demerger created a phantom -10R loss. Eight were found. Any vendor feed will "
         "keep producing these; the detector should run on every refresh and the flagged list reviewed."),
        ("Do not tune on this report",
         "Section 4 is the demonstration. Every change suggested by the miss data failed validation. "
         "Treat section 3 as a description of the strategy's blind spot, not a to-do list."),
    ], start=1):
        st += [KeepTogether([para(f"{n}. {h}", H2), para(b, BODY)])]

    st += [Spacer(1, 8), para(
        "Limits of this report. The universe is today's Nifty 500, so companies that dropped out or "
        "delisted are absent and every figure is optimistic. Fills are assumed at the next open with "
        "no slippage beyond the cost assumption, no circuit limits and no impact. When a stop and "
        "target fall on the same day the stop is assumed. The window covers a falling market - the "
        "Nifty lost 7.7% - and momentum strategies are known to struggle in those conditions, so this "
        "is not a verdict on the approach across all regimes. Research output, not investment advice.",
        NOTE)]

    doc.build(st)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--analysis", default=str(ROOT / "output" / "analysis" / "analysis.json"))
    ap.add_argument("--sweep", default=str(ROOT / "output" / "analysis" / "sweep.json"))
    ap.add_argument("--lift", default=str(ROOT / "output" / "analysis" / "lift.json"))
    ap.add_argument("--out", default=str(ROOT / "output" / "analysis" /
                                         "momentum_engine_period_report.pdf"))
    a = ap.parse_args()

    analysis = json.loads(Path(a.analysis).read_text(encoding="utf-8"))
    sweep = json.loads(Path(a.sweep).read_text(encoding="utf-8")) if Path(a.sweep).exists() else None
    lift = json.loads(Path(a.lift).read_text(encoding="utf-8")) if Path(a.lift).exists() else None

    nifty = None
    nf = ROOT / "output" / "analysis" / "nifty.csv"
    if nf.exists():
        nifty = pd.read_csv(nf, parse_dates=["date"]).set_index("date")["close"]

    out = build(analysis, sweep, nifty, Path(a.out), lift)
    print(f"wrote {out} ({out.stat().st_size/1024:.0f} KB)")


if __name__ == "__main__":
    main()
