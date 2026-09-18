"""What separated the winning trades from the losing ones?

Joins every simulated trade back to what the stock looked like at entry, then compares
the two groups.

A warning that governs how the output should be read: this comparison is descriptive,
not prescriptive. Splitting finished trades by outcome and hunting for differences is
the purest form of hindsight - the winners are winners BECAUSE they went up, and any
characteristic correlated with going up will show a difference. A real discriminator
has to survive being applied as a filter on data it did not come from, which the
closing section actually tests rather than assumes.

  python tools/win_loss_autopsy.py --model trend_lowbeta --start 2024-09-01
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from engine import backtest as bt          # noqa: E402
from engine import data                    # noqa: E402

log = logging.getLogger("autopsy")

# Entry characteristics worth comparing, with a readable label and a formatter.
FEATURES = [
    ("mom_6m_skip", "6M momentum at entry", "pct"),
    ("ret_20", "1M return at entry", "pct"),
    ("rs_63", "RS vs Nifty (63d)", "pct"),
    ("dist_high", "% of 52-week high", "pct_raw"),
    ("dist_low", "x above 52-week low", "num"),
    ("atr_pct", "daily ATR as % of price", "pct"),
    ("ext_atr", "ATR above 20DMA", "num"),
    ("ext_50", "% above 50DMA", "pct"),
    ("rvol", "relative volume", "num"),
    ("sector_rank", "sector strength percentile", "pct_raw"),
    ("med_value_cr", "20d median traded value (Rs cr)", "num"),
]


def fmt(v, kind):
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "-"
    if kind == "pct":
        return f"{v * 100:+.1f}%"
    if kind == "pct_raw":
        return f"{v * 100:.0f}%"
    return f"{v:.2f}"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(ROOT / "config.yaml"))
    ap.add_argument("--model", default="trend_lowbeta")
    ap.add_argument("--start", default="2024-09-01")
    ap.add_argument("--end", default=None)
    ap.add_argument("--out", default=str(ROOT / "output" / "analysis_final" / "win_loss.json"))
    a = ap.parse_args()
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")

    cfg = yaml.safe_load(open(a.config))
    cfg["backtest"]["start"] = a.start
    # Load the FULL panel first. backtest.end also bounds the download range, so setting
    # it here made the loader re-request every symbol that had not yet listed by the
    # analysis end date - hundreds of failed fetches for no benefit.
    cfg["backtest"]["end"] = None
    uni = data.load_universe(cfg["run"]["universe_file"], cfg["run"].get("max_symbols"))
    panel = data.load_panel(cfg, uni)
    if a.end:
        cfg["backtest"]["end"] = a.end          # bound the scan dates, not the download
    res = bt.run_backtest(panel, uni, cfg)

    tr = res["trades"]
    sg = res["signals"]
    tr = tr[tr["model"] == a.model].copy()
    sg = sg[sg["model"] == a.model].copy()
    if tr.empty:
        raise SystemExit(f"no trades for {a.model}")

    keys = ["date", "model", "symbol"]
    cols = keys + [c for c, _, _ in FEATURES if c in sg.columns] + ["industry", "rank"]
    df = tr.merge(sg[cols].drop_duplicates(keys), on=keys, how="left")
    df["win"] = df["ret"] > 0
    w, l = df[df["win"]], df[~df["win"]]

    out = {"model": a.model, "window": [a.start, a.end or str(panel.dates[-1].date())],
           "trades": len(df), "winners": len(w), "losers": len(l),
           "win_rate": round(len(w) / len(df), 4)}

    print(f"\n{'='*78}\n{a.model}   {a.start} -> {out['window'][1]}   "
          f"{len(df)} trades, {len(w)} winners / {len(l)} losers\n{'='*78}")

    # ── how trades ended ─────────────────────────────────────────────────────
    print("\nHOW TRADES ENDED")
    print(f"{'exit':<12}{'count':>7}{'share':>8}{'win rate':>10}{'avg return':>12}{'avg hold':>10}")
    ex_rows = []
    for r, g in df.groupby("exit_reason"):
        ex_rows.append({"exit": r, "n": len(g), "share": len(g) / len(df),
                        "win_rate": float((g["ret"] > 0).mean()),
                        "avg_ret": float(g["ret"].mean()),
                        "avg_hold": float(g["hold_days"].mean())})
    for r in sorted(ex_rows, key=lambda x: -x["n"]):
        print(f"{r['exit']:<12}{r['n']:>7}{r['share']:>8.1%}{r['win_rate']:>10.1%}"
              f"{r['avg_ret']:>+12.1%}{r['avg_hold']:>10.1f}")
    out["exits"] = ex_rows

    # ── the shape of the outcome distribution ────────────────────────────────
    print("\nWHERE THE MONEY CAME FROM")
    s = df["ret"].sort_values(ascending=False)
    tot = float(s.sum())
    gains = float(s[s > 0].sum())
    losses = float(-s[s < 0].sum())
    # Expressing the top trades as a share of the NET total gives absurd figures when
    # the net is near zero (403 points of gain against a net of 29 reads as "1373%").
    # Share of gross gains is the meaningful quantity.
    print(f"  gross gains {gains:+.2f} | gross losses -{losses:.2f} | net {tot:+.2f} "
          f"(summed trade returns)")
    for k in (5, 10, 20):
        print(f"  top {k:>2} trades = {s.head(k).sum():+.2f}, "
              f"{s.head(k).sum() / gains:.0%} of all gains")
    need = int((s.cumsum() >= losses).argmax()) + 1 if (s.cumsum() >= losses).any() else None
    out["concentration"] = {"gross_gains": round(gains, 3), "gross_losses": round(losses, 3),
                            "net": round(tot, 3),
                            **{f"top_{k}_share_of_gains": round(float(s.head(k).sum() / gains), 3)
                               for k in (5, 10, 20)},
                            "trades_to_cover_all_losses": need}
    print(f"  best trade {s.iloc[0]:+.1%} | worst {s.iloc[-1]:+.1%} | "
          f"median {s.median():+.1%}")
    print(f"  the best {need} trades alone cover every loss the strategy took "
          f"({need / len(df):.1%} of all trades)")

    # ── entry characteristics ────────────────────────────────────────────────
    print("\nWHAT WINNERS AND LOSERS LOOKED LIKE AT ENTRY (medians)")
    print(f"{'characteristic':<34}{'winners':>12}{'losers':>12}{'gap':>10}")
    feat_rows = []
    for c, label, kind in FEATURES:
        if c not in df.columns:
            continue
        mw, ml = w[c].median(), l[c].median()
        if not np.isfinite(mw) or not np.isfinite(ml):
            continue
        # standardised gap, so characteristics on different scales are comparable
        sd = df[c].std()
        z = (mw - ml) / sd if sd and np.isfinite(sd) else np.nan
        feat_rows.append({"feature": c, "label": label, "winners": float(mw),
                          "losers": float(ml), "z": float(z) if np.isfinite(z) else None})
        print(f"{label:<34}{fmt(mw, kind):>12}{fmt(ml, kind):>12}"
              f"{(f'{z:+.2f}' if np.isfinite(z) else '-'):>10}")
    out["features"] = feat_rows
    print("  gap = (winner median - loser median) / standard deviation. |gap| under about")
    print("  0.2 is noise; this is a post-hoc split, so treat even large gaps as descriptive.")

    # ── regime ───────────────────────────────────────────────────────────────
    if "regime" in df.columns:
        print("\nBY MARKET REGIME AT ENTRY")
        print(f"{'regime':<12}{'trades':>8}{'win rate':>10}{'avg return':>12}{'total':>10}")
        reg_rows = []
        for r, g in df.groupby("regime"):
            reg_rows.append({"regime": r, "n": len(g), "win_rate": float((g["ret"] > 0).mean()),
                             "avg_ret": float(g["ret"].mean()), "sum": float(g["ret"].sum())})
        for r in sorted(reg_rows, key=lambda x: -x["n"]):
            print(f"{r['regime']:<12}{r['n']:>8}{r['win_rate']:>10.1%}"
                  f"{r['avg_ret']:>+12.1%}{r['sum']:>+10.1f}")
        out["regime"] = reg_rows

    # ── rank within the list ─────────────────────────────────────────────────
    if "rank" in df.columns and df["rank"].notna().any():
        print("\nDOES THE SCORE RANK PREDICT THE OUTCOME?")
        df["rank_band"] = pd.cut(df["rank"], [0, 5, 10, 15, 20],
                                 labels=["1-5", "6-10", "11-15", "16-20"])
        print(f"{'rank':<12}{'trades':>8}{'win rate':>10}{'avg return':>12}")
        rk = []
        for r, g in df.groupby("rank_band", observed=True):
            rk.append({"band": str(r), "n": len(g), "win_rate": float((g["ret"] > 0).mean()),
                       "avg_ret": float(g["ret"].mean())})
            print(f"{str(r):<12}{len(g):>8}{(g['ret'] > 0).mean():>10.1%}"
                  f"{g['ret'].mean():>+12.1%}")
        out["by_rank"] = rk

    # ── holding time ─────────────────────────────────────────────────────────
    print("\nHOLDING TIME")
    print(f"  winners held {w['hold_days'].mean():.1f} days on average, "
          f"losers {l['hold_days'].mean():.1f}")
    print(f"  median winner {w['hold_days'].median():.0f} days, "
          f"median loser {l['hold_days'].median():.0f}")
    out["hold"] = {"winner_mean": float(w["hold_days"].mean()),
                   "loser_mean": float(l["hold_days"].mean()),
                   "winner_median": float(w["hold_days"].median()),
                   "loser_median": float(l["hold_days"].median())}

    # ── worst and best, named ────────────────────────────────────────────────
    print("\nTHE FIVE WORST TRADES")
    cols_show = ["date", "symbol", "industry", "ret", "r_multiple", "hold_days", "exit_reason"]
    cols_show = [c for c in cols_show if c in df.columns]
    worst = df.nsmallest(5, "ret")[cols_show]
    print(worst.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print("\nTHE FIVE BEST TRADES")
    best = df.nlargest(5, "ret")[cols_show]
    print(best.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    out["worst"] = worst.astype(str).to_dict(orient="records")
    out["best"] = best.astype(str).to_dict(orient="records")

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=1, default=str), encoding="utf-8")
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
