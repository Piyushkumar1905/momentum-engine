"""Collect every number the detailed report needs, in one pass.

Produces facts.json: the selection funnel, trade frequency, risk per trade in rupees,
stop and exit distributions, and the winner/loser comparison - so the report generator
only formats numbers rather than computing them.

  python tools/collect_report_facts.py --config config_research.yaml --start 2009-01-01
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
from engine import data, rules             # noqa: E402
from engine.factor_model import build_score, load_spec   # noqa: E402

log = logging.getLogger("facts")


def funnel(panel, F, base, spec, i) -> dict:
    """How many stocks survive each stage on one scan date."""
    total = len(panel.symbols)
    have_data = int(np.isfinite(panel.close.to_numpy(float)[i]).sum())
    hist = int((F["history"].to_numpy(float)[i] >= 252).sum())
    liq = int(((F["history"].to_numpy(float)[i] >= 252) &
               (F["med_value_cr"].to_numpy(float)[i] >= 10)).sum())
    elig = int(base[i].sum())
    score = build_score(spec, panel, F, base)
    scored = int(np.isfinite(score[i]).sum())
    return {"date": str(panel.dates[i].date()), "in_index": total, "with_price_data": have_data,
            "enough_history": hist, "liquid_enough": liq, "passed_all_hygiene": elig,
            "scored": scored, "selected": min(int(spec.get("top_n", 20)), scored)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=str(ROOT / "config.yaml"))
    ap.add_argument("--start", default="2009-01-01")
    ap.add_argument("--end", default=None)
    ap.add_argument("--capital", type=float, default=1_000_000)
    ap.add_argument("--risk-pct", type=float, default=0.01)
    ap.add_argument("--out", default=str(ROOT / "output" / "analysis_final" / "facts.json"))
    a = ap.parse_args()
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")

    cfg = yaml.safe_load(open(a.config))
    cfg["backtest"]["start"] = a.start
    cfg["backtest"]["end"] = None
    uni = data.load_universe(cfg["run"]["universe_file"], cfg["run"].get("max_symbols"))
    panel = data.load_panel(cfg, uni)
    if a.end:
        cfg["backtest"]["end"] = a.end
    prep = bt.prepare(panel, uni, cfg)
    spec = load_spec(cfg["run"]["strategy_spec"])
    F, base = prep["F"], prep["base"].to_numpy(bool)
    res = bt.run_backtest(panel, uni, cfg, prep)

    tr = res["trades"]
    tr = tr[tr["model"] == spec["name"]].copy()
    sg = res["signals"]
    sg = sg[sg["model"] == spec["name"]].copy()
    tr["date"] = pd.to_datetime(tr["date"])
    sg["date"] = pd.to_datetime(sg["date"])

    facts: dict = {"spec": spec, "window": [a.start, a.end or str(panel.dates[-1].date())],
                   "capital": a.capital, "risk_pct": a.risk_pct}

    # ── selection funnel, averaged and on the latest date ────────────────────
    idxs = np.where((panel.dates >= pd.Timestamp(a.start)))[0][::5]
    facts["funnel_latest"] = funnel(panel, F, base, spec, int(idxs[-1]))
    sample = [funnel(panel, F, base, spec, int(i)) for i in idxs[::10]]
    facts["funnel_average"] = {k: (int(np.mean([s[k] for s in sample]))
                                   if k != "date" else f"average of {len(sample)} scans")
                               for k in sample[0]}

    # ── what the filters actually cost ───────────────────────────────────────
    H = F["history"].to_numpy(float)
    V = F["med_value_cr"].to_numpy(float)
    C = F["close"].to_numpy(float)
    drop = {"too little history (<252 sessions)": [], "too illiquid (<Rs 10cr/day)": [],
            "price below Rs 50": []}
    for i in idxs[::10]:
        live = np.isfinite(C[i])
        drop["too little history (<252 sessions)"].append(int((live & (H[i] < 252)).sum()))
        drop["too illiquid (<Rs 10cr/day)"].append(int((live & (H[i] >= 252) & (V[i] < 10)).sum()))
        drop["price below Rs 50"].append(
            int((live & (H[i] >= 252) & (V[i] >= 10) & (C[i] < 50)).sum()))
    facts["filter_cost"] = {k: int(np.mean(v)) for k, v in drop.items()}

    # ── trade frequency ──────────────────────────────────────────────────────
    scan_dates = sorted(sg["date"].unique())
    years = (pd.Timestamp(scan_dates[-1]) - pd.Timestamp(scan_dates[0])).days / 365.25
    per_scan = tr.groupby("date").size()
    facts["frequency"] = {
        "scan_dates": len(scan_dates), "years": round(years, 2),
        "scans_per_year": round(len(scan_dates) / years, 1),
        "signals_total": len(sg), "trades_total": len(tr),
        "new_trades_per_scan_mean": round(float(per_scan.mean()), 1),
        "new_trades_per_scan_median": float(per_scan.median()),
        "trades_per_year": round(len(tr) / years, 1),
        "trades_per_month": round(len(tr) / (years * 12), 1),
        "trading_days_per_year": 250,
        "new_trades_per_trading_day": round(len(tr) / (years * 250), 2),
    }
    # concurrent positions: a trade is open from entry to exit. run_backtest records
    # hold_days rather than an exit date, so derive it from the trading calendar.
    pos_of = {d: k for k, d in enumerate(panel.dates)}
    tr["exit_dt"] = [panel.dates[min(pos_of[pd.Timestamp(d)] + int(h), len(panel.dates) - 1)]
                     for d, h in zip(tr["date"], tr["hold_days"])]
    days = pd.date_range(tr["date"].min(), tr["exit_dt"].max(), freq="B")
    open_count = pd.Series(0, index=days)
    for _, r in tr.iterrows():
        open_count.loc[r["date"]:r["exit_dt"]] += 1
    facts["frequency"]["concurrent_positions_mean"] = round(float(open_count.mean()), 1)
    facts["frequency"]["concurrent_positions_max"] = int(open_count.max())

    # ── risk per trade ───────────────────────────────────────────────────────
    tr["risk_frac"] = (tr["entry"] - tr["stop"]) / tr["entry"]
    q = tr["risk_frac"].quantile([0.1, 0.25, 0.5, 0.75, 0.9]).to_dict()
    facts["risk"] = {
        "stop_distance_pct": {f"p{int(k*100)}": round(float(v), 4) for k, v in q.items()},
        "stop_distance_mean": round(float(tr["risk_frac"].mean()), 4),
        "rupee_risk_per_trade": round(a.capital * a.risk_pct, 0),
        "position_value_median": round(float(a.capital * a.risk_pct / tr["risk_frac"].median()), 0),
        "position_pct_of_capital_median": round(
            float(a.risk_pct / tr["risk_frac"].median()), 4),
        "position_pct_of_capital_p10": round(float(a.risk_pct / q[0.9]), 4),
        "position_pct_of_capital_p90": round(float(a.risk_pct / q[0.1]), 4),
    }
    pp = ROOT / "output" / "analysis_final" / "portfolio_positions.csv"
    if pp.exists():
        pos = pd.read_csv(pp)
        facts["risk"]["actual_position_value_median"] = round(float(pos["value"].median()), 0)
        facts["risk"]["actual_position_value_mean"] = round(float(pos["value"].mean()), 0)
        facts["risk"]["actual_qty_median"] = float(pos["qty"].median())

    # ── how trades ended ─────────────────────────────────────────────────────
    ex = []
    for r, g in tr.groupby("exit_reason"):
        ex.append({"exit": r, "n": len(g), "share": round(len(g) / len(tr), 4),
                   "win_rate": round(float((g["ret"] > 0).mean()), 4),
                   "avg_ret": round(float(g["ret"].mean()), 4),
                   "median_ret": round(float(g["ret"].median()), 4),
                   "avg_hold": round(float(g["hold_days"].mean()), 1),
                   "avg_R": round(float(g["r_multiple"].mean()), 3)})
    facts["exits"] = sorted(ex, key=lambda x: -x["n"])

    # R-multiple distribution: where losses and gains actually land
    bins = [-np.inf, -2, -1.5, -1.0, -0.5, 0, 0.5, 1, 2, 3, 5, np.inf]
    labels = ["worse than -2R", "-2 to -1.5R", "-1.5 to -1R", "-1 to -0.5R", "-0.5 to 0R",
              "0 to +0.5R", "+0.5 to +1R", "+1 to +2R", "+2 to +3R", "+3 to +5R", "above +5R"]
    cut = pd.cut(tr["r_multiple"], bins=bins, labels=labels)
    facts["r_distribution"] = [{"band": str(k), "n": int(v), "share": round(v / len(tr), 4)}
                               for k, v in cut.value_counts().reindex(labels).fillna(0).items()]
    facts["r_worse_than_minus_1"] = round(float((tr["r_multiple"] < -1.001).mean()), 4)

    hold = tr["hold_days"]
    facts["hold_distribution"] = {
        "p10": float(hold.quantile(0.1)), "p25": float(hold.quantile(0.25)),
        "median": float(hold.median()), "p75": float(hold.quantile(0.75)),
        "p90": float(hold.quantile(0.9)), "max": int(hold.max()),
        "share_under_10d": round(float((hold < 10).mean()), 4),
        "share_full_60d": round(float((hold >= 60).mean()), 4)}

    # ── winners vs losers on entry characteristics ───────────────────────────
    feats = ["mom_6m_skip", "ret_20", "rs_63", "dist_high", "dist_low", "atr_pct",
             "ext_atr", "ext_50", "rvol", "sector_rank", "med_value_cr"]
    keys = ["date", "model", "symbol"]
    have = [f for f in feats if f in sg.columns]
    df = tr.merge(sg[keys + have + ["rank"]].drop_duplicates(keys), on=keys, how="left")
    df["win"] = df["ret"] > 0
    w, l = df[df["win"]], df[~df["win"]]
    comp = []
    for f in have:
        mw, ml, sd = w[f].median(), l[f].median(), df[f].std()
        if not np.isfinite(mw) or not np.isfinite(ml) or not sd:
            continue
        comp.append({"feature": f, "winners": round(float(mw), 4), "losers": round(float(ml), 4),
                     "gap_sd": round(float((mw - ml) / sd), 3)})
    facts["winners_vs_losers"] = sorted(comp, key=lambda r: -abs(r["gap_sd"]))
    facts["outcome"] = {
        "trades": len(df), "winners": len(w), "losers": len(l),
        "win_rate": round(len(w) / len(df), 4),
        "winner_hold_median": float(w["hold_days"].median()),
        "loser_hold_median": float(l["hold_days"].median()),
        "winner_avg_ret": round(float(w["ret"].mean()), 4),
        "loser_avg_ret": round(float(l["ret"].mean()), 4),
        "gross_gains": round(float(df.loc[df["ret"] > 0, "ret"].sum()), 2),
        "gross_losses": round(float(-df.loc[df["ret"] < 0, "ret"].sum()), 2),
    }
    facts["by_rank"] = [
        {"band": str(b), "n": int(len(g)), "win_rate": round(float((g["ret"] > 0).mean()), 4),
         "avg_ret": round(float(g["ret"].mean()), 4)}
        for b, g in df.assign(band=pd.cut(df["rank"], [0, 5, 10, 15, 20],
                                          labels=["1-5", "6-10", "11-15", "16-20"]))
        .groupby("band", observed=True)]

    # data for charts
    facts["_hist_r"] = [round(float(x), 3) for x in tr["r_multiple"].clip(-3, 8)]
    facts["_hist_hold"] = [int(x) for x in tr["hold_days"]]

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(facts, indent=1, default=str), encoding="utf-8")
    print(f"wrote {a.out}")
    print(f"  funnel (latest): {facts['funnel_latest']}")
    print(f"  frequency      : {facts['frequency']['trades_per_month']} trades/month, "
          f"{facts['frequency']['concurrent_positions_mean']} open on average")
    print(f"  median stop    : {facts['risk']['stop_distance_pct']['p50']:.1%} below entry")
    print(f"  worse than -1R : {facts['r_worse_than_minus_1']:.1%} of trades")


if __name__ == "__main__":
    main()
