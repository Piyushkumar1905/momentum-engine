"""Period autopsy: what the rules actually did, and what they missed.

Answers, for an arbitrary window (default 2024-09 -> latest):

  1. Portfolio outcome  - deployed, returned, alpha vs Nifty, equity curve
  2. Trade statistics   - win rate, R, hold time, best/worst, win/loss streaks
  3. Mistakes           - where the rules lost money they did not have to
  4. Misses             - stocks the filters rejected that then ran, attributed to
                          the single rule that rejected each one

(4) is the part a normal backtest never shows. A backtest reports how the trades
it took performed; it is silent about the trades the filter refused, which is
where the cost of an over-tight rule actually hides.

  python tools/analyze_period.py --start 2024-09-01
  python tools/analyze_period.py --start 2024-09-01 --capital 1000000 --risk-pct 0.01
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from engine import backtest as bt          # noqa: E402
from engine import data, rules             # noqa: E402
from engine.features import compute_features, market_regime   # noqa: E402

log = logging.getLogger("analyze_period")
IST = timezone(timedelta(hours=5, minutes=30))


# ─────────────────────────────── portfolio simulation ───────────────────────────────

def simulate_portfolio(trades: pd.DataFrame, capital: float, risk_pct: float,
                       max_positions: int, max_weight: float) -> dict:
    """Event-driven portfolio over the engine's simulated trades.

    Sizing is risk-based: each position risks `risk_pct` of CURRENT equity across the
    distance from entry to stop, capped at `max_weight` of equity so a tight stop
    cannot produce an absurd position. Trades are taken in score order on each date
    and skipped when no slot or no cash is free - which is what actually happens to a
    real account, and is invisible in a signal-level backtest.
    """
    if trades.empty:
        return {"error": "no trades in window"}

    t = trades.sort_values(["date", "score"], ascending=[True, False]).reset_index(drop=True)
    equity = capital
    peak_equity = capital
    max_dd = 0.0
    open_pos: list[dict] = []
    closed: list[dict] = []
    deployed_total = 0.0
    skipped_no_slot = 0
    skipped_no_cash = 0
    curve = []

    # Every date on which something opens or closes.
    events = sorted(set(t["date"]) | set(t["exit_date"]))
    for today in events:
        # close first, freeing capital and slots for the same day's entries
        still_open = []
        for p in open_pos:
            if p["exit_date"] <= today:
                pnl = p["qty"] * (p["exit"] - p["entry"]) - p["cost"]
                equity += pnl
                closed.append({**p, "pnl": pnl, "pct": pnl / p["value"]})
            else:
                still_open.append(p)
        open_pos = still_open

        cash_free = equity - sum(p["value"] for p in open_pos)
        for _, r in t[t["date"] == today].iterrows():
            if len(open_pos) >= max_positions:
                skipped_no_slot += 1
                continue
            risk_per_share = r["entry"] - r["stop"]
            if risk_per_share <= 0:
                continue
            qty = int((equity * risk_pct) / risk_per_share)
            value = qty * r["entry"]
            if value > equity * max_weight:
                qty = int((equity * max_weight) / r["entry"])
                value = qty * r["entry"]
            if qty <= 0:
                continue
            if value > cash_free:
                skipped_no_cash += 1
                continue
            cash_free -= value
            deployed_total += value
            open_pos.append({"symbol": r["symbol"], "model": r["model"], "date": today,
                             "exit_date": r["exit_date"], "entry": r["entry"], "stop": r["stop"],
                             "exit": r["exit"], "qty": qty, "value": value,
                             "cost": value * r["cost_pct"] if "cost_pct" in r else value * 0.0025,
                             "exit_reason": r["exit_reason"], "hold_days": r["hold_days"],
                             "r_multiple": r["r_multiple"]})

        exposure = sum(p["value"] for p in open_pos)
        peak_equity = max(peak_equity, equity)
        max_dd = min(max_dd, equity / peak_equity - 1)
        curve.append({"date": str(pd.Timestamp(today).date()), "equity": round(equity, 2),
                      "open_positions": len(open_pos), "exposure": round(exposure, 2)})

    c = pd.DataFrame(closed)
    return {
        "starting_capital": capital,
        "ending_equity": round(equity, 2),
        "net_pnl": round(equity - capital, 2),
        "total_return_pct": round(equity / capital - 1, 4),
        "total_deployed": round(deployed_total, 2),
        "turnover_x": round(deployed_total / capital, 2),
        "positions_taken": len(closed),
        "skipped_no_slot": skipped_no_slot,
        "skipped_no_cash": skipped_no_cash,
        "max_drawdown_pct": round(max_dd, 4),
        "peak_equity": round(peak_equity, 2),
        "avg_position_value": round(c["value"].mean(), 2) if len(c) else None,
        "curve": curve,
        "closed": c,
    }


# ─────────────────────────────── trade statistics ───────────────────────────────

def streaks(wins: list[bool]) -> dict:
    """Longest runs of consecutive wins and losses, in chronological order."""
    best_w = best_l = cur_w = cur_l = 0
    for w in wins:
        if w:
            cur_w += 1; cur_l = 0
        else:
            cur_l += 1; cur_w = 0
        best_w, best_l = max(best_w, cur_w), max(best_l, cur_l)
    return {"longest_win_streak": best_w, "longest_loss_streak": best_l}


def trade_report(t: pd.DataFrame) -> dict:
    """Signal-level statistics. Independent of position sizing, so these hold
    whatever capital the account actually runs."""
    if t.empty:
        return {}
    t = t.sort_values("date")
    r = t["ret"]
    w, l = r[r > 0], r[r <= 0]
    best, worst = t.loc[r.idxmax()], t.loc[r.idxmin()]
    # Average reward:risk realised, not the 2.0 the plan assumes
    avg_win_r = t.loc[r > 0, "r_multiple"].mean() if len(w) else 0.0
    avg_loss_r = t.loc[r <= 0, "r_multiple"].mean() if len(l) else 0.0
    return {
        "trades": len(t),
        "win_rate": round(len(w) / len(t), 4),
        "avg_return_pct": round(r.mean(), 4),
        "median_return_pct": round(r.median(), 4),
        "avg_win_pct": round(w.mean(), 4) if len(w) else 0.0,
        "avg_loss_pct": round(l.mean(), 4) if len(l) else 0.0,
        "expectancy_pct": round(r.mean(), 4),
        "profit_factor": round(w.sum() / -l.sum(), 3) if l.sum() < 0 else None,
        "avg_R": round(t["r_multiple"].mean(), 3),
        "avg_win_R": round(avg_win_r, 3),
        "avg_loss_R": round(avg_loss_r, 3),
        "realised_reward_risk": round(abs(avg_win_r / avg_loss_r), 2) if avg_loss_r else None,
        "avg_hold_days": round(t["hold_days"].mean(), 2),
        "median_hold_days": float(t["hold_days"].median()),
        "avg_hold_days_winners": round(t.loc[r > 0, "hold_days"].mean(), 2) if len(w) else None,
        "avg_hold_days_losers": round(t.loc[r <= 0, "hold_days"].mean(), 2) if len(l) else None,
        "best_trade": {"symbol": best["symbol"], "date": str(pd.Timestamp(best["date"]).date()),
                       "return_pct": round(best["ret"], 4), "R": round(best["r_multiple"], 2),
                       "hold_days": int(best["hold_days"]), "exit": best["exit_reason"]},
        "worst_trade": {"symbol": worst["symbol"], "date": str(pd.Timestamp(worst["date"]).date()),
                        "return_pct": round(worst["ret"], 4), "R": round(worst["r_multiple"], 2),
                        "hold_days": int(worst["hold_days"]), "exit": worst["exit_reason"]},
        "exit_mix": {k: round(v, 4) for k, v in t["exit_reason"].value_counts(normalize=True).items()},
        **streaks(list(r > 0)),
    }


# ─────────────────────────────── mistake analysis ───────────────────────────────

def stop_autopsy(t: pd.DataFrame, panel, max_hold: int) -> dict:
    """Of the trades stopped out, how many would have reached target had the stop
    not been there? That is the direct, measurable cost of the stop distance."""
    stopped = t[t["exit_reason"] == "stop"]
    if stopped.empty:
        return {}
    dates = panel.dates
    recovered = would_have_won = 0
    for _, r in stopped.iterrows():
        j = panel.close.columns.get_loc(r["symbol"])
        i = dates.get_loc(pd.Timestamp(r["date"]))
        end = min(i + 1 + max_hold, len(dates))
        hi = panel.high.iloc[i + 1:end, j].max()
        cl = panel.close.iloc[end - 1, j] if end - 1 < len(dates) else np.nan
        if pd.notna(hi) and hi >= r["target"]:
            would_have_won += 1
        if pd.notna(cl) and cl > r["entry"]:
            recovered += 1
    return {
        "stopped_trades": len(stopped),
        "would_have_hit_target_without_stop": would_have_won,
        "would_have_hit_target_pct": round(would_have_won / len(stopped), 4),
        "closed_above_entry_by_horizon": recovered,
        "closed_above_entry_pct": round(recovered / len(stopped), 4),
        "avg_loss_on_stops_pct": round(stopped["ret"].mean(), 4),
    }


# ─────────────────────────────── miss analysis ───────────────────────────────

def missed_opportunities(F, regime, cfg, base, panel, scan_idx, horizon=20,
                         big_move=0.15) -> dict:
    """For every scan date, look at the liquid universe the screen could have chosen
    from, and ask which single rule rejected the stocks that then ran.

    'Blocked solely by rule R' means the stock failed R and passed every other rule -
    so relaxing R alone would have surfaced it. That is the actionable number; a
    stock failing four rules tells you nothing about any one of them.
    """
    conds = rules.momentum_conditions(F, regime, cfg)
    names = list(conds)
    C = np.stack([conds[n].to_numpy(bool) for n in names])        # (rules, dates, stocks)
    B = base.to_numpy(bool)

    entry = panel.open.shift(-1).to_numpy(float)
    fwd = (panel.close.shift(-horizon) / panel.open.shift(-1) - 1).to_numpy(float)

    n_fail = (~C).sum(axis=0)
    passed_all = n_fail == 0

    rows = []
    tally = {n: {"blocked_solely": 0, "big_movers_blocked": 0, "sum_ret": 0.0,
                 "captured_ret": []} for n in names}
    universe_big = selected_big = 0

    for i in scan_idx:
        elig = B[i] & np.isfinite(fwd[i]) & np.isfinite(entry[i])
        if not elig.any():
            continue
        big = elig & (fwd[i] >= big_move)
        universe_big += int(big.sum())
        selected_big += int((big & passed_all[i]).sum())
        # attribute each blocked big mover to the single rule that stopped it
        solo = elig & (n_fail[i] == 1)
        for k, n in enumerate(names):
            blocked_by_k = solo & ~C[k, i]
            tally[n]["blocked_solely"] += int(blocked_by_k.sum())
            bigk = blocked_by_k & (fwd[i] >= big_move)
            tally[n]["big_movers_blocked"] += int(bigk.sum())
            if blocked_by_k.any():
                tally[n]["sum_ret"] += float(np.nansum(fwd[i][blocked_by_k]))
                tally[n]["captured_ret"].extend(fwd[i][blocked_by_k].tolist())
        # record the biggest individual misses for the report
        miss = elig & ~passed_all[i] & (fwd[i] >= big_move)
        for j in np.where(miss)[0]:
            rows.append({"date": str(panel.dates[i].date()), "symbol": panel.symbols[j],
                         "fwd_return": round(float(fwd[i, j]), 4),
                         "rules_failed": int(n_fail[i, j]),
                         "failed": ",".join(n for k, n in enumerate(names) if not C[k, i, j])})

    summary = []
    for n in names:
        v = tally[n]
        arr = np.array(v["captured_ret"], dtype=float)
        arr = arr[np.isfinite(arr)]
        summary.append({
            "rule": n,
            "blocked_solely": v["blocked_solely"],
            "big_movers_blocked": v["big_movers_blocked"],
            "avg_fwd_return_of_blocked": round(float(arr.mean()), 4) if arr.size else None,
            "hit_rate_of_blocked": round(float((arr >= big_move).mean()), 4) if arr.size else None,
        })
    summary.sort(key=lambda r: -(r["big_movers_blocked"] or 0))

    top = pd.DataFrame(rows).sort_values("fwd_return", ascending=False).head(25) if rows else pd.DataFrame()
    return {
        "horizon_days": horizon,
        "big_move_threshold": big_move,
        "universe_big_movers": universe_big,
        "captured_by_screen": selected_big,
        "capture_rate": round(selected_big / universe_big, 4) if universe_big else None,
        "by_rule": summary,
        "top_misses": top.to_dict(orient="records") if len(top) else [],
    }


# ─────────────────────────────── main ───────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(ROOT / "config.yaml"))
    ap.add_argument("--start", default="2024-09-01")
    ap.add_argument("--end", default=None)
    ap.add_argument("--capital", type=float, default=1_000_000)
    ap.add_argument("--risk-pct", type=float, default=0.01)
    ap.add_argument("--max-positions", type=int, default=10)
    ap.add_argument("--max-weight", type=float, default=0.20)
    ap.add_argument("--out", default=str(ROOT / "output" / "analysis"))
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    cfg = yaml.safe_load(open(a.config))
    cfg["backtest"]["start"] = a.start
    if a.end:
        cfg["backtest"]["end"] = a.end

    uni = data.load_universe(cfg["run"]["universe_file"], cfg["run"].get("max_symbols"))
    panel = data.load_panel(cfg, uni)
    F = compute_features(panel, uni.set_index("symbol")["industry"])
    regime = market_regime(panel, F)
    base = rules.base_mask(F, cfg)
    prep = bt.prepare(panel, uni, cfg)

    res = bt.run_backtest(panel, uni, cfg, prep)
    trades = res["trades"].copy()
    if trades.empty:
        raise SystemExit("no trades in window")

    # exit_date is needed by the portfolio sim; derive it from hold_days
    dates = panel.dates
    idx = {d: k for k, d in enumerate(dates)}
    trades["exit_date"] = [dates[min(idx[pd.Timestamp(d)] + int(h), len(dates) - 1)]
                           for d, h in zip(trades["date"], trades["hold_days"])]
    trades["cost_pct"] = float(cfg["backtest"]["cost_pct"])

    window = {"start": a.start, "end": str(dates[-1].date()),
              "scan_dates": len(res["scan_dates"]),
              "trading_days": int(((dates >= pd.Timestamp(a.start)) & (dates <= dates[-1])).sum())}

    # benchmark over the same window
    b = panel.bench.reindex(dates).dropna()
    b = b[b.index >= pd.Timestamp(a.start)]
    years = (b.index[-1] - b.index[0]).days / 365.25
    nifty_ret = float(b.iloc[-1] / b.iloc[0] - 1)

    out = {"generated_at": datetime.now(IST).isoformat(timespec="seconds"), "window": window,
           "assumptions": {"capital": a.capital, "risk_pct": a.risk_pct,
                           "max_positions": a.max_positions, "max_weight": a.max_weight,
                           "cost_pct": cfg["backtest"]["cost_pct"],
                           "entry": "next session open", "sizing": "risk-based on stop distance"},
           "benchmark": {"nifty_return_pct": round(nifty_ret, 4),
                         "nifty_cagr_pct": round((1 + nifty_ret) ** (1 / years) - 1, 4),
                         "years": round(years, 2)}}

    out["by_model"] = {}
    for model in sorted(trades["model"].unique()):
        out["by_model"][model] = trade_report(trades[trades["model"] == model])

    main_models = trades[trades["model"].isin(["momentum", "reversal"])]
    out["all_signals"] = trade_report(main_models)

    port = simulate_portfolio(main_models, a.capital, a.risk_pct, a.max_positions, a.max_weight)
    closed = port.pop("closed", pd.DataFrame())
    pr = port["total_return_pct"]
    port["cagr_pct"] = round((1 + pr) ** (1 / years) - 1, 4)
    port["alpha_vs_nifty_pct"] = round(pr - nifty_ret, 4)
    port["alpha_cagr_pct"] = round(port["cagr_pct"] - out["benchmark"]["nifty_cagr_pct"], 4)
    out["portfolio"] = port

    out["mistakes"] = {"stops": stop_autopsy(main_models, panel, int(cfg["backtest"]["max_hold_days"]))}

    scan_idx = np.where((dates >= pd.Timestamp(a.start)) & (dates <= dates[-1]))[0][
        ::int(cfg["backtest"]["rebalance_every"])]
    out["missed"] = missed_opportunities(F, regime, cfg, base, panel, scan_idx)

    dest = Path(a.out)
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "analysis.json").write_text(json.dumps(out, indent=1, default=str), encoding="utf-8")
    trades.to_csv(dest / "trades_window.csv", index=False)
    if len(closed):
        closed.to_csv(dest / "portfolio_positions.csv", index=False)
    log.info("wrote %s", dest / "analysis.json")

    p = out["portfolio"]
    print(f"\n{'='*70}\nWINDOW {window['start']} -> {window['end']}  ({window['scan_dates']} scans)")
    print(f"{'='*70}")
    print(f"Capital {a.capital:,.0f} -> {p['ending_equity']:,.0f}  "
          f"({p['total_return_pct']:+.1%}, CAGR {p['cagr_pct']:+.1%})")
    print(f"Nifty {nifty_ret:+.1%} (CAGR {out['benchmark']['nifty_cagr_pct']:+.1%})  "
          f"-> ALPHA {p['alpha_vs_nifty_pct']:+.1%}")
    print(f"Deployed {p['total_deployed']:,.0f} across {p['positions_taken']} positions "
          f"({p['turnover_x']}x capital), max DD {p['max_drawdown_pct']:.1%}")
    s = out["all_signals"]
    print(f"\nWin rate {s['win_rate']:.1%} | avg R {s['avg_R']} | realised R:R {s['realised_reward_risk']} "
          f"| avg hold {s['avg_hold_days']}d")
    print(f"Streaks: {s['longest_win_streak']}W / {s['longest_loss_streak']}L")
    print(f"Best  {s['best_trade']['symbol']} {s['best_trade']['return_pct']:+.1%}")
    print(f"Worst {s['worst_trade']['symbol']} {s['worst_trade']['return_pct']:+.1%}")
    m = out["missed"]
    print(f"\nCapture rate: {m['captured_by_screen']}/{m['universe_big_movers']} "
          f"({m['capture_rate']:.1%}) of +{m['big_move_threshold']:.0%} moves")
    print("Rules blocking the most winners on their own:")
    for r in m["by_rule"][:4]:
        print(f"  {r['rule']:<20} blocked {r['big_movers_blocked']:>4} big movers "
              f"(of {r['blocked_solely']} solo rejections)")


if __name__ == "__main__":
    main()
