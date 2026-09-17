"""Point-in-time event-study backtest.

For each scan date t (every `rebalance_every` sessions):
  1. rank stocks with features known at the close of t
  2. take the top N per model
  3. enter at the OPEN of t+1 (no same-bar look-ahead)
  4. measure forward return, max upside (MFE) and max drawdown (MAE) over each horizon
  5. simulate a rules-based trade (ATR stop, R-multiple target, time exit)
  6. compare with the whole liquid universe on the same dates and with a naive 52W-low control
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from . import rules
from .data import Panel
from .features import compute_features, market_regime

log = logging.getLogger(__name__)
ROW_KEYS = ["close", "atr", "atr_pct", "lo252", "hi252", "dist_high", "dist_low", "mom_6m_skip",
            "ret_20", "rs_63", "rvol", "ext_atr", "ext_50", "sector_rank", "brk_recent", "brk_rvol",
            "brk_close_loc", "base_depth", "atr_contraction", "med_value_cr", "drawdown", "rs_div",
            "higher_low_atr", "support_days", "cap_rvol", "updown_20", "rs_vs_sector", "low15"]


def prepare(panel: Panel, universe: pd.DataFrame, cfg: dict) -> dict:
    industry = universe.set_index("symbol")["industry"]
    F = compute_features(panel, industry)
    regime = market_regime(panel, F)
    base = rules.base_mask(F, cfg)
    scores, used = {}, {}
    if cfg["momentum"].get("enabled", True):
        scores["momentum"], _, used["momentum"] = rules.momentum_model(F, regime, cfg, base)
    if cfg["reversal"].get("enabled", True):
        scores["reversal"], _, used["reversal"] = rules.reversal_model(F, regime, cfg, base)
    scores["control_naive_low"] = rules.naive_low_control(F, cfg, base)
    log.info("Weights actually used (fundamentals/catalyst re-distributed): %s", used)
    return {"F": F, "regime": regime, "base": base, "scores": scores, "weights_used": used,
            "industry": industry}


def _top(score_row: np.ndarray, n: int) -> np.ndarray:
    ok = np.where(~np.isnan(score_row))[0]
    if ok.size == 0:
        return ok
    order = ok[np.argsort(-score_row[ok], kind="stable")]
    return order[:n]


def _forward(panel: Panel, horizons) -> dict:
    entry = panel.open.shift(-1)
    out = {}
    for H in horizons:
        fmax = panel.high.shift(-H).rolling(H, min_periods=H).max()
        fmin = panel.low.shift(-H).rolling(H, min_periods=H).min()
        out[H] = {"ret": (panel.close.shift(-H) / entry - 1).to_numpy(),
                  "mfe": (fmax / entry - 1).to_numpy(),
                  "mae": (fmin / entry - 1).to_numpy()}
    b = panel.bench.reindex(panel.dates)
    nifty = {H: (b.shift(-H) / b - 1).to_numpy() for H in horizons}
    return {"stock": out, "nifty": nifty}


def _simulate(i, j, stop, tgt_r, O, Hh, L, C, max_hold, cost, ca=None):
    T = O.shape[0]
    if i + 1 >= T or np.isnan(O[i + 1, j]):
        return None
    # A demerger or unadjusted rebasing inside the holding window makes the price
    # series incomparable across the gap. Excluding the trade is the honest choice:
    # booking the printed drop would record a loss the holder never took.
    if ca is not None and ca[i + 1:min(i + 1 + max_hold, T), j].any():
        return None
    entry = O[i + 1, j]
    risk = entry - stop
    if not np.isfinite(risk) or risk <= 0:
        return None
    target = entry + tgt_r * risk
    last = i + max_hold
    if last >= T:
        return None                     # trade would still be open at the end of data
    for k in range(i + 1, last + 1):
        o, h, lo, c = O[k, j], Hh[k, j], L[k, j], C[k, j]
        if np.isnan(c):
            continue
        if lo <= stop:                  # stop checked first (conservative when both hit)
            px = min(o, stop) if not np.isnan(o) else stop
            reason = "stop"
            break
        if h >= target:
            px = max(o, target) if not np.isnan(o) else target
            reason = "target"
            break
    else:
        k, px, reason = last, C[last, j], "time"
        if np.isnan(px):
            return None
    return {"entry": entry, "stop": stop, "target": target, "exit": px, "exit_reason": reason,
            "hold_days": k - i, "exit_idx": k, "ret": px / entry - 1 - cost,
            "r_multiple": (px - entry) / risk}


def trade_stats(r: pd.Series) -> dict:
    r = r.dropna()
    if r.empty:
        return {"trades": 0}
    w, l_ = r[r > 0], r[r <= 0]
    return {"trades": len(r), "win_rate": len(w) / len(r),
            "avg_win": w.mean() if len(w) else 0.0, "avg_loss": l_.mean() if len(l_) else 0.0,
            "expectancy": r.mean(), "median": r.median(),
            "profit_factor": (w.sum() / -l_.sum()) if l_.sum() < 0 else np.nan,
            "worst": r.min(), "best": r.max()}


def run_backtest(panel: Panel, universe: pd.DataFrame, cfg: dict, prep: dict | None = None) -> dict:
    bt = cfg["backtest"]
    prep = prep or prepare(panel, universe, cfg)
    F, regime, base, scores = prep["F"], prep["regime"], prep["base"], prep["scores"]
    dates, syms = panel.dates, np.array(panel.symbols)
    horizons = [int(h) for h in bt["horizons"]]
    fwd = _forward(panel, horizons)
    A = {k: F[k].to_numpy(dtype=float) for k in ROW_KEYS}
    B = base.to_numpy(dtype=bool)
    O, Hh, L, C = (x.to_numpy(float) for x in (panel.open, panel.high, panel.low, panel.close))
    CA = panel.corp_action.to_numpy(bool) if panel.corp_action is not None else None
    start = pd.Timestamp(bt["start"])
    end = pd.Timestamp(bt["end"]) if bt.get("end") else dates[-1]
    idxs = np.where((dates >= start) & (dates <= end))[0][::int(bt["rebalance_every"])]
    ind = prep["industry"]
    reg_arr = regime["regime"].to_numpy()

    sig_rows, trades = [], []
    open_until: dict = {}
    for i in idxs:
        # universe baseline (all liquid stocks) for this date
        ub = {}
        for H in horizons:
            v = fwd["stock"][H]["ret"][i][B[i]]
            m = fwd["stock"][H]["mfe"][i][B[i]]
            ub[H] = (np.nanmean(v) if np.isfinite(v).any() else np.nan,
                     np.nanmean(m) if np.isfinite(m).any() else np.nan)
        for model, S in scores.items():
            row_scores = S.iloc[i].to_numpy(float)
            for rank, j in enumerate(_top(row_scores, int(bt["top_n"])), start=1):
                x = {k: A[k][i, j] for k in ROW_KEYS}
                rec = {"date": dates[i], "model": model, "rank": rank, "symbol": syms[j],
                       "industry": ind.get(syms[j], "Unknown"), "score": row_scores[j],
                       "regime": reg_arr[i], **x}
                for H in horizons:
                    for m in ("ret", "mfe", "mae"):
                        rec[f"{m}_{H}d"] = fwd["stock"][H][m][i, j]
                    rec[f"universe_ret_{H}d"], rec[f"universe_mfe_{H}d"] = ub[H]
                    rec[f"nifty_ret_{H}d"] = fwd["nifty"][H][i]
                    rec[f"excess_{H}d"] = rec[f"ret_{H}d"] - ub[H][0]
                sig_rows.append(rec)
                # trade simulation
                if bt.get("skip_if_open", True) and open_until.get((model, j), -1) >= i + 1:
                    continue
                t = _simulate(i, j, rules.stop_price(model, x, cfg), rules.target_r(model, cfg),
                              O, Hh, L, C, int(bt["max_hold_days"]), float(bt["cost_pct"]), CA)
                if t:
                    open_until[(model, j)] = t.pop("exit_idx")
                    trades.append({"date": dates[i], "model": model, "symbol": syms[j],
                                   "regime": reg_arr[i], "score": row_scores[j], **t})
    signals = pd.DataFrame(sig_rows)
    trades = pd.DataFrame(trades)
    log.info("Backtest: %d scan dates, %d signals, %d simulated trades", len(idxs), len(signals), len(trades))
    return {"signals": signals, "trades": trades, "summary": summarize(signals, trades, cfg),
            "regime": regime, "scan_dates": dates[idxs], "weights_used": prep["weights_used"]}


def _period_label(d, periods):
    for p in periods:
        s = pd.Timestamp(p["start"])
        e = pd.Timestamp(p["end"]) if p.get("end") else pd.Timestamp.max
        if s <= d <= e:
            return p["name"]
    return "other"


def summarize(signals: pd.DataFrame, trades: pd.DataFrame, cfg: dict) -> dict:
    out = {}
    if signals.empty:
        return out
    horizons = [int(h) for h in cfg["backtest"]["horizons"]]
    periods = cfg["backtest"].get("periods") or []
    rows = []
    for (model, H) in [(m, h) for m in signals["model"].unique() for h in horizons]:
        s = signals[signals["model"] == model].dropna(subset=[f"ret_{H}d"])
        if s.empty:
            continue
        rows.append({"model": model, "horizon_days": H, "signals": len(s),
                     "avg_return": s[f"ret_{H}d"].mean(), "median_return": s[f"ret_{H}d"].median(),
                     "hit_rate_positive": (s[f"ret_{H}d"] > 0).mean(),
                     "avg_max_upside_MFE": s[f"mfe_{H}d"].mean(),
                     "median_max_upside_MFE": s[f"mfe_{H}d"].median(),
                     "share_reaching_+5pct": (s[f"mfe_{H}d"] >= 0.05).mean(),
                     "share_reaching_+10pct": (s[f"mfe_{H}d"] >= 0.10).mean(),
                     "avg_max_drawdown_MAE": s[f"mae_{H}d"].mean(),
                     "universe_avg_return": s[f"universe_ret_{H}d"].mean(),
                     "universe_avg_MFE": s[f"universe_mfe_{H}d"].mean(),
                     "avg_excess_vs_universe": s[f"excess_{H}d"].mean(),
                     "nifty_avg_return": s[f"nifty_ret_{H}d"].mean()})
    out["forward"] = pd.DataFrame(rows)

    if not trades.empty:
        t = trades.copy()
        t["period"] = t["date"].apply(lambda d: _period_label(d, periods))
        t["year"] = t["date"].dt.year
        def grp(keys):
            recs = []
            for k, g in t.groupby(keys):
                k = k if isinstance(k, tuple) else (k,)
                st = trade_stats(g["ret"])
                st.update(dict(zip(keys, k)))
                st["avg_hold_days"] = g["hold_days"].mean()
                st["target_hit"] = (g["exit_reason"] == "target").mean()
                st["stop_hit"] = (g["exit_reason"] == "stop").mean()
                st["time_exit"] = (g["exit_reason"] == "time").mean()
                st["avg_R"] = g["r_multiple"].mean()
                recs.append(st)
            df = pd.DataFrame(recs)
            return df[keys + [c for c in df.columns if c not in keys]]
        out["trades"] = grp(["model"])
        out["by_regime"] = grp(["model", "regime"])
        out["by_period"] = grp(["model", "period"])
        out["by_year"] = grp(["model", "year"])
    return out


def scan_latest(panel: Panel, universe: pd.DataFrame, cfg: dict, prep: dict | None = None) -> pd.DataFrame:
    """Today's candidate lists (last date in the data) with reasons, flags and trade plan."""
    prep = prep or prepare(panel, universe, cfg)
    F, regime, scores = prep["F"], prep["regime"], prep["scores"]
    i = len(panel.dates) - 1
    reg = regime["regime"].iloc[i]
    rows = []
    for model in ("momentum", "reversal"):
        if model not in scores:
            continue
        S = scores[model].iloc[i].to_numpy(float)
        for rank, j in enumerate(_top(S, int(cfg["backtest"]["top_n"])), start=1):
            sym = panel.symbols[j]
            x = {k: float(F[k].iat[i, j]) for k in ROW_KEYS}
            stop = rules.stop_price(model, x, cfg)
            risk = x["close"] - stop
            tgt = x["close"] + rules.target_r(model, cfg) * risk
            why, flags = rules.explain(model, x, reg)
            rows.append({"as_of": panel.dates[i].date(), "model": model, "rank": rank, "symbol": sym,
                         "industry": prep["industry"].get(sym, "Unknown"), "score": round(S[j], 3),
                         "setup": rules.momentum_setup(x) if model == "momentum" else "Confirmed reversal",
                         "last_close": round(x["close"], 2), "stop_ref": round(stop, 2),
                         "risk_pct": round(risk / x["close"], 4), "target_ref": round(tgt, 2),
                         "reward_to_risk": rules.target_r(model, cfg), "why": why, "risk_flags": flags,
                         "regime": reg, "breadth_50dma": round(float(regime["breadth_50dma"].iloc[i]), 3),
                         "median_value_cr": round(x["med_value_cr"], 1)})
    return pd.DataFrame(rows)


def track_picks(saved: pd.DataFrame, panel: Panel, horizons) -> pd.DataFrame:
    """Forward-track previously saved lists: entry = next open after the list date."""
    out = []
    dates = panel.dates
    for r in saved.itertuples():
        if r.symbol not in panel.close.columns:
            continue
        pos = dates.searchsorted(pd.Timestamp(r.as_of), side="right")   # first session after list date
        if pos >= len(dates):
            out.append({**r._asdict(), "status": "not started"})
            continue
        entry = panel.open[r.symbol].iloc[pos]
        rec = {**r._asdict(), "entry_date": dates[pos].date(), "entry": entry}
        hi, lo, cl = panel.high[r.symbol].iloc[pos:], panel.low[r.symbol].iloc[pos:], panel.close[r.symbol].iloc[pos:]
        for H in horizons:
            if len(cl) >= H:
                rec[f"ret_{H}d"] = cl.iloc[H - 1] / entry - 1
                rec[f"mfe_{H}d"] = hi.iloc[:H].max() / entry - 1
                rec[f"mae_{H}d"] = lo.iloc[:H].min() / entry - 1
        rec["days_elapsed"] = len(cl)
        s_hit = np.flatnonzero((lo <= r.stop_ref).to_numpy())
        t_hit = np.flatnonzero((hi >= r.target_ref).to_numpy())
        s_day = s_hit[0] if s_hit.size else np.inf
        t_day = t_hit[0] if t_hit.size else np.inf
        rec["first_hit"] = ("none yet" if s_day == t_day == np.inf else
                            "stop" if s_day <= t_day else "target")   # same day counts as stop
        rec["days_to_first_hit"] = None if rec["first_hit"] == "none yet" else int(min(s_day, t_day)) + 1
        rec["return_to_date"] = cl.iloc[-1] / entry - 1
        out.append(rec)
    return pd.DataFrame(out)
