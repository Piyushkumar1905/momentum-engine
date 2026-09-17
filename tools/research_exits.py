"""Is the problem the entries or the exits?

Holds the entry rule fixed and varies only the exit policy. If a losing strategy
becomes profitable under a different exit, the filters were never the problem and
tuning them is wasted effort.

Everything here runs on the TRAIN window only. Later windows stay untouched so they
can still judge whatever this suggests.

  python tools/research_exits.py --train-start 2021-01-01 --train-end 2023-12-31
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

log = logging.getLogger("research_exits")


def simulate(i, j, entry_px, atr, O, H, L, C, CA, policy) -> dict | None:
    """One trade under an arbitrary exit policy.

    policy keys:
      stop_atr      initial stop distance in ATR (None = no stop)
      target_r      profit target in R (None = no target, let it run)
      max_hold      time exit in sessions
      trail_atr     chandelier trail in ATR from the highest close seen (None = off)
      breakeven_r   move stop to entry once this many R is reached (None = off)
    """
    T = O.shape[0]
    if i + 1 >= T:
        return None
    entry = O[i + 1, j]
    if not np.isfinite(entry) or not np.isfinite(atr) or atr <= 0:
        return None
    max_hold = policy["max_hold"]
    last = i + max_hold
    if last >= T:
        return None
    if CA is not None and CA[i + 1:last + 1, j].any():
        return None

    stop = entry - policy["stop_atr"] * atr if policy.get("stop_atr") else -np.inf
    risk = entry - stop if np.isfinite(stop) else policy.get("nominal_risk_atr", 1.5) * atr
    if risk <= 0:
        return None
    target = entry + policy["target_r"] * risk if policy.get("target_r") else np.inf
    peak = entry
    moved_be = False

    for k in range(i + 1, last + 1):
        o, h, lo, c = O[k, j], H[k, j], L[k, j], C[k, j]
        if not np.isfinite(c):
            continue
        if lo <= stop:
            px = min(o, stop) if np.isfinite(o) else stop
            return _res(entry, stop, risk, px, "stop", k - i)
        if h >= target:
            px = max(o, target) if np.isfinite(o) else target
            return _res(entry, stop, risk, px, "target", k - i)
        peak = max(peak, c)
        if policy.get("breakeven_r") and not moved_be and (peak - entry) >= policy["breakeven_r"] * risk:
            stop, moved_be = max(stop, entry), True
        if policy.get("trail_atr"):
            stop = max(stop, peak - policy["trail_atr"] * atr)
    px = C[last, j]
    if not np.isfinite(px):
        return None
    return _res(entry, stop, risk, px, "time", max_hold)


def _res(entry, stop, risk, px, reason, held):
    return {"entry": entry, "exit": px, "exit_reason": reason, "hold_days": held,
            "ret": px / entry - 1, "r_multiple": (px - entry) / risk}


POLICIES = {
    "baseline_2R_20d":        {"stop_atr": 1.5, "target_r": 2.0, "max_hold": 20},
    "target_3R_20d":          {"stop_atr": 1.5, "target_r": 3.0, "max_hold": 20},
    "no_target_20d":          {"stop_atr": 1.5, "target_r": None, "max_hold": 20},
    "no_target_40d":          {"stop_atr": 1.5, "target_r": None, "max_hold": 40},
    "no_target_60d":          {"stop_atr": 1.5, "target_r": None, "max_hold": 60},
    "no_target_120d":         {"stop_atr": 1.5, "target_r": None, "max_hold": 120},
    "trail3_60d":             {"stop_atr": 1.5, "target_r": None, "max_hold": 60, "trail_atr": 3.0},
    "trail3_120d":            {"stop_atr": 1.5, "target_r": None, "max_hold": 120, "trail_atr": 3.0},
    "trail4_120d":            {"stop_atr": 1.5, "target_r": None, "max_hold": 120, "trail_atr": 4.0},
    "wide_stop3_trail4_120d": {"stop_atr": 3.0, "target_r": None, "max_hold": 120, "trail_atr": 4.0},
    "be_trail3_120d":         {"stop_atr": 1.5, "target_r": None, "max_hold": 120, "trail_atr": 3.0,
                               "breakeven_r": 1.0},
    "timeonly_60d":           {"stop_atr": None, "target_r": None, "max_hold": 60,
                               "nominal_risk_atr": 1.5},
    "timeonly_120d":          {"stop_atr": None, "target_r": None, "max_hold": 120,
                               "nominal_risk_atr": 1.5},
}


def stats(t: pd.DataFrame, cost: float) -> dict:
    if t.empty:
        return {"trades": 0}
    r = t["ret"] - cost
    w, l = r[r > 0], r[r <= 0]
    return {"trades": len(t), "win_rate": round(len(w) / len(t), 3),
            "expectancy": round(float(r.mean()), 5),
            "profit_factor": round(float(w.sum() / -l.sum()), 3) if l.sum() < 0 else None,
            "avg_R": round(float(t["r_multiple"].mean()), 3),
            "avg_hold": round(float(t["hold_days"].mean()), 1),
            "best": round(float(r.max()), 3), "worst": round(float(r.min()), 3)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(ROOT / "config.yaml"))
    ap.add_argument("--train-start", default="2021-01-01")
    ap.add_argument("--train-end", default="2023-12-31")
    ap.add_argument("--out", default=str(ROOT / "output" / "research" / "exits.json"))
    a = ap.parse_args()
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")

    cfg = yaml.safe_load(open(a.config))
    cfg["backtest"]["start"] = a.train_start
    uni = data.load_universe(cfg["run"]["universe_file"], cfg["run"].get("max_symbols"))
    panel = data.load_panel(cfg, uni)
    prep = bt.prepare(panel, uni, cfg)

    dates = panel.dates
    sel = (dates >= pd.Timestamp(a.train_start)) & (dates <= pd.Timestamp(a.train_end))
    idxs = np.where(sel)[0][::int(cfg["backtest"]["rebalance_every"])]
    O, H, L, C = (x.to_numpy(float) for x in (panel.open, panel.high, panel.low, panel.close))
    CA = panel.corp_action.to_numpy(bool) if panel.corp_action is not None else None
    ATR = prep["F"]["atr"].to_numpy(float)
    cost = float(cfg["backtest"]["cost_pct"])
    top_n = int(cfg["backtest"]["top_n"])

    # Control arm: the same number of picks per date, drawn at random from the liquid
    # universe. Without it a long no-stop hold in a rising market looks like skill when
    # it is only beta - which is exactly how this kind of search manufactures an edge.
    rng = np.random.default_rng(7)
    B = prep["base"].to_numpy(bool)

    results = {}
    for model in ("momentum", "reversal", "random_control"):
        picks = []
        if model == "random_control":
            for i in idxs:
                pool = np.where(B[i] & np.isfinite(O[min(i + 1, len(dates) - 1)]))[0]
                if pool.size:
                    for j in rng.choice(pool, size=min(top_n, pool.size), replace=False):
                        picks.append((i, int(j)))
        else:
            S = prep["scores"][model]
            for i in idxs:
                row = S.iloc[i].to_numpy(float)
                for j in bt._top(row, top_n):
                    picks.append((i, j))
        rows = {name: [] for name in POLICIES}
        for i, j in picks:
            for name, pol in POLICIES.items():
                t = simulate(i, j, None, ATR[i, j], O, H, L, C, CA, pol)
                if t:
                    rows[name].append(t)
        results[model] = {name: stats(pd.DataFrame(v), cost) for name, v in rows.items()}

        print(f"\n{'='*94}\n{model.upper()}  ({len(picks):,} picks, train {a.train_start} -> {a.train_end})\n{'='*94}")
        print(f"{'policy':<24}{'trades':>8}{'win':>8}{'expect':>10}{'PF':>8}{'avgR':>8}{'hold':>7}{'best':>8}")
        for name in POLICIES:
            s = results[model][name]
            if not s.get("trades"):
                continue
            print(f"{name:<24}{s['trades']:>8,}{s['win_rate']:>8.1%}{s['expectancy']:>10.4f}"
                  f"{(s['profit_factor'] or 0):>8.3f}{s['avg_R']:>8.2f}{s['avg_hold']:>7.1f}{s['best']:>8.1%}")

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps({"train": [a.train_start, a.train_end],
                                       "policies": POLICIES, "results": results},
                                      indent=1), encoding="utf-8")
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
