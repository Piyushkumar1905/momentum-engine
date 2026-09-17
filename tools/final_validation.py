"""Statistically honest re-test of a strategy spec.

Fixes three flaws that the adversarial review found in the earlier harness, each of
which independently inflated apparent edges:

1. ZERO-NOISE BENCHMARK. The old control was a single 20-stock random draw whose
   sampling SD was 55-65% of the edges being measured. Here the benchmark is the mean
   of EVERY eligible name on the same date under the same exit policy, so there is no
   sampling noise in the comparison at all.

2. NO SILENT TRUNCATION. simulate() returns None when the holding window runs past the
   end of data, so long holds quietly dropped the last months of the holdout - and how
   much they dropped scaled with max_hold, the parameter being chosen. Dates without a
   full window are now excluded from every arm equally and the count is reported.

3. OVERLAP-AWARE INFERENCE. Weekly rebalancing with a 60-day hold means consecutive
   observations share ~92% of their window; they are not independent. The per-date edge
   series gets a Newey-West t-statistic at the overlap lag, plus a non-overlapping
   cohort check, because 5,000 overlapping trades can be 11 real observations.

  python tools/final_validation.py --spec-json '{"name":"x","factors":{"mom_12m_skip1":1}}'
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
from tools.research_exits import simulate   # noqa: E402
from tools.strategy_lab import (DEFAULT_EXIT, FULL_END, FULL_START, WINDOWS,  # noqa: E402
                                build_score, load_context)

log = logging.getLogger("final_validation")


def newey_west_t(x: np.ndarray, lags: int) -> float:
    """t-statistic of the mean, robust to the autocorrelation overlapping holds create."""
    x = x[np.isfinite(x)]
    n = len(x)
    if n < 5:
        return float("nan")
    mu = x.mean()
    e = x - mu
    g0 = (e @ e) / n
    var = g0
    for k in range(1, min(lags, n - 1) + 1):
        gk = (e[k:] @ e[:-k]) / n
        var += 2 * (1 - k / (lags + 1)) * gk        # Bartlett kernel
    if var <= 0:
        return float("nan")
    return float(mu / np.sqrt(var / n))


def per_date_edge(spec: dict, ctx: dict) -> pd.DataFrame:
    """Mean picked-stock outcome minus mean eligible-universe outcome, per scan date."""
    panel, F, base_np, dates = ctx["panel"], ctx["F"], ctx["base"], ctx["dates"]
    O, H, L, C, CA, ATR = ctx["O"], ctx["H"], ctx["L"], ctx["C"], ctx["CA"], ctx["ATR"]
    exit_pol = {**DEFAULT_EXIT, **(spec.get("exit") or {})}
    top_n = int(spec.get("top_n", 20))
    max_hold = int(exit_pol["max_hold"])
    score = build_score(spec, panel, F, base_np)
    T = len(dates)

    rows = []
    for i in ctx["idxs"]:
        # every arm needs a full holding window, or the comparison is not like-for-like
        if i + 1 + max_hold >= T:
            continue
        elig = np.where(base_np[i] & np.isfinite(O[i + 1]))[0]
        if elig.size < 50:
            continue
        picks = bt._top(score[i], top_n)
        if picks.size == 0:
            continue
        uni_r, pick_r = [], []
        pick_set = set(int(j) for j in picks)
        for j in elig:
            t = simulate(i, int(j), None, ATR[i, int(j)], O, H, L, C, CA, exit_pol)
            if not t:
                continue
            uni_r.append(t["ret"])
            if int(j) in pick_set:
                pick_r.append(t["ret"])
        if len(pick_r) < max(3, top_n // 3) or len(uni_r) < 50:
            continue
        rows.append({"date": dates[i], "strategy": float(np.mean(pick_r)),
                     "universe": float(np.mean(uni_r)), "n_picks": len(pick_r),
                     "n_universe": len(uni_r)})
    df = pd.DataFrame(rows)
    if len(df):
        df["edge"] = df["strategy"] - df["universe"]
    return df


def summarise(df: pd.DataFrame, max_hold: int, rebalance: int = 5) -> dict:
    """Window-level statistics with overlap-aware inference."""
    out = {}
    overlap = max(1, int(np.ceil(max_hold / rebalance)))
    for name, (s, e) in WINDOWS.items():
        w = df[(df["date"] >= pd.Timestamp(s)) & (df["date"] <= pd.Timestamp(e))]
        if w.empty:
            out[name] = {"scan_dates": 0}
            continue
        x = w["edge"].to_numpy(float)
        ind = x[::overlap]                     # non-overlapping cohorts
        out[name] = {
            "scan_dates": len(w),
            "mean_edge": round(float(x.mean()), 5),
            "median_edge": round(float(np.median(x)), 5),
            "share_positive": round(float((x > 0).mean()), 3),
            "naive_t": round(float(x.mean() / (x.std(ddof=1) / np.sqrt(len(x)))), 2)
                        if len(x) > 2 and x.std() else None,
            "newey_west_t": round(newey_west_t(x, overlap), 2),
            "independent_cohorts": len(ind),
            "cohort_mean_edge": round(float(ind.mean()), 5) if len(ind) else None,
            "cohort_share_positive": round(float((ind > 0).mean()), 3) if len(ind) else None,
            "strategy_mean": round(float(w["strategy"].mean()), 5),
            "universe_mean": round(float(w["universe"].mean()), 5),
        }
    x = df["edge"].to_numpy(float)
    out["full"] = {"scan_dates": len(df), "mean_edge": round(float(x.mean()), 5),
                   "newey_west_t": round(newey_west_t(x, overlap), 2),
                   "share_positive": round(float((x > 0).mean()), 3)}
    out["overlap_lag"] = overlap
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(ROOT / "config.yaml"))
    ap.add_argument("--spec", default=None)
    ap.add_argument("--spec-json", default=None)
    ap.add_argument("--out", default=str(ROOT / "output" / "research" / "final_validation.json"))
    a = ap.parse_args()
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")

    specs = []
    if a.spec_json:
        specs.append(json.loads(a.spec_json))
    if a.spec:
        p = Path(a.spec)
        files = sorted(p.glob("*.json")) if p.is_dir() else [p]
        specs += [json.loads(f.read_text(encoding="utf-8")) for f in files]
    if not specs:
        raise SystemExit("give --spec or --spec-json")

    ctx = load_context(a.config)
    results = []
    for spec in specs:
        df = per_date_edge(spec, ctx)
        if df.empty:
            print(f"{spec['name']}: no usable scan dates")
            continue
        exit_pol = {**DEFAULT_EXIT, **(spec.get("exit") or {})}
        s = summarise(df, int(exit_pol["max_hold"]))
        results.append({"name": spec["name"], "spec": spec, "stats": s})

        print(f"\n{'='*92}\n{spec['name']}   (benchmark = ALL eligible names, "
              f"overlap lag {s['overlap_lag']})\n{'='*92}")
        print(f"{'window':<11}{'scans':>7}{'mean edge':>11}{'%pos':>7}{'naive t':>9}"
              f"{'NW t':>7}{'indep':>7}{'cohort edge':>13}{'%pos':>7}")
        for w in ("train", "validate", "holdout", "full"):
            v = s.get(w, {})
            if not v.get("scan_dates"):
                continue
            print(f"{w:<11}{v['scan_dates']:>7}{v.get('mean_edge', 0):>+11.5f}"
                  f"{(v.get('share_positive') or 0):>7.2f}{(v.get('naive_t') or 0):>9.2f}"
                  f"{(v.get('newey_west_t') or 0):>7.2f}"
                  f"{v.get('independent_cohorts', 0):>7}"
                  f"{(v.get('cohort_mean_edge') or 0):>+13.5f}"
                  f"{(v.get('cohort_share_positive') or 0):>7.2f}")

    dest = Path(a.out)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(results, indent=1, default=str), encoding="utf-8")
    print(f"\nwrote {dest}")


if __name__ == "__main__":
    main()
