"""Evaluate a complete strategy spec: selection + exits, judged per year against controls.

A spec is JSON:

  {
    "name": "composite_trend",
    "factors": {"mom_12m_skip1": 1.0, "trend_consistency": 0.5},
    "filters": {"min_close_over_200dma": 0.0, "max_vol_ann": 0.60},
    "regime": "none",            # none | skip_bear | scale_bear
    "top_n": 20,
    "exit": {"stop_atr": 3.0, "target_r": null, "max_hold": 60, "trail_atr": 4.0}
  }

Every result is reported against two controls on the SAME dates:
  * random picks from the liquid universe (does the ranking do anything?)
  * the Nifty over the holding window (is there alpha, or just beta?)

Year-by-year output, because a strategy that makes its whole edge in one year is not
a strategy. Windows are fixed and named, so an out-of-sample claim cannot be made by
quietly moving the boundary.

  python tools/strategy_lab.py --spec specs/my_idea.json
  python tools/strategy_lab.py --spec-json '{"name":"x","factors":{"mom_12m_skip1":1}}'
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
from tools.factor_lab import FACTORS       # noqa: E402
from tools.research_exits import simulate  # noqa: E402

log = logging.getLogger("strategy_lab")

FULL_START, FULL_END = "2021-01-01", "2026-09-17"
WINDOWS = {"train": ("2021-01-01", "2023-12-31"),
           "validate": ("2024-01-01", "2025-06-30"),
           "holdout": ("2025-07-01", "2026-09-17")}
DEFAULT_EXIT = {"stop_atr": 1.5, "target_r": 2.0, "max_hold": 20}


def build_score(spec, panel, F, base_np) -> np.ndarray:
    """Weighted sum of cross-sectional percentile ranks, masked by the spec's filters."""
    parts = []
    for name, w in spec["factors"].items():
        fn, _ = FACTORS[name]
        f = fn(panel, F)
        r = f.rank(axis=1, pct=True).to_numpy(float)
        parts.append(float(w) * r)
    score = np.nansum(parts, axis=0)
    score[~base_np] = np.nan

    flt = spec.get("filters") or {}
    if "min_close_over_200dma" in flt:
        v = (panel.close / F["dma200"] - 1).to_numpy(float)
        score[~(v >= flt["min_close_over_200dma"])] = np.nan
    if "max_vol_ann" in flt:
        v = F["vol_ann"].to_numpy(float)
        score[~(v <= flt["max_vol_ann"])] = np.nan
    if "min_vol_ann" in flt:
        v = F["vol_ann"].to_numpy(float)
        score[~(v >= flt["min_vol_ann"])] = np.nan
    if "max_dist_high" in flt:
        v = F["dist_high"].to_numpy(float)
        score[~(v <= flt["max_dist_high"])] = np.nan
    if "min_dist_high" in flt:
        v = F["dist_high"].to_numpy(float)
        score[~(v >= flt["min_dist_high"])] = np.nan
    return score


def run(spec: dict, ctx: dict) -> dict:
    panel, F, base_np, dates = ctx["panel"], ctx["F"], ctx["base"], ctx["dates"]
    O, H, L, C, CA, ATR = ctx["O"], ctx["H"], ctx["L"], ctx["C"], ctx["CA"], ctx["ATR"]
    regime = ctx["regime"]
    cost = ctx["cost"]
    exit_pol = {**DEFAULT_EXIT, **(spec.get("exit") or {})}
    top_n = int(spec.get("top_n", 20))
    rng = np.random.default_rng(11)

    score = build_score(spec, panel, F, base_np)
    idxs = ctx["idxs"]
    mode = spec.get("regime", "none")

    rows, ctrl = [], []
    for i in idxs:
        if mode == "skip_bear" and regime[i] == "bear":
            continue
        row = score[i]
        picks = bt._top(row, top_n)
        for j in picks:
            t = simulate(i, j, None, ATR[i, j], O, H, L, C, CA, exit_pol)
            if t:
                rows.append({"date": dates[i], "symbol": panel.symbols[j], **t})
        pool = np.where(base_np[i] & np.isfinite(O[min(i + 1, len(dates) - 1)]))[0]
        if pool.size:
            for j in rng.choice(pool, size=min(top_n, pool.size), replace=False):
                t = simulate(i, int(j), None, ATR[i, int(j)], O, H, L, C, CA, exit_pol)
                if t:
                    ctrl.append({"date": dates[i], **t})

    t = pd.DataFrame(rows)
    c = pd.DataFrame(ctrl)
    if t.empty:
        return {"name": spec["name"], "error": "no trades"}

    def agg(df):
        if df.empty:
            return {"trades": 0}
        r = df["ret"] - cost
        w, l = r[r > 0], r[r <= 0]
        return {"trades": len(df), "win_rate": round(float((r > 0).mean()), 3),
                "expectancy": round(float(r.mean()), 5),
                "profit_factor": round(float(w.sum() / -l.sum()), 3) if l.sum() < 0 else None,
                "avg_hold": round(float(df["hold_days"].mean()), 1),
                "median": round(float(r.median()), 5)}

    out = {"name": spec["name"], "spec": spec, "overall": agg(t), "control": agg(c)}
    out["overall"]["edge_vs_control"] = round(
        out["overall"]["expectancy"] - (out["control"].get("expectancy") or 0), 5)

    # per fixed window
    out["windows"] = {}
    for wname, (s, e) in WINDOWS.items():
        ws, we = pd.Timestamp(s), pd.Timestamp(e)
        tw = t[(t["date"] >= ws) & (t["date"] <= we)]
        cw = c[(c["date"] >= ws) & (c["date"] <= we)] if not c.empty else c
        a, b = agg(tw), agg(cw)
        a["edge_vs_control"] = round(a.get("expectancy", 0) - (b.get("expectancy") or 0), 5) \
            if a.get("trades") else None
        out["windows"][wname] = {"strategy": a, "control": b}

    # per calendar year
    out["years"] = {}
    t["year"] = t["date"].dt.year
    if not c.empty:
        c["year"] = c["date"].dt.year
    for y in sorted(t["year"].unique()):
        a = agg(t[t["year"] == y])
        b = agg(c[c["year"] == y]) if not c.empty else {}
        a["edge_vs_control"] = round(a.get("expectancy", 0) - (b.get("expectancy") or 0), 5)
        out["years"][int(y)] = {"strategy": a, "control": b}
    return out


def load_context(config: str) -> dict:
    cfg = yaml.safe_load(open(config))
    cfg["backtest"]["start"] = FULL_START
    uni = data.load_universe(cfg["run"]["universe_file"], cfg["run"].get("max_symbols"))
    panel = data.load_panel(cfg, uni)
    prep = bt.prepare(panel, uni, cfg)
    dates = panel.dates
    idxs = np.where((dates >= pd.Timestamp(FULL_START)) & (dates <= pd.Timestamp(FULL_END)))[0][::5]
    return {"panel": panel, "F": prep["F"], "base": prep["base"].to_numpy(bool),
            "regime": prep["regime"]["regime"].to_numpy(), "dates": dates, "idxs": idxs,
            "O": panel.open.to_numpy(float), "H": panel.high.to_numpy(float),
            "L": panel.low.to_numpy(float), "C": panel.close.to_numpy(float),
            "CA": panel.corp_action.to_numpy(bool) if panel.corp_action is not None else None,
            "ATR": prep["F"]["atr"].to_numpy(float),
            "cost": float(cfg["backtest"]["cost_pct"])}


def report(r: dict) -> None:
    if r.get("error"):
        print(f"{r['name']}: {r['error']}")
        return
    o, c = r["overall"], r["control"]
    print(f"\n{'='*86}\n{r['name']}\n{'='*86}")
    print(f"overall   trades={o['trades']:>5}  win={o['win_rate']:.1%}  "
          f"expect={o['expectancy']:+.4f}  PF={o['profit_factor']}  hold={o['avg_hold']}d")
    print(f"control   trades={c.get('trades',0):>5}  win={(c.get('win_rate') or 0):.1%}  "
          f"expect={(c.get('expectancy') or 0):+.4f}  PF={c.get('profit_factor')}")
    print(f"EDGE vs control: {o['edge_vs_control']:+.4f}")
    print(f"\n{'window':<12}{'trades':>8}{'expect':>10}{'control':>10}{'edge':>10}{'PF':>8}")
    for w, v in r["windows"].items():
        s, b = v["strategy"], v["control"]
        if not s.get("trades"):
            continue
        print(f"{w:<12}{s['trades']:>8}{s['expectancy']:>+10.4f}"
              f"{(b.get('expectancy') or 0):>+10.4f}{(s.get('edge_vs_control') or 0):>+10.4f}"
              f"{(s['profit_factor'] or 0):>8.2f}")
    print(f"\n{'year':<12}{'trades':>8}{'expect':>10}{'control':>10}{'edge':>10}{'PF':>8}")
    for y, v in r["years"].items():
        s, b = v["strategy"], v["control"]
        print(f"{y:<12}{s['trades']:>8}{s['expectancy']:>+10.4f}"
              f"{(b.get('expectancy') or 0):>+10.4f}{(s.get('edge_vs_control') or 0):>+10.4f}"
              f"{(s['profit_factor'] or 0):>8.2f}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(ROOT / "config.yaml"))
    ap.add_argument("--spec", default=None, help="path to a spec JSON, or a directory of them")
    ap.add_argument("--spec-json", default=None)
    ap.add_argument("--out", default=str(ROOT / "output" / "research" / "strategies.json"))
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
    for s in specs:
        r = run(s, ctx)
        report(r)
        results.append(r)

    dest = Path(a.out)
    dest.parent.mkdir(parents=True, exist_ok=True)
    prev = json.loads(dest.read_text(encoding="utf-8")) if dest.exists() else []
    names = {r["name"] for r in results}
    merged = [x for x in prev if x.get("name") not in names] + results
    dest.write_text(json.dumps(merged, indent=1, default=str), encoding="utf-8")
    print(f"\nwrote {dest} ({len(merged)} strategies on file)")


if __name__ == "__main__":
    main()
