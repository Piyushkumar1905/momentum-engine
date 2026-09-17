"""Threshold variants, judged in-sample and out-of-sample separately.

The miss analysis says two momentum rules reject stocks that go on to outperform.
That observation is in-sample by construction, and acting on it directly is the
overfitting the project's own discipline forbids.

So each variant is fitted on the FIRST half and judged on the SECOND, which it never
informed. A change is only worth keeping if it improves both. Anything that improves
in-sample and decays out-of-sample is a curve fit, and is reported as one.

  python tools/sweep_thresholds.py --start 2024-09-01 --split 2025-09-01
"""
from __future__ import annotations

import argparse
import copy
import json
import logging
import sys
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from engine import backtest as bt          # noqa: E402
from engine import data                    # noqa: E402

log = logging.getLogger("sweep")

VARIANTS = {
    "baseline": {},
    "relax_ext_50dma": {"momentum.max_ext_50dma": 0.35},
    "relax_52w_high": {"momentum.min_dist_52w_high": 0.75},
    "relax_both": {"momentum.max_ext_50dma": 0.35, "momentum.min_dist_52w_high": 0.75},
    "drop_ext_50dma": {"momentum.max_ext_50dma": 99.0},
    "tighten_rs": {"momentum.min_rs_63": 0.05},
}


def apply(cfg: dict, patch: dict) -> dict:
    c = copy.deepcopy(cfg)
    for path, v in patch.items():
        sec, key = path.split(".")
        c[sec][key] = v
    return c


def stats(t: pd.DataFrame) -> dict:
    if t.empty:
        return {"trades": 0, "win_rate": None, "expectancy": None, "profit_factor": None}
    r = t["ret"]
    w, l = r[r > 0], r[r <= 0]
    return {"trades": len(t), "win_rate": round(len(w) / len(t), 4),
            "expectancy": round(r.mean(), 5),
            "profit_factor": round(w.sum() / -l.sum(), 3) if l.sum() < 0 else None,
            "avg_R": round(t["r_multiple"].mean(), 3)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(ROOT / "config.yaml"))
    ap.add_argument("--start", default="2024-09-01")
    ap.add_argument("--split", default="2025-09-01")
    ap.add_argument("--model", default="momentum")
    ap.add_argument("--out", default=str(ROOT / "output" / "analysis" / "sweep.json"))
    a = ap.parse_args()
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")

    base_cfg = yaml.safe_load(open(a.config))
    base_cfg["backtest"]["start"] = a.start
    uni = data.load_universe(base_cfg["run"]["universe_file"], base_cfg["run"].get("max_symbols"))
    panel = data.load_panel(base_cfg, uni)          # loaded once, reused by every variant

    split = pd.Timestamp(a.split)
    rows = []
    for name, patch in VARIANTS.items():
        cfg = apply(base_cfg, patch)
        res = bt.run_backtest(panel, uni, cfg)
        t = res["trades"]
        t = t[t["model"] == a.model] if len(t) else t
        ins = t[t["date"] < split] if len(t) else t
        oos = t[t["date"] >= split] if len(t) else t
        row = {"variant": name, "patch": json.dumps(patch) if patch else "-",
               "in_sample": stats(ins), "out_of_sample": stats(oos), "full": stats(t)}
        rows.append(row)
        i, o = row["in_sample"], row["out_of_sample"]
        print(f"{name:<18} IS: n={i['trades']:<5} PF={i['profit_factor']} exp={i['expectancy']}"
              f"   OOS: n={o['trades']:<5} PF={o['profit_factor']} exp={o['expectancy']}")

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(
        {"model": a.model, "start": a.start, "split": a.split, "variants": rows},
        indent=1), encoding="utf-8")
    print(f"\nwrote {a.out}")

    base = next(r for r in rows if r["variant"] == "baseline")
    print("\nVerdict (a change must beat baseline in BOTH halves to be worth keeping):")
    for r in rows:
        if r["variant"] == "baseline":
            continue
        bi, bo = base["in_sample"], base["out_of_sample"]
        ri, ro = r["in_sample"], r["out_of_sample"]
        if None in (ri["expectancy"], ro["expectancy"], bi["expectancy"], bo["expectancy"]):
            continue
        better_is, better_oos = ri["expectancy"] > bi["expectancy"], ro["expectancy"] > bo["expectancy"]
        verdict = ("KEEP - improves both" if better_is and better_oos else
                   "REJECT - curve fit (in-sample only)" if better_is else
                   "REJECT - worse in-sample" if better_oos else "REJECT - worse in both")
        print(f"  {r['variant']:<18} {verdict}")


if __name__ == "__main__":
    main()
