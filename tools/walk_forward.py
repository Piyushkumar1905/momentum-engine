"""Six-month block analysis and parameter-stability mapping.

Two questions a single backtest number cannot answer:

  1. "If I run this for the next six months, what actually happens?"
     Answered by splitting history into consecutive six-month blocks and reporting the
     DISTRIBUTION of outcomes, not the average. An average of +0.7% built from +6% and
     -4% is a different proposition from one built from +0.8% and +0.6%.

  2. "Are these the right parameter values?"
     Answered by mapping the neighbourhood. A parameter sitting on a sharp peak was
     fitted; one sitting on a plateau is robust. Deploy from the middle of a plateau,
     never from a peak, even when the peak scores higher.

The spec is held FIXED across every block - no refitting. That is deliberate: it
measures whether one fixed configuration survives regime change, which is what running
it for six months actually is.

  python tools/walk_forward.py --spec specs/RECOMMENDED.json --blocks
  python tools/walk_forward.py --spec specs/RECOMMENDED.json --sweep top_n=10,20,30,40
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.final_validation import newey_west_t, per_date_edge   # noqa: E402
from tools.strategy_lab import DEFAULT_EXIT, load_context        # noqa: E402

log = logging.getLogger("walk_forward")


def blocks(df: pd.DataFrame, months: int = 6) -> pd.DataFrame:
    """Consecutive non-overlapping calendar blocks of `months`.

    Built by integer-dividing the month index rather than pd.to_period(f"{months}M"),
    which silently returns MONTHLY periods and quietly turned 33 six-month blocks into
    196 one-month ones.
    """
    d = df.copy()
    mi = d["date"].dt.year * 12 + (d["date"].dt.month - 1)
    grp = (mi - mi.min()) // months
    lab = (mi.min() + grp * months)
    d["block"] = [f"{v // 12}-{v % 12 + 1:02d}" for v in lab]
    g = d.groupby("block").agg(
        scans=("edge", "size"),
        mean_edge=("edge", "mean"),
        median_edge=("edge", "median"),
        pct_positive=("edge", lambda x: float((x > 0).mean())),
        strategy=("strategy", "mean"),
        universe=("universe", "mean"),
        start=("date", "min"), end=("date", "max"))
    return g.reset_index()


def sweep_axis(spec: dict, ctx: dict, axis: str, values: list) -> list:
    """Vary one parameter, hold everything else fixed, report the surface."""
    out = []
    for v in values:
        s = json.loads(json.dumps(spec))
        s["name"] = f"{spec['name']}__{axis}={v}"
        if axis in ("top_n",):
            s[axis] = int(v)
        elif axis in ("max_hold", "stop_atr", "trail_atr"):
            s.setdefault("exit", {})[axis] = (int(v) if axis == "max_hold" else float(v))
        elif axis in ("max_vol_ann", "min_close_over_200dma"):
            s.setdefault("filters", {})[axis] = float(v)
        else:
            raise SystemExit(f"unknown sweep axis {axis}")
        df = per_date_edge(s, ctx)
        if df.empty:
            out.append({"value": v, "scans": 0})
            continue
        mh = int({**DEFAULT_EXIT, **(s.get("exit") or {})}["max_hold"])
        lag = max(1, int(np.ceil(mh / 5)))
        x = df["edge"].to_numpy(float)
        b = blocks(df)
        out.append({"value": v, "scans": len(df),
                    "mean_edge": round(float(x.mean()), 5),
                    "nw_t": round(newey_west_t(x, lag), 2),
                    "pct_scans_positive": round(float((x > 0).mean()), 3),
                    "blocks": len(b),
                    "pct_blocks_positive": round(float((b["mean_edge"] > 0).mean()), 3),
                    "worst_block": round(float(b["mean_edge"].min()), 5)})
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(ROOT / "config.yaml"))
    ap.add_argument("--start", default=None, help="widen the analysis range, e.g. 2009-01-01")
    ap.add_argument("--end", default=None)
    ap.add_argument("--spec", required=True)
    ap.add_argument("--blocks", action="store_true")
    ap.add_argument("--months", type=int, default=6)
    ap.add_argument("--sweep", default=None, help="axis=v1,v2,v3 (repeatable, comma-joined)")
    ap.add_argument("--out", default=str(ROOT / "output" / "research" / "walk_forward.json"))
    a = ap.parse_args()
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")

    spec = json.loads(Path(a.spec).read_text(encoding="utf-8"))
    ctx = load_context(a.config, a.start, a.end)
    result = {"spec": spec}

    if a.blocks or not a.sweep:
        df = per_date_edge(spec, ctx)
        if df.empty:
            raise SystemExit("no usable scan dates")
        mh = int({**DEFAULT_EXIT, **(spec.get("exit") or {})}["max_hold"])
        lag = max(1, int(np.ceil(mh / 5)))
        b = blocks(df, a.months)
        x = df["edge"].to_numpy(float)
        pos = int((b["mean_edge"] > 0).sum())
        result["blocks"] = b.to_dict(orient="records")
        result["summary"] = {
            "blocks": len(b), "blocks_positive": pos,
            "share_blocks_positive": round(pos / len(b), 3),
            "mean_block_edge": round(float(b["mean_edge"].mean()), 5),
            "median_block_edge": round(float(b["mean_edge"].median()), 5),
            "worst_block": round(float(b["mean_edge"].min()), 5),
            "best_block": round(float(b["mean_edge"].max()), 5),
            "block_sd": round(float(b["mean_edge"].std()), 5),
            "full_nw_t": round(newey_west_t(x, lag), 2),
        }
        print(f"\n{spec['name']}  -  {a.months}-month blocks "
              f"(edge = picks minus every eligible name, same dates)")
        print(f"{'block':<12}{'window':<26}{'scans':>7}{'mean edge':>12}{'%dates +':>10}")
        print("-" * 68)
        for r in result["blocks"]:
            win = f"{pd.Timestamp(r['start']).date()} to {pd.Timestamp(r['end']).date()}"
            print(f"{r['block']:<12}{win:<26}{r['scans']:>7}{r['mean_edge']:>+12.5f}"
                  f"{r['pct_positive']:>10.2f}")
        s = result["summary"]
        print("-" * 68)
        print(f"{s['blocks_positive']}/{s['blocks']} blocks positive "
              f"({s['share_blocks_positive']:.0%})   mean {s['mean_block_edge']:+.4f}   "
              f"median {s['median_block_edge']:+.4f}")
        print(f"worst block {s['worst_block']:+.4f}   best {s['best_block']:+.4f}   "
              f"sd {s['block_sd']:.4f}   full NW t {s['full_nw_t']}")

    if a.sweep:
        result["sweeps"] = {}
        for part in a.sweep.split(";"):
            axis, vals = part.split("=")
            values = [float(v) if "." in v else int(v) for v in vals.split(",")]
            rows = sweep_axis(spec, ctx, axis.strip(), values)
            result["sweeps"][axis.strip()] = rows
            print(f"\nSensitivity: {axis}")
            print(f"{'value':>10}{'mean edge':>12}{'NW t':>8}{'%dates +':>10}"
                  f"{'%blocks +':>11}{'worst block':>13}")
            print("-" * 64)
            for r in rows:
                if not r.get("scans"):
                    print(f"{r['value']:>10}   (no data)")
                    continue
                print(f"{r['value']:>10}{r['mean_edge']:>+12.5f}{r['nw_t']:>8.2f}"
                      f"{r['pct_scans_positive']:>10.2f}{r['pct_blocks_positive']:>11.2f}"
                      f"{r['worst_block']:>+13.5f}")

    dest = Path(a.out)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(result, indent=1, default=str), encoding="utf-8")
    print(f"\nwrote {dest}")


if __name__ == "__main__":
    main()
