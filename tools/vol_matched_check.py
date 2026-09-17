"""Is the edge selection skill, or just compensation for holding riskier stocks?

Comparing picks against the whole eligible universe is the right test ONLY if the picks
carry similar risk. If the ranking systematically selects high-volatility names, part of
the measured edge is a volatility premium the strategy is being paid for bearing - not
skill, and it reverses hard in a crash.

The test: for every scan date, benchmark each pick against eligible names in the SAME
ex-ante volatility decile, then average. If the edge survives volatility matching, the
ranking is finding something beyond risk. If it collapses, it is a leveraged beta bet.

  python tools/vol_matched_check.py --config config_research.yaml --start 2009-01-01 \
      --spec specs/RECOMMENDED.json
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

from engine import backtest as bt          # noqa: E402
from engine.factor_model import build_score   # noqa: E402
from tools.final_validation import newey_west_t   # noqa: E402
from tools.research_exits import simulate   # noqa: E402
from tools.strategy_lab import DEFAULT_EXIT, load_context   # noqa: E402

log = logging.getLogger("vol_matched")


def run(spec: dict, ctx: dict, deciles: int = 10) -> dict:
    panel, F, base, dates = ctx["panel"], ctx["F"], ctx["base"], ctx["dates"]
    O, H, L, C, CA, ATR = ctx["O"], ctx["H"], ctx["L"], ctx["C"], ctx["CA"], ctx["ATR"]
    ex = {**DEFAULT_EXIT, **(spec.get("exit") or {})}
    mh, top_n = int(ex["max_hold"]), int(spec.get("top_n", 20))
    score = build_score(spec, panel, F, base)
    VOL = F["vol_ann"].to_numpy(float)
    T = len(dates)

    rows = []
    for i in ctx["idxs"]:
        if i + 1 + mh >= T:
            continue
        elig = np.where(base[i] & np.isfinite(O[i + 1]) & np.isfinite(VOL[i]))[0]
        if elig.size < 100:
            continue
        picks = bt._top(score[i], top_n)
        picks = np.array([j for j in picks if np.isfinite(VOL[i, j])], dtype=int)
        if picks.size < 5:
            continue

        ret = {}
        for j in elig:
            t = simulate(i, int(j), None, ATR[i, int(j)], O, H, L, C, CA, ex)
            if t:
                ret[int(j)] = t["ret"]
        if len(ret) < 80:
            continue
        have = np.array([j for j in elig if j in ret], dtype=int)
        v = VOL[i, have]
        # decile edges from the eligible pool on this date
        cuts = np.quantile(v, np.linspace(0, 1, deciles + 1))
        binid = np.clip(np.searchsorted(cuts, v, side="right") - 1, 0, deciles - 1)
        bin_mean = {b: float(np.mean([ret[j] for j, bb in zip(have, binid) if bb == b]))
                    for b in range(deciles)
                    if any(bb == b for bb in binid)}

        pk = [j for j in picks if j in ret]
        if len(pk) < 5:
            continue
        raw = float(np.mean([ret[j] for j in pk]))
        uni = float(np.mean(list(ret.values())))
        matched = []
        for j in pk:
            b = int(np.clip(np.searchsorted(cuts, VOL[i, j], side="right") - 1, 0, deciles - 1))
            if b in bin_mean:
                matched.append(ret[j] - bin_mean[b])
        if not matched:
            continue
        rows.append({"date": dates[i],
                     "edge_raw": raw - uni,
                     "edge_vol_matched": float(np.mean(matched)),
                     "pick_vol": float(np.mean(VOL[i, pk])),
                     "universe_vol": float(np.mean(v))})
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(ROOT / "config.yaml"))
    ap.add_argument("--start", default=None)
    ap.add_argument("--end", default=None)
    ap.add_argument("--spec", required=True)
    ap.add_argument("--out", default=str(ROOT / "output" / "research" / "vol_matched.json"))
    a = ap.parse_args()
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")

    spec = json.loads(Path(a.spec).read_text(encoding="utf-8"))
    ctx = load_context(a.config, a.start, a.end)
    df = run(spec, ctx)
    if df.empty:
        raise SystemExit("no usable dates")

    ex = {**DEFAULT_EXIT, **(spec.get("exit") or {})}
    lag = max(1, int(np.ceil(int(ex["max_hold"]) / 5)))
    raw, vm = df["edge_raw"].to_numpy(float), df["edge_vol_matched"].to_numpy(float)
    ratio = float(df["pick_vol"].mean() / df["universe_vol"].mean())

    out = {"spec": spec["name"], "scans": len(df),
           "pick_vol": round(float(df["pick_vol"].mean()), 4),
           "universe_vol": round(float(df["universe_vol"].mean()), 4),
           "vol_ratio": round(ratio, 3),
           "edge_raw": round(float(raw.mean()), 5), "raw_nw_t": round(newey_west_t(raw, lag), 2),
           "edge_vol_matched": round(float(vm.mean()), 5),
           "matched_nw_t": round(newey_west_t(vm, lag), 2),
           "share_retained": round(float(vm.mean() / raw.mean()), 3) if raw.mean() else None}

    print(f"\n{spec['name']}  -  volatility-matched benchmark ({len(df)} scan dates)")
    print("=" * 72)
    print(f"picks average {out['pick_vol']:.1%} annualised volatility; "
          f"universe {out['universe_vol']:.1%}  ->  ratio {out['vol_ratio']}x")
    print()
    print(f"{'benchmark':<28}{'mean edge':>12}{'NW t':>9}")
    print("-" * 49)
    print(f"{'whole eligible universe':<28}{out['edge_raw']:>+12.5f}{out['raw_nw_t']:>9.2f}")
    print(f"{'same volatility decile':<28}{out['edge_vol_matched']:>+12.5f}{out['matched_nw_t']:>9.2f}")
    print()
    if out["share_retained"] is not None:
        print(f"{out['share_retained']:.0%} of the edge survives volatility matching.")
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
