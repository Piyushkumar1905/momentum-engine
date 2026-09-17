"""Spec-driven scoring model - the deployed strategy.

A spec (specs/RECOMMENDED.json) describes a complete selection rule: which factors to
rank by, which filters constrain the selectable universe, and how many names to take.
The research harness and the live daily scan both score through this function, so what
is published is the thing that was validated, not a re-implementation of it.

Scoring is a weighted sum of cross-sectional percentile ranks. Ranks rather than raw
values, because raw factor scales are incomparable and one extreme value would
otherwise dominate the blend.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from .factors import FACTORS

log = logging.getLogger(__name__)

# Filters constrain the universe; they never rank. Each maps to (feature, comparison).
_FILTERS = {
    "max_vol_ann":            ("vol_ann", "le"),
    "min_vol_ann":            ("vol_ann", "ge"),
    "max_dist_high":          ("dist_high", "le"),
    "min_dist_high":          ("dist_high", "ge"),
}


def load_spec(path: str | Path) -> dict | None:
    p = Path(path)
    if not p.exists():
        return None
    spec = json.loads(p.read_text(encoding="utf-8"))
    missing = [f for f in spec.get("factors", {}) if f not in FACTORS]
    if missing:
        raise ValueError(f"{p}: unknown factor(s) {missing}")
    return spec


def build_score(spec: dict, panel, F: dict, base: np.ndarray) -> np.ndarray:
    """Weighted rank score, NaN wherever the stock is not selectable."""
    parts = []
    for name, w in spec.get("factors", {}).items():
        fn, _ = FACTORS[name]
        parts.append(float(w) * fn(panel, F).rank(axis=1, pct=True).to_numpy(float))
    if not parts:
        raise ValueError("spec has no factors")
    score = np.nansum(parts, axis=0)
    score[~base] = np.nan

    flt = spec.get("filters") or {}
    for key, val in flt.items():
        if key == "min_close_over_200dma":
            v = (panel.close / F["dma200"] - 1).to_numpy(float)
            score[~(v >= float(val))] = np.nan
            continue
        if key not in _FILTERS:
            raise ValueError(f"unknown filter {key}")
        feat, how = _FILTERS[key]
        v = F[feat].to_numpy(float)
        ok = v <= float(val) if how == "le" else v >= float(val)
        score[~ok] = np.nan
    return score


def score_frame(spec: dict, panel, F: dict, base: pd.DataFrame) -> pd.DataFrame:
    """build_score as a DataFrame, matching the shape the other models return."""
    arr = build_score(spec, panel, F, base.to_numpy(bool))
    return pd.DataFrame(arr, index=panel.close.index, columns=panel.close.columns)


def stop_and_target(spec: dict, row: dict) -> tuple[float, float]:
    """Trade plan from the spec's exit block. No profit target means the trail decides,
    so the 'target' shown is a reference at the trail distance, not an order."""
    ex = spec.get("exit") or {}
    atr = row["atr"]
    stop = row["close"] - float(ex.get("stop_atr", 2.5)) * atr
    if ex.get("target_r"):
        risk = row["close"] - stop
        return stop, row["close"] + float(ex["target_r"]) * risk
    trail = float(ex.get("trail_atr", 3.5))
    return stop, row["close"] + trail * atr


def explain(spec: dict, row: dict) -> tuple[str, str]:
    # Label each number as the quantity actually being shown. An earlier version
    # printed the 6-month feature under a "12M" label and an ATR-derived figure under
    # "annualised volatility", which read as violating the very cap that selected it.
    why = []
    if row.get("mom_12m") is not None:
        why.append(f"12M momentum (ex-last month) {row['mom_12m']:+.0%}")
    if row.get("mom_6m_skip") is not None:
        why.append(f"6M momentum {row['mom_6m_skip']:+.0%}")
    if row.get("dma200"):
        why.append(f"{row['close'] / row['dma200'] - 1:+.0%} vs 200DMA")
    if row.get("vol_ann") is not None:
        why.append(f"{row['vol_ann']:.0%} annualised volatility")
    why.append(f"{row['dist_high']:.0%} of 52W high")
    flags = []
    ex = spec.get("exit") or {}
    flags.append(f"time exit after {ex.get('max_hold', 40)} sessions")
    if not ex.get("target_r"):
        flags.append("no fixed target - trailing stop decides the exit")
    flags.append("candidate strategy, not significant in every window - size accordingly")
    return "; ".join(x for x in why if x), "; ".join(flags)
