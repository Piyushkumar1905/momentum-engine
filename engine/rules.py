"""Screening rules, scoring and explanations for the two setup models.

Model A - momentum swing (strength continues)
Model B - confirmed 52W-low reversal (weakness exhausted)
Control - naive 52W-low buying (expected to be the weakest; tests the anchoring hypothesis)
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)
NOT_IN_V1 = {"fundamentals", "catalyst"}   # need non-price data; weights re-distributed


def base_mask(F: dict, cfg: dict) -> pd.DataFrame:
    f = cfg["filters"]
    return ((F["history"] >= f["min_history_days"])
            & (F["med_value_cr"] >= f["min_median_value_cr"])
            & (F["close"] >= f["min_price"]))


def _threshold(cfg_model: dict, key: str, regime: pd.Series, mode: str) -> pd.Series:
    th = pd.Series(float(cfg_model[key]), index=regime.index)
    over = (cfg_model.get("bear_overrides") or {})
    if mode == "tighten" and key in over:
        th[regime == "bear"] = float(over[key])
    return th


def _rank(x: pd.DataFrame, mask: pd.DataFrame) -> pd.DataFrame:
    return x.where(mask).rank(axis=1, pct=True)


def _combine(components: dict, weights: dict) -> tuple[pd.DataFrame, dict]:
    used = {k: float(w) for k, w in weights.items() if k in components and k not in NOT_IN_V1}
    total = sum(used.values())
    score = sum(components[k].fillna(0) * w for k, w in used.items()) / total
    return score, {k: round(w / total, 3) for k, w in used.items()}


# --------------------------------------------------------------------------- Model A
def momentum_model(F: dict, regime: pd.DataFrame, cfg: dict, base: pd.DataFrame):
    m, mode = cfg["momentum"], cfg["regime"]["mode"]
    reg = regime["regime"]
    th = lambda k: _threshold(m, k, reg, mode)
    elig = (base
            & F["dist_high"].ge(th("min_dist_52w_high"), axis=0)
            & (F["atr_pct"] >= m["min_atr_pct"]) & (F["atr_pct"] <= m["max_atr_pct"])
            & (F["ext_atr"] <= m["max_ext_atr"])
            & (F["ext_50"] <= m["max_ext_50dma"])
            & F["rs_63"].ge(th("min_rs_63"), axis=0))
    if m.get("require_ma_alignment", True):
        elig &= F["ma_aligned"]
    if mode == "block":
        elig = elig.mul((reg != "bear").astype(bool), axis=0).astype(bool)

    R = lambda x: _rank(x, base)
    brk_quality = (F["brk_recent"].fillna(0)
                   * (0.5 * F["brk_close_loc"].fillna(0) + 0.5 * (F["brk_rvol"].fillna(0) / 2).clip(upper=1)))
    comps = {
        "price_momentum": (R(F["mom_6m_skip"]) + R(F["vam"]) + R(F["ret_20"])) / 3,
        "relative_strength": (R(F["rs_63"]) + R(F["rs_near_high"])) / 2,
        "volume": (R(F["rvol"].clip(upper=5)) + R(F["updown_50"]) + (1 - R(F["dryup"]))) / 3,
        "structure": (R(F["slope200"]) + (1 - R(F["atr_contraction"])) + (1 - R(F["ext_atr"].abs()))) / 3,
        "breakout": brk_quality,
        "sector": F["sector_rank"],
    }
    score, used = _combine(comps, m["weights"])
    return score.where(elig), comps, used


def momentum_setup(row) -> str:
    if row.get("brk_recent", 0) == 1:
        rv, cl, depth = row.get("brk_rvol", 0) or 0, row.get("brk_close_loc", 0) or 0, row.get("base_depth", 1)
        grade = "A" if (rv >= 2 and cl >= 0.75 and depth <= 0.25) else "B" if rv >= 1.5 else "C (weak volume)"
        return f"Breakout - grade {grade}"
    if row.get("ext_atr", 9) <= 1 and row.get("dist_high", 1) < 0.95:
        return "Pullback in uptrend"
    if row.get("atr_contraction", 1) < 0.75:
        return "Volatility contraction"
    return "Momentum continuation"


# --------------------------------------------------------------------------- Model B
def reversal_model(F: dict, regime: pd.DataFrame, cfg: dict, base: pd.DataFrame):
    r, mode = cfg["reversal"], cfg["regime"]["mode"]
    reg = regime["regime"]
    elig = (base
            & (F["drawdown"] >= r["min_drawdown"])
            & F["low_in_window"]
            & (F["higher_low_atr"] >= r["higher_low_atr"])
            & (F["support_days"] >= r["min_support_days"])
            & (F["atr_pct"] >= 0.01) & (F["atr_pct"] <= 0.08))
    if r.get("require_reclaim", True):
        elig &= F["reclaimed"]
    if r.get("require_rs_divergence", True):
        elig &= F["rs_div"] > 0
    if mode == "block":
        elig = elig.mul((reg != "bear").astype(bool), axis=0).astype(bool)
    R = lambda x: _rank(x, base)
    comps = {
        "rs_divergence": R(F["rs_div"]),
        "reclaim": R(F["reclaim_strength"].clip(upper=0.15)),
        "accumulation": R(F["updown_20"]),
        "higher_low": R(F["higher_low_atr"].clip(upper=6)),
        "capitulation": R(F["cap_rvol"]),
        "sector": F["sector_rank"],
    }
    score, used = _combine(comps, r["weights"])
    return score.where(elig), comps, used


# --------------------------------------------------------------------------- control
def naive_low_control(F: dict, cfg: dict, base: pd.DataFrame):
    within = cfg["controls"]["naive_low_within"]
    elig = base & (F["dist_low"] <= 1 + within)
    return (-F["dist_low"]).where(elig)         # closest to the low ranks first


# --------------------------------------------------------------------------- stops / text
def stop_price(model: str, row: dict, cfg: dict) -> float:
    if model == "momentum":
        return row["close"] - cfg["momentum"]["stop_atr"] * row["atr"]
    if model == "reversal":
        r = cfg["reversal"]
        anchor = row["low15"] if r.get("stop_basis", "higher_low") == "higher_low" else row["lo252"]
        return anchor - r["stop_atr_below_low"] * row["atr"]
    return row["close"] - 1.5 * row["atr"]


def target_r(model: str, cfg: dict) -> float:
    return {"momentum": cfg["momentum"]["target_r"], "reversal": cfg["reversal"]["target_r"]}.get(model, 2.0)


def explain(model: str, x: dict, regime: str) -> tuple[str, str]:
    """Plain-language reasons and risk flags (observations from price data only)."""
    why, flags = [], []
    if model == "momentum":
        why += [f"6M momentum (ex-last month) {x['mom_6m_skip']:+.0%}",
                f"RS vs Nifty 63D {x['rs_63']:+.1%}",
                f"at {x['dist_high']:.0%} of 52W high",
                f"RVOL {x['rvol']:.1f}x", f"{x['ext_atr']:.1f} ATR above 20DMA",
                f"sector rank {x['sector_rank']:.0%}"]
        if x["ext_atr"] > 2.5:
            flags.append("near extension limit")
        if x["rvol"] < 1 and x.get("brk_recent", 0) == 1:
            flags.append("breakout on below-average volume")
    elif model == "reversal":
        why += [f"fell {x['drawdown']:.0%} before basing",
                f"RS higher low {x['rs_div']:+.1%}",
                f"{x['higher_low_atr']:.1f} ATR above the low",
                f"{int(x['support_days'])} prior sessions tested the zone",
                f"capitulation RVOL {x['cap_rvol']:.1f}x",
                f"up/down volume 20D {x['updown_20']:.2f}"]
        if x["cap_rvol"] < 2:
            flags.append("no clear capitulation volume")
        if x["sector_rank"] < 0.33:
            flags.append("weak sector")
    if x["atr_pct"] > 0.045:
        flags.append("high volatility")
    if regime == "bear":
        flags.append("bear regime - smaller size")
    flags.append("check results date, pledge, news (not in v1 data)")
    return "; ".join(why), "; ".join(flags)
