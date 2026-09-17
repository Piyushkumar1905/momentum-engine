"""Feature engineering. Every feature at date t uses data up to and including t only.
(Forward-looking outcome columns live in backtest.py, never here.)"""
from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

from .data import Panel


def _true_range(p: Panel) -> pd.DataFrame:
    prev = p.close.shift(1)
    return np.fmax(p.high - p.low, np.fmax((p.high - prev).abs(), (p.low - prev).abs()))


def _updown_ratio(ret, vol, n):
    up = vol.where(ret > 0, 0.0).rolling(n, min_periods=n).sum()
    dn = vol.where(ret < 0, 0.0).rolling(n, min_periods=n).sum()
    return up / dn.replace(0, np.nan)


def _count_near_level(low: pd.DataFrame, thr: pd.DataFrame, lag_from: int, lag_to: int) -> pd.DataFrame:
    """For each t: number of sessions in [t-lag_to, t-lag_from] whose low <= thr[t]."""
    L = low.to_numpy(float)
    TH = thr.to_numpy(float)
    T, N = L.shape
    W = lag_to - lag_from + 1
    out = np.full((T, N), np.nan)
    if T <= lag_to:
        return pd.DataFrame(out, index=low.index, columns=low.columns)
    win = sliding_window_view(L, W, axis=0)          # win[s] = rows s .. s+W-1
    step = 200
    for s0 in range(0, T - lag_to, step):
        s1 = min(s0 + step, T - lag_to)
        t0, t1 = s0 + lag_to, s1 + lag_to
        cmp = win[s0:s1] <= TH[t0:t1, :, None]
        out[t0:t1] = cmp.sum(axis=2)
    out[np.isnan(TH)] = np.nan
    return pd.DataFrame(out, index=low.index, columns=low.columns)


def compute_features(p: Panel, industry: pd.Series) -> dict[str, pd.DataFrame]:
    c, h, l, o, v = p.close, p.high, p.low, p.open, p.volume
    F: dict[str, pd.DataFrame] = {}
    ret = c.pct_change(fill_method=None)

    # --- momentum
    F["ret_20"] = c / c.shift(20) - 1
    F["ret_63"] = c / c.shift(63) - 1
    F["mom_6m_skip"] = c.shift(21) / c.shift(126) - 1          # skip last month
    F["vol_ann"] = ret.rolling(126, min_periods=100).std() * np.sqrt(252)
    F["vam"] = F["mom_6m_skip"] / F["vol_ann"]
    hi252 = h.rolling(252, min_periods=240).max()
    lo252 = l.rolling(252, min_periods=240).min()
    F["hi252"], F["lo252"] = hi252, lo252
    F["dist_high"] = c / hi252
    F["dist_low"] = c / lo252

    # --- trend
    for n in (20, 50, 200):
        F[f"dma{n}"] = c.rolling(n, min_periods=n).mean()
    F["slope200"] = F["dma200"] / F["dma200"].shift(20) - 1
    F["slope20"] = F["dma20"] / F["dma20"].shift(5) - 1
    tr = _true_range(p)
    F["atr"] = tr.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    F["atr_pct"] = F["atr"] / c
    F["atr_contraction"] = tr.rolling(10).mean() / tr.rolling(50).mean()
    F["ext_atr"] = (c - F["dma20"]) / F["atr"]
    F["ext_50"] = c / F["dma50"] - 1
    F["ma_aligned"] = (c > F["dma50"]) & (F["dma50"] > F["dma200"]) & (F["slope200"] > 0)

    # --- volume
    avg50_prev = v.shift(1).rolling(50, min_periods=40).mean()
    F["rvol"] = v / avg50_prev.replace(0, np.nan)
    F["dryup"] = v.rolling(10).mean() / v.rolling(50).mean().replace(0, np.nan)
    F["updown_50"] = _updown_ratio(ret, v, 50)
    F["updown_20"] = _updown_ratio(ret, v, 20)
    rng = (h - l).replace(0, np.nan)
    F["close_loc"] = ((c - l) / rng).clip(0, 1)
    F["med_value_cr"] = (c * v).rolling(20, min_periods=15).median() / 1e7

    # --- breakout / base
    prior_high50 = h.shift(1).rolling(50, min_periods=50).max()
    prior_low50 = l.shift(1).rolling(50, min_periods=50).min()
    brk = (c > prior_high50).astype(float).where(prior_high50.notna())
    F["brk_recent"] = brk.rolling(3, min_periods=1).max()
    F["brk_rvol"] = F["rvol"].where(brk == 1).rolling(3, min_periods=1).max()
    F["brk_close_loc"] = F["close_loc"].where(brk == 1).rolling(3, min_periods=1).max()
    F["base_depth"] = (prior_high50 - prior_low50) / prior_high50

    # --- relative strength
    rs = c.div(p.bench.reindex(c.index), axis=0)
    F["rs_63"] = rs / rs.shift(63) - 1
    F["rs_near_high"] = rs / rs.rolling(252, min_periods=200).max()

    # --- sector (derived: median of universe members sharing an industry)
    ind = industry.reindex(c.columns).fillna("Unknown")
    r63 = F["ret_63"]
    F["rs_vs_sector"] = r63 - r63.T.groupby(ind).transform("median").T
    sec = r63.T.groupby(ind).median().T.rank(axis=1, pct=True)
    F["sector_rank"] = pd.DataFrame(sec.reindex(columns=ind.values).to_numpy(), index=c.index, columns=c.columns)

    # --- reversal-specific (low made 20-60 sessions ago)
    low_window_min = l.shift(20).rolling(41, min_periods=41).min()
    recent_min20 = l.rolling(20, min_periods=20).min()
    F["low_in_window"] = (low_window_min <= lo252 * (1 + 1e-9)) & (recent_min20 > lo252)
    pre_high = h.shift(61).rolling(191, min_periods=150).max()        # high made before the low
    F["drawdown"] = 1 - lo252 / pre_high
    F["low15"] = l.rolling(15, min_periods=15).min()
    F["higher_low_atr"] = (F["low15"] - lo252) / F["atr"]
    thr = lo252 + F["atr"] * 1.0
    F["support_days"] = _count_near_level(l, thr, 61, 250)
    F["cap_rvol"] = F["rvol"].shift(20).rolling(41, min_periods=20).max()
    rs_recent = rs.rolling(20, min_periods=20).min()
    rs_at_low = rs.shift(20).rolling(41, min_periods=41).min()
    F["rs_div"] = rs_recent / rs_at_low - 1
    F["reclaimed"] = (c > F["dma20"]) & (c > F["dma50"]) & (F["slope20"] > 0)
    F["reclaim_strength"] = c / F["dma50"] - 1

    # --- hygiene
    F["history"] = c.notna().cumsum()
    F["close"] = c
    return F


def market_regime(p: Panel, F: dict) -> pd.DataFrame:
    """bull / neutral / bear from Nifty trend, plus breadth (% of universe above 50DMA)."""
    b = p.bench.reindex(p.dates)
    b50, b200 = b.rolling(50).mean(), b.rolling(200).mean()
    reg = pd.Series("neutral", index=p.dates, dtype=object)
    reg[(b > b200) & (b50 > b200)] = "bull"
    reg[(b < b200) & (b50 < b200)] = "bear"
    reg[b200.isna()] = "unknown"
    above = (p.close > F["dma50"]).where(F["dma50"].notna())
    breadth = above.sum(axis=1) / above.notna().sum(axis=1).replace(0, np.nan)
    return pd.DataFrame({"regime": reg, "breadth_50dma": breadth, "nifty": b})
