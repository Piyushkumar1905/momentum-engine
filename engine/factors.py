"""Factor definitions - the single place a factor is defined.

Both the research harness (tools/factor_lab.py) and the live scoring model
(engine/factor_model.py) import from here, so a factor cannot mean one thing in a
backtest and another in production.

A factor is a function (panel, F) -> DataFrame aligned to panel.close, where HIGHER is
more attractive. Point-in-time only: using data after date t invalidates every result
computed from it.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

FACTORS: dict[str, tuple] = {}


def factor(name: str, note: str = ""):
    def deco(fn):
        FACTORS[name] = (fn, note or fn.__doc__ or "")
        return fn
    return deco


def _safe(df):
    return df.replace([np.inf, -np.inf], np.nan)


# --- classic momentum family
@factor("mom_12m_skip1", "12-month return skipping the most recent month")
def f_mom12(p, F):
    return _safe(p.close.shift(21) / p.close.shift(252) - 1)


@factor("mom_6m_skip1", "6-month return skipping the most recent month")
def f_mom6(p, F):
    return _safe(F["mom_6m_skip"])


@factor("mom_3m", "3-month return")
def f_mom3(p, F):
    return _safe(p.close / p.close.shift(63) - 1)


@factor("vol_adj_mom_12m", "12-month momentum divided by realised volatility")
def f_vam12(p, F):
    return _safe((p.close.shift(21) / p.close.shift(252) - 1) / F["vol_ann"])


# --- short-horizon reversal family (a well-documented anomaly)
@factor("reversal_1m", "NEGATIVE of last month's return: buy recent losers")
def f_rev1m(p, F):
    return _safe(-(p.close / p.close.shift(21) - 1))


@factor("reversal_1w", "NEGATIVE of last week's return")
def f_rev1w(p, F):
    return _safe(-(p.close / p.close.shift(5) - 1))


# --- low volatility / quality-of-trend
@factor("low_vol", "NEGATIVE of annualised volatility: prefer calm stocks")
def f_lowvol(p, F):
    return _safe(-F["vol_ann"])


@factor("trend_consistency", "share of up days over 126 sessions")
def f_consist(p, F):
    r = p.close.pct_change(fill_method=None)
    return _safe((r > 0).rolling(126, min_periods=100).mean())


@factor("dd_from_high", "NEGATIVE of drawdown from the 252-day high")
def f_dd(p, F):
    return _safe(-(1 - p.close / p.high.rolling(252, min_periods=200).max()))


@factor("above_200dma", "distance above the 200-day average")
def f_a200(p, F):
    return _safe(p.close / F["dma200"] - 1)


# --- volume / participation
@factor("volume_trend", "20-day average volume over 100-day average")
def f_voltrend(p, F):
    return _safe(p.volume.rolling(20).mean() / p.volume.rolling(100).mean())


@factor("updown_volume", "up-day volume over down-day volume, 50 sessions")
def f_ud(p, F):
    return _safe(F["updown_50"])


@factor("illiquidity", "Amihud: |return| per unit of traded value; higher = more illiquid")
def f_amihud(p, F):
    r = p.close.pct_change(fill_method=None).abs()
    val = (p.close * p.volume).replace(0, np.nan)
    return _safe((r / val).rolling(21, min_periods=15).mean() * 1e9)


# --- relative strength
@factor("rs_vs_sector", "63-day return minus the sector median")
def f_rssec(p, F):
    return _safe(F["rs_vs_sector"])


@factor("rs_nifty_63", "63-day change in the stock/Nifty ratio")
def f_rs63(p, F):
    return _safe(F["rs_63"])


# --- volatility compression
@factor("atr_compression", "NEGATIVE of 10-day over 50-day true range")
def f_atrc(p, F):
    return _safe(-F["atr_contraction"])


@factor("near_52w_high", "close over the 252-day high")
def f_n52(p, F):
    return _safe(F["dist_high"])


# ───────────────────── extended library (regime-diversification candidates) ─────────

@factor("lottery_max5", "NEGATIVE of the largest daily gain in 21 sessions (lottery demand)")
def f_lottery(p, F):
    r = p.close.pct_change(fill_method=None)
    return _safe(-r.rolling(21, min_periods=15).max())


@factor("idio_vol", "NEGATIVE of residual volatility after removing market beta")
def f_idio(p, F):
    r = p.close.pct_change(fill_method=None)
    m = p.bench.reindex(p.dates).pct_change(fill_method=None)
    cov = r.mul(m, axis=0).rolling(126, min_periods=100).mean() - \
        r.rolling(126, min_periods=100).mean().mul(m.rolling(126, min_periods=100).mean(), axis=0)
    var = m.rolling(126, min_periods=100).var()
    beta = cov.div(var, axis=0)
    resid = r.sub(beta.mul(m, axis=0))
    return _safe(-resid.rolling(126, min_periods=100).std())


@factor("low_beta", "NEGATIVE of 252-day beta to the Nifty")
def f_beta(p, F):
    r = p.close.pct_change(fill_method=None)
    m = p.bench.reindex(p.dates).pct_change(fill_method=None)
    cov = r.mul(m, axis=0).rolling(252, min_periods=200).mean() - \
        r.rolling(252, min_periods=200).mean().mul(m.rolling(252, min_periods=200).mean(), axis=0)
    return _safe(-cov.div(m.rolling(252, min_periods=200).var(), axis=0))


@factor("ret_accel", "3-month return minus 6-month return: improving trend")
def f_accel(p, F):
    return _safe((p.close / p.close.shift(63) - 1) - (p.close / p.close.shift(126) - 1))


@factor("sharpe_126", "126-day return divided by its own volatility")
def f_sharpe(p, F):
    r = p.close.pct_change(fill_method=None)
    return _safe(r.rolling(126, min_periods=100).mean() / r.rolling(126, min_periods=100).std())


@factor("dist_20dma", "NEGATIVE distance above the 20-day average: prefer pullbacks")
def f_d20(p, F):
    return _safe(-(p.close / F["dma20"] - 1))


@factor("skew_126", "NEGATIVE of return skewness: avoid lottery-like payoffs")
def f_skew(p, F):
    return _safe(-p.close.pct_change(fill_method=None).rolling(126, min_periods=100).skew())


@factor("value_proxy_rev5y", "NEGATIVE of 5-year return: long-horizon mean reversion")
def f_rev5(p, F):
    return _safe(-(p.close / p.close.shift(1250) - 1))
