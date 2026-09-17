"""Factor evaluation harness.

Scores a candidate signal on its ability to rank stocks by forward return, with the
guards that make such a search honest:

  * TRAIN-only evaluation. Later windows are never touched here.
  * Every metric is relative to the liquid universe, so market drift cannot be
    mistaken for skill.
  * Rank IC (Spearman) rather than return spreads, so one runaway stock cannot
    carry a factor.
  * Turnover is reported: a factor that only works at 300% monthly turnover does
    not survive costs.

A factor is a function (panel, F) -> DataFrame aligned to panel.close, where a HIGHER
value means a MORE attractive stock. Point-in-time only: anything using data after t
invalidates the whole exercise.

  python tools/factor_lab.py --list
  python tools/factor_lab.py --window train
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

log = logging.getLogger("factor_lab")

WINDOWS = {
    "train":    ("2021-01-01", "2023-12-31"),
    "validate": ("2024-01-01", "2025-06-30"),
    "holdout":  ("2025-07-01", "2026-09-17"),
}

# ───────────────────────────── factor library ─────────────────────────────
# Each entry: name -> (callable, "higher is better" description)
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


# ───────────────────────────── evaluation ─────────────────────────────

def evaluate(fac: pd.DataFrame, panel, base: np.ndarray, idxs, horizons=(20, 60, 120),
             deciles=10) -> dict:
    """Rank IC, top-decile lift and turnover for one factor on the given scan dates."""
    A = fac.to_numpy(float)
    out = {"coverage": None}
    fwd = {h: (panel.close.shift(-h) / panel.open.shift(-1) - 1).to_numpy(float)
           for h in horizons}
    cov = []
    for h in horizons:
        ics, tops, bots, bases, hits, bhits = [], [], [], [], [], []
        for i in idxs:
            m = base[i] & np.isfinite(A[i]) & np.isfinite(fwd[h][i])
            n = int(m.sum())
            if n < 50:
                continue
            x, y = A[i][m], fwd[h][i][m]
            xr = pd.Series(x).rank().to_numpy()
            yr = pd.Series(y).rank().to_numpy()
            if xr.std() == 0 or yr.std() == 0:
                continue
            ics.append(float(np.corrcoef(xr, yr)[0, 1]))
            k = max(1, n // deciles)
            order = np.argsort(-x)
            tops.append(float(np.nanmean(y[order[:k]])))
            bots.append(float(np.nanmean(y[order[-k:]])))
            bases.append(float(np.nanmean(y)))
            hits.append(float(np.nanmean(y[order[:k]] >= 0.15)))
            bhits.append(float(np.nanmean(y >= 0.15)))
            if h == horizons[0]:
                cov.append(n)
        if not ics:
            continue
        ic = np.array(ics)
        out[f"h{h}"] = {
            "ic_mean": round(float(ic.mean()), 4),
            "ic_std": round(float(ic.std()), 4),
            # t-stat of the mean IC: the honest test of whether a factor ranks at all
            "ic_t": round(float(ic.mean() / (ic.std() / np.sqrt(len(ic)))), 2) if ic.std() else None,
            "ic_hit": round(float((ic > 0).mean()), 3),
            "top_decile": round(float(np.mean(tops)), 4),
            "bottom_decile": round(float(np.mean(bots)), 4),
            "universe": round(float(np.mean(bases)), 4),
            "top_minus_universe": round(float(np.mean(tops) - np.mean(bases)), 4),
            "long_short": round(float(np.mean(tops) - np.mean(bots)), 4),
            "big_move_lift": round(float(np.mean(hits) / np.mean(bhits)), 3) if np.mean(bhits) else None,
            "periods": len(ic),
        }
    out["coverage"] = int(np.mean(cov)) if cov else 0

    # turnover of the top decile between consecutive scans
    prev, turns = None, []
    for i in idxs:
        m = base[i] & np.isfinite(A[i])
        if m.sum() < 50:
            continue
        k = max(1, int(m.sum()) // deciles)
        cur = set(np.where(m)[0][np.argsort(-A[i][m])[:k]])
        if prev:
            turns.append(1 - len(cur & prev) / max(1, len(cur)))
        prev = cur
    out["turnover"] = round(float(np.mean(turns)), 3) if turns else None
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(ROOT / "config.yaml"))
    ap.add_argument("--window", default="train", choices=list(WINDOWS))
    ap.add_argument("--only", default=None, help="comma-separated factor names")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")

    if a.list:
        for n, (_, note) in FACTORS.items():
            print(f"{n:<22} {note}")
        return

    start, end = WINDOWS[a.window]
    cfg = yaml.safe_load(open(a.config))
    cfg["backtest"]["start"] = start
    uni = data.load_universe(cfg["run"]["universe_file"], cfg["run"].get("max_symbols"))
    panel = data.load_panel(cfg, uni)
    prep = bt.prepare(panel, uni, cfg)
    F, base = prep["F"], prep["base"].to_numpy(bool)
    dates = panel.dates
    idxs = np.where((dates >= pd.Timestamp(start)) & (dates <= pd.Timestamp(end)))[0][::5]

    names = [n.strip() for n in a.only.split(",")] if a.only else list(FACTORS)
    rows = {}
    for n in names:
        fn, note = FACTORS[n]
        try:
            rows[n] = evaluate(fn(panel, F), panel, base, idxs)
            rows[n]["note"] = note
        except Exception as e:                       # a broken factor must not kill the sweep
            log.warning("%s failed: %s", n, e)

    print(f"\nWindow {a.window}: {start} -> {end}   ({len(idxs)} scans)")
    print(f"{'factor':<22}{'IC20':>8}{'t':>7}{'IC60':>8}{'t':>7}{'IC120':>8}{'t':>7}"
          f"{'top-uni(60d)':>14}{'lift':>7}{'turn':>7}")
    print("-" * 95)
    def key(r):
        return -(r[1].get("h60", {}).get("ic_t") or -99)
    for n, r in sorted(rows.items(), key=key):
        h20, h60, h120 = r.get("h20", {}), r.get("h60", {}), r.get("h120", {})
        print(f"{n:<22}{h20.get('ic_mean', 0):>8.3f}{(h20.get('ic_t') or 0):>7.1f}"
              f"{h60.get('ic_mean', 0):>8.3f}{(h60.get('ic_t') or 0):>7.1f}"
              f"{h120.get('ic_mean', 0):>8.3f}{(h120.get('ic_t') or 0):>7.1f}"
              f"{h60.get('top_minus_universe', 0):>14.4f}"
              f"{(h60.get('big_move_lift') or 0):>7.2f}{(r.get('turnover') or 0):>7.2f}")

    dest = Path(a.out or ROOT / "output" / "research" / f"factors_{a.window}.json")
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps({"window": a.window, "range": [start, end],
                                "scans": len(idxs), "factors": rows}, indent=1), encoding="utf-8")
    print(f"\nwrote {dest}")


if __name__ == "__main__":
    main()


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
