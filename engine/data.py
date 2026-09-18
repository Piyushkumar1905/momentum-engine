"""Data layer: universe, price sources (Yahoo / CSV / synthetic), caching and alignment.

All prices are daily. The engine never mixes intraday and EOD data.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)
FIELDS = ["Open", "High", "Low", "Close", "Volume"]


# --------------------------------------------------------------------------- universe
def load_universe(path: str | Path, max_symbols: int | None = None) -> pd.DataFrame:
    """Read a niftyindices.com constituent CSV (Company Name, Industry, Symbol, Series, ISIN Code)
    or any CSV with a Symbol column (+ optional Industry/Sector)."""
    df = pd.read_csv(path)
    df.columns = [str(c).strip() for c in df.columns]
    lower = {c.lower(): c for c in df.columns}
    if "symbol" not in lower:
        raise ValueError(f"{path}: needs a 'Symbol' column")
    sym = df[lower["symbol"]].astype(str).str.strip()
    ind_col = lower.get("industry") or lower.get("sector")
    ind = df[ind_col].astype(str).str.strip() if ind_col else pd.Series("Unknown", index=df.index)
    if "series" in lower:
        keep = df[lower["series"]].astype(str).str.strip().eq("EQ")
        sym, ind = sym[keep], ind[keep]
    out = pd.DataFrame({"symbol": sym.values, "industry": ind.values}).drop_duplicates("symbol")
    # NSE constituent files carry placeholder rows (e.g. DUMMYHEG / "Dummy HEG Ltd.") for
    # corporate actions; they are not tradable and have no price history anywhere.
    out = out[~out["symbol"].str.upper().str.startswith("DUMMY")]
    if max_symbols:
        out = out.head(int(max_symbols))
    log.info("Universe: %d symbols from %s", len(out), path)
    return out.reset_index(drop=True)


# --------------------------------------------------------------------------- panel
@dataclass
class Panel:
    open: pd.DataFrame
    high: pd.DataFrame
    low: pd.DataFrame
    close: pd.DataFrame
    volume: pd.DataFrame
    bench: pd.Series          # benchmark close
    source: str = "unknown"
    corp_action: pd.DataFrame | None = None   # True where the price series is discontinuous

    @property
    def dates(self) -> pd.DatetimeIndex:
        return self.close.index

    @property
    def symbols(self) -> list[str]:
        return list(self.close.columns)

    def truncate(self, end) -> "Panel":
        """Copy of the panel with data only up to `end` (used by look-ahead tests)."""
        sl = slice(None, pd.Timestamp(end))
        ca = self.corp_action.loc[sl] if self.corp_action is not None else None
        return Panel(self.open.loc[sl], self.high.loc[sl], self.low.loc[sl],
                     self.close.loc[sl], self.volume.loc[sl], self.bench.loc[sl], self.source, ca)


def detect_corporate_actions(op: pd.DataFrame, close: pd.DataFrame,
                             drop: float = -0.30, recovery: float = 0.5) -> pd.DataFrame:
    """Flag overnight gaps that are price rebasings, not market moves.

    `auto_adjust` handles splits and dividends but NOT demergers, where value leaves
    the listed entity for a separately-listed one. The price series then shows a
    catastrophic loss the shareholder never took - Vedanta's 2026 demerger prints as
    -63% overnight, and a backtest books it as a real -10R trade.

    Adjusting the history would mean inventing prices; the demerged entity's value is
    simply not in this series. So these bars are flagged and the affected trades are
    excluded instead, which neither invents a gain nor books a phantom loss.

    A genuine crash usually retraces some of the fall; a rebasing never does. That is
    the test used to separate them.
    """
    gap = op / close.shift(1) - 1
    suspect = gap <= drop
    flags = pd.DataFrame(False, index=op.index, columns=op.columns)
    if not suspect.any().any():
        return flags
    for sym in op.columns[suspect.any()]:
        prev = close[sym].shift(1)
        for dt in op.index[suspect[sym].fillna(False)]:
            fwd = close[sym].loc[dt:].head(11)
            if not len(fwd) or not np.isfinite(prev.loc[dt]):
                continue
            retrace = fwd.max() / prev.loc[dt] - 1
            if retrace < drop * recovery:        # never came back -> rebasing
                flags.loc[dt, sym] = True
                log.warning("%s: %.0f%% gap on %s looks like a corporate action; "
                            "trades spanning it are excluded", sym, gap.loc[dt, sym] * 100,
                            dt.date())
    return flags


def build_panel(per_symbol: dict[str, pd.DataFrame], bench: pd.Series, source: str,
                max_ffill: int = 5) -> Panel:
    """Align every symbol to the benchmark trading calendar.
    Short gaps (<= max_ffill sessions) are filled as flat, zero-volume days; longer gaps stay NaN."""
    bench = bench.dropna().sort_index()
    idx = pd.DatetimeIndex(bench.index).normalize()
    bench.index = idx
    frames = {f: {} for f in FIELDS}
    for sym, df in per_symbol.items():
        if df is None or df.empty:
            continue
        d = df.copy()
        d.index = pd.DatetimeIndex(d.index).tz_localize(None).normalize()
        d = d[~d.index.duplicated(keep="last")].sort_index()
        d = d.reindex(idx)
        c = d["Close"].ffill(limit=max_ffill)
        filled = d["Close"].isna() & c.notna()
        for f in ("Open", "High", "Low"):
            d[f] = d[f].where(~filled, c)
        d["Close"] = c
        d["Volume"] = d["Volume"].where(~filled, 0.0)
        # basic validation: drop impossible bars
        bad = (d["High"] < d["Low"]) | (d["Close"] <= 0)
        if bad.any():
            log.warning("%s: %d invalid bars set to NaN", sym, int(bad.sum()))
            d.loc[bad, FIELDS] = np.nan
        for f in FIELDS:
            frames[f][sym] = d[f].astype(float)
    mk = {f: pd.DataFrame(frames[f], index=idx) for f in FIELDS}
    if mk["Close"].empty:
        raise RuntimeError("No price data loaded - check symbols / data source")
    ca = detect_corporate_actions(mk["Open"], mk["Close"])
    n = int(ca.to_numpy().sum())
    if n:
        log.warning("Flagged %d corporate-action discontinuities across %d symbols",
                    n, int(ca.any().sum()))
    return Panel(mk["Open"], mk["High"], mk["Low"], mk["Close"], mk["Volume"], bench, source, ca)


# --------------------------------------------------------------------------- sources
def _retry(fn, tries=3, wait=3):
    for k in range(tries):
        try:
            return fn()
        except Exception as e:  # network errors, rate limits
            if k == tries - 1:
                raise
            log.warning("retry %d after error: %s", k + 1, e)
            time.sleep(wait * (k + 1))


def fetch_yahoo(symbols, start, end, cache_dir, benchmark="^NSEI", chunk=40) -> Panel:
    """Daily OHLCV from Yahoo Finance via yfinance (auto-adjusted for splits/dividends).
    For personal research; check the provider's terms before any commercial use."""
    import yfinance as yf  # imported lazily so tests run without it

    cache = Path(cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    today = pd.Timestamp.today().normalize()
    end_ts = pd.Timestamp(end) if end else today
    start_ts = pd.Timestamp(start)

    def path(ticker):
        return cache / f"{ticker.replace('^', '_')}.pkl"

    def cached(ticker):
        """Cached frame, or None if absent/unreadable/not covering the requested start.

        Freshness is judged from the DATA, never from the file's mtime: a cache restored
        by CI has a fresh mtime and stale contents, which would silently freeze the site.
        The attempt date travels inside the frame so it survives that restore.
        """
        p = path(ticker)
        if not p.exists():
            return None
        try:
            df = pd.read_pickle(p)
        except Exception as e:                       # pickle written by another pandas
            log.warning("%s: unreadable cache (%s), refetching", ticker, e)
            return None
        if df is None or not len(df):
            return None
        # A stock that listed after `start` legitimately has no earlier history, so
        # "does the cache reach back to start?" would reject every post-IPO listing on
        # every run. Compare against the start we ASKED for last time instead.
        asked = df.attrs.get("start")
        if asked is not None:
            return df if pd.Timestamp(asked) <= start_ts else None
        if df.index.min() > start_ts + pd.Timedelta(days=7):   # legacy cache, no marker
            return None
        return df

    def tried_today(df) -> bool:
        return df is not None and df.attrs.get("fetched_on") == str(today.date())

    def save(ticker, df):
        """Write the cache, never shrinking it.

        A request for a narrower window (say --end 2020) must not destroy history
        already on disk. Without this merge, one analysis run bounded to a past date
        replaced the benchmark and 299 symbols with series ending in 2020, silently
        truncating every later panel built from them.
        """
        p_ = path(ticker)
        if p_.exists():
            try:
                old = pd.read_pickle(p_)
                if old is not None and len(old):
                    both = pd.concat([old, df])
                    df = both[~both.index.duplicated(keep="last")].sort_index()
                    prev = old.attrs.get("start")
                    if prev and pd.Timestamp(prev) < start_ts:
                        df.attrs["start"] = prev      # keep the widest coverage claimed
            except Exception:
                pass
        df.attrs.setdefault("start", str(start_ts.date()))
        df.attrs["fetched_on"] = str(today.date())
        df.to_pickle(p_)

    def download(tickers, since):
        raw = _retry(lambda: yf.download(
            tickers, start=pd.Timestamp(since).strftime("%Y-%m-%d"),
            end=(end_ts + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
            auto_adjust=True, group_by="ticker", threads=True, progress=False))
        out = {}
        if raw is None or raw.empty:
            return out
        if not isinstance(raw.columns, pd.MultiIndex):
            raw.columns = pd.MultiIndex.from_product([[tickers[0]], raw.columns])
        for t in tickers:
            if t in raw.columns.get_level_values(0):
                df = raw[t][FIELDS].dropna(how="all")
                if len(df):
                    out[t] = df
        return out

    def merge(old, new):
        if old is None:
            return new
        both = pd.concat([old, new])
        return both[~both.index.duplicated(keep="last")].sort_index()

    # --- benchmark first: its last session defines the trading calendar for everyone else
    b = cached(benchmark)
    if b is None or (b.index.max() < end_ts and not tried_today(b)):
        since = start if b is None else b.index.max() + pd.Timedelta(days=1)
        got = download([benchmark], since)
        if benchmark in got:
            b = merge(b, got[benchmark])
        elif b is None:
            raise RuntimeError(f"Could not download benchmark {benchmark}")
        save(benchmark, b)
    last_session = b.index.max()

    # --- symbols: full download if absent, otherwise only the missing tail
    per_symbol, full, tail = {}, [], []
    for s in symbols:
        t = f"{s}.NS"
        df = cached(t)
        if df is None:
            full.append(t)
        elif df.index.max() < last_session and not tried_today(df):
            per_symbol[s] = df
            tail.append(t)
        else:
            per_symbol[s] = df

    def run(tickers, since, what):
        for i in range(0, len(tickers), chunk):
            part = tickers[i:i + chunk]
            log.info("%s %d-%d of %d tickers (from %s)", what, i + 1, i + len(part),
                     len(tickers), pd.Timestamp(since).date())
            got = download(part, since)
            for t, df in got.items():
                merged = merge(per_symbol.get(t[:-3]), df)
                save(t, merged)
                per_symbol[t[:-3]] = merged
            for t in part:                            # record the attempt even if empty
                if t[:-3] in per_symbol:
                    save(t, per_symbol[t[:-3]])
            time.sleep(1)

    if full:
        run(full, start, "Downloading")
    if tail:
        since = min(per_symbol[t[:-3]].index.max() for t in tail) + pd.Timedelta(days=1)
        log.info("Incremental update for %d cached tickers", len(tail))
        run(tail, since, "Updating")
    missing = sorted(set(symbols) - set(per_symbol))
    if missing:
        log.warning("No data for %d symbols (first few: %s)", len(missing), missing[:10])
    return build_panel(per_symbol, b["Close"], source="yahoo (auto-adjusted EOD)")


def fetch_csv(symbols, csv_dir) -> Panel:
    """Per-symbol CSVs with Date,Open,High,Low,Close,Volume plus _BENCHMARK.csv.
    Use this for data you already hold (e.g. built from NSE bhavcopies or a licensed vendor).
    Prices must already be adjusted for corporate actions."""
    d = Path(csv_dir)

    def read(p):
        df = pd.read_csv(p, parse_dates=["Date"]).set_index("Date")
        df.columns = [c.strip().title() for c in df.columns]
        return df[FIELDS]

    bench = read(d / "_BENCHMARK.csv")["Close"]
    per = {s: read(d / f"{s}.csv") for s in symbols if (d / f"{s}.csv").exists()}
    return build_panel(per, bench, source=f"csv ({csv_dir})")


def synthetic_universe(n=120, n_ind=10, seed=7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame({"symbol": [f"SYN{i:03d}" for i in range(n)],
                         "industry": [f"Industry {k}" for k in rng.integers(0, n_ind, n)]})


def fetch_synthetic(universe: pd.DataFrame, start="2019-01-01", end="2026-09-16", seed=7) -> Panel:
    """Random-walk market with regime switches and persistent stock-specific drifts.
    ONLY for testing the plumbing - results on this data mean nothing."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start, end)
    T = len(dates)
    regime_drift = np.repeat(rng.choice([0.0008, 0.0, -0.0007], size=T // 90 + 1), 90)[:T]
    mkt = regime_drift + rng.normal(0, 0.010, T)
    bench = pd.Series(15000 * np.exp(np.cumsum(mkt)), index=dates)
    inds = universe["industry"].unique()
    ind_drift = {k: np.repeat(rng.normal(0, 0.0008, T // 120 + 1), 120)[:T] for k in inds}
    per = {}
    for s, ind in zip(universe["symbol"], universe["industry"]):
        beta = rng.uniform(0.6, 1.4)
        own = np.repeat(rng.normal(0, 0.0012, T // 60 + 1), 60)[:T]
        r = beta * mkt + ind_drift[ind] + own + rng.normal(0, rng.uniform(0.012, 0.025), T)
        close = rng.uniform(80, 3000) * np.exp(np.cumsum(r))
        rng_pct = np.abs(rng.normal(0.012, 0.006, T)) + np.abs(r) * 0.5
        high = close * (1 + rng_pct * rng.uniform(0.2, 0.8, T))
        low = close * (1 - rng_pct * rng.uniform(0.2, 0.8, T))
        open_ = np.clip(np.r_[close[0], close[:-1]] * (1 + rng.normal(0, 0.004, T)), low, high)
        vol = rng.uniform(2e5, 3e6) * np.exp(rng.normal(0, 0.35, T)) * (1 + 25 * np.abs(r))
        per[s] = pd.DataFrame({"Open": open_, "High": high, "Low": low, "Close": close,
                               "Volume": vol}, index=dates)
    return build_panel(per, bench, source="synthetic (TEST DATA ONLY)")


def load_panel(cfg: dict, universe: pd.DataFrame) -> Panel:
    run, bt = cfg["run"], cfg["backtest"]
    start = (pd.Timestamp(bt["start"]) - pd.Timedelta(days=int(bt["warmup_days"]))).strftime("%Y-%m-%d")
    src = run["data_source"]
    if src == "yahoo":
        return fetch_yahoo(list(universe["symbol"]), start, bt.get("end"), run["cache_dir"], run["benchmark"])
    if src == "csv":
        return fetch_csv(list(universe["symbol"]), run["csv_dir"])
    if src == "synthetic":
        return fetch_synthetic(universe, start=start, end=bt.get("end") or "2026-09-16")
    raise ValueError(f"Unknown data_source {src}")
