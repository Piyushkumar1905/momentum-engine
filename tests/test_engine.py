import copy

import numpy as np
import pandas as pd
import pytest
import yaml

from engine import backtest as bt
from engine import data, rules
from engine.features import compute_features

CFG = yaml.safe_load(open("config.yaml"))


@pytest.fixture(scope="module")
def synth():
    cfg = copy.deepcopy(CFG)
    cfg["run"]["data_source"] = "synthetic"
    cfg["filters"]["min_median_value_cr"] = 0.5
    cfg["backtest"]["start"] = "2023-01-01"
    uni = data.synthetic_universe(n=40)
    panel = data.fetch_synthetic(uni, start="2021-06-01", end="2024-12-31")
    return cfg, uni, panel


def test_features_have_no_lookahead(synth):
    """Features up to a cut-off date must be identical whether or not later data exists."""
    cfg, uni, panel = synth
    cut = panel.dates[600]
    full = compute_features(panel, uni.set_index("symbol")["industry"])
    part = compute_features(panel.truncate(cut), uni.set_index("symbol")["industry"])
    for k, df in part.items():
        a = full[k].loc[:cut].astype(float)
        b = df.astype(float)
        pd.testing.assert_frame_equal(a, b, check_exact=False, rtol=1e-9, obj=k)


def test_scores_have_no_lookahead(synth):
    cfg, uni, panel = synth
    cut = panel.dates[600]
    s_full = bt.prepare(panel, uni, cfg)["scores"]
    s_part = bt.prepare(panel.truncate(cut), uni, cfg)["scores"]
    for m in s_part:
        pd.testing.assert_frame_equal(s_full[m].loc[:cut], s_part[m], check_exact=False, rtol=1e-9)


def _bars(rows):
    a = np.array(rows, float)          # columns: open, high, low, close ; one stock
    return a[:, [0]], a[:, [1]], a[:, [2]], a[:, [3]]


def test_trade_hits_stop_first_when_both_touched():
    O, H, L, C = _bars([[100, 101, 99, 100],     # signal day i=0
                        [100, 130, 80, 100],     # entry day: both stop (90) and target touched
                        [100, 100, 100, 100]])
    t = bt._simulate(0, 0, 90.0, 2.0, O, H, L, C, max_hold=2, cost=0.0)
    assert t["exit_reason"] == "stop" and t["exit"] == 90.0


def test_trade_gap_down_exits_at_open():
    O, H, L, C = _bars([[100, 101, 99, 100], [100, 102, 98, 101], [85, 86, 84, 85], [85, 85, 85, 85]])
    t = bt._simulate(0, 0, 95.0, 2.0, O, H, L, C, max_hold=3, cost=0.0)
    assert t["exit_reason"] == "stop" and t["exit"] == 85.0


def test_trade_target_and_time_exit():
    O, H, L, C = _bars([[100, 101, 99, 100], [100, 103, 99, 102], [102, 125, 101, 120], [120, 121, 119, 120]])
    t = bt._simulate(0, 0, 95.0, 2.0, O, H, L, C, max_hold=3, cost=0.0)
    assert t["exit_reason"] == "target" and t["exit"] == 110.0
    O, H, L, C = _bars([[100, 101, 99, 100], [100, 103, 99, 102], [102, 104, 101, 103], [103, 104, 102, 104]])
    t = bt._simulate(0, 0, 95.0, 2.0, O, H, L, C, max_hold=3, cost=0.0)
    assert t["exit_reason"] == "time" and t["exit"] == 104.0


def test_open_trade_at_end_is_not_counted():
    O, H, L, C = _bars([[100, 101, 99, 100], [100, 101, 99, 100]])
    assert bt._simulate(0, 0, 95.0, 2.0, O, H, L, C, max_hold=5, cost=0.0) is None


def test_top_ignores_nan_and_orders():
    row = np.array([np.nan, 0.2, 0.9, np.nan, 0.5])
    assert list(bt._top(row, 2)) == [2, 4]


def test_pipeline_runs(synth):
    cfg, uni, panel = synth
    prep = bt.prepare(panel, uni, cfg)
    res = bt.run_backtest(panel, uni, cfg, prep)
    assert not res["signals"].empty and "forward" in res["summary"]
    # entries are strictly after the scan date: forward return uses next open
    s = res["signals"].iloc[0]
    j = panel.symbols.index(s["symbol"])
    i = panel.dates.get_loc(s["date"])
    exp = panel.close.iat[i + 5, j] / panel.open.iat[i + 1, j] - 1
    assert s["ret_5d"] == pytest.approx(exp)
    picks = bt.scan_latest(panel, uni, cfg, prep)
    assert {"why", "risk_flags", "stop_ref"} <= set(picks.columns)
    assert (picks["stop_ref"] < picks["last_close"]).all()


def test_eligibility_respects_thresholds(synth):
    cfg, uni, panel = synth
    prep = bt.prepare(panel, uni, cfg)
    F, S = prep["F"], prep["scores"]["momentum"]
    picked = S.notna().to_numpy()
    assert picked.sum() > 0
    assert (F["dist_high"].to_numpy()[picked] >= 0.85 - 1e-12).all()
    assert (F["ext_atr"].to_numpy()[picked] <= 3.0 + 1e-12).all()
    assert F["ma_aligned"].to_numpy()[picked].all()
    bear = (prep["regime"]["regime"] == "bear").to_numpy()[:, None] & picked
    assert (F["dist_high"].to_numpy()[bear] >= 0.90 - 1e-12).all()   # tightened in bear regime


def test_universe_loader(tmp_path):
    p = tmp_path / "u.csv"
    p.write_text("Company Name,Industry,Symbol,Series,ISIN Code\n"
                 "A Ltd,Healthcare,AAA,EQ,X1\nB Ltd,Power,BBB,BE,X2\nC Ltd,Power,CCC,EQ,X3\n")
    u = data.load_universe(p)
    assert list(u["symbol"]) == ["AAA", "CCC"] and list(u["industry"]) == ["Healthcare", "Power"]


def test_yahoo_parser_with_mock(tmp_path, monkeypatch):
    """yfinance output format (group_by='ticker') is parsed and cached correctly."""
    import sys
    import types
    uni = data.synthetic_universe(n=3)
    syn = data.fetch_synthetic(uni, start="2023-01-01", end="2023-12-31")

    def fake_download(tickers, **kw):
        parts = {}
        for t in tickers:
            if t == "^NSEI":
                c = syn.bench
                parts[t] = pd.DataFrame({"Open": c, "High": c, "Low": c, "Close": c, "Volume": 0.0})
            elif t != "SYN002.NS":                       # simulate one missing symbol
                s = t[:-3]
                parts[t] = pd.DataFrame({f: getattr(syn, f.lower())[s] for f in data.FIELDS})
        return pd.concat(parts, axis=1)

    monkeypatch.setitem(sys.modules, "yfinance", types.SimpleNamespace(download=fake_download))
    p = data.fetch_yahoo(["SYN000", "SYN001", "SYN002"], "2023-01-01", "2023-12-31", tmp_path)
    assert p.symbols == ["SYN000", "SYN001"]
    pd.testing.assert_series_equal(p.close["SYN000"], syn.close["SYN000"], check_names=False, check_freq=False)
    assert (tmp_path / "SYN000.NS.pkl").exists() and (tmp_path / "_NSEI.pkl").exists()
    # second call is served from cache (download would now fail)
    monkeypatch.setitem(sys.modules, "yfinance", types.SimpleNamespace(download=lambda *a, **k: 1 / 0))
    p2 = data.fetch_yahoo(["SYN000", "SYN001"], "2023-01-01", "2023-12-31", tmp_path)
    assert p2.symbols == ["SYN000", "SYN001"]


def test_csv_source(tmp_path):
    uni = data.synthetic_universe(n=2)
    syn = data.fetch_synthetic(uni, start="2023-01-01", end="2023-06-30")
    for s in syn.symbols:
        df = pd.DataFrame({f: getattr(syn, f.lower())[s] for f in data.FIELDS})
        df.index.name = "Date"
        df.to_csv(tmp_path / f"{s}.csv")
    b = pd.DataFrame({f: syn.bench for f in data.FIELDS}); b.index.name = "Date"
    b.to_csv(tmp_path / "_BENCHMARK.csv")
    p = data.fetch_csv(syn.symbols, tmp_path)
    assert p.symbols == syn.symbols and len(p.dates) == len(syn.dates)


def test_tracker_reports_first_hit(synth):
    cfg, uni, panel = synth
    old = panel.truncate(panel.dates[-31])
    picks = bt.scan_latest(old, uni, cfg)
    tr = bt.track_picks(picks, panel, [5, 10, 20])
    assert len(tr) == len(picks)
    assert set(tr["first_hit"]) <= {"stop", "target", "none yet"}
    assert (tr["days_elapsed"] == 30).all()


def _yahoo_stub(syn, calls):
    """Stand-in for yfinance.download that serves slices of a synthetic 'server' panel
    and records the start date of every request."""
    import pandas as _pd

    def fake_download(tickers, **kw):
        calls.append(kw.get("start"))
        if isinstance(tickers, str):
            tickers = [tickers]
        lo, hi = _pd.Timestamp(kw["start"]), _pd.Timestamp(kw["end"])
        parts = {}
        for t in tickers:
            if t == "^NSEI":
                c = syn.bench
                df = _pd.DataFrame({"Open": c, "High": c, "Low": c, "Close": c, "Volume": 0.0})
            else:
                s = t[:-3]
                if s not in syn.close.columns:
                    continue
                df = _pd.DataFrame({f: getattr(syn, f.lower())[s] for f in data.FIELDS})
            parts[t] = df[(df.index >= lo) & (df.index < hi)]
        return _pd.concat(parts, axis=1) if parts else _pd.DataFrame()
    return fake_download


def test_cache_updates_incrementally_and_survives_ci_restore(tmp_path, monkeypatch):
    """A restored CI cache has a fresh mtime but stale contents.

    Freshness must be judged from the data, not the file timestamp, or the scheduled
    refresh silently serves frozen prices forever. The top-up must also fetch only the
    missing tail rather than re-downloading full history.
    """
    import sys
    import types
    uni = data.synthetic_universe(n=3)
    server = data.fetch_synthetic(uni, start="2023-01-01", end="2023-07-31")
    calls = []
    monkeypatch.setitem(sys.modules, "yfinance",
                        types.SimpleNamespace(download=_yahoo_stub(server, calls)))

    syms = list(server.close.columns)
    p1 = data.fetch_yahoo(syms, "2023-01-01", "2023-06-30", tmp_path)
    assert p1.dates.max() <= pd.Timestamp("2023-06-30")
    first_round = len(calls)

    # Same day, nothing new to get -> must not hit the network again.
    data.fetch_yahoo(syms, "2023-01-01", "2023-06-30", tmp_path)
    assert len(calls) == first_round, "re-downloaded despite an up-to-date cache"

    # Simulate CI restoring the cache on a later day: mtime is now, data is stale.
    for f in tmp_path.glob("*.pkl"):
        df = pd.read_pickle(f)
        df.attrs["fetched_on"] = "1999-01-01"
        df.to_pickle(f)
        f.touch()

    calls.clear()
    p2 = data.fetch_yahoo(syms, "2023-01-01", "2023-07-31", tmp_path)
    assert p2.dates.max() > pd.Timestamp("2023-06-30"), "stale cache was served as fresh"
    assert len(p2.dates) > len(p1.dates)
    # every request started near the cached edge, not at the original history start
    assert calls and all(pd.Timestamp(c) > pd.Timestamp("2023-06-01") for c in calls), \
        f"expected incremental tail fetches, got starts {calls}"


def test_post_ipo_listings_are_not_refetched_every_run(tmp_path, monkeypatch):
    """A stock that listed after `start` has no earlier history by definition.

    Judging cache validity by 'does it reach back to start?' rejects every such listing
    on every run, re-downloading a quarter of the universe daily and inviting rate limits.
    """
    import sys
    import types
    uni = data.synthetic_universe(n=3)
    server = data.fetch_synthetic(uni, start="2023-01-01", end="2023-06-30")
    late = server.close.columns[1]                      # pretend this one listed in April
    calls = []
    base = _yahoo_stub(server, calls)

    def fake_download(tickers, **kw):
        raw = base(tickers, **kw)
        col = f"{late}.NS"
        if isinstance(raw, pd.DataFrame) and col in raw.columns.get_level_values(0):
            raw.loc[raw.index < pd.Timestamp("2023-04-01"), col] = float("nan")
        return raw

    monkeypatch.setitem(sys.modules, "yfinance", types.SimpleNamespace(download=fake_download))
    syms = list(server.close.columns)

    data.fetch_yahoo(syms, "2023-01-01", "2023-06-30", tmp_path)
    cached_late = pd.read_pickle(tmp_path / f"{late}.NS.pkl")
    assert cached_late.index.min() >= pd.Timestamp("2023-04-01"), "fixture did not shorten history"

    calls.clear()
    data.fetch_yahoo(syms, "2023-01-01", "2023-06-30", tmp_path)
    assert calls == [], f"re-downloaded an up-to-date cache: {calls}"
