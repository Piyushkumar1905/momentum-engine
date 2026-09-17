"""Export engine results to the static JSON the site reads.

This is the ONLY thing that runs on a schedule. It executes the rules, writes plain
JSON files, and exits. Nothing serves requests; the site is flat files.

  python tools/export_site.py                 # full: backtest + today's lists
  python tools/export_site.py --daily         # today's lists + tracking only (fast)

`--daily` deliberately leaves backtest.json untouched. Re-running a 2021->today
backtest every morning barely moves any statistic and invites tuning on recent data,
which config.yaml's own discipline forbids. The backtest refreshes when the rules
change (its config hash) or on an explicit full run.
"""
from __future__ import annotations

import argparse
import copy
import glob
import hashlib
import json
import logging
import math
import re
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from engine import backtest as bt          # noqa: E402
from engine import data                    # noqa: E402

log = logging.getLogger("export_site")
IST = timezone(timedelta(hours=5, minutes=30))

# Fields published per pick. Deliberately excludes raw OHLCV: the site publishes
# derived output only, never the vendor's price series.
PICK_FIELDS = ["rank", "symbol", "industry", "score", "setup", "last_close", "stop_ref",
               "risk_pct", "target_ref", "reward_to_risk", "why", "risk_flags",
               "median_value_cr"]


def clean(o):
    """Recursively replace NaN/Inf with None so the result is valid JSON."""
    if isinstance(o, dict):
        return {k: clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, float):
        return None if (math.isnan(o) or math.isinf(o)) else round(o, 6)
    if isinstance(o, (pd.Timestamp, datetime)):
        return str(o.date() if hasattr(o, "date") else o)
    if hasattr(o, "item"):                 # numpy scalar
        return clean(o.item())
    return o


def records(df: pd.DataFrame | None) -> list:
    if df is None or not len(df):
        return []
    return clean(df.to_dict(orient="records"))


def rules_hash(cfg: dict) -> str:
    """Fingerprint of the rule-bearing config only, so the site can say whether the
    displayed backtest was computed from the rules currently in force."""
    rules_only = {k: cfg[k] for k in ("filters", "regime", "momentum", "reversal", "controls",
                                      "backtest") if k in cfg}
    blob = json.dumps(rules_only, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode()).hexdigest()[:12]


def stamp_assets(site: Path) -> None:
    """Rewrite index.html asset references to app.js?v=<content hash>.

    GitHub Pages serves assets with max-age=600, so after a deploy a visitor keeps
    running the previous JS for up to ten minutes and a fix appears to have silently
    not worked. Hashing the content into the URL makes a changed file a different URL,
    so the new version is fetched at once while an unchanged one still caches.

    Plain string replacement, not a regex: an earlier regex attempt consumed the
    filename itself and emitted href="?v=..." , which would have shipped a page with
    no stylesheet and no script.
    """
    html = site / "index.html"
    if not html.exists():
        return
    text = original = html.read_text(encoding="utf-8")
    for name in ("app.js", "styles.css"):
        f = site / name
        if not f.exists():
            continue
        digest = hashlib.sha256(f.read_bytes()).hexdigest()[:8]
        for quote in ('"', "'"):
            base = f"{quote}{name}{quote}"
            if base in text:
                text = text.replace(base, f"{quote}{name}?v={digest}{quote}")
                continue
            # already stamped: swap the existing version for the current one
            i = text.find(f"{quote}{name}?v=")
            while i != -1:
                j = text.find(quote, i + 1)
                text = text[:i] + f"{quote}{name}?v={digest}" + text[j:]
                i = text.find(f"{quote}{name}?v=", i + 1)
    if text != original:
        html.write_text(text, encoding="utf-8")
        log.info("stamped asset versions into index.html")


def write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(clean(payload), indent=1, allow_nan=False), encoding="utf-8")
    log.info("wrote %s (%.1f KB)", path.name, path.stat().st_size / 1024)


def archive_universe(universe_file: Path, site_dir: Path, as_of: str) -> dict:
    """Append-only dated snapshot of the constituent list.

    Overwriting would throw away the only cure for the survivorship bias the README
    names as limitation #1. Each snapshot is kept so the universe can later be
    replayed as it actually stood on each past scan date.
    """
    snaps = ROOT / "universe" / "snapshots"
    snaps.mkdir(parents=True, exist_ok=True)
    dest = snaps / f"{universe_file.stem}_{as_of}.csv"
    if not dest.exists() and universe_file.exists():
        dest.write_bytes(universe_file.read_bytes())
        log.info("archived universe snapshot %s", dest.name)
    have = sorted(p.name for p in snaps.glob("*.csv"))
    return {"file": universe_file.name, "snapshots": len(have),
            "first_snapshot": have[0].split("_")[-1].replace(".csv", "") if have else None}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(ROOT / "config.yaml"))
    ap.add_argument("--daily", action="store_true",
                    help="skip the backtest; refresh today's lists and tracking only")
    ap.add_argument("--site-dir", default=str(ROOT / "site"))
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    cfg = yaml.safe_load(open(a.config))
    site = Path(a.site_dir)
    out = site / "data"
    universe_file = ROOT / cfg["run"]["universe_file"]

    universe = data.load_universe(universe_file, cfg["run"].get("max_symbols"))
    panel = data.load_panel(cfg, universe)
    as_of = str(panel.dates[-1].date())
    log.info("panel: %d symbols, %s -> %s", len(panel.symbols), panel.dates[0].date(), as_of)

    prep = bt.prepare(panel, universe, cfg)
    picks = bt.scan_latest(panel, universe, cfg, prep)
    regime_row = prep["regime"].iloc[-1]

    stamp_assets(site)
    uni_meta = archive_universe(universe_file, site, as_of)

    # Persist today's list before anything else reads it back. Forward tracking is the only
    # genuinely out-of-sample record this project produces, and it exists only if every
    # published list is kept. Written once per date, never overwritten.
    picks_dir = ROOT / cfg["run"]["output_dir"]
    picks_dir.mkdir(parents=True, exist_ok=True)
    picks_csv = picks_dir / f"picks_{as_of}.csv"
    if len(picks) and not picks_csv.exists():
        picks.to_csv(picks_csv, index=False)
        log.info("saved %s (%d rows)", picks_csv.name, len(picks))

    # ---- latest.json : Tier 2, refreshed every trading day
    models = [m for m in picks["model"].unique()] if len(picks) else []
    by_model = {m: records(picks[picks["model"] == m][PICK_FIELDS]) for m in models}
    write(out / "latest.json", {
        "generated_at": datetime.now(IST).isoformat(timespec="seconds"),
        "as_of": as_of,
        "data_source": panel.source,
        "universe": {**uni_meta, "symbols": len(panel.symbols),
                     "eligible_today": int(prep["base"].iloc[-1].sum())},
        "regime": {"label": regime_row["regime"],
                   "breadth_50dma": float(regime_row["breadth_50dma"]),
                   "nifty": float(regime_row["nifty"])},
        "counts": {m: int(prep["scores"][m].iloc[-1].notna().sum()) for m in prep["scores"]},
        "strategy": ({"name": prep["spec"]["name"], "spec": prep["spec"]}
                     if prep.get("spec") else None),
        "picks": by_model,
    })

    # ---- tracking.json : Tier 2, forward outcomes of every list saved so far
    saved = sorted(glob.glob(str(ROOT / cfg["run"]["output_dir"] / "picks_*.csv")))
    if saved:
        hist = pd.concat([pd.read_csv(f) for f in saved], ignore_index=True)
        tracked = bt.track_picks(hist, panel, [int(h) for h in cfg["backtest"]["horizons"]])
        keep = [c for c in ["as_of", "model", "symbol", "entry_date", "entry", "return_to_date",
                            "ret_5d", "ret_10d", "ret_20d", "mfe_20d", "mae_20d",
                            "first_hit", "days_to_first_hit", "days_elapsed"]
                if c in tracked.columns]
        write(out / "tracking.json", {"generated_at": datetime.now(IST).isoformat(timespec="seconds"),
                                      "lists": len(saved), "rows": records(tracked[keep])})

    # ---- backtest.json : Tier 1, deliberately stagnant
    bfile = out / "backtest.json"
    h = rules_hash(cfg)
    if a.daily and bfile.exists():
        prev = json.loads(bfile.read_text(encoding="utf-8"))
        if prev.get("rules_hash") == h:
            log.info("--daily and rules unchanged -> backtest.json left as is (%s)", prev.get("as_of"))
            _write_meta(out, cfg, h, as_of, prev.get("as_of"))
            return
        log.info("rules changed (%s -> %s) -> rebuilding backtest", prev.get("rules_hash"), h)

    res = bt.run_backtest(panel, universe, cfg, prep)
    s = res["summary"]
    write(bfile, {
        "generated_at": datetime.now(IST).isoformat(timespec="seconds"),
        "as_of": as_of,
        "rules_hash": h,
        "window": {"start": cfg["backtest"]["start"], "end": as_of},
        "scan_dates": len(res["scan_dates"]),
        "rebalance_every": cfg["backtest"]["rebalance_every"],
        "top_n": cfg["backtest"]["top_n"],
        "cost_pct": cfg["backtest"]["cost_pct"],
        "weights_used": clean(res["weights_used"]),
        "forward": records(s.get("forward")),
        "trades": records(s.get("trades")),
        "by_period": records(s.get("by_period")),
        "by_regime": records(s.get("by_regime")),
        "by_year": records(s.get("by_year")),
    })
    _write_meta(out, cfg, h, as_of, as_of)


def _write_meta(out: Path, cfg: dict, h: str, as_of: str, backtest_as_of: str | None) -> None:
    """Freshness manifest. The site reads this to show what is current and what is frozen."""
    write(out / "meta.json", {
        "generated_at": datetime.now(IST).isoformat(timespec="seconds"),
        "as_of": as_of,
        "rules_hash": h,
        "backtest_as_of": backtest_as_of,
        "backtest_is_stale": backtest_as_of != as_of,
        "thresholds": {
            "momentum": {k: cfg["momentum"][k] for k in
                         ("min_dist_52w_high", "min_atr_pct", "max_atr_pct", "max_ext_atr",
                          "max_ext_50dma", "min_rs_63", "stop_atr", "target_r")},
            "reversal": {k: cfg["reversal"][k] for k in
                         ("min_drawdown", "low_age_min", "low_age_max", "higher_low_atr",
                          "min_support_days", "stop_atr_below_low", "target_r")},
            "filters": cfg["filters"],
        },
    })


if __name__ == "__main__":
    main()
