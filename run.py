"""Command line entry point.

  python run.py scan       -> today's momentum + reversal lists (top N each)
  python run.py backtest   -> point-in-time backtest of the rules + today's lists
  python run.py track      -> update outcomes of previously saved lists (forward test)
  python run.py demo       -> full pipeline on synthetic data (checks your install; results meaningless)
"""
from __future__ import annotations

import argparse
import copy
import glob
import logging
from pathlib import Path

import pandas as pd
import yaml

from engine import backtest as bt
from engine import data, report


def load_cfg(path):
    with open(path) as f:
        return yaml.safe_load(f)


def get_universe(cfg):
    if cfg["run"]["data_source"] == "synthetic":
        return data.synthetic_universe()
    return data.load_universe(cfg["run"]["universe_file"], cfg["run"].get("max_symbols"))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["scan", "backtest", "track", "demo"])
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("-v", "--verbose", action="store_true")
    a = ap.parse_args()
    logging.basicConfig(level=logging.DEBUG if a.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    cfg = load_cfg(a.config)
    if a.command == "demo":
        cfg = copy.deepcopy(cfg)
        cfg["run"]["data_source"] = "synthetic"
        cfg["run"]["output_dir"] = str(Path(cfg["run"]["output_dir"]) / "demo")
        cfg["filters"]["min_median_value_cr"] = 0.5
    universe = get_universe(cfg)
    panel = data.load_panel(cfg, universe)
    logging.info("Data: %s | %d symbols | %s to %s", panel.source, len(panel.symbols),
                 panel.dates[0].date(), panel.dates[-1].date())
    out_dir = cfg["run"]["output_dir"]

    if a.command == "track":
        files = sorted(glob.glob(str(Path(out_dir) / "picks_*.csv")))
        if not files:
            raise SystemExit("No saved lists found - run `scan` first.")
        saved = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
        tracked = bt.track_picks(saved, panel, [int(h) for h in cfg["backtest"]["horizons"]])
        dest = Path(out_dir) / "tracking.csv"
        tracked.to_csv(dest, index=False)
        cols = [c for c in ["as_of", "model", "symbol", "entry", "return_to_date", "mfe_5d", "mfe_10d",
                            "mfe_20d", "first_hit", "days_to_first_hit"] if c in tracked.columns]
        print(tracked[cols].to_string(index=False))
        print(f"\nSaved {dest}")
        return

    prep = bt.prepare(panel, universe, cfg)
    picks = bt.scan_latest(panel, universe, cfg, prep)
    res = bt.run_backtest(panel, universe, cfg, prep) if a.command in ("backtest", "demo") else None
    report.print_summary(res or {}, picks)
    tag = f"{a.command}_{panel.dates[-1].date()}"
    path = report.write_outputs(out_dir, res, picks, cfg, tag)
    print(f"\nReport: {path}")


if __name__ == "__main__":
    main()
