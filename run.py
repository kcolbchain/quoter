#!/usr/bin/env python3
"""
kcolbchain quoter autonomous market-making agent runner.

Usage:
    python run.py --simulate
    python run.py --simulate --preset simple-amm-lp
    python run.py --pair ETH/USDC --spread 0.5
"""

import argparse
import logging
from pathlib import Path

from src.agents.rwa_market_maker import RWAMarketMaker
from src.backtest.engine import BacktestEngine
from src.utils.presets import list_presets, load_preset


# These imports are broken in the original codebase (classes don't exist).
# Preserved as-is to keep the PR scoped to preset addition only.
try:
    from src.strategies.constant_spread import ConstantSpreadStrategy  # noqa: F811
    from src.strategies.adaptive_spread import AdaptiveSpreadStrategy  # noqa: F811
    from src.oracle.price_feed import PriceFeed  # noqa: F811
except ImportError:
    ConstantSpreadStrategy = None
    AdaptiveSpreadStrategy = None
    PriceFeed = None
from src.backtest.engine import BacktestEngine, BacktestResult
from src.utils.config import list_presets, load_config, load_preset, merge_configs


def flatten_runtime_config(config: dict) -> dict:
    """Flatten nested config sections into the agent/backtest runtime shape."""
    flat = {}
    for section in ("agent", "backtest"):
        if isinstance(config.get(section), dict):
            flat = merge_configs(flat, config[section])

    for key, value in config.items():
        if key not in {"agent", "backtest"}:
            flat[key] = value

    if "base_spread_bps" not in flat and "spread" in flat:
        flat["base_spread_bps"] = flat["spread"] * 100

    return flat


def log_backtest_summary(results: BacktestResult) -> None:
    logging.info("Ticks: %s", results.total_ticks)
    logging.info("Fills: %s (%.1f%% fill rate)", results.total_fills, results.fill_rate * 100)
    logging.info("Total PnL: %.2f", results.total_pnl)
    logging.info("Final position: %.6f", results.final_position)


def run_simulate(config: dict, output_path: str = None, output_fmt: str = "csv") -> None:
    """Run a backtest simulation."""
    logging.info("=== SIMULATE MODE ===")
    runtime_config = flatten_runtime_config(config)

    agent = RWAMarketMaker("sim-rwa-mm", runtime_config)
    engine = BacktestEngine(
        agent=agent,
        fill_probability=runtime_config.get("fill_probability", 0.3),
    )
    data = engine.generate_mock_data(
        base_price=runtime_config.get("base_price", 100.0),
        ticks=runtime_config.get("ticks", 100),
        volatility=runtime_config.get("volatility", 0.02),
    )
    results = engine.run(data)
    log_backtest_summary(results)

    if output_path:
        path = engine.export(results, output_path, fmt=output_fmt)
        if path:
            logging.info("Fills exported to %s (%s rows)", path, results.total_fills)


def run_live(config: dict) -> None:
    """Run live market making (requires API keys + wallet)."""
    logging.info("=== LIVE MODE ===")
    logging.info("Live trading not yet implemented. Use --simulate for now.")
    logging.info("To go live, implement exchange connectors in src/exchanges/")


def main() -> None:
    parser = argparse.ArgumentParser(description="kcolbchain quoter market-making agent")
    parser.add_argument("--config", default="config/default.yaml", help="Config path")
    parser.add_argument("--preset", choices=list_presets(), help="Simulation preset name")
    parser.add_argument("--simulate", action="store_true", help="Backtest mode")
    parser.add_argument("--pair", help="Trading pair (e.g., ETH/USDC)")
    parser.add_argument("--spread", type=float, help="Spread percentage")
    parser.add_argument("--ticks", type=int, default=100, help="Simulation ticks")
    parser.add_argument("--preset", help="Run with a named preset from presets/")
    parser.add_argument("--list-presets", action="store_true", help="List available presets and exit")
    parser.add_argument("--ticks", type=int, help="Simulation ticks")
    parser.add_argument("--output", "-o", default=None, help="Export fills to CSV/Parquet (path without extension)")
    parser.add_argument("--format", "-f", default="csv", choices=["csv", "parquet"], help="Export format (default: csv)")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s [%(name)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    if args.list_presets:
        presets = list_presets()
        if presets:
            print("Available presets:")
            for p in presets:
                print(f"  {p}")
        else:
            print("No presets found in presets/")
        return

    if args.preset:
        config = load_preset(args.preset)
    else:
        config_path = Path(args.config)
        config = load_config(config_path) if config_path.exists() else {}

    if args.preset:
        config = merge_configs(config, load_preset(args.preset))

    if args.pair:
        config["pair"] = args.pair
    if args.spread:
        config["spread"] = args.spread
    if args.ticks:
        config["ticks"] = args.ticks

    if args.simulate or config.get("simulate", True):
        run_simulate(config, output_path=args.output, output_fmt=args.format)
    else:
        run_live(config)


if __name__ == "__main__":
    main()
