#!/usr/bin/env python3
"""
kcolbchain quoter — autonomous market-making agent runner.

Usage:
    python run.py --simulate                    # backtest mode
    python run.py --config config/live.yaml     # live mode
    python run.py --pair ETH/USDC --spread 0.5  # quick test
"""
import argparse
import logging
import yaml
from pathlib import Path

from src.agents.rwa_market_maker import RWAMarketMaker
from src.strategies.constant_spread import ConstantSpreadStrategy
from src.strategies.adaptive_spread import AdaptiveSpreadStrategy
from src.oracle.price_feed import PriceFeed
from src.backtest.engine import BacktestEngine


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def run_simulate(config: dict, output_path: str = None, output_fmt: str = "csv"):
    """Run a backtest simulation."""
    logging.info("=== SIMULATE MODE ===")
    pair = config.get("pair", "ETH/USDC")
    strategy_name = config.get("strategy", "constant_spread")

    oracle = PriceFeed(source="mock", pair=pair)

    if strategy_name == "adaptive":
        strategy = AdaptiveSpreadStrategy(
            base_spread=config.get("spread", 0.5),
            volatility_window=config.get("vol_window", 20),
        )
    else:
        strategy = ConstantSpreadStrategy(spread_pct=config.get("spread", 0.5))

    agent = RWAMarketMaker(
        strategy=strategy,
        oracle=oracle,
        config=config,
    )

    engine = BacktestEngine(agent=agent, oracle=oracle)
    results = engine.run(ticks=config.get("ticks", 100))
    engine.print_summary(results)

    if output_path:
        path = engine.export(results, output_path, fmt=output_fmt)
        if path:
            logging.info(f"Fills exported to {path} ({results.total_fills} rows)")


def run_live(config: dict):
    """Run live market making (requires API keys + wallet)."""
    logging.info("=== LIVE MODE ===")
    logging.info("Live trading not yet implemented — use --simulate for now")
    logging.info("To go live, implement exchange connectors in src/exchanges/")


def main():
    parser = argparse.ArgumentParser(description="kcolbchain quoter — market-making agent")
    parser.add_argument("--config", default="config/default.yaml", help="Config path")
    parser.add_argument("--simulate", action="store_true", help="Backtest mode")
    parser.add_argument("--pair", help="Trading pair (e.g., ETH/USDC)")
    parser.add_argument("--spread", type=float, help="Spread percentage")
    parser.add_argument("--ticks", type=int, default=100, help="Simulation ticks")
    parser.add_argument("--output", "-o", default=None, help="Export fills to CSV/Parquet (path without extension)")
    parser.add_argument("--format", "-f", default="csv", choices=["csv", "parquet"], help="Export format (default: csv)")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s [%(name)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    config_path = Path(args.config)
    config = load_config(config_path) if config_path.exists() else {}

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
