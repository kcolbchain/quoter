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
from src.oracle.price_feed import MockPriceFeed
from src.backtest.engine import BacktestEngine
from src.utils.config import load_config, load_preset, merge_configs


def run_simulate(config: dict, output_path: str = None, output_fmt: str = "csv"):
    """Run a backtest simulation."""
    logging.info("=== SIMULATE MODE ===")
    pair = config.get("pair", "ETH/USDC")
    strategy_name = config.get("strategy", "constant_spread")

    oracle = MockPriceFeed(base_prices={pair: 100.0})

    agent = RWAMarketMaker(
        agent_id="sim-agent",
        oracle=oracle,
        config=config,
    )

    engine = BacktestEngine(agent=agent)
    data = BacktestEngine.generate_mock_data(base_price=100.0, ticks=config.get("ticks", 100))
    results = engine.run(data)
    logging.info(f"Backtest completed: {results.total_fills} fills, PnL: {results.total_pnl:.2f}")

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
    parser.add_argument("--preset", help="Load a registered preset config")
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

    config = {}
    if args.preset:
        config = load_preset(args.preset)
    else:
        config_path = Path(args.config)
        if config_path.exists():
            config = load_config(str(config_path))

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
