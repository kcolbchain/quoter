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
from src.backtest.engine import BacktestEngine


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def run_simulate(config: dict, output: str = None):
    """Run a backtest simulation."""
    logging.info("=== SIMULATE MODE ===")

    agent = RWAMarketMaker(
        agent_id="backtest-rwa",
        config={
            "initial_quote": config.get("initial_quote", 10000),
            "base_spread_bps": int(config.get("spread", 0.5) * 100),
            "max_order_size_pct": config.get("max_order_size_pct", 0.1),
            "max_exposure": config.get("max_exposure", 50),
        },
    )

    engine = BacktestEngine(agent=agent, fill_probability=config.get("fill_probability", 0.3))
    data = BacktestEngine.generate_mock_data(
        base_price=config.get("base_price", 100.0),
        ticks=config.get("ticks", 100),
    )
    results = engine.run(data)

    logging.info(f"Ticks:        {results.total_ticks}")
    logging.info(f"Fills:        {results.total_fills} ({results.fill_rate:.1%} fill rate)")
    logging.info(f"Realized PnL: {results.realized_pnl:.2f}")
    logging.info(f"Unrealized:   {results.unrealized_pnl:.2f}")
    logging.info(f"Total PnL:    {results.total_pnl:.2f}")
    logging.info(f"Max Drawdown: {results.max_drawdown:.2f}")
    logging.info(f"Sharpe Ratio: {results.sharpe_ratio:.3f}")

    if output:
        engine.export_results(output)
        print(f"Results exported to {output}")


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
    parser.add_argument("--output", help="Export backtest results to file (.csv or .parquet)")
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
        run_simulate(config, output=args.output)
    else:
        run_live(config)


if __name__ == "__main__":
    main()
