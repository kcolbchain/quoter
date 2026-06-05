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
import math
import random
import yaml
from pathlib import Path

from src.agents.rwa_market_maker import RWAMarketMaker
from src.backtest.engine import BacktestEngine, BacktestTick


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def _apply_amm_lp_preset(config: dict) -> dict:
    """Translate AMM LP preset parameters into the simulator config shape."""
    amm_lp = config.get("amm_lp") or {}
    if not amm_lp:
        return config

    agent = config.setdefault("agent", {})
    backtest = config.setdefault("backtest", {})

    agent.setdefault("initial_base", amm_lp.get("initial_base_liquidity", 0))
    agent.setdefault("initial_quote", amm_lp.get("initial_quote_liquidity", 0))
    agent.setdefault("max_inventory_pct", amm_lp.get("target_allocation_pct", 0.5))
    agent.setdefault("base_spread_bps", amm_lp.get("fee_bps", 30))
    backtest.setdefault("base_price", amm_lp.get("initial_price", 100.0))
    backtest.setdefault("fee_bps", amm_lp.get("fee_bps", 30))
    return config


def _build_synthetic_ticks(config: dict) -> list[BacktestTick]:
    """Build deterministic synthetic ticks from config/backtest parameters."""
    backtest = config.get("backtest", {}) or {}
    ticks = int(config.get("ticks", backtest.get("ticks", 100)))
    base_price = float(backtest.get("base_price", config.get("base_price", 100.0)))
    volatility = float(backtest.get("volatility", config.get("volatility", 0.02)))
    volume = float(config.get("agent", {}).get("liquid_volume_threshold", 1_000_000))
    rng = random.Random(42)

    data = []
    price = base_price
    for i in range(ticks):
        # Smooth deterministic wave plus tiny seeded noise for repeatable demos.
        drift = math.sin(i / 12) * volatility * 0.4
        noise = rng.uniform(-volatility, volatility) * 0.2
        price = max(0.01, price * (1 + drift + noise))
        data.append(BacktestTick(
            timestamp=f"tick-{i + 1}",
            oracle_price=price,
            on_chain_price=price * (1 + math.sin(i / 8) * 0.001),
            volume_24h=volume,
        ))
    return data


def run_simulate(config: dict, output_path: str = None, output_fmt: str = "csv"):
    """Run a backtest simulation."""
    logging.info("=== SIMULATE MODE ===")
    config = _apply_amm_lp_preset(config)
    agent_config = config.get("agent", {}) or {}
    agent = RWAMarketMaker(agent_id="quoter-sim", config=agent_config)
    fill_probability = (config.get("backtest", {}) or {}).get("fill_probability", 0.3)
    engine = BacktestEngine(agent=agent, fill_probability=fill_probability)
    results = engine.run(_build_synthetic_ticks(config))
    logging.info("Backtest Results:")
    logging.info(f"  Ticks:        {results.total_ticks}")
    logging.info(f"  Fills:        {results.total_fills} ({results.fill_rate:.1%} fill rate)")
    logging.info(f"  Realized PnL: {results.realized_pnl:.2f}")
    logging.info(f"  Unrealized:   {results.unrealized_pnl:.2f}")
    logging.info(f"  Total PnL:    {results.total_pnl:.2f}")
    logging.info(f"  Max Drawdown: {results.max_drawdown:.2f}")
    logging.info(f"  Sharpe Ratio: {results.sharpe_ratio:.3f}")

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
    parser.add_argument("--ticks", type=int, default=None, help="Simulation ticks")
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
        config.setdefault("agent", {})["base_spread_bps"] = int(args.spread * 100)
    if args.ticks:
        config["ticks"] = args.ticks
        config.setdefault("backtest", {})["ticks"] = args.ticks

    if args.simulate or config.get("simulate", True):
        run_simulate(config, output_path=args.output, output_fmt=args.format)
    else:
        run_live(config)


if __name__ == "__main__":
    main()
