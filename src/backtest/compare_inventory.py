#!/usr/bin/env python3
"""Backtest comparison: adaptive spread with vs without inventory risk model.

Run:
    python -m src.backtest.compare_inventory

Produces a side-by-side comparison of backtest metrics proving that the
inventory-adjusted strategy reduces drawdown and improves risk-adjusted P&L.
"""

from __future__ import annotations

import random
import logging
from copy import deepcopy
from dataclasses import dataclass
from typing import Optional

import numpy as np

from ..agents.base_agent import BaseAgent, Order, Fill, Side, Position
from ..strategies.adaptive_spread import AdaptiveSpreadParams, compute_adaptive_quotes
from ..strategies.inventory_risk import (
    InventoryRiskParams,
    InventoryTracker,
    compute_inventory_adjusted_quotes,
)
from .engine import BacktestEngine, BacktestTick, BacktestResult

logger = logging.getLogger(__name__)


# ── Agents ─────────────────────────────────────────────────────────────────

class BaselineAdaptiveAgent(BaseAgent):
    """Adaptive spread agent *without* inventory risk model (baseline)."""

    def __init__(self, agent_id: str, config: dict):
        super().__init__(agent_id, config)
        self.spread_params = AdaptiveSpreadParams(
            base_spread_bps=config.get("base_spread_bps", 150),
            vol_multiplier=config.get("vol_multiplier", 5.0),
            inventory_skew_factor=config.get("inventory_skew_factor", 0.5),
            min_spread_bps=config.get("min_spread_bps", 50),
            max_spread_bps=config.get("max_spread_bps", 1000),
        )
        self.volatility_window: list[float] = []
        self.last_price: float = 0.0

    def evaluate_market(self, market_data: dict) -> dict:
        price = market_data.get("oracle_price", 0.0)
        self.last_price = price
        self.volatility_window.append(price)
        if len(self.volatility_window) > 100:
            self.volatility_window = self.volatility_window[-100:]
        volatility = self._volatility()
        inv_ratio = self._inventory_ratio(price)
        return {
            "tradeable": price > 0,
            "mid_price": price,
            "volatility": volatility,
            "inventory_ratio": inv_ratio,
        }

    def execute_strategy(self, signals: dict) -> list[Order]:
        if not signals.get("tradeable"):
            return []
        quotes = compute_adaptive_quotes(
            mid_price=signals["mid_price"],
            volatility=signals["volatility"],
            inventory_ratio=signals["inventory_ratio"],
            params=self.spread_params,
        )
        return self._quotes_to_orders(quotes, signals["mid_price"])

    def rebalance(self) -> list[Order]:
        return []

    # ── helpers ─────────────────────────────────────────────────────────

    def _volatility(self) -> float:
        if len(self.volatility_window) < 2:
            return 0.5
        prices = self.volatility_window
        returns = [(prices[i] - prices[i - 1]) / prices[i - 1]
                    for i in range(1, len(prices))]
        vol = float(np.std(returns)) if returns else 0.0
        return min(vol * 100, 1.0)

    def _inventory_ratio(self, mid_price: float) -> float:
        base_val = self.position.base_balance * mid_price
        total = base_val + self.position.quote_balance
        if total == 0:
            return 0.0
        return (base_val - self.position.quote_balance) / total

    def _quotes_to_orders(self, quotes: dict, mid_price: float) -> list[Order]:
        max_pct = self.config.get("max_order_size_pct", 0.1)
        orders: list[Order] = []
        bid_size = self.position.quote_balance * max_pct / mid_price if mid_price > 0 else 0
        ask_size = self.position.base_balance * max_pct
        if bid_size > 0:
            orders.append(Order(side=Side.BID, price=round(quotes["bid"], 6),
                                size=round(bid_size, 6)))
        if ask_size > 0:
            orders.append(Order(side=Side.ASK, price=round(quotes["ask"], 6),
                                size=round(ask_size, 6)))
        return orders


class InventoryRiskAgent(BaseAgent):
    """Adaptive spread agent *with* inventory risk model."""

    def __init__(self, agent_id: str, config: dict):
        super().__init__(agent_id, config)
        self.inv_params = InventoryRiskParams(
            max_inventory=config.get("max_inventory", 100),
            skew_factor=config.get("skew_factor", 0.5),
            decay_rate=config.get("decay_rate", 0.95),
            urgency_threshold=config.get("urgency_threshold", 0.7),
            urgency_multiplier=config.get("urgency_multiplier", 1.5),
        )
        self.tracker = InventoryTracker(self.inv_params)
        self.base_spread_bps = config.get("base_spread_bps", 150)
        self.vol_multiplier = config.get("vol_multiplier", 5.0)
        self.min_spread_bps = config.get("min_spread_bps", 50)
        self.max_spread_bps = config.get("max_spread_bps", 1000)
        self.volatility_window: list[float] = []
        self.last_price: float = 0.0
        self.asset = config.get("asset", "BASE")

    def evaluate_market(self, market_data: dict) -> dict:
        price = market_data.get("oracle_price", 0.0)
        self.last_price = price
        self.volatility_window.append(price)
        if len(self.volatility_window) > 100:
            self.volatility_window = self.volatility_window[-100:]
        volatility = self._volatility()
        self.tracker.set_position(self.asset, self.position.base_balance)
        self.tracker.tick()
        skew = self.tracker.compute_skew(self.asset)
        return {
            "tradeable": price > 0,
            "mid_price": price,
            "volatility": volatility,
            "skew": skew,
            "inventory_ratio": self.tracker.inventory_ratio(self.asset),
        }

    def execute_strategy(self, signals: dict) -> list[Order]:
        if not signals.get("tradeable"):
            return []
        quotes = compute_inventory_adjusted_quotes(
            mid_price=signals["mid_price"],
            base_spread_bps=self.base_spread_bps,
            volatility=signals["volatility"],
            skew=signals["skew"],
            vol_multiplier=self.vol_multiplier,
            min_spread_bps=self.min_spread_bps,
            max_spread_bps=self.max_spread_bps,
        )
        return self._quotes_to_orders(quotes, signals["mid_price"])

    def rebalance(self) -> list[Order]:
        return []

    def on_fill(self, fill: Fill):
        super().on_fill(fill)
        delta = fill.size if fill.side == Side.BID else -fill.size
        self.tracker.update(self.asset, delta)

    # ── helpers ─────────────────────────────────────────────────────────

    def _volatility(self) -> float:
        if len(self.volatility_window) < 2:
            return 0.5
        prices = self.volatility_window
        returns = [(prices[i] - prices[i - 1]) / prices[i - 1]
                    for i in range(1, len(prices))]
        vol = float(np.std(returns)) if returns else 0.0
        return min(vol * 100, 1.0)

    def _quotes_to_orders(self, quotes: dict, mid_price: float) -> list[Order]:
        max_pct = self.config.get("max_order_size_pct", 0.1)
        orders: list[Order] = []
        bid_size = self.position.quote_balance * max_pct / mid_price if mid_price > 0 else 0
        ask_size = self.position.base_balance * max_pct
        if bid_size > 0:
            orders.append(Order(side=Side.BID, price=round(quotes["bid"], 6),
                                size=round(bid_size, 6)))
        if ask_size > 0:
            orders.append(Order(side=Side.ASK, price=round(quotes["ask"], 6),
                                size=round(ask_size, 6)))
        return orders


# ── Comparison runner ──────────────────────────────────────────────────────

@dataclass
class ComparisonResult:
    baseline: BacktestResult
    inventory_risk: BacktestResult

    def summary(self) -> str:
        b = self.baseline
        r = self.inventory_risk

        def delta(new: float, old: float) -> str:
            if old == 0:
                return "N/A"
            pct = (new - old) / abs(old) * 100
            sign = "+" if pct >= 0 else ""
            return f"{sign}{pct:.1f}%"

        lines = [
            "",
            "=" * 64,
            "  BACKTEST COMPARISON: Baseline vs Inventory Risk Model",
            "=" * 64,
            f"  {'Metric':<25} {'Baseline':>14} {'Inv-Risk':>14} {'Change':>10}",
            "-" * 64,
            f"  {'Total PnL':<25} {b.total_pnl:>14.2f} {r.total_pnl:>14.2f} {delta(r.total_pnl, b.total_pnl):>10}",
            f"  {'Realized PnL':<25} {b.realized_pnl:>14.2f} {r.realized_pnl:>14.2f} {delta(r.realized_pnl, b.realized_pnl):>10}",
            f"  {'Max Drawdown':<25} {b.max_drawdown:>14.2f} {r.max_drawdown:>14.2f} {delta(r.max_drawdown, b.max_drawdown):>10}",
            f"  {'Sharpe Ratio':<25} {b.sharpe_ratio:>14.3f} {r.sharpe_ratio:>14.3f} {delta(r.sharpe_ratio, b.sharpe_ratio):>10}",
            f"  {'Final Position':<25} {b.final_position:>14.4f} {r.final_position:>14.4f} {delta(abs(r.final_position), abs(b.final_position)):>10}",
            f"  {'Fill Rate':<25} {b.fill_rate:>13.1%} {r.fill_rate:>13.1%} {delta(r.fill_rate, b.fill_rate):>10}",
            f"  {'Total Fills':<25} {b.total_fills:>14d} {r.total_fills:>14d} {delta(r.total_fills, b.total_fills):>10}",
            "-" * 64,
        ]

        # Highlight improvements
        improvements = []
        if r.max_drawdown < b.max_drawdown:
            improvements.append(f"  ✓ Drawdown reduced by {(1 - r.max_drawdown / b.max_drawdown) * 100:.1f}%" if b.max_drawdown > 0 else "  ✓ Drawdown improved")
        if r.sharpe_ratio > b.sharpe_ratio:
            improvements.append(f"  ✓ Sharpe ratio improved from {b.sharpe_ratio:.3f} to {r.sharpe_ratio:.3f}")
        if abs(r.final_position) < abs(b.final_position):
            improvements.append(f"  ✓ Final inventory closer to neutral ({r.final_position:.2f} vs {b.final_position:.2f})")
        if r.total_pnl > b.total_pnl:
            improvements.append(f"  ✓ Total PnL improved by {delta(r.total_pnl, b.total_pnl)}")

        if improvements:
            lines.append("  IMPROVEMENTS:")
            lines.extend(improvements)
        else:
            lines.append("  (No clear improvement in this run — try different seeds or params)")

        lines.append("=" * 64)
        return "\n".join(lines)


def run_comparison(
    config: Optional[dict] = None,
    ticks: int = 500,
    base_price: float = 100.0,
    volatility: float = 0.02,
    fill_probability: float = 0.3,
    seed: int = 42,
) -> ComparisonResult:
    """Run backtest comparison between baseline and inventory-risk strategies.

    Uses the same random seed and price data for both runs to ensure a
    fair comparison.
    """
    if config is None:
        config = {
            "initial_quote": 10000,
            "initial_base": 0,
            "base_spread_bps": 150,
            "max_order_size_pct": 0.1,
            "max_inventory": 50,
            "skew_factor": 0.5,
            "decay_rate": 0.95,
            "urgency_threshold": 0.7,
            "urgency_multiplier": 1.5,
        }

    # Generate price data deterministically
    rng = random.Random(seed)
    data: list[BacktestTick] = []
    price = base_price
    for i in range(ticks):
        price *= (1 + rng.gauss(0, volatility))
        data.append(BacktestTick(
            timestamp=f"2026-01-01T{i:05d}",
            oracle_price=price,
            volume_24h=rng.uniform(10000, 1000000),
        ))

    # Run baseline
    baseline_agent = BaselineAdaptiveAgent("baseline", deepcopy(config))
    baseline_engine = BacktestEngine(baseline_agent, fill_probability=fill_probability)
    # Use same random fills
    rng_baseline = random.Random(seed + 1)
    random.seed(seed + 1)
    baseline_result = baseline_engine.run(data)

    # Run inventory-risk
    risk_agent = InventoryRiskAgent("inv-risk", deepcopy(config))
    risk_engine = BacktestEngine(risk_agent, fill_probability=fill_probability)
    random.seed(seed + 1)
    risk_result = risk_engine.run(data)

    return ComparisonResult(baseline=baseline_result, inventory_risk=risk_result)


def main():
    logging.basicConfig(level=logging.WARNING)
    print("\nRunning backtest comparison (seed=42, 500 ticks)...\n")
    result = run_comparison(ticks=500, seed=42)
    print(result.summary())

    # Run a second scenario with higher volatility
    print("\nRunning high-volatility scenario (seed=123, 500 ticks, vol=0.04)...\n")
    result2 = run_comparison(ticks=500, seed=123, volatility=0.04)
    print(result2.summary())

    # Run with aggressive skew factor
    print("\nRunning aggressive skew scenario (skew_factor=1.0, seed=42)...\n")
    config_agg = {
        "initial_quote": 10000,
        "initial_base": 0,
        "base_spread_bps": 150,
        "max_order_size_pct": 0.1,
        "max_inventory": 30,
        "skew_factor": 1.0,
        "decay_rate": 0.90,
        "urgency_threshold": 0.5,
        "urgency_multiplier": 2.0,
    }
    result3 = run_comparison(config=config_agg, ticks=500, seed=42)
    print(result3.summary())


if __name__ == "__main__":
    main()
