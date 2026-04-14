#!/usr/bin/env python3
"""Backtest comparison: with vs without inventory risk model."""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import random
from src.strategies.adaptive_spread import AdaptiveSpreadParams, compute_adaptive_quotes
from src.backtest.engine import BacktestEngine, BacktestTick
from src.agents.base_agent import BaseAgent, Order, Side
from dataclasses import dataclass
import logging

logging.basicConfig(level=logging.WARNING)


@dataclass
class ComparisonResult:
    baseline_total_pnl: float
    enhanced_total_pnl: float
    pnl_improvement: float
    pnl_improvement_pct: float
    baseline_max_drawdown: float
    enhanced_max_drawdown: float
    drawdown_improvement: float
    baseline_sharpe: float
    enhanced_sharpe: float
    sharpe_improvement: float
    baseline_fill_rate: float
    enhanced_fill_rate: float
    baseline_final_position: float
    enhanced_final_position: float


class SimpleAgent(BaseAgent):
    """Fixed spread without inventory risk."""

    def __init__(self, agent_id, config):
        super().__init__(agent_id, config)
        self.base_spread_bps = config.get("base_spread_bps", 200)
        self.max_order_size_pct = config.get("max_order_size_pct", 0.1)

    def evaluate_market(self, market_data):
        return {"mid": market_data.get("oracle_price", 0)}

    def execute_strategy(self, signals):
        mid = signals["mid"]
        if mid <= 0:
            return []
        spread = self.base_spread_bps / 10000
        size = max(self.position.quote_balance * self.max_order_size_pct / mid, 0.001)
        return [
            Order(side=Side.BID, price=mid * (1 - spread / 2), size=size),
            Order(side=Side.ASK, price=mid * (1 + spread / 2), size=size),
        ]

    def rebalance(self):
        return []


class InventoryAwareAgent(BaseAgent):
    """Adaptive spread with full inventory risk model."""

    def __init__(self, agent_id, config):
        super().__init__(agent_id, config)
        self.params = AdaptiveSpreadParams(
            base_spread_bps=config.get("base_spread_bps", 200),
            vol_multiplier=config.get("vol_multiplier", 5.0),
            inventory_skew_factor=config.get("inventory_skew_factor", 0.5),
            max_inventory_ratio=config.get("max_inventory_ratio", 0.3),
            risk_adjustment_factor=config.get("risk_adjustment_factor", 2.0),
            min_spread_bps=config.get("min_spread_bps", 50),
            max_spread_bps=config.get("max_spread_bps", 1000),
        )
        self.max_order_size_pct = config.get("max_order_size_pct", 0.1)
        self.volatility_estimate = 0.02
        self._mid = 0.0

    def evaluate_market(self, market_data):
        mid = market_data.get("oracle_price", 0)
        self._mid = mid
        volume = market_data.get("volume_24h", 100000)
        self.volatility_estimate = min(0.5, max(0.001, 10000 / volume))
        total_value = self.position.base_balance * mid + self.position.quote_balance
        inventory_ratio = (self.position.base_balance * mid) / total_value if total_value > 0 else 0.0
        return {"mid": mid, "inventory_ratio": inventory_ratio}

    def execute_strategy(self, signals):
        mid = signals["mid"]
        if mid <= 0:
            return []

        quotes = compute_adaptive_quotes(
            mid_price=mid, volatility=self.volatility_estimate,
            inventory_ratio=signals["inventory_ratio"], params=self.params,
        )

        base_size = max(self.position.quote_balance * self.max_order_size_pct / mid, 0.001)
        risk_reduction = max(0.3, 1.0 - quotes["risk_score"] * 0.7)
        bid_size = base_size * risk_reduction
        ask_size = base_size * risk_reduction

        ir = signals["inventory_ratio"]
        if ir > 0.5:
            bid_size *= 0.5
            ask_size *= 1.5
        elif ir < -0.5:
            bid_size *= 1.5
            ask_size *= 0.5

        return [
            Order(side=Side.BID, price=quotes["bid"], size=bid_size),
            Order(side=Side.ASK, price=quotes["ask"], size=ask_size),
        ]

    def rebalance(self):
        return []


def run_comparison(base_price=100.0, ticks=2000, fill_probability=0.3, seed=42):
    random.seed(seed)
    np.random.seed(seed)

    data = []
    price = base_price
    for i in range(ticks):
        regime_vol = 0.02
        trend = 0.0
        if 500 < i < 800:
            regime_vol = 0.05
        elif 1200 < i < 1500:
            regime_vol = 0.04
            trend = 0.001
        price *= (1 + random.gauss(trend, regime_vol))
        price = max(price, 1.0)
        data.append(BacktestTick(
            timestamp=f"2026-01-{(i // 96) + 1:02d}T{(i % 96) // 4:02d}:{(i % 4) * 15:02d}",
            oracle_price=price, volume_24h=random.uniform(50000, 500000),
        ))

    config = {"initial_quote": 10000, "base_spread_bps": 150, "max_order_size_pct": 0.05, "max_exposure": 50}
    enhanced_config = {
        **config, "inventory_skew_factor": 0.5, "max_inventory_ratio": 0.3,
        "risk_adjustment_factor": 2.0, "vol_multiplier": 5.0,
        "min_spread_bps": 50, "max_spread_bps": 1000,
    }

    random.seed(seed); np.random.seed(seed)
    baseline_result = BacktestEngine(SimpleAgent("baseline", config), fill_probability=fill_probability).run(data)

    random.seed(seed); np.random.seed(seed)
    enhanced_result = BacktestEngine(InventoryAwareAgent("enhanced", enhanced_config), fill_probability=fill_probability).run(data)

    pnl_imp = enhanced_result.total_pnl - baseline_result.total_pnl
    pnl_pct = (pnl_imp / abs(baseline_result.total_pnl)) * 100 if baseline_result.total_pnl != 0 else 0

    return ComparisonResult(
        baseline_total_pnl=baseline_result.total_pnl,
        enhanced_total_pnl=enhanced_result.total_pnl,
        pnl_improvement=pnl_imp,
        pnl_improvement_pct=pnl_pct,
        baseline_max_drawdown=baseline_result.max_drawdown,
        enhanced_max_drawdown=enhanced_result.max_drawdown,
        drawdown_improvement=baseline_result.max_drawdown - enhanced_result.max_drawdown,
        baseline_sharpe=baseline_result.sharpe_ratio,
        enhanced_sharpe=enhanced_result.sharpe_ratio,
        sharpe_improvement=enhanced_result.sharpe_ratio - baseline_result.sharpe_ratio,
        baseline_fill_rate=baseline_result.fill_rate,
        enhanced_fill_rate=enhanced_result.fill_rate,
        baseline_final_position=baseline_result.final_position,
        enhanced_final_position=enhanced_result.final_position,
    )
