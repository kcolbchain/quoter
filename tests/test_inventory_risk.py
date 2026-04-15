"""Tests for inventory risk model in adaptive spread strategy."""

import pytest
import numpy as np
from src.strategies.adaptive_spread import (
    AdaptiveSpreadParams,
    compute_adaptive_quotes,
)
from src.agents.rwa_market_maker import RWAMarketMaker
from src.agents.base_agent import Fill, Side
from src.backtest.engine import BacktestEngine, BacktestTick


class TestInventorySkewDirection:
    """Verify that inventory skew reduces exposure correctly."""

    def test_long_inventory_widens_ask_tightens_bid(self):
        """When long, ask should be further from mid, bid closer to mid."""
        params = AdaptiveSpreadParams(base_spread_bps=200, inventory_skew_factor=0.5)
        mid = 100.0

        # Neutral inventory
        neutral = compute_adaptive_quotes(mid, 0.0, 0.0, params)
        # Long inventory
        long_inv = compute_adaptive_quotes(mid, 0.0, 0.8, params)

        # Ask moves further from mid when long (wider ask)
        assert long_inv["ask"] > neutral["ask"]
        # Bid moves closer to mid when long (tighter bid)
        assert long_inv["bid"] > neutral["bid"]

    def test_short_inventory_widens_bid_tightens_ask(self):
        """When short, bid should be further from mid, ask closer to mid."""
        params = AdaptiveSpreadParams(base_spread_bps=200, inventory_skew_factor=0.5)
        mid = 100.0

        neutral = compute_adaptive_quotes(mid, 0.0, 0.0, params)
        short_inv = compute_adaptive_quotes(mid, 0.0, -0.8, params)

        # Bid moves further from mid when short (wider bid)
        assert short_inv["bid"] < neutral["bid"]
        # Ask moves closer to mid when short (tighter ask)
        assert short_inv["ask"] < neutral["ask"]

    def test_neutral_inventory_equal_spreads(self):
        """With zero inventory, bid and ask spreads should be equal."""
        params = AdaptiveSpreadParams(base_spread_bps=200)
        result = compute_adaptive_quotes(100.0, 0.0, 0.0, params)
        assert result["bid_spread_bps"] == pytest.approx(result["ask_spread_bps"])

    def test_symmetry(self):
        """Equal magnitude long/short should produce symmetric spreads."""
        params = AdaptiveSpreadParams(base_spread_bps=200, inventory_skew_factor=0.5)
        long_q = compute_adaptive_quotes(100.0, 0.0, 0.5, params)
        short_q = compute_adaptive_quotes(100.0, 0.0, -0.5, params)

        # Long bid_spread = Short ask_spread and vice versa
        assert long_q["bid_spread_bps"] == pytest.approx(short_q["ask_spread_bps"])
        assert long_q["ask_spread_bps"] == pytest.approx(short_q["bid_spread_bps"])


class TestInventorySkewBounds:
    """Verify spreads stay within valid bounds."""

    def test_spreads_always_positive(self):
        """Bid and ask spreads should never go negative."""
        params = AdaptiveSpreadParams(base_spread_bps=200, inventory_skew_factor=0.8)
        for inv in [-0.99, -0.5, 0.0, 0.5, 0.99]:
            result = compute_adaptive_quotes(100.0, 0.3, inv, params)
            assert result["bid_spread_bps"] > 0, f"Negative bid spread at inv={inv}"
            assert result["ask_spread_bps"] > 0, f"Negative ask spread at inv={inv}"

    def test_bid_below_ask(self):
        """Bid price should always be below ask price."""
        params = AdaptiveSpreadParams(base_spread_bps=100, inventory_skew_factor=0.9)
        for inv in [-0.99, -0.5, 0.0, 0.5, 0.99]:
            result = compute_adaptive_quotes(100.0, 0.5, inv, params)
            assert result["bid"] < result["ask"], f"Bid >= Ask at inv={inv}"

    def test_skew_factor_zero_no_effect(self):
        """With zero skew factor, inventory should not affect spreads."""
        params = AdaptiveSpreadParams(base_spread_bps=200, inventory_skew_factor=0.0)
        neutral = compute_adaptive_quotes(100.0, 0.0, 0.0, params)
        long_inv = compute_adaptive_quotes(100.0, 0.0, 0.8, params)
        assert neutral["bid"] == pytest.approx(long_inv["bid"])
        assert neutral["ask"] == pytest.approx(long_inv["ask"])


class TestConfigurableSkewFactor:
    """Verify that inventory_skew_factor is configurable."""

    def test_higher_skew_factor_more_aggressive(self):
        """Higher skew factor should produce more spread asymmetry."""
        params_low = AdaptiveSpreadParams(base_spread_bps=200, inventory_skew_factor=0.2)
        params_high = AdaptiveSpreadParams(base_spread_bps=200, inventory_skew_factor=0.8)

        low = compute_adaptive_quotes(100.0, 0.0, 0.5, params_low)
        high = compute_adaptive_quotes(100.0, 0.0, 0.5, params_high)

        # Higher skew = bigger difference between bid and ask spreads
        low_diff = abs(low["ask_spread_bps"] - low["bid_spread_bps"])
        high_diff = abs(high["ask_spread_bps"] - high["bid_spread_bps"])
        assert high_diff > low_diff


class TestBacktestComparison:
    """Backtest: inventory-skewed strategy vs. constant spread."""

    def _run_backtest(self, agent, seed=42):
        """Run a backtest with deterministic data."""
        import random
        random.seed(seed)
        engine = BacktestEngine(agent, fill_probability=0.3)
        data = engine.generate_mock_data(100.0, 1000, volatility=0.015)
        return engine.run(data)

    def test_skewed_strategy_lower_max_drawdown(self):
        """Inventory-aware strategy should have lower max drawdown than constant."""
        # Agent with inventory skew
        config_skew = {
            "initial_quote": 10000,
            "base_spread_bps": 200,
            "max_order_size_pct": 0.1,
            "max_inventory_pct": 0.3,
        }
        agent_skew = RWAMarketMaker("skew", config_skew)

        # Agent without inventory awareness (very low skew in execute_strategy)
        config_flat = {
            "initial_quote": 10000,
            "base_spread_bps": 200,
            "max_order_size_pct": 0.1,
            "max_inventory_pct": 0.99,  # effectively no inventory limit
        }
        agent_flat = RWAMarketMaker("flat", config_flat)

        result_skew = self._run_backtest(agent_skew)
        result_flat = self._run_backtest(agent_flat)

        # Skewed agent should have lower or comparable drawdown
        # (inventory management reduces directional risk)
        assert result_skew.max_drawdown <= result_flat.max_drawdown * 1.1  # allow 10% margin

    def test_skewed_strategy_smaller_final_position(self):
        """Inventory-aware strategy should end with less directional exposure."""
        config_skew = {
            "initial_quote": 10000,
            "base_spread_bps": 200,
            "max_order_size_pct": 0.1,
            "max_inventory_pct": 0.3,
        }
        config_flat = {
            "initial_quote": 10000,
            "base_spread_bps": 200,
            "max_order_size_pct": 0.1,
            "max_inventory_pct": 0.99,
        }

        result_skew = self._run_backtest(RWAMarketMaker("s", config_skew))
        result_flat = self._run_backtest(RWAMarketMaker("f", config_flat))

        # Inventory-managed agent should have smaller absolute position
        assert abs(result_skew.final_position) <= abs(result_flat.final_position) * 1.1
