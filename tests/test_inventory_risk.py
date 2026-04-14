"""Tests for inventory risk model in adaptive spread strategy."""

import pytest
import numpy as np

from src.strategies.adaptive_spread import (
    AdaptiveSpreadParams,
    compute_adaptive_quotes,
    compute_inventory_risk_score,
    compute_value_at_risk,
    compute_risk_adjusted_spread,
)


class TestInventoryRiskScore:
    """Test inventory risk score computation."""

    def test_zero_inventory_zero_risk(self):
        params = AdaptiveSpreadParams()
        score = compute_inventory_risk_score(0.0, 0.02, params)
        assert score == 0.0

    def test_max_inventory_caps_at_one(self):
        params = AdaptiveSpreadParams(max_inventory_ratio=0.3)
        score = compute_inventory_risk_score(1.0, 0.02, params)
        assert score <= 1.0

    def test_higher_volatility_increases_risk(self):
        params = AdaptiveSpreadParams()
        low_vol = compute_inventory_risk_score(0.15, 0.01, params)
        high_vol = compute_inventory_risk_score(0.15, 0.1, params)
        assert high_vol > low_vol

    def test_negative_inventory_same_risk(self):
        """Risk should be symmetric for long/short."""
        params = AdaptiveSpreadParams()
        long_risk = compute_inventory_risk_score(0.15, 0.02, params)
        short_risk = compute_inventory_risk_score(-0.15, 0.02, params)
        assert abs(long_risk - short_risk) < 0.2


class TestValueAtRisk:
    """Test VaR computation."""

    def test_zero_inventory_zero_var(self):
        params = AdaptiveSpreadParams()
        var = compute_value_at_risk(0.0, 0.02, 100.0, params)
        assert var == 0.0

    def test_positive_inventory_positive_var(self):
        params = AdaptiveSpreadParams()
        var = compute_value_at_risk(0.15, 0.02, 100.0, params)
        assert var > 0.0

    def test_var_increases_with_position(self):
        params = AdaptiveSpreadParams()
        small_var = compute_value_at_risk(0.1, 0.02, 100.0, params)
        large_var = compute_value_at_risk(0.3, 0.02, 100.0, params)
        assert large_var > small_var

    def test_var_increases_with_volatility(self):
        params = AdaptiveSpreadParams()
        low_vol_var = compute_value_at_risk(0.15, 0.01, 100.0, params)
        high_vol_var = compute_value_at_risk(0.15, 0.1, 100.0, params)
        assert high_vol_var > low_vol_var


class TestRiskAdjustedSpread:
    """Test risk-adjusted spread computation."""

    def test_no_risk_no_adjustment(self):
        params = AdaptiveSpreadParams(min_spread_bps=50, max_spread_bps=1000)
        spread = compute_risk_adjusted_spread(100.0, 0.0, params)
        assert spread == 100.0

    def test_high_risk_widens_spread(self):
        params = AdaptiveSpreadParams(min_spread_bps=50, max_spread_bps=1000)
        low_risk = compute_risk_adjusted_spread(100.0, 0.1, params)
        high_risk = compute_risk_adjusted_spread(100.0, 0.9, params)
        assert high_risk > low_risk

    def test_spread_capped_at_max(self):
        params = AdaptiveSpreadParams(min_spread_bps=50, max_spread_bps=500)
        spread = compute_risk_adjusted_spread(400.0, 1.0, params)
        assert spread <= 500.0

    def test_spread_floored_at_min(self):
        params = AdaptiveSpreadParams(min_spread_bps=100, max_spread_bps=1000)
        spread = compute_risk_adjusted_spread(50.0, 0.0, params)
        assert spread >= 100.0


class TestAdaptiveQuotes:
    """Test full adaptive quote computation."""

    def test_symmetric_when_flat(self):
        """With no inventory, bid/ask should be roughly symmetric."""
        params = AdaptiveSpreadParams()
        quotes = compute_adaptive_quotes(
            mid_price=100.0, volatility=0.02,
            inventory_ratio=0.0, params=params,
        )
        mid_spread = quotes["spread_bps"] / 10000 * 100.0
        assert abs(quotes["bid"] - (100.0 - mid_spread / 2)) < 0.2
        assert abs(quotes["ask"] - (100.0 + mid_spread / 2)) < 0.2

    def test_long_inventory_widens_ask(self):
        """When long, ask spread should be wider than bid spread."""
        params = AdaptiveSpreadParams()
        flat = compute_adaptive_quotes(100.0, 0.02, 0.0, params)
        long_pos = compute_adaptive_quotes(100.0, 0.02, 0.5, params)
        # When long, bid should be lower (wider on bid side)
        assert long_pos["bid"] < flat["bid"]

    def test_short_inventory_widens_bid(self):
        """When short, bid spread should be wider than ask spread."""
        params = AdaptiveSpreadParams()
        flat = compute_adaptive_quotes(100.0, 0.02, 0.0, params)
        short_pos = compute_adaptive_quotes(100.0, 0.02, -0.5, params)
        # When short, ask should be higher (wider on ask side)
        assert short_pos["ask"] > flat["ask"]

    def test_high_volatility_widens_spread(self):
        params = AdaptiveSpreadParams()
        low_vol = compute_adaptive_quotes(100.0, 0.01, 0.0, params)
        high_vol = compute_adaptive_quotes(100.0, 0.1, 0.0, params)
        assert high_vol["spread_bps"] > low_vol["spread_bps"]

    def test_returns_risk_metrics(self):
        quotes = compute_adaptive_quotes(100.0, 0.02, 0.15, params=AdaptiveSpreadParams())
        assert "risk_score" in quotes
        assert "var_amount" in quotes
        assert "risk_premium" in quotes
        assert "skew" in quotes

    def test_configurable_skew_factor(self):
        """Test that inventory_skew_factor is configurable."""
        low_skew_params = AdaptiveSpreadParams(inventory_skew_factor=0.1)
        high_skew_params = AdaptiveSpreadParams(inventory_skew_factor=2.0)
        
        low = compute_adaptive_quotes(100.0, 0.02, 0.5, low_skew_params)
        high = compute_adaptive_quotes(100.0, 0.02, 0.5, high_skew_params)
        
        # Higher skew factor should produce more skew
        assert abs(high["skew"]) > abs(low["skew"])
