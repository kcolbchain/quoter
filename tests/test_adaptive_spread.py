"""Tests for adaptive spread strategy — individual computational functions."""

import pytest
import numpy as np
from src.strategies.adaptive_spread import (
    AdaptiveSpreadParams,
    compute_inventory_risk_score,
    compute_value_at_risk,
    compute_risk_adjusted_spread,
)


class TestComputeInventoryRiskScore:
    def test_zero_inventory_zero_risk(self):
        params = AdaptiveSpreadParams()
        score = compute_inventory_risk_score(0.0, 0.0, params)
        assert score == 0.0

    def test_at_max_inventory_ratio(self):
        params = AdaptiveSpreadParams(max_inventory_ratio=0.3)
        score = compute_inventory_risk_score(0.3, 0.0, params)
        assert score == pytest.approx(1.0)

    def test_clamped_at_one(self):
        params = AdaptiveSpreadParams(max_inventory_ratio=0.3)
        score = compute_inventory_risk_score(1.0, 0.0, params)
        assert score == 1.0

    def test_volatility_increases_risk(self):
        params = AdaptiveSpreadParams(max_inventory_ratio=0.5)
        low_vol = compute_inventory_risk_score(0.3, 0.0, params)
        high_vol = compute_inventory_risk_score(0.3, 0.5, params)
        assert high_vol > low_vol

    def test_negative_inventory_same_as_positive(self):
        params = AdaptiveSpreadParams(max_inventory_ratio=0.5)
        pos = compute_inventory_risk_score(0.4, 0.0, params)
        neg = compute_inventory_risk_score(-0.4, 0.0, params)
        assert pos == pytest.approx(neg)

    def test_risk_score_range(self):
        params = AdaptiveSpreadParams()
        for inv in [0.0, 0.1, 0.5, 0.9]:
            for vol in [0.0, 0.2, 0.8]:
                score = compute_inventory_risk_score(inv, vol, params)
                assert 0.0 <= score <= 1.0


class TestComputeValueAtRisk:
    def test_zero_inventory_zero_var(self):
        params = AdaptiveSpreadParams()
        var = compute_value_at_risk(0.0, 0.5, 100.0, params)
        assert var == 0.0

    def test_positive_for_positive_inventory(self):
        params = AdaptiveSpreadParams(var_confidence=0.95)
        var = compute_value_at_risk(0.5, 0.02, 100.0, params)
        assert var > 0

    def test_higher_confidence_higher_var(self):
        params_low = AdaptiveSpreadParams(var_confidence=0.90)
        params_high = AdaptiveSpreadParams(var_confidence=0.99)
        var_low = compute_value_at_risk(0.5, 0.02, 100.0, params_low)
        var_high = compute_value_at_risk(0.5, 0.02, 100.0, params_high)
        assert var_high > var_low

    def test_higher_volatility_higher_var(self):
        params = AdaptiveSpreadParams(var_confidence=0.95)
        var_low = compute_value_at_risk(0.5, 0.01, 100.0, params)
        var_high = compute_value_at_risk(0.5, 0.05, 100.0, params)
        assert var_high > var_low

    def test_higher_price_higher_var(self):
        params = AdaptiveSpreadParams(var_confidence=0.95)
        var_low = compute_value_at_risk(0.5, 0.02, 50.0, params)
        var_high = compute_value_at_risk(0.5, 0.02, 200.0, params)
        assert var_high > var_low


class TestComputeRiskAdjustedSpread:
    def test_baseline_with_zero_risk(self):
        params = AdaptiveSpreadParams()
        result = compute_risk_adjusted_spread(100, 0.0, params)
        assert result == pytest.approx(100)

    def test_increases_with_risk(self):
        params = AdaptiveSpreadParams()
        low = compute_risk_adjusted_spread(100, 0.0, params)
        high = compute_risk_adjusted_spread(100, 0.5, params)
        assert high > low

    def test_clamped_to_min(self):
        params = AdaptiveSpreadParams(min_spread_bps=50)
        result = compute_risk_adjusted_spread(10, 0.0, params)
        assert result >= 50

    def test_clamped_to_max(self):
        params = AdaptiveSpreadParams(max_spread_bps=500)
        result = compute_risk_adjusted_spread(100, 1.0, params)
        assert result <= 500

    def test_risk_premium_per_unit(self):
        params = AdaptiveSpreadParams()
        result = compute_risk_adjusted_spread(100, 0.5, params)
        expected = 100 + 0.5 * 200
        assert result == pytest.approx(expected)

    def test_return_type(self):
        params = AdaptiveSpreadParams()
        result = compute_risk_adjusted_spread(100, 0.3, params)
        assert isinstance(result, (float, np.floating))
