"""Tests for constant spread strategy."""

import pytest
from src.strategies.constant_spread import ConstantSpreadParams, compute_quotes


class TestConstantSpreadParams:
    def test_default_spread_bps(self):
        params = ConstantSpreadParams()
        assert params.spread_bps == 200

    def test_default_order_size_pct(self):
        params = ConstantSpreadParams()
        assert params.order_size_pct == 0.1

    def test_custom_params(self):
        params = ConstantSpreadParams(spread_bps=50, order_size_pct=0.05)
        assert params.spread_bps == 50
        assert params.order_size_pct == 0.05


class TestComputeQuotes:
    def test_bid_below_ask(self):
        params = ConstantSpreadParams(spread_bps=200)
        result = compute_quotes(100.0, params)
        assert result["bid"] < result["ask"]

    def test_symmetric_around_mid(self):
        params = ConstantSpreadParams(spread_bps=200)
        result = compute_quotes(100.0, params)
        mid = 100.0
        assert (mid - result["bid"]) == pytest.approx(result["ask"] - mid)

    def test_spread_matches_params(self):
        params = ConstantSpreadParams(spread_bps=100)
        result = compute_quotes(100.0, params)
        assert result["spread_bps"] == 100

    def test_half_spread_calculation(self):
        params = ConstantSpreadParams(spread_bps=200)
        result = compute_quotes(100.0, params)
        half = 100.0 * (200 / 10000) / 2
        assert result["bid"] == pytest.approx(100.0 - half)
        assert result["ask"] == pytest.approx(100.0 + half)

    def test_zero_spread(self):
        params = ConstantSpreadParams(spread_bps=0)
        result = compute_quotes(100.0, params)
        assert result["bid"] == pytest.approx(100.0)
        assert result["ask"] == pytest.approx(100.0)

    def test_wide_spread(self):
        params = ConstantSpreadParams(spread_bps=1000)
        result = compute_quotes(100.0, params)
        assert result["bid"] == pytest.approx(95.0)
        assert result["ask"] == pytest.approx(105.0)

    def test_low_price(self):
        params = ConstantSpreadParams(spread_bps=200)
        result = compute_quotes(0.01, params)
        assert result["bid"] < result["ask"]
        assert result["bid"] > 0

    def test_high_price(self):
        params = ConstantSpreadParams(spread_bps=200)
        result = compute_quotes(100000.0, params)
        assert result["bid"] < result["ask"]
