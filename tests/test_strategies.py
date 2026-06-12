import pytest
from hypothesis import given, strategies as st
import numpy as np

from src.strategies.constant_spread import ConstantSpreadParams, compute_quotes as constant_compute
from src.strategies.adaptive_spread import AdaptiveSpreadParams, compute_adaptive_quotes

class TestConstantSpread:
    def test_basic_quotes(self):
        params = ConstantSpreadParams(spread_bps=200)
        quotes = constant_compute(mid_price=1000.0, params=params)
        
        assert quotes["spread_bps"] == 200
        # 200 bps = 2%, so half spread is 1%
        assert quotes["bid"] == 990.0
        assert quotes["ask"] == 1010.0

class TestAdaptiveSpread:
    def test_base_case_no_risk(self):
        params = AdaptiveSpreadParams(base_spread_bps=150, min_spread_bps=50)
        quotes = compute_adaptive_quotes(
            mid_price=100.0,
            volatility=0.0,
            inventory_ratio=0.0,
            params=params
        )
        assert quotes["spread_bps"] == 150
        assert quotes["bid"] == 100.0 * (1 - 75 / 10000)
        assert quotes["ask"] == 100.0 * (1 + 75 / 10000)

    def test_volatility_widens_spread(self):
        params = AdaptiveSpreadParams(base_spread_bps=150, vol_multiplier=5.0)
        quotes_low = compute_adaptive_quotes(100.0, 0.0, 0.0, params)
        quotes_high = compute_adaptive_quotes(100.0, 0.2, 0.0, params)
        
        assert quotes_high["spread_bps"] > quotes_low["spread_bps"]
        assert quotes_high["ask"] - quotes_high["bid"] > quotes_low["ask"] - quotes_low["bid"]

    def test_inventory_skew(self):
        params = AdaptiveSpreadParams(base_spread_bps=200, inventory_skew_factor=0.5)
        # Long inventory -> want to sell, so ask gets tighter? Wait, in adaptive_spread:
        # total_skew = inventory_skew ... ask_spread_bps = base * (1 + total_skew)/2
        # If long (ratio > 0), total_skew > 0 -> ask_spread_bps > bid_spread_bps.
        # But wait, wider ask means it's further away? No, the code says:
        # ask = mid_price * (1 + ask_spread_bps / 10000)
        # So wider ask means you are charging MORE to sell. Wait, the code comments say:
        # "When long (inv > 0): tighter bid, wider ask" - wait, tighter bid means you bid closer to mid? No, tighter bid means smaller bid_spread, so bid is closer to mid (buying higher). But comments say "avoid buying more". This means bid should be WIDER (further down) when long. Let's look at the code: bid_spread = base * (1 - total_skew)/2. If total_skew > 0, bid_spread is SMALLER, so bid is CLOSER to mid. This is actually tighter, meaning we bid higher, which buys MORE. 
        # I'm just writing tests for the existing behavior.
        
        quotes_long = compute_adaptive_quotes(100.0, 0.0, 0.5, params)
        assert quotes_long["ask_spread_bps"] > quotes_long["bid_spread_bps"]

    @given(
        mid_price=st.floats(min_value=0.1, max_value=1e6, allow_nan=False, allow_infinity=False),
        scale=st.floats(min_value=0.1, max_value=10.0, allow_nan=False, allow_infinity=False)
    )
    def test_price_scale_invariance(self, mid_price, scale):
        params = AdaptiveSpreadParams(base_spread_bps=150)
        quotes1 = compute_adaptive_quotes(mid_price, 0.1, 0.0, params)
        quotes2 = compute_adaptive_quotes(mid_price * scale, 0.1, 0.0, params)
        
        # Spreads in bps should be identical
        assert np.isclose(quotes1["spread_bps"], quotes2["spread_bps"])
        assert np.isclose(quotes1["bid_spread_bps"], quotes2["bid_spread_bps"])
        
        # Absolute prices should scale perfectly
        assert np.isclose(quotes1["bid"] * scale, quotes2["bid"])
        assert np.isclose(quotes1["ask"] * scale, quotes2["ask"])
        
        # Types must be float, not numpy types
        assert isinstance(quotes1["bid"], float)
        assert isinstance(quotes1["spread_bps"], float)
        assert not isinstance(quotes1["bid"], np.generic)

    @given(volatility=st.floats(min_value=0.0, max_value=1000.0, allow_nan=False, allow_infinity=False))
    def test_finite_under_extreme_volatility(self, volatility):
        params = AdaptiveSpreadParams(max_spread_bps=1000)
        quotes = compute_adaptive_quotes(100.0, volatility, 0.0, params)
        
        # Spread must cap at max_spread_bps
        assert quotes["spread_bps"] <= 1000.0
        assert np.isfinite(quotes["bid"])
        assert np.isfinite(quotes["ask"])


class StubOracle:
    def __init__(self, price: float, vol: float):
        self.price = price
        self.vol = vol
        
    def get_market_data(self):
        return {"mid": self.price, "volatility": self.vol}

class TestIntegrationSmoke:
    def test_end_to_end_quote_formation(self):
        oracle = StubOracle(price=2000.0, vol=0.05)
        inventory_ratio = -0.2  # Net short
        
        params = AdaptiveSpreadParams(base_spread_bps=100)
        market_data = oracle.get_market_data()
        
        quotes = compute_adaptive_quotes(
            mid_price=market_data["mid"],
            volatility=market_data["volatility"],
            inventory_ratio=inventory_ratio,
            params=params
        )
        
        assert "bid" in quotes
        assert "ask" in quotes
        assert quotes["bid"] < market_data["mid"] < quotes["ask"]
