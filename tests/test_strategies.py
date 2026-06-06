"""Tests for quoting strategies, unit and property-based."""

from hypothesis import given, strategies as st, settings
from src.strategies.constant_spread import ConstantSpreadParams, compute_quotes
from src.strategies.adaptive_spread import AdaptiveSpreadParams, compute_adaptive_quotes


def test_constant_spread_unit():
    params = ConstantSpreadParams(spread_bps=200)
    quotes = compute_quotes(mid_price=1000.0, params=params)
    
    assert quotes["spread_bps"] == 200
    assert quotes["bid"] == 990.0
    assert quotes["ask"] == 1010.0


def test_adaptive_spread_unit():
    params = AdaptiveSpreadParams(base_spread_bps=150, min_spread_bps=50, max_spread_bps=1000)
    quotes = compute_adaptive_quotes(
        mid_price=1000.0,
        volatility=0.01,
        inventory_ratio=0.0,
        params=params
    )
    
    assert quotes["bid"] < 1000.0
    assert quotes["ask"] > 1000.0
    # At zero skew, spreads should be symmetric
    assert abs(quotes["bid_spread_bps"] - quotes["ask_spread_bps"]) < 1e-6
    assert type(quotes["spread_bps"]) is float


@settings(deadline=None)
@given(
    mid_price=st.floats(min_value=1.0, max_value=1e6, allow_nan=False, allow_infinity=False),
    volatility=st.floats(min_value=0.0, max_value=10.0, allow_nan=False, allow_infinity=False),
    inventory_ratio=st.floats(min_value=-1.0, max_value=1.0, allow_nan=False, allow_infinity=False),
)
def test_adaptive_spread_properties(mid_price, volatility, inventory_ratio):
    params = AdaptiveSpreadParams()
    quotes = compute_adaptive_quotes(
        mid_price=mid_price,
        volatility=volatility,
        inventory_ratio=inventory_ratio,
        params=params
    )
    
    # 1. Check types: must be standard python float, not numpy scalar
    # (Important for JSON serialization to on-chain format)
    for k, v in quotes.items():
        assert type(v) is float, f"Value for {k} is {type(v)}, expected float"
        
    # 2. Invariants: Bid and Ask must wrap the mid price safely
    assert quotes["bid"] <= mid_price
    assert quotes["ask"] >= mid_price
    
    # 3. Prices must remain positive
    assert quotes["bid"] > 0
    assert quotes["ask"] > 0
    
    # 4. Total spread should respect minimums
    assert quotes["bid_spread_bps"] >= params.min_spread_bps / 2
    assert quotes["ask_spread_bps"] >= params.min_spread_bps / 2
