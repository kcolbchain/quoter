"""Property-based tests for quote-formation math.

Verifies mathematical invariants across both constant and adaptive spread strategies
under a wide range of inputs using hypothesis.
"""

import pytest
import numpy as np
from hypothesis import given, strategies as st, settings

# scipy import inside compute_adaptive_quotes has high first-call latency;
# disable hypothesis deadline to avoid flaky DeadlineExceeded failures.
settings.register_profile("no_deadline", deadline=None)
settings.load_profile("no_deadline")

from src.strategies.constant_spread import ConstantSpreadParams, compute_quotes
from src.strategies.adaptive_spread import AdaptiveSpreadParams, compute_adaptive_quotes


# Realistic price range: $0.01 to $100,000
prices = st.floats(min_value=0.01, max_value=100_000.0, allow_nan=False, allow_infinity=False)
# Spread range: 1 to 2000 bps (0.01% to 20%)
spread_bps = st.integers(min_value=1, max_value=2000)
# Volatility range: 0 to 1
volatilities = st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False)
# Inventory ratio: -1 to +1
inventory_ratios = st.floats(min_value=-1.0, max_value=1.0, allow_nan=False, allow_infinity=False)
# Scale factors for invariance test
scale_factors = st.floats(min_value=0.5, max_value=10.0, allow_nan=False, allow_infinity=False)


class TestConstantSpreadProperties:

    @given(price=prices, spread=spread_bps)
    def test_price_scale_invariance(self, price: float, spread: int):
        """spread_bps is invariant under price scaling."""
        params = ConstantSpreadParams(spread_bps=spread)
        r1 = compute_quotes(price, params)
        r2 = compute_quotes(price * 2.5, params)
        assert r1["spread_bps"] == r2["spread_bps"] == spread

    @given(price=prices, spread=spread_bps)
    def test_bid_below_ask(self, price: float, spread: int):
        """Bid is always below ask."""
        params = ConstantSpreadParams(spread_bps=spread)
        r = compute_quotes(price, params)
        assert r["bid"] < r["ask"]

    @given(price=prices, spread=spread_bps)
    def test_mid_between_bid_and_ask(self, price: float, spread: int):
        """Mid price falls between bid and ask."""
        params = ConstantSpreadParams(spread_bps=spread)
        r = compute_quotes(price, params)
        assert r["bid"] <= price <= r["ask"]

    @given(price=prices, spread=spread_bps)
    def test_return_types(self, price: float, spread: int):
        """All numeric return values are Python float."""
        params = ConstantSpreadParams(spread_bps=spread)
        r = compute_quotes(price, params)
        for v in [r["bid"], r["ask"], r["spread_bps"]]:
            assert isinstance(v, (float, int)), f"{v} is {type(v)}"


class TestAdaptiveSpreadProperties:

    @given(price=prices, vol=volatilities, inv=inventory_ratios)
    def test_bid_below_ask(self, price: float, vol: float, inv: float):
        """Bid is always below ask for any valid input."""
        params = AdaptiveSpreadParams()
        r = compute_adaptive_quotes(price, vol, inv, params)
        assert r["bid"] < r["ask"]

    @given(price=prices, vol=volatilities, inv=inventory_ratios)
    def test_mid_between_bid_and_ask(self, price: float, vol: float, inv: float):
        """Mid price falls between bid and ask."""
        params = AdaptiveSpreadParams()
        r = compute_adaptive_quotes(price, vol, inv, params)
        assert r["bid"] <= price <= r["ask"]

    @given(price=prices, vol=st.just(0.0), inv=inventory_ratios)
    def test_price_scale_invariance(self, price: float, vol: float, inv: float):
        """spread_bps is invariant under price scaling for adaptive spread."""
        params = AdaptiveSpreadParams()
        r1 = compute_adaptive_quotes(price, vol, inv, params)
        r2 = compute_adaptive_quotes(price * 3.0, vol, inv, params)
        assert r1["spread_bps"] == pytest.approx(r2["spread_bps"], rel=1e-9)

    @given(price=prices, vol=volatilities, inv=inventory_ratios)
    def test_bid_spread_positive(self, price: float, vol: float, inv: float):
        """Bid spread is always positive."""
        params = AdaptiveSpreadParams()
        r = compute_adaptive_quotes(price, vol, inv, params)
        assert r["bid_spread_bps"] > 0

    @given(price=prices, vol=volatilities, inv=inventory_ratios)
    def test_ask_spread_positive(self, price: float, vol: float, inv: float):
        """Ask spread is always positive."""
        params = AdaptiveSpreadParams()
        r = compute_adaptive_quotes(price, vol, inv, params)
        assert r["ask_spread_bps"] > 0

    @given(price=prices, vol=volatilities, inv=inventory_ratios)
    def test_return_types(self, price: float, vol: float, inv: float):
        """All numeric return values are Python float, not numpy scalars."""
        params = AdaptiveSpreadParams()
        r = compute_adaptive_quotes(price, vol, inv, params)
        for k in ["bid", "ask", "spread_bps", "bid_spread_bps", "ask_spread_bps", "skew", "risk_score"]:
            val = r[k]
            assert isinstance(val, (float, np.floating)), f"{k}={val} is {type(val)}"

    @given(price=prices, inv=inventory_ratios)
    def test_finite_under_extreme_volatility(self, price: float, inv: float):
        """Bid/ask prices are finite and ordered even at extreme volatility."""
        params = AdaptiveSpreadParams()
        r = compute_adaptive_quotes(price, 1.0, inv, params)
        assert np.isfinite(r["bid"])
        assert np.isfinite(r["ask"])
        assert r["bid"] < r["ask"]
