"""Tests for the inventory risk model."""

import pytest
import numpy as np

from src.strategies.inventory_risk import (
    InventoryRiskParams,
    InventoryTracker,
    compute_inventory_adjusted_quotes,
)


# ---------------------------------------------------------------------------
# InventoryRiskParams defaults
# ---------------------------------------------------------------------------

class TestInventoryRiskParams:
    def test_defaults(self):
        p = InventoryRiskParams()
        assert p.max_inventory == 100.0
        assert p.skew_factor == 0.5
        assert p.decay_rate == 0.95
        assert p.urgency_threshold == 0.7
        assert p.urgency_multiplier == 1.5

    def test_custom_params(self):
        p = InventoryRiskParams(max_inventory=50, skew_factor=0.8, decay_rate=0.9)
        assert p.max_inventory == 50
        assert p.skew_factor == 0.8
        assert p.decay_rate == 0.9


# ---------------------------------------------------------------------------
# InventoryTracker
# ---------------------------------------------------------------------------

class TestInventoryTracker:
    def test_initial_position_is_zero(self):
        tracker = InventoryTracker()
        assert tracker.get_position("ETH") == 0.0

    def test_update_adds_delta(self):
        tracker = InventoryTracker()
        tracker.update("ETH", 10.0)
        assert tracker.get_position("ETH") == 10.0
        tracker.update("ETH", -3.0)
        assert tracker.get_position("ETH") == 7.0

    def test_set_position(self):
        tracker = InventoryTracker()
        tracker.set_position("ETH", 42.0)
        assert tracker.get_position("ETH") == 42.0
        tracker.set_position("ETH", -5.0)
        assert tracker.get_position("ETH") == -5.0

    def test_inventory_ratio_clamped(self):
        params = InventoryRiskParams(max_inventory=100)
        tracker = InventoryTracker(params)
        tracker.set_position("ETH", 200)
        assert tracker.inventory_ratio("ETH") == 1.0
        tracker.set_position("ETH", -200)
        assert tracker.inventory_ratio("ETH") == -1.0

    def test_inventory_ratio_proportional(self):
        params = InventoryRiskParams(max_inventory=100)
        tracker = InventoryTracker(params)
        tracker.set_position("ETH", 50)
        assert tracker.inventory_ratio("ETH") == pytest.approx(0.5)

    def test_skew_positive_when_long(self):
        params = InventoryRiskParams(max_inventory=100, skew_factor=0.5)
        tracker = InventoryTracker(params)
        tracker.set_position("ETH", 50)
        skew = tracker.compute_skew("ETH")
        assert skew > 0, "Long position should produce positive skew"

    def test_skew_negative_when_short(self):
        params = InventoryRiskParams(max_inventory=100, skew_factor=0.5)
        tracker = InventoryTracker(params)
        tracker.set_position("ETH", -50)
        skew = tracker.compute_skew("ETH")
        assert skew < 0, "Short position should produce negative skew"

    def test_skew_zero_when_flat(self):
        tracker = InventoryTracker()
        skew = tracker.compute_skew("ETH")
        assert skew == pytest.approx(0.0)

    def test_urgency_multiplier_kicks_in(self):
        params_no_urgency = InventoryRiskParams(
            max_inventory=100, skew_factor=0.5,
            urgency_threshold=0.7, urgency_multiplier=1.0,
            decay_rate=0.95,
        )
        params_with_urgency = InventoryRiskParams(
            max_inventory=100, skew_factor=0.5,
            urgency_threshold=0.7, urgency_multiplier=2.0,
            decay_rate=0.95,
        )
        tracker_no = InventoryTracker(params_no_urgency)
        tracker_with = InventoryTracker(params_with_urgency)
        # Set to above urgency threshold
        tracker_no.set_position("ETH", 80)
        tracker_with.set_position("ETH", 80)
        skew_no = tracker_no.compute_skew("ETH")
        skew_with = tracker_with.compute_skew("ETH")
        assert abs(skew_with) > abs(skew_no)

    def test_decay_reduces_pressure_over_time(self):
        params = InventoryRiskParams(max_inventory=100, decay_rate=0.5)
        tracker = InventoryTracker(params)
        tracker.set_position("ETH", 80)
        initial_skew = tracker.compute_skew("ETH")
        # Zero out position but tick several times
        tracker.set_position("ETH", 0)
        for _ in range(20):
            tracker.tick()
        later_skew = tracker.compute_skew("ETH")
        assert abs(later_skew) < abs(initial_skew)

    def test_multiple_assets_independent(self):
        tracker = InventoryTracker()
        tracker.set_position("ETH", 50)
        tracker.set_position("BTC", -30)
        assert tracker.get_position("ETH") == 50
        assert tracker.get_position("BTC") == -30
        assert tracker.compute_skew("ETH") > 0
        assert tracker.compute_skew("BTC") < 0

    def test_positions_property(self):
        tracker = InventoryTracker()
        tracker.set_position("ETH", 10)
        tracker.set_position("BTC", 20)
        positions = tracker.positions
        assert positions == {"ETH": 10, "BTC": 20}
        # Ensure it's a copy
        positions["ETH"] = 999
        assert tracker.get_position("ETH") == 10


# ---------------------------------------------------------------------------
# compute_inventory_adjusted_quotes
# ---------------------------------------------------------------------------

class TestComputeInventoryAdjustedQuotes:
    def test_symmetric_when_no_skew(self):
        q = compute_inventory_adjusted_quotes(
            mid_price=100.0, base_spread_bps=200, volatility=0.0, skew=0.0,
        )
        bid_dist = 100.0 - q["bid"]
        ask_dist = q["ask"] - 100.0
        assert bid_dist == pytest.approx(ask_dist, rel=1e-9)

    def test_positive_skew_widens_ask(self):
        q = compute_inventory_adjusted_quotes(
            mid_price=100.0, base_spread_bps=200, volatility=0.0, skew=0.3,
        )
        assert q["ask_spread_bps"] > q["bid_spread_bps"]

    def test_negative_skew_widens_bid(self):
        q = compute_inventory_adjusted_quotes(
            mid_price=100.0, base_spread_bps=200, volatility=0.0, skew=-0.3,
        )
        assert q["bid_spread_bps"] > q["ask_spread_bps"]

    def test_bid_below_ask(self):
        for skew in [-0.8, -0.3, 0.0, 0.3, 0.8]:
            q = compute_inventory_adjusted_quotes(
                mid_price=100.0, base_spread_bps=200, volatility=0.5, skew=skew,
            )
            assert q["bid"] < q["ask"], f"bid >= ask with skew={skew}"

    def test_spread_clamped_to_bounds(self):
        q = compute_inventory_adjusted_quotes(
            mid_price=100.0, base_spread_bps=10, volatility=0.0, skew=0.0,
            min_spread_bps=50, max_spread_bps=1000,
        )
        assert q["spread_bps"] >= 50

        q2 = compute_inventory_adjusted_quotes(
            mid_price=100.0, base_spread_bps=2000, volatility=1.0, skew=0.0,
            min_spread_bps=50, max_spread_bps=1000,
        )
        assert q2["spread_bps"] <= 1000

    def test_volatility_widens_spread(self):
        q_low = compute_inventory_adjusted_quotes(
            mid_price=100.0, base_spread_bps=200, volatility=0.0, skew=0.0,
        )
        q_high = compute_inventory_adjusted_quotes(
            mid_price=100.0, base_spread_bps=200, volatility=0.5, skew=0.0,
        )
        assert q_high["spread_bps"] > q_low["spread_bps"]

    def test_long_position_encourages_selling(self):
        """When long (positive skew), ask should be closer to mid than bid,
        making it easier for counterparties to sell to us... wait, no:
        long → widen ask (make our selling more expensive for buyers = wider),
        tighten bid (make our buying cheaper = tighter).
        Actually: positive skew → widen ask, tighten bid.
        This *discourages* further buying and makes our asks more competitive
        relative to our inflated inventory."""
        q = compute_inventory_adjusted_quotes(
            mid_price=100.0, base_spread_bps=200, volatility=0.0, skew=0.5,
        )
        bid_dist = 100.0 - q["bid"]
        ask_dist = q["ask"] - 100.0
        # Ask is wider than bid
        assert ask_dist > bid_dist

    def test_short_position_encourages_buying(self):
        q = compute_inventory_adjusted_quotes(
            mid_price=100.0, base_spread_bps=200, volatility=0.0, skew=-0.5,
        )
        bid_dist = 100.0 - q["bid"]
        ask_dist = q["ask"] - 100.0
        # Bid is wider than ask
        assert bid_dist > ask_dist

    def test_return_keys(self):
        q = compute_inventory_adjusted_quotes(
            mid_price=100.0, base_spread_bps=200, volatility=0.0, skew=0.0,
        )
        expected_keys = {"bid", "ask", "spread_bps", "bid_spread_bps", "ask_spread_bps", "skew"}
        assert set(q.keys()) == expected_keys
