"""Tests for multi-venue execution coordinator."""

import pytest
from src.execution.coordinator import (
    MultiVenueCoordinator, VenueConnector, Quote, Fill, Side, VenueInventory
)


class MockVenue(VenueConnector):
    """Mock venue for testing with configurable quotes."""
    
    def __init__(self, venue_id: str, bids: list, asks: list, fee_rate: float = 0.001):
        super().__init__(venue_id)
        self._bids = bids
        self._asks = asks
        self._fee_rate = fee_rate
        self.execution_count = 0
    
    def get_quotes(self, symbol: str) -> list[Quote]:
        quotes = []
        for price, size in self._bids:
            quotes.append(Quote(self.venue_id, Side.BUY, price, size))
        for price, size in self._asks:
            quotes.append(Quote(self.venue_id, Side.SELL, price, size))
        return quotes
    
    def execute(self, side: Side, symbol: str, size: float, price: float) -> Fill:
        self.execution_count += 1
        fee = price * size * self._fee_rate
        return Fill(self.venue_id, side, price, size, fee)
    
    def get_fee_rate(self) -> float:
        return self._fee_rate


class TestVenueInventory:
    def test_initial_state(self):
        inv = VenueInventory("test")
        assert inv.base_balance == 0.0
        assert inv.quote_balance == 0.0
        assert inv.net_exposure == 0.0
    
    def test_net_exposure(self):
        inv = VenueInventory("test", base_balance=10.0, quote_balance=-5000.0)
        assert inv.net_exposure == 10.0


class TestMultiVenueCoordinator:
    def setup_method(self):
        # Venue 1: bid 100 / ask 101
        self.venue1 = MockVenue("venue1", [(100.0, 5.0)], [(101.0, 5.0)])
        # Venue 2: bid 99 / ask 102
        self.venue2 = MockVenue("venue2", [(99.0, 3.0)], [(102.0, 3.0)])
        self.coordinator = MultiVenueCoordinator([self.venue1, self.venue2])
    
    def test_consolidated_book(self):
        book = self.coordinator.get_consolidated_book("BTC/USDT")
        assert book["venue_count"] == 2
        assert len(book["bids"]) == 2
        assert len(book["asks"]) == 2
        # Bids sorted descending
        assert book["bids"][0].price == 100.0
        assert book["bids"][1].price == 99.0
        # Asks sorted ascending
        assert book["asks"][0].price == 101.0
        assert book["asks"][1].price == 102.0
    
    def test_best_bid(self):
        best = self.coordinator.get_best_bid("BTC/USDT")
        assert best is not None
        assert best.venue_id == "venue1"
        assert best.price == 100.0
    
    def test_best_ask(self):
        best = self.coordinator.get_best_ask("BTC/USDT")
        assert best is not None
        assert best.venue_id == "venue1"
        assert best.price == 101.0
    
    def test_buy_routes_to_cheapest(self):
        fill = self.coordinator.execute_buy("BTC/USDT", size=1.0)
        assert fill is not None
        assert fill.venue_id == "venue1"  # Cheapest ask
        assert fill.price == 101.0
        assert fill.size == 1.0
    
    def test_sell_routes_to_highest(self):
        fill = self.coordinator.execute_sell("BTC/USDT", size=1.0)
        assert fill is not None
        assert fill.venue_id == "venue1"  # Highest bid
        assert fill.price == 100.0
        assert fill.size == 1.0
    
    def test_size_filtering(self):
        # Request 6.0 but max available is 5.0
        best = self.coordinator.get_best_bid("BTC/USDT", size=6.0)
        assert best is None
    
    def test_inventory_tracking(self):
        self.coordinator.execute_buy("BTC/USDT", size=2.0)
        inv = self.coordinator.get_aggregate_inventory()
        assert inv["total_base_balance"] == 2.0
        assert inv["total_quote_balance"] == pytest.approx(-202.0)  # 2 * 101
    
    def test_sell_updates_inventory(self):
        self.coordinator.execute_sell("BTC/USDT", size=1.0)
        inv = self.coordinator.get_aggregate_inventory()
        assert inv["total_base_balance"] == -1.0
        assert inv["total_quote_balance"] == pytest.approx(100.0)
    
    def test_fill_history(self):
        self.coordinator.execute_buy("BTC/USDT", size=1.0)
        self.coordinator.execute_sell("BTC/USDT", size=0.5)
        history = self.coordinator.get_fill_history()
        assert len(history) == 2
    
    def test_fee_calculation(self):
        fill = self.coordinator.execute_buy("BTC/USDT", size=1.0)
        expected_fee = 101.0 * 1.0 * 0.001
        assert fill.fee == pytest.approx(expected_fee)
    
    def test_add_remove_venue(self):
        venue3 = MockVenue("venue3", [(98.0, 10.0)], [(103.0, 10.0)])
        self.coordinator.add_venue(venue3)
        assert "venue3" in self.coordinator.venue_ids
        assert self.coordinator.get_consolidated_book("BTC/USDT")["venue_count"] == 3
        
        self.coordinator.remove_venue("venue3")
        assert "venue3" not in self.coordinator.venue_ids
        assert self.coordinator.get_consolidated_book("BTC/USDT")["venue_count"] == 2
    
    def test_max_price_constraint(self):
        # max_price=100.5 should filter out venue1 ask (101) and venue2 ask (102)
        fill = self.coordinator.execute_buy("BTC/USDT", size=1.0, max_price=100.5)
        assert fill is None
    
    def test_min_price_constraint(self):
        # min_price=100.5 should filter out both bids (100, 99)
        fill = self.coordinator.execute_sell("BTC/USDT", size=1.0, min_price=100.5)
        assert fill is None
    
    def test_empty_book(self):
        empty_coordinator = MultiVenueCoordinator([])
        book = empty_coordinator.get_consolidated_book("BTC/USDT")
        assert book["bids"] == []
        assert book["asks"] == []
        assert book["venue_count"] == 0
    
    def test_venue_failure_handling(self):
        class FailingVenue(VenueConnector):
            def __init__(self):
                super().__init__("failing")
            def get_quotes(self, symbol):
                raise ConnectionError("Venue down")
            def execute(self, side, symbol, size, price):
                return None
        
        coord = MultiVenueCoordinator([FailingVenue(), self.venue1])
        book = coord.get_consolidated_book("BTC/USDT")
        assert book["venue_count"] == 2
        assert len(book["bids"]) == 1  # Only from venue1
