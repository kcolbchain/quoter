"""Multi-venue execution coordinator.

Aggregates quotes from multiple venue connectors into a consolidated order book,
routes fills to the best-priced venue with sufficient depth, and tracks
inventory across all venues for unified position management.

Usage:
    coordinator = MultiVenueConnector([uniswap_connector, cex_connector])
    best_bid = coordinator.get_best_bid("BTC/USDT", size=1.0)
    fill = coordinator.execute_buy("BTC/USDT", size=1.0)
    total_inv = coordinator.get_aggregate_inventory("BTC")
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

logger = logging.getLogger(__name__)


class Side(Enum):
    BUY = "buy"
    SELL = "sell"


@dataclass
class Quote:
    """A single quote from a venue."""
    venue_id: str
    side: Side
    price: float
    size: float
    timestamp: float = field(default_factory=time.time)


@dataclass
class Fill:
    """An executed fill from a venue."""
    venue_id: str
    side: Side
    price: float
    size: float
    fee: float = 0.0
    timestamp: float = field(default_factory=time.time)


@dataclass
class VenueInventory:
    """Inventory tracking for a single venue."""
    venue_id: str
    base_balance: float = 0.0
    quote_balance: float = 0.0
    total_fees: float = 0.0

    @property
    def net_exposure(self) -> float:
        return self.base_balance


class VenueConnector:
    """Abstract venue connector interface.
    
    Implement this class to connect to any trading venue
    (CEX, DEX, RFQ provider, etc.).
    """
    
    def __init__(self, venue_id: str):
        self.venue_id = venue_id
    
    def get_quotes(self, symbol: str) -> list[Quote]:
        """Return current bid/ask quotes for a symbol."""
        raise NotImplementedError
    
    def execute(self, side: Side, symbol: str, size: float, price: float) -> Optional[Fill]:
        """Execute a trade on this venue."""
        raise NotImplementedError
    
    def get_fee_rate(self) -> float:
        """Return the fee rate for this venue."""
        return 0.001  # Default 0.1%


class MultiVenueCoordinator:
    """Coordinates execution across multiple trading venues.
    
    - Aggregates quotes from N venues into a consolidated order book
    - Routes fills to the venue with best price + sufficient depth
    - Tracks inventory across all venues
    - Provides unified inventory view for strategy agents
    """
    
    def __init__(self, venues: list[VenueConnector]):
        self.venues = {v.venue_id: v for v in venues}
        self.inventories: dict[str, VenueInventory] = {
            v.venue_id: VenueInventory(venue_id=v.venue_id) for v in venues
        }
        self._fill_history: list[Fill] = []
    
    @property
    def venue_ids(self) -> list[str]:
        return list(self.venues.keys())
    
    def get_consolidated_book(self, symbol: str) -> dict:
        """Aggregate quotes from all venues into a consolidated order book.
        
        Returns:
            {
                "bids": [Quote, ...],  # Sorted descending by price
                "asks": [Quote, ...],  # Sorted ascending by price
                "venue_count": int
            }
        """
        all_bids: list[Quote] = []
        all_asks: list[Quote] = []
        
        for venue_id, venue in self.venues.items():
            try:
                quotes = venue.get_quotes(symbol)
                for q in quotes:
                    if q.side == Side.BUY:
                        all_bids.append(q)
                    else:
                        all_asks.append(q)
            except Exception as e:
                logger.warning(f"Venue {venue_id} quote fetch failed: {e}")
        
        all_bids.sort(key=lambda q: q.price, reverse=True)
        all_asks.sort(key=lambda q: q.price)
        
        return {
            "bids": all_bids,
            "asks": all_asks,
            "venue_count": len(self.venues),
        }
    
    def get_best_bid(self, symbol: str, size: float = 0.0) -> Optional[Quote]:
        """Get the best bid quote with sufficient depth across all venues.
        
        Args:
            symbol: Trading pair symbol
            size: Minimum size required
            
        Returns:
            Best bid Quote or None if no sufficient liquidity
        """
        book = self.get_consolidated_book(symbol)
        for bid in book["bids"]:
            if bid.size >= size:
                return bid
        return None
    
    def get_best_ask(self, symbol: str, size: float = 0.0) -> Optional[Quote]:
        """Get the best ask quote with sufficient depth across all venues.
        
        Args:
            symbol: Trading pair symbol
            size: Minimum size required
            
        Returns:
            Best ask Quote or None if no sufficient liquidity
        """
        book = self.get_consolidated_book(symbol)
        for ask in book["asks"]:
            if ask.size >= size:
                return ask
        return None
    
    def execute_buy(self, symbol: str, size: float, max_price: Optional[float] = None) -> Optional[Fill]:
        """Execute a buy order, routing to the best-priced venue.
        
        Args:
            symbol: Trading pair symbol
            size: Amount to buy
            max_price: Maximum acceptable price (None = market)
            
        Returns:
            Fill or None if no venue has sufficient liquidity
        """
        book = self.get_consolidated_book(symbol)
        asks = book["asks"]
        if max_price is not None:
            asks = [a for a in asks if a.price <= max_price]
        
        if not asks:
            logger.warning(f"No asks available for {symbol} (max_price={max_price})")
            return None
        
        # Route to best ask
        best_ask = asks[0]
        venue = self.venues[best_ask.venue_id]
        
        fill = venue.execute(Side.BUY, symbol, size, best_ask.price)
        if fill:
            self._record_fill(fill)
            return fill
        
        logger.warning(f"Execution failed on venue {best_ask.venue_id}")
        return None
    
    def execute_sell(self, symbol: str, size: float, min_price: Optional[float] = None) -> Optional[Fill]:
        """Execute a sell order, routing to the best-priced venue.
        
        Args:
            symbol: Trading pair symbol
            size: Amount to sell
            min_price: Minimum acceptable price (None = market)
            
        Returns:
            Fill or None if no venue has sufficient liquidity
        """
        book = self.get_consolidated_book(symbol)
        bids = book["bids"]
        if min_price is not None:
            bids = [b for b in bids if b.price >= min_price]
        
        if not bids:
            logger.warning(f"No bids available for {symbol} (min_price={min_price})")
            return None
        
        # Route to best bid
        best_bid = bids[0]
        venue = self.venues[best_bid.venue_id]
        
        fill = venue.execute(Side.SELL, symbol, size, best_bid.price)
        if fill:
            self._record_fill(fill)
            return fill
        
        logger.warning(f"Execution failed on venue {best_bid.venue_id}")
        return None
    
    def _record_fill(self, fill: Fill) -> None:
        """Record a fill and update inventory."""
        self._fill_history.append(fill)
        inv = self.inventories[fill.venue_id]
        inv.total_fees += fill.fee
        
        if fill.side == Side.BUY:
            inv.base_balance += fill.size
            inv.quote_balance -= fill.price * fill.size
        else:
            inv.base_balance -= fill.size
            inv.quote_balance += fill.price * fill.size
    
    def get_aggregate_inventory(self, base_symbol: Optional[str] = None) -> dict:
        """Get unified inventory across all venues.
        
        Returns:
            {
                "total_base_balance": float,
                "total_quote_balance": float,
                "total_fees": float,
                "per_venue": {venue_id: VenueInventory, ...}
            }
        """
        total_base = sum(inv.base_balance for inv in self.inventories.values())
        total_quote = sum(inv.quote_balance for inv in self.inventories.values())
        total_fees = sum(inv.total_fees for inv in self.inventories.values())
        
        return {
            "total_base_balance": total_base,
            "total_quote_balance": total_quote,
            "total_fees": total_fees,
            "net_exposure": total_base,
            "per_venue": dict(self.inventories),
        }
    
    def get_fill_history(self, venue_id: Optional[str] = None) -> list[Fill]:
        """Get fill history, optionally filtered by venue."""
        if venue_id:
            return [f for f in self._fill_history if f.venue_id == venue_id]
        return list(self._fill_history)
    
    def add_venue(self, venue: VenueConnector) -> None:
        """Add a new venue to the coordinator."""
        self.venues[venue.venue_id] = venue
        self.inventories[venue.venue_id] = VenueInventory(venue_id=venue.venue_id)
    
    def remove_venue(self, venue_id: str) -> None:
        """Remove a venue from the coordinator."""
        self.venues.pop(venue_id, None)
        self.inventories.pop(venue_id, None)
