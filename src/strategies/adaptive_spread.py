"""Adaptive spread strategy -- adjusts to volatility and inventory.

Provides two complementary APIs:

Legacy functional API (preserved for backward-compatibility):
    - AdaptiveSpreadParams  -- configuration dataclass
    - compute_adaptive_quotes  -- stateless quote computation

Class-based API with inventory risk model (new):
    - MarketData  -- market snapshot input type
    - Strategy  -- abstract base class
    - AdaptiveSpreadStrategy  -- stateful strategy with inventory tracking
"""

from dataclasses import dataclass
from typing import Tuple
import numpy as np


# ── Legacy functional API (preserved) ────────────────────────────────────────

@dataclass
class AdaptiveSpreadParams:
    base_spread_bps: float = 150
    vol_multiplier: float = 5.0
    inventory_skew_factor: float = 0.5
    min_spread_bps: float = 50
    max_spread_bps: float = 1000


def compute_adaptive_quotes(
    mid_price: float,
    volatility: float,
    inventory_ratio: float,
    params: AdaptiveSpreadParams,
) -> dict:
    """Compute adaptive bid/ask quotes.

    Args:
        mid_price: Current mid price
        volatility: Normalized volatility (0-1)
        inventory_ratio: -1 (all quote) to +1 (all base)
        params: Strategy parameters

    Returns:
        dict with keys: bid, ask, spread_bps, skew
    """
    spread_bps = params.base_spread_bps + volatility * params.vol_multiplier * 100
    spread_bps = np.clip(spread_bps, params.min_spread_bps, params.max_spread_bps)

    skew = inventory_ratio * params.inventory_skew_factor
    bid_spread = spread_bps * (1 + skew) / 10000
    ask_spread = spread_bps * (1 - skew) / 10000

    return {
        "bid": mid_price * (1 - bid_spread / 2),
        "ask": mid_price * (1 + ask_spread / 2),
        "spread_bps": float(spread_bps),
        "skew": skew,
    }


# ── Class-based API with inventory risk model (new) ───────────────────────────

class MarketData:
    """Minimal market snapshot used as strategy input."""

    def __init__(self, mid_price: float, bid_price: float, ask_price: float):
        self.mid_price = mid_price
        self.bid_price = bid_price
        self.ask_price = ask_price

    @property
    def spread(self) -> float:
        return self.ask_price - self.bid_price

    @property
    def volatility_proxy(self) -> float:
        """Spread-to-mid ratio as a simple volatility proxy (0-1)."""
        if self.mid_price == 0:
            return 0.0
        return min(self.spread / self.mid_price, 1.0)


class Strategy:
    """Abstract base for trading strategies."""

    def __init__(self, instrument_id: str):
        self.instrument_id = instrument_id

    def get_quotes(self, market_data: MarketData) -> Tuple[float, float]:
        """Return (bid_price, ask_price) for the given market snapshot."""
        raise NotImplementedError

    def on_order_executed(self, quantity: float, price: float) -> None:
        """Called when an order is executed; quantity > 0 is a buy."""
        pass


class AdaptiveSpreadStrategy(Strategy):
    """Stateful adaptive-spread strategy with inventory risk model.

    Adjusts quotes based on current inventory level:
    - Long inventory  -> shift quotes up  (wider ask, tighter bid) to encourage selling
    - Short inventory -> shift quotes down (wider bid, tighter ask) to encourage buying

    The inventory penalty magnitude is controlled by ``inventory_skew_factor``.
    """

    def __init__(
        self,
        instrument_id: str,
        base_spread: float,
        inventory_skew_factor: float = 0.0,
        initial_inventory: float = 0.0,
    ):
        super().__init__(instrument_id)
        if not isinstance(base_spread, (int, float)) or base_spread <= 0:
            raise ValueError("base_spread must be a positive number")
        if not isinstance(inventory_skew_factor, (int, float)) or inventory_skew_factor < 0:
            raise ValueError("inventory_skew_factor must be non-negative")

        self.base_spread = base_spread
        self.inventory_skew_factor = inventory_skew_factor
        self.inventory = initial_inventory

    def get_quotes(self, market_data: MarketData) -> Tuple[float, float]:
        """Compute inventory-adjusted bid/ask quotes.

        Returns:
            Tuple of (bid_price, ask_price).
        """
        half_spread = self.base_spread / 2.0
        inventory_penalty = self.inventory * self.inventory_skew_factor

        bid = market_data.mid_price - half_spread - inventory_penalty
        ask = market_data.mid_price + half_spread - inventory_penalty
        return bid, ask

    def on_order_executed(self, quantity: float, price: float) -> None:
        """Update inventory on execution.  Positive quantity = buy."""
        self.inventory += quantity

    @property
    def inventory_ratio(self) -> float:
        """Normalised inventory proxy (unbounded; informational only)."""
        return self.inventory
