from typing import Tuple

# Minimal representation of market data, typically found in a core models module.
# Defined here for self-containment if not globally available.
class MarketData:
    """Minimal representation of market data for strategy input."""
    def __init__(self, mid_price: float, bid_price: float, ask_price: float):
        self.mid_price = mid_price
        self.bid_price = bid_price
        self.ask_price = ask_price

class Strategy:
    """Base class for trading strategies."""
    def __init__(self, instrument_id: str):
        self.instrument_id = instrument_id

    def get_quotes(self, market_data: MarketData) -> Tuple[float, float]:
        """
        Calculates and returns bid and ask prices based on market data.
        Returns (bid_price, ask_price).
        """
        raise NotImplementedError("Subclasses must implement get_quotes method.")

    def on_order_executed(self, quantity: float, price: float):
        """
        Called when an order for this strategy's instrument is executed.
        quantity: positive for buy, negative for sell.
        price: execution price.
        """
        pass

class AdaptiveSpreadStrategy(Strategy):
    """
    An adaptive spread strategy that adjusts quotes based on market conditions
    and an inventory risk model.

    The strategy applies an inventory penalty by shifting its quoting range.
    When long (inventory > 0), quotes shift upwards (widen ask, tighten bid).
    When short (inventory < 0), quotes shift downwards (widen bid, tighten ask).
    The magnitude of this shift is determined by the inventory_skew_factor.
    """
    def __init__(self,
                 instrument_id: str,
                 base_spread: float,
                 inventory_skew_factor: float = 0.0,
                 initial_inventory: float = 0.0):
        super().__init__(instrument_id)
        if not isinstance(base_spread, (int, float)) or base_spread <= 0:
            raise ValueError("Base spread must be a positive number.")
        if not isinstance(inventory_skew_factor, (int, float)) or inventory_skew_factor < 0:
            raise ValueError("Inventory skew factor cannot be negative.")
        if not isinstance(initial_inventory, (int, float)):
            raise ValueError("Initial inventory must be a number.")

        self.base_spread = base_spread
        self.inventory_skew_factor = inventory_skew_factor
        self.inventory = initial_inventory

    def get_quotes(self, market_data: MarketData) -> Tuple[float, float]:
        """
        Calculates bid and ask prices, applying an inventory penalty if
        inventory_skew_factor is configured.
        """
        mid_price = market_data.mid_price

        # Calculate base bid and ask based on mid-price and base spread
        half_spread = self.base_spread / 2.0
        base_bid = mid_price - half_spread
        base_ask = mid_price + half_spread

        adjusted_bid = base_bid
        adjusted_ask = base_ask

        # Apply inventory risk adjustment based on skew factor
        if self.inventory_skew_factor > 0:
            # Calculate the inventory-based offset
            # Positive inventory_skew_factor ensures the direction of shift depends on inventory sign.
            # Example: inventory=10, skew_factor=0.1 -> offset=1.0 (shift up)
            #          inventory=-10, skew_factor=0.1 -> offset=-1.0 (shift down)
            inventory_offset = self.inventory * self.inventory_skew_factor

            # Shift both bid and ask prices by the calculated offset
            # This implements:
            # - When long (inventory > 0), quotes shift up (widen ask, tighten bid relative to original mid).
            # - When short (inventory < 0), quotes shift down (widen bid, tighten ask relative to original mid).
            adjusted_bid = base_bid + inventory_offset
            adjusted_ask = base_ask + inventory_offset
        
        # Note: With the current symmetric shift, the spread (adjusted_ask - adjusted_bid)
        # remains equal to base_spread. The `base_spread > 0` check in __init__ prevents inversion.
        # Additional checks for minimum spread or max offset could be added for robustness
        # if the strategy logic were more complex (e.g., asymmetric adjustments).

        return adjusted_bid, adjusted_ask

    def on_order_executed(self, quantity: float, price: float):
        """
        Updates the strategy's current inventory based on an executed order.
        Positive quantity for buy, negative for sell.
        """
        self.inventory += quantity
