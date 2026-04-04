"""Inventory risk model for adaptive spread adjustment.

Applies an inventory penalty to bid/ask spreads:
  - Long position  → widen ask, tighten bid (encourage selling)
  - Short position → tighten ask, widen bid (encourage buying)

Configurable via skew_factor, max_inventory, and decay_rate.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class InventoryRiskParams:
    """Parameters controlling inventory-based spread adjustment."""

    max_inventory: float = 100.0
    """Maximum allowed inventory (base units) before full penalty applies."""

    skew_factor: float = 0.5
    """Strength of inventory skew. Higher values apply more aggressive
    adjustment.  At skew_factor=1.0, a fully skewed position doubles
    the spread on the penalized side."""

    decay_rate: float = 0.95
    """Exponential decay applied to inventory pressure each tick.
    Values closer to 1.0 mean slower decay (more persistent pressure)."""

    urgency_threshold: float = 0.7
    """Inventory ratio (0-1) above which urgency bonus kicks in."""

    urgency_multiplier: float = 1.5
    """Extra skew multiplier applied when inventory exceeds urgency_threshold."""


class InventoryTracker:
    """Track per-asset inventory and compute risk-adjusted skew."""

    def __init__(self, params: Optional[InventoryRiskParams] = None):
        self.params = params or InventoryRiskParams()
        self._positions: dict[str, float] = {}
        self._pressure: dict[str, float] = {}

    @property
    def positions(self) -> dict[str, float]:
        return dict(self._positions)

    def update(self, asset: str, quantity_delta: float) -> None:
        """Update inventory for *asset* by *quantity_delta*."""
        self._positions[asset] = self._positions.get(asset, 0.0) + quantity_delta
        self._apply_decay(asset)

    def set_position(self, asset: str, quantity: float) -> None:
        """Set absolute inventory level for *asset*."""
        self._positions[asset] = quantity
        self._apply_decay(asset)

    def get_position(self, asset: str) -> float:
        return self._positions.get(asset, 0.0)

    def inventory_ratio(self, asset: str) -> float:
        """Return inventory as a fraction of max_inventory, clamped to [-1, 1].

        Positive means long, negative means short.
        """
        pos = self.get_position(asset)
        if self.params.max_inventory == 0:
            return 0.0
        return float(np.clip(pos / self.params.max_inventory, -1.0, 1.0))

    def compute_skew(self, asset: str) -> float:
        """Compute the signed skew value used to shift bid/ask spreads.

        Positive skew → widen ask, tighten bid (reduce long exposure).
        Negative skew → widen bid, tighten ask (reduce short exposure).
        """
        ratio = self.inventory_ratio(asset)
        pressure = self._pressure.get(asset, 0.0)
        effective_ratio = float(np.clip(ratio + pressure, -1.0, 1.0))

        skew = effective_ratio * self.params.skew_factor

        if abs(effective_ratio) > self.params.urgency_threshold:
            skew *= self.params.urgency_multiplier

        return skew

    def _apply_decay(self, asset: str) -> None:
        """Update pressure with exponential decay towards current inventory."""
        ratio = self.inventory_ratio(asset)
        prev = self._pressure.get(asset, 0.0)
        self._pressure[asset] = prev * self.params.decay_rate + ratio * (1 - self.params.decay_rate)

    def tick(self) -> None:
        """Advance one time step — decay all pressure values."""
        for asset in list(self._pressure):
            self._pressure[asset] *= self.params.decay_rate


def compute_inventory_adjusted_quotes(
    mid_price: float,
    base_spread_bps: float,
    volatility: float,
    skew: float,
    vol_multiplier: float = 5.0,
    min_spread_bps: float = 50.0,
    max_spread_bps: float = 1000.0,
) -> dict:
    """Compute bid/ask quotes with inventory risk adjustment.

    Args:
        mid_price: Current mid-market price.
        base_spread_bps: Base spread in basis points.
        volatility: Normalised volatility (0–1).
        skew: Inventory skew from :meth:`InventoryTracker.compute_skew`.
            Positive → widen ask, tighten bid.
        vol_multiplier: Volatility scaling factor.
        min_spread_bps: Floor for effective spread.
        max_spread_bps: Ceiling for effective spread.

    Returns:
        dict with keys: bid, ask, spread_bps, bid_spread_bps,
        ask_spread_bps, skew.
    """
    spread_bps = base_spread_bps + volatility * vol_multiplier * 100
    spread_bps = float(np.clip(spread_bps, min_spread_bps, max_spread_bps))

    # Apply skew: positive skew widens ask, tightens bid
    bid_spread_bps = spread_bps * (1 - skew)
    ask_spread_bps = spread_bps * (1 + skew)

    # Ensure non-negative spreads
    bid_spread_bps = max(bid_spread_bps, min_spread_bps * 0.1)
    ask_spread_bps = max(ask_spread_bps, min_spread_bps * 0.1)

    bid = mid_price * (1 - bid_spread_bps / 10000 / 2)
    ask = mid_price * (1 + ask_spread_bps / 10000 / 2)

    return {
        "bid": bid,
        "ask": ask,
        "spread_bps": spread_bps,
        "bid_spread_bps": bid_spread_bps,
        "ask_spread_bps": ask_spread_bps,
        "skew": skew,
    }
