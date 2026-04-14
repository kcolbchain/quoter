"""Adaptive spread strategy — adjusts to volatility and inventory risk."""

from dataclasses import dataclass
import numpy as np


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
    """Compute adaptive bid/ask quotes with inventory risk penalty.

    The inventory skew incentivizes reducing directional exposure:
      - Long inventory (inv > 0): widen ask (discount to sell), tighten bid (avoid buying more)
      - Short inventory (inv < 0): widen bid (premium to buy), tighten ask (avoid selling more)

    Args:
        mid_price: Current mid price
        volatility: Normalized volatility (0-1)
        inventory_ratio: -1 (all quote/short) to +1 (all base/long)
        params: Strategy parameters
    """
    # Base spread + volatility component
    spread_bps = params.base_spread_bps + volatility * params.vol_multiplier * 100
    spread_bps = float(np.clip(spread_bps, params.min_spread_bps, params.max_spread_bps))

    # Inventory risk penalty: skew prices to reduce directional exposure
    # When long (inv > 0): make selling more attractive (wider ask) and buying less attractive (tighter bid)
    # When short (inv < 0): make buying more attractive (wider bid) and selling less attractive (tighter ask)
    skew = inventory_ratio * params.inventory_skew_factor
    bid_spread_bps = spread_bps * (1 - skew) / 2  # tighter bid when long, wider when short
    ask_spread_bps = spread_bps * (1 + skew) / 2  # wider ask when long, tighter when short

    return {
        "bid": mid_price * (1 - bid_spread_bps / 10000),
        "ask": mid_price * (1 + ask_spread_bps / 10000),
        "spread_bps": float(spread_bps),
        "bid_spread_bps": float(bid_spread_bps),
        "ask_spread_bps": float(ask_spread_bps),
        "skew": float(skew),
    }