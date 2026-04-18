"""Adaptive spread strategy — adjusts to volatility and inventory risk."""

from dataclasses import dataclass
import numpy as np
from typing import Tuple


@dataclass
class AdaptiveSpreadParams:
    base_spread_bps: float = 150
    vol_multiplier: float = 5.0
    inventory_skew_factor: float = 0.5
    min_spread_bps: float = 50
    max_spread_bps: float = 1000
    
    # Inventory risk management parameters
    max_inventory_ratio: float = 0.3  # Max position as fraction of total value
    risk_adjustment_factor: float = 2.0  # Multiplier for risk-based spread widening
    var_confidence: float = 0.95  # Confidence level for VaR calculation
    correlation_factor: float = 0.1  # Correlation between price and inventory risk


def compute_inventory_risk_score(
    inventory_ratio: float,
    volatility: float,
    params: AdaptiveSpreadParams,
) -> float:
    """Compute inventory risk score based on position size and volatility.
    
    Returns: Risk score from 0 to 1, where 1 is maximum risk
    """
    # Base risk from position size
    position_risk = abs(inventory_ratio) / params.max_inventory_ratio
    position_risk = min(position_risk, 1.0)
    
    # Volatility amplifies risk
    vol_multiplier = 1 + (volatility * params.risk_adjustment_factor)
    
    # Combined risk score
    risk_score = position_risk * vol_multiplier
    return min(risk_score, 1.0)


def compute_value_at_risk(
    inventory_ratio: float,
    volatility: float,
    mid_price: float,
    params: AdaptiveSpreadParams,
) -> float:
    """Calculate Value at Risk for current inventory position.
    
    Returns: VaR amount (absolute value)
    """
    if abs(inventory_ratio) < 1e-6:
        return 0.0
    
    # Simplified VaR calculation using normal distribution
    # VaR = position_value * z_score * volatility
    from scipy.stats import norm
    z_score = norm.ppf(params.var_confidence)
    position_value = abs(inventory_ratio) * mid_price
    var = position_value * z_score * volatility
    return var


def compute_risk_adjusted_spread(
    base_spread_bps: float,
    risk_score: float,
    params: AdaptiveSpreadParams,
) -> float:
    """Adjust spread based on inventory risk."""
    risk_premium = risk_score * 200  # Add up to 200 bps for high risk
    adjusted_spread = base_spread_bps + risk_premium
    return np.clip(adjusted_spread, params.min_spread_bps, params.max_spread_bps)


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
    # Compute inventory risk components
    risk_score = compute_inventory_risk_score(inventory_ratio, volatility, params)
    var_amount = compute_value_at_risk(inventory_ratio, volatility, mid_price, params)
    
    # Base spread + volatility component
    base_spread = params.base_spread_bps + volatility * params.vol_multiplier * 100

    # Risk-adjusted spread (widens total spread when risk is high)
    spread_bps = compute_risk_adjusted_spread(base_spread, risk_score, params)
    risk_premium = spread_bps - base_spread

    # Inventory skew with risk consideration
    inventory_skew = inventory_ratio * params.inventory_skew_factor
    # Risk skew amplifies existing inventory skew (no effect if skew_factor is 0)
    risk_skew = risk_score * params.correlation_factor * inventory_ratio / max(abs(inventory_ratio), 1e-9)
    risk_skew = risk_skew * params.inventory_skew_factor / max(params.inventory_skew_factor, 1e-9) if params.inventory_skew_factor > 0 else 0
    total_skew = inventory_skew + risk_skew

    # Distribute base spread with skew
    # When long (inv > 0): tighter bid, wider ask
    # When short (inv < 0): wider bid, tighter ask
    bid_spread_bps = base_spread * (1 - total_skew) / 2
    ask_spread_bps = base_spread * (1 + total_skew) / 2

    # Add risk premium to the exit side only when inventory skew is active
    if params.inventory_skew_factor > 0:
        if inventory_ratio >= 0:
            ask_spread_bps += risk_premium
        else:
            bid_spread_bps += risk_premium

    # Clamp to prevent negative spreads
    bid_spread_bps = max(bid_spread_bps, params.min_spread_bps / 2)
    ask_spread_bps = max(ask_spread_bps, params.min_spread_bps / 2)
    return {
        "bid": mid_price * (1 - bid_spread_bps / 10000),
        "ask": mid_price * (1 + ask_spread_bps / 10000),
        "spread_bps": float(spread_bps),
        "bid_spread_bps": float(bid_spread_bps),
        "ask_spread_bps": float(ask_spread_bps),
        "skew": float(total_skew),
        "risk_score": float(risk_score),
        "var_amount": float(var_amount),
    }
