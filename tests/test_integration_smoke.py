"""Integration smoke test — quote-to-fill against stub oracle."""

import pytest
from src.agents.rwa_market_maker import RWAMarketMaker
from src.agents.base_agent import Side
from src.oracle.price_feed import MockPriceFeed


@pytest.fixture
def agent():
    config = {
        "initial_quote": 10000,
        "initial_base": 1.0,
        "base_spread_bps": 200,
        "max_order_size_pct": 0.1,
        "max_inventory_pct": 0.3,
        "liquid_volume_threshold": 500000,
    }
    return RWAMarketMaker("smoke-test", config)


@pytest.fixture
def oracle():
    return MockPriceFeed(base_prices={"ETH/USDC": 3000.0}, volatility=0.01)


def test_oracle_feeds_price_to_agent(agent, oracle):
    """MockPriceFeed produces a price the agent can consume."""
    price_point = oracle.get_price("ETH/USDC")
    assert price_point is not None
    assert price_point.price > 0

    market_data = {
        "oracle_price": price_point.price,
        "on_chain_price": price_point.price,
        "volume_24h": 500000,
    }
    orders = agent.tick(market_data)
    assert len(orders) >= 1
    for o in orders:
        assert o.price > 0
        assert o.size > 0


def test_quote_to_fill_roundtrip(agent):
    """Agent produces orders, fill updates position and PnL."""
    market_data = {
        "oracle_price": 100.0,
        "on_chain_price": 100.1,
        "volume_24h": 500000,
    }
    orders = agent.tick(market_data)
    assert len(orders) >= 1

    before_pnl = agent.get_pnl(100.0)
    for o in orders:
        import copy
        fill = copy.copy(o)
        from src.agents.base_agent import Fill
        fill = Fill(side=o.side, price=o.price, size=o.size, fee=o.price * o.size * 0.001)
        agent.on_fill(fill)

    after_pnl = agent.get_pnl(100.0)
    assert after_pnl["realized"] >= before_pnl["realized"]
    assert after_pnl["quote_balance"] != before_pnl["quote_balance"]


def test_multi_tick_smoke(agent):
    """Agent survives multiple ticks, producing orders each time."""
    for i in range(10):
        price = 100.0 + i * 0.5
        market_data = {
            "oracle_price": price,
            "on_chain_price": price * 1.001,
            "volume_24h": 500000,
        }
        orders = agent.tick(market_data)

        for o in orders:
            assert o.price > 0
            assert o.size > 0

        pnl = agent.get_pnl(price)
        assert isinstance(pnl["total"], (float, int))
        assert isinstance(pnl["quote_balance"], (float, int))

    pnl = agent.get_pnl(price)
    assert "total" in pnl
    assert "realized" in pnl
    assert "quote_balance" in pnl
