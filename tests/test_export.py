"""Tests for CSV/Parquet backtest export (issue #3)."""

import os
import tempfile

import pytest

from src.agents.rwa_market_maker import RWAMarketMaker
from src.backtest.engine import BacktestEngine, BacktestTick


@pytest.fixture
def agent_and_engine():
    config = {
        "initial_quote": 10000,
        "base_spread_bps": 200,
        "max_order_size_pct": 0.1,
        "max_exposure": 50,
    }
    agent = RWAMarketMaker("test-agent", config)
    engine = BacktestEngine(agent, fill_probability=0.5)
    return agent, engine


@pytest.fixture
def sample_data():
    return BacktestEngine.generate_mock_data(100.0, 50, volatility=0.01)


def test_run_produces_fills(agent_and_engine, sample_data):
    """Backtest should produce fills when fill_probability > 0."""
    agent, engine = agent_and_engine
    result = engine.run(sample_data)
    assert result.total_ticks == 50
    assert result.total_fills > 0
    assert len(result.fills) == result.total_fills


def test_fill_record_has_required_columns(agent_and_engine, sample_data):
    """Each fill record should have all required columns."""
    _, engine = agent_and_engine
    result = engine.run(sample_data)
    if result.fills:
        record_keys = set(result.fills[0].keys())
        assert "timestamp" in record_keys
        assert "action" in record_keys
        assert "price" in record_keys
        assert "size" in record_keys
        assert "inventory" in record_keys
        assert "total_pnl" in record_keys


def test_export_csv(agent_and_engine, sample_data):
    """Export to CSV should create a valid file with correct rows."""
    _, engine = agent_and_engine
    result = engine.run(sample_data)

    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "test_output")
        resolved = engine.export(result, path, fmt="csv")

        assert resolved.endswith(".csv")
        assert os.path.exists(resolved)

        import pandas as pd
        df = pd.read_csv(resolved)
        assert len(df) == result.total_fills
        assert "timestamp" in df.columns
        assert "action" in df.columns
        assert "price" in df.columns
        assert "size" in df.columns
        assert "inventory" in df.columns
        assert "total_pnl" in df.columns


def test_export_parquet(agent_and_engine, sample_data):
    """Export to Parquet should create a valid file with correct rows."""
    _, engine = agent_and_engine
    result = engine.run(sample_data)

    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "test_output")
        resolved = engine.export(result, path, fmt="parquet")

        assert resolved.endswith(".parquet")
        assert os.path.exists(resolved)

        import pandas as pd
        df = pd.read_parquet(resolved)
        assert len(df) == result.total_fills
        assert "timestamp" in df.columns
        assert "action" in df.columns
        assert "price" in df.columns


def test_export_empty_fills():
    """Export with no fills should return empty string."""
    config = {"initial_quote": 10000, "base_spread_bps": 200}
    agent = RWAMarketMaker("empty-agent", config)
    engine = BacktestEngine(agent, fill_probability=0.0)

    data = BacktestEngine.generate_mock_data(100.0, 10)
    result = engine.run(data)

    assert result.total_fills == 0
    with tempfile.TemporaryDirectory() as tmpdir:
        resolved = engine.export(result, os.path.join(tmpdir, "empty"), fmt="csv")
        assert resolved == ""


def test_export_creates_parent_dirs(agent_and_engine, sample_data):
    """Export should create parent directories if they don't exist."""
    _, engine = agent_and_engine
    result = engine.run(sample_data)

    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "nested", "dir", "output")
        resolved = engine.export(result, path, fmt="csv")
        assert os.path.exists(resolved)
