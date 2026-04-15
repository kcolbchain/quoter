"""Tests for Binance WebSocket connector and recorded feed."""

import asyncio
import json
import time
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.connectors.binance_ws import (
    BinanceTradeMessage,
    BinanceWebSocketConnector,
    create_binance_connector,
)
from src.connectors.recorded_feed import (
    RecordedFeedConnector,
    create_sample_trace,
)
from src.oracle.price_feed import PricePoint


# --- BinanceTradeMessage tests ---


class TestBinanceTradeMessage:
    def test_from_json(self):
        data = {
            "e": "trade",
            "E": 1672515782136,
            "s": "BTCUSDT",
            "t": 12345,
            "p": "50000.00",
            "q": "0.001",
            "T": 1672515782136,
            "m": True,
        }
        msg = BinanceTradeMessage.from_json(data)
        assert msg.event_type == "trade"
        assert msg.symbol == "BTCUSDT"
        assert msg.price == 50000.0
        assert msg.quantity == 0.001
        assert msg.is_buyer_maker is True
        assert msg.trade_id == 12345

    def test_from_json_missing_fields(self):
        data = {"e": "trade"}
        msg = BinanceTradeMessage.from_json(data)
        assert msg.price == 0.0
        assert msg.quantity == 0.0
        assert msg.symbol == ""

    def test_raw_preserved(self):
        data = {"e": "trade", "custom": "field"}
        msg = BinanceTradeMessage.from_json(data)
        assert msg.raw["custom"] == "field"


# --- BinanceWebSocketConnector tests ---


class TestBinanceWebSocketConnector:
    def test_stream_url(self):
        conn = BinanceWebSocketConnector(symbol="BTCUSDT")
        assert conn.stream_url == "wss://stream.binance.com:9443/ws/btcusdt@trade"

    def test_stream_url_lowercase(self):
        conn = BinanceWebSocketConnector(symbol="btcusdt")
        assert "btcusdt@trade" in conn.stream_url

    def test_stream_url_custom(self):
        conn = BinanceWebSocketConnector(
            symbol="ETHUSDT",
            base_url="wss://test.example.com/ws",
        )
        assert conn.stream_url == "wss://test.example.com/ws/ethusdt@trade"

    def test_initial_state(self):
        conn = BinanceWebSocketConnector(symbol="BTCUSDT")
        assert not conn.is_connected
        assert conn.stats["trades_received"] == 0
        assert conn.stats["ticks_emitted"] == 0

    def test_handle_valid_message(self):
        conn = BinanceWebSocketConnector(symbol="BTCUSDT", aggregate_interval=0)
        data = {
            "e": "trade",
            "E": 1672515782136,
            "s": "BTCUSDT",
            "t": 12345,
            "p": "50000.00",
            "q": "0.001",
            "T": 1672515782136,
            "m": False,
        }
        asyncio.get_event_loop().run_until_complete(
            conn._handle_message(json.dumps(data))
        )
        assert conn.stats["trades_received"] == 1
        assert conn.stats["ticks_emitted"] == 1

    def test_handle_invalid_json(self):
        conn = BinanceWebSocketConnector(symbol="BTCUSDT")
        asyncio.get_event_loop().run_until_complete(
            conn._handle_message("not json")
        )
        assert conn.stats["trades_received"] == 0

    def test_handle_non_trade_event(self):
        conn = BinanceWebSocketConnector(symbol="BTCUSDT")
        # Non-trade events still get parsed but trade fields are zero
        data = {"e": "depthUpdate", "E": 123, "s": "BTCUSDT"}
        asyncio.get_event_loop().run_until_complete(
            conn._handle_message(json.dumps(data))
        )
        assert conn.stats["trades_received"] == 1
        # Price is 0 but it still emits a tick
        assert conn.stats["ticks_emitted"] == 1

    def test_aggregation_skips_rapid_ticks(self):
        conn = BinanceWebSocketConnector(
            symbol="BTCUSDT", aggregate_interval=1.0  # 1 second
        )
        data = {
            "e": "trade", "E": 1672515782136, "s": "BTCUSDT",
            "t": 1, "p": "50000.00", "q": "0.001", "T": 1672515782136, "m": False,
        }
        loop = asyncio.get_event_loop()
        # First tick should go through
        loop.run_until_complete(conn._handle_message(json.dumps(data)))
        assert conn.stats["ticks_emitted"] == 1
        # Second tick immediately — should be skipped
        data["t"] = 2
        loop.run_until_complete(conn._handle_message(json.dumps(data)))
        assert conn.stats["ticks_emitted"] == 1
        assert conn.stats["trades_received"] == 2

    def test_get_latest_tick(self):
        conn = BinanceWebSocketConnector(symbol="BTCUSDT")
        assert conn.get_latest_tick() is None

        data = {
            "e": "trade", "E": 1672515782136, "s": "BTCUSDT",
            "t": 1, "p": "50000.00", "q": "0.001", "T": 1672515782136, "m": False,
        }
        asyncio.get_event_loop().run_until_complete(
            conn._handle_message(json.dumps(data))
        )
        tick = conn.get_latest_tick()
        assert tick is not None
        assert tick.price == 50000.0
        assert tick.asset == "BTCUSDT"
        assert tick.source == "binance_ws"

    def test_to_market_data(self):
        conn = BinanceWebSocketConnector(symbol="BTCUSDT")
        tick = PricePoint(
            asset="BTCUSDT",
            price=50000.0,
            currency="USD",
            source="binance_ws",
            timestamp=datetime(2023, 1, 1, 12, 0, 0),
            confidence=1.0,
        )
        md = conn.to_market_data(tick)
        assert md["price"] == 50000.0
        assert md["source"] == "binance_ws"
        assert md["symbol"] == "BTCUSDT"
        assert "timestamp" in md

    def test_stats_tracking(self):
        conn = BinanceWebSocketConnector(symbol="BTCUSDT")
        data = {
            "e": "trade", "E": 1672515782136, "s": "BTCUSDT",
            "t": 1, "p": "50000.00", "q": "0.001", "T": 1672515782136, "m": False,
        }
        loop = asyncio.get_event_loop()
        for i in range(5):
            data["t"] = i
            loop.run_until_complete(conn._handle_message(json.dumps(data)))
        stats = conn.stats
        assert stats["trades_received"] == 5
        assert stats["ticks_emitted"] == 5


# --- RecordedFeedConnector tests ---


class TestRecordedFeedConnector:
    def test_from_json_string(self):
        trace = json.dumps([
            {
                "e": "trade", "E": 1672515782136, "s": "BTCUSDT",
                "t": 1, "p": "50000.00", "q": "0.001", "T": 1672515782136, "m": False,
            },
            {
                "e": "trade", "E": 1672515782186, "s": "BTCUSDT",
                "t": 2, "p": "50001.00", "q": "0.002", "T": 1672515782186, "m": True,
            },
        ])
        connector = RecordedFeedConnector.from_json_string(trace)
        assert connector.message_count == 2

    def test_from_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            RecordedFeedConnector.from_file("/nonexistent/trace.json")

    def test_from_file_invalid_format(self):
        import tempfile
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            f.write('"not an array"')
            f.flush()
            with pytest.raises(ValueError, match="JSON array"):
                RecordedFeedConnector.from_file(f.name)

    def test_replay_ticks(self):
        trace = json.dumps([
            {
                "e": "trade", "E": 1672515782136, "s": "BTCUSDT",
                "t": 1, "p": "50000.00", "q": "0.001", "T": 1672515782136, "m": False,
            },
            {
                "e": "trade", "E": 1672515782236, "s": "BTCUSDT",
                "t": 2, "p": "50100.00", "q": "0.002", "T": 1672515782236, "m": True,
            },
        ])
        connector = RecordedFeedConnector.from_json_string(trace, speed=1000)

        ticks = []
        async def collect():
            async for tick in connector.ticks():
                ticks.append(tick)

        asyncio.get_event_loop().run_until_complete(collect())

        assert len(ticks) == 2
        assert ticks[0].price == 50000.0
        assert ticks[1].price == 50100.0
        assert ticks[0].source == "recorded"

    def test_symbol_override(self):
        trace = json.dumps([
            {
                "e": "trade", "E": 1672515782136, "s": "BTCUSDT",
                "t": 1, "p": "50000.00", "q": "0.001", "T": 1672515782136, "m": False,
            },
        ])
        connector = RecordedFeedConnector.from_json_string(
            trace, symbol_override="ETHUSDT"
        )

        ticks = []
        async def collect():
            async for tick in connector.ticks():
                ticks.append(tick)

        asyncio.get_event_loop().run_until_complete(collect())
        assert ticks[0].asset == "ETHUSDT"

    def test_stop_replay(self):
        trace = json.dumps([
            {
                "e": "trade", "E": 1000 + i * 100, "s": "BTCUSDT",
                "t": i, "p": "50000.00", "q": "0.001", "T": 1000 + i * 100, "m": False,
            }
            for i in range(100)
        ])
        connector = RecordedFeedConnector.from_json_string(trace, speed=1000)

        ticks = []
        async def collect():
            async for tick in connector.ticks():
                ticks.append(tick)
                if len(ticks) >= 3:
                    connector.stop()

        asyncio.get_event_loop().run_until_complete(collect())
        assert len(ticks) <= 5  # May get a couple more before stop takes effect

    def test_message_count(self):
        trace = json.dumps([
            {
                "e": "trade", "E": 1000 + i * 100, "s": "BTCUSDT",
                "t": i, "p": "50000.00", "q": "0.001", "T": 1000 + i * 100, "m": False,
            }
            for i in range(10)
        ])
        connector = RecordedFeedConnector.from_json_string(trace)
        assert connector.message_count == 10


# --- create_sample_trace tests ---


class TestCreateSampleTrace:
    def test_creates_50_messages(self):
        trace = create_sample_trace()
        assert len(trace) == 50

    def test_messages_have_required_fields(self):
        trace = create_sample_trace()
        for msg in trace:
            assert "e" in msg
            assert "s" in msg
            assert "p" in msg
            assert "q" in msg
            assert msg["s"] == "BTCUSDT"

    def test_prices_vary(self):
        trace = create_sample_trace()
        prices = [float(msg["p"]) for msg in trace]
        assert min(prices) != max(prices)  # Should have variation

    def test_timing_increases(self):
        trace = create_sample_trace()
        times = [msg["E"] for msg in trace]
        for i in range(1, len(times)):
            assert times[i] > times[i - 1]


# --- create_binance_connector factory tests ---


class TestCreateBinanceConnector:
    def test_creates_connector(self):
        conn = create_binance_connector("BTCUSDT")
        assert isinstance(conn, BinanceWebSocketConnector)
        assert conn.symbol == "btcusdt"

    def test_default_aggregate(self):
        conn = create_binance_connector("BTCUSDT")
        assert conn.aggregate_interval == 0

    def test_custom_aggregate(self):
        conn = create_binance_connector("BTCUSDT", aggregate_ms=100)
        assert conn.aggregate_interval == 0.1
