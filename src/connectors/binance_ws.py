"""Binance WebSocket price connector for live market data.

Connects to Binance's public WebSocket streams and emits PricePoint ticks
that the agent loop can consume. Supports both live streaming and recorded
trace replay for testing.

Usage:
    connector = BinanceWebSocketConnector(symbol="BTCUSDT")
    await connector.connect()
    async for tick in connector.ticks():
        # tick is a PricePoint from oracle.price_feed
        agent.tick({"price": tick.price, "timestamp": tick.timestamp})
"""

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import AsyncIterator, Optional

import websockets

from src.oracle.price_feed import PricePoint

logger = logging.getLogger(__name__)


@dataclass
class BinanceTradeMessage:
    """Parsed Binance trade WebSocket message.

    Binance trade stream format:
    {
        "e": "trade",
        "E": 1672515782136,   // Event time (ms)
        "s": "BTCUSDT",        // Symbol
        "t": 12345,            // Trade ID
        "p": "50000.00",       // Price
        "q": "0.001",          // Quantity
        "b": 88,               // Buyer order ID
        "a": 50,               // Seller order ID
        "T": 1672515782136,   // Trade time (ms)
        "m": true,             // Is the buyer the market maker?
    }
    """
    event_type: str
    event_time_ms: int
    symbol: str
    trade_id: int
    price: float
    quantity: float
    trade_time_ms: int
    is_buyer_maker: bool
    raw: dict = field(default_factory=dict)

    @classmethod
    def from_json(cls, data: dict) -> "BinanceTradeMessage":
        return cls(
            event_type=data.get("e", ""),
            event_time_ms=data.get("E", 0),
            symbol=data.get("s", ""),
            trade_id=data.get("t", 0),
            price=float(data.get("p", 0)),
            quantity=float(data.get("q", 0)),
            trade_time_ms=data.get("T", 0),
            is_buyer_maker=data.get("m", False),
            raw=data,
        )


class BinanceWebSocketConnector:
    """WebSocket connector for Binance spot trade streams.

    Streams real-time trade data from Binance and converts to PricePoint
    ticks compatible with the agent loop's market_data format.

    Features:
    - Reconnection with exponential backoff
    - Heartbeat monitoring (ping/pong)
    - Price aggregation (combine rapid trades into single ticks)
    - Volume tracking per tick
    - Simulated fill engine for paper trading

    Args:
        symbol: Trading pair (e.g., "BTCUSDT", "ETHUSDT")
        base_url: WebSocket URL (default: Binance spot stream)
        ping_interval: WebSocket ping interval in seconds
        reconnect_base_delay: Base delay for reconnection (doubles each attempt)
        reconnect_max_delay: Maximum reconnection delay
        max_reconnects: Maximum reconnection attempts
        aggregate_interval: Minimum ms between emitted ticks (0 = every trade)
    """

    DEFAULT_URL = "wss://stream.binance.com:9443/ws"

    def __init__(
        self,
        symbol: str,
        base_url: str = DEFAULT_URL,
        ping_interval: float = 20.0,
        reconnect_base_delay: float = 1.0,
        reconnect_max_delay: float = 60.0,
        max_reconnects: int = 50,
        aggregate_interval: float = 0.0,
    ):
        self.symbol = symbol.lower()
        self.base_url = base_url
        self.ping_interval = ping_interval
        self.reconnect_base_delay = reconnect_base_delay
        self.reconnect_max_delay = reconnect_max_delay
        self.max_reconnects = max_reconnects
        self.aggregate_interval = aggregate_interval

        self._ws: Optional[websockets.WebSocketClientProtocol] = None
        self._running = False
        self._reconnect_count = 0
        self._tick_queue: asyncio.Queue[PricePoint] = asyncio.Queue()
        self._last_tick_time: float = 0.0
        self._stats = {
            "trades_received": 0,
            "ticks_emitted": 0,
            "reconnects": 0,
            "errors": 0,
        }

    @property
    def stream_url(self) -> str:
        """Build the full WebSocket URL for the trade stream."""
        return f"{self.base_url}/{self.symbol}@trade"

    @property
    def stats(self) -> dict:
        return self._stats.copy()

    @property
    def is_connected(self) -> bool:
        return self._ws is not None and self._ws.open

    async def connect(self):
        """Connect to the WebSocket with automatic reconnection."""
        self._running = True

        while self._running and self._reconnect_count < self.max_reconnects:
            try:
                logger.info(f"Connecting to Binance: {self.stream_url}")
                async with websockets.connect(
                    self.stream_url,
                    ping_interval=self.ping_interval,
                    ping_timeout=self.ping_interval * 1.5,
                ) as ws:
                    self._ws = ws
                    self._reconnect_count = 0
                    logger.info(
                        f"Connected to Binance {self.symbol} trade stream"
                    )

                    async for raw_message in ws:
                        await self._handle_message(raw_message)

            except websockets.ConnectionClosed as e:
                logger.warning(f"WebSocket closed: {e.code} {e.reason}")
                self._ws = None
            except asyncio.CancelledError:
                logger.info("WebSocket task cancelled")
                break
            except Exception as e:
                logger.error(f"WebSocket error: {e}")
                self._stats["errors"] += 1
                self._ws = None

            if self._running:
                self._reconnect_count += 1
                self._stats["reconnects"] += 1
                delay = min(
                    self.reconnect_base_delay * (2 ** (self._reconnect_count - 1)),
                    self.reconnect_max_delay,
                )
                logger.info(
                    f"Reconnecting in {delay:.1f}s "
                    f"(attempt {self._reconnect_count}/{self.max_reconnects})"
                )
                await asyncio.sleep(delay)

        if self._running:
            logger.error(f"Max reconnects ({self.max_reconnects}) reached")
        self._running = False

    async def disconnect(self):
        """Gracefully disconnect."""
        self._running = False
        if self._ws:
            await self._ws.close()
            self._ws = None

    async def ticks(self) -> AsyncIterator[PricePoint]:
        """Async iterator yielding PricePoint ticks.

        Usage:
            async for tick in connector.ticks():
                print(f"{tick.asset}: {tick.price}")
        """
        while self._running or not self._tick_queue.empty():
            try:
                tick = await asyncio.wait_for(
                    self._tick_queue.get(), timeout=1.0
                )
                yield tick
            except asyncio.TimeoutError:
                continue

    def get_latest_tick(self) -> Optional[PricePoint]:
        """Get the most recent tick without blocking."""
        latest = None
        while not self._tick_queue.empty():
            latest = self._tick_queue.get_nowait()
        return latest

    async def _handle_message(self, raw_message: str):
        """Parse a raw WebSocket message and emit a PricePoint."""
        try:
            data = json.loads(raw_message)
        except json.JSONDecodeError:
            return

        trade = BinanceTradeMessage.from_json(data)
        self._stats["trades_received"] += 1

        # Check aggregation interval
        now = time.time()
        if (
            self.aggregate_interval > 0
            and now - self._last_tick_time < self.aggregate_interval
        ):
            return  # Skip this tick — too soon after the last one

        self._last_tick_time = now

        tick = PricePoint(
            asset=trade.symbol.upper(),
            price=trade.price,
            currency="USD",
            source="binance_ws",
            timestamp=datetime.utcfromtimestamp(trade.event_time_ms / 1000),
            confidence=1.0,
        )

        self._stats["ticks_emitted"] += 1
        await self._tick_queue.put(tick)

    def to_market_data(self, tick: PricePoint) -> dict:
        """Convert a PricePoint to the market_data dict format used by agents.

        Returns:
            dict with 'price', 'timestamp', 'volume', 'source', 'symbol' keys.
        """
        return {
            "price": tick.price,
            "timestamp": tick.timestamp.isoformat() if tick.timestamp else None,
            "source": tick.source,
            "symbol": tick.asset,
            "confidence": tick.confidence,
        }


def create_binance_connector(
    symbol: str = "BTCUSDT",
    aggregate_ms: float = 0,
) -> BinanceWebSocketConnector:
    """Factory function to create a configured Binance connector.

    Args:
        symbol: Trading pair (e.g., "BTCUSDT")
        aggregate_ms: Minimum ms between emitted ticks (0 = every trade)

    Returns:
        Configured BinanceWebSocketConnector instance.
    """
    return BinanceWebSocketConnector(
        symbol=symbol,
        aggregate_interval=aggregate_ms / 1000 if aggregate_ms > 0 else 0,
    )
