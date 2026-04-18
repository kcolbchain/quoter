"""Recorded feed connector for replaying saved WebSocket traces.

Loads a JSON file of recorded Binance trade messages and replays them
as PricePoint ticks. Used for testing strategies against historical data
without requiring a live WebSocket connection.

Trace format (JSON array):
[
    {
        "e": "trade",
        "E": 1672515782136,
        "s": "BTCUSDT",
        "t": 12345,
        "p": "50000.00",
        "q": "0.001",
        "T": 1672515782136,
        "m": true
    },
    ...
]

Usage:
    connector = RecordedFeedConnector.from_file("trace.json", speed=10)
    async for tick in connector.ticks():
        agent.tick({"price": tick.price})
"""

import asyncio
import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import AsyncIterator, Optional

from src.connectors.binance_ws import BinanceTradeMessage
from src.oracle.price_feed import PricePoint

logger = logging.getLogger(__name__)


class RecordedFeedConnector:
    """Replays recorded WebSocket trade messages as PricePoint ticks.

    Args:
        messages: List of raw trade message dicts.
        speed: Playback speed multiplier (1 = real-time, 10 = 10x faster).
        symbol_override: Override the symbol from the trace (useful for testing).
    """

    def __init__(
        self,
        messages: list[dict],
        speed: float = 1.0,
        symbol_override: Optional[str] = None,
    ):
        self._messages = messages
        self.speed = speed
        self._symbol_override = symbol_override
        self._running = False

    @classmethod
    def from_file(
        cls,
        path: str,
        speed: float = 1.0,
        symbol_override: Optional[str] = None,
    ) -> "RecordedFeedConnector":
        """Load a recorded trace from a JSON file.

        Args:
            path: Path to JSON file containing array of trade messages.
            speed: Playback speed multiplier.
            symbol_override: Override symbol from trace.

        Returns:
            Configured RecordedFeedConnector.
        """
        file_path = Path(path)
        if not file_path.exists():
            raise FileNotFoundError(f"Trace file not found: {path}")

        with open(file_path, "r") as f:
            messages = json.load(f)

        if not isinstance(messages, list):
            raise ValueError("Trace file must contain a JSON array of messages")

        logger.info(f"Loaded {len(messages)} messages from {path}")
        return cls(messages, speed=speed, symbol_override=symbol_override)

    @classmethod
    def from_json_string(
        cls,
        json_str: str,
        speed: float = 1.0,
        symbol_override: Optional[str] = None,
    ) -> "RecordedFeedConnector":
        """Create from a JSON string (useful for inline test data)."""
        messages = json.loads(json_str)
        return cls(messages, speed=speed, symbol_override=symbol_override)

    async def ticks(self) -> AsyncIterator[PricePoint]:
        """Async iterator yielding PricePoint ticks at replay speed.

        Respects original timing between messages (adjusted by speed).
        Messages with no timing info are emitted with a 10ms gap.
        """
        self._running = True
        prev_time_ms: Optional[int] = None

        for msg in self._messages:
            if not self._running:
                break

            trade = BinanceTradeMessage.from_json(msg)

            # Calculate delay from original timing
            if prev_time_ms is not None and trade.event_time_ms > prev_time_ms:
                delay_ms = (trade.event_time_ms - prev_time_ms) / self.speed
                if delay_ms > 0:
                    await asyncio.sleep(delay_ms / 1000)
            else:
                await asyncio.sleep(0.01)  # 10ms default gap

            prev_time_ms = trade.event_time_ms

            symbol = self._symbol_override or trade.symbol

            tick = PricePoint(
                asset=symbol.upper(),
                price=trade.price,
                currency="USD",
                source="recorded",
                timestamp=datetime.utcfromtimestamp(
                    trade.event_time_ms / 1000
                ),
                confidence=1.0,
            )

            yield tick

        self._running = False

    def stop(self):
        """Stop the replay."""
        self._running = False

    @property
    def message_count(self) -> int:
        return len(self._messages)


def create_sample_trace() -> list[dict]:
    """Generate a small sample trace for testing.

    Returns:
        List of 50 simulated BTCUSDT trade messages over 5 seconds.
    """
    import random

    base_time = int(time.time() * 1000)
    base_price = 50000.0
    messages = []

    for i in range(50):
        price = base_price + random.gauss(0, 50)  # ±$50 volatility
        messages.append({
            "e": "trade",
            "E": base_time + i * 100,  # 100ms apart
            "s": "BTCUSDT",
            "t": 1000 + i,
            "p": f"{price:.2f}",
            "q": f"{random.uniform(0.001, 0.1):.6f}",
            "T": base_time + i * 100,
            "m": random.random() > 0.5,
        })

    return messages
