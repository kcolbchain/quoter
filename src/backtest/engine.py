"""Backtesting engine — replay historical data through agents."""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

from ..agents.base_agent import BaseAgent, Fill, Side

logger = logging.getLogger(__name__)


@dataclass
class BacktestTick:
    timestamp: str
    oracle_price: float
    on_chain_price: Optional[float] = None
    volume_24h: float = 0.0


@dataclass
class BacktestResult:
    total_ticks: int
    total_fills: int
    realized_pnl: float
    unrealized_pnl: float
    total_pnl: float
    max_drawdown: float
    sharpe_ratio: float
    final_position: float
    fill_rate: float  # fills per tick
    fills: list[dict] = field(default_factory=list)  # per-fill record


class BacktestEngine:
    """Replay historical data through an agent and measure performance."""

    def __init__(self, agent: BaseAgent, fill_probability: float = 0.3):
        self.agent = agent
        self.fill_probability = fill_probability

    def run(self, data: list[BacktestTick]) -> BacktestResult:
        """Run backtest over historical data.

        Returns a BacktestResult with summary metrics and a per-fill record list.
        """
        pnl_curve = []
        peak_pnl = 0.0
        max_drawdown = 0.0
        total_fills = 0
        fill_records: list[dict] = []

        for tick in data:
            market_data = {
                "oracle_price": tick.oracle_price,
                "on_chain_price": tick.on_chain_price or tick.oracle_price,
                "volume_24h": tick.volume_24h,
            }

            orders = self.agent.tick(market_data)

            # Simulate fills with probability
            import random
            for order in orders:
                if random.random() < self.fill_probability:
                    fill = Fill(
                        side=order.side,
                        price=order.price,
                        size=order.size,
                        fee=order.price * order.size * 0.001,  # 10bps fee
                    )
                    self.agent.on_fill(fill)
                    total_fills += 1

                    pnl = self.agent.get_pnl(tick.oracle_price)
                    fill_records.append({
                        "timestamp": tick.timestamp,
                        "action": fill.side.value,
                        "price": round(fill.price, 6),
                        "size": round(fill.size, 6),
                        "fee": round(fill.fee, 6),
                        "spread": round(abs(market_data["on_chain_price"] - market_data["oracle_price"]), 6) if market_data.get("on_chain_price") else 0.0,
                        "inventory": round(self.agent.position.base_balance, 6),
                        "quote_balance": round(self.agent.position.quote_balance, 2),
                        "realized_pnl": round(pnl["realized"], 4),
                        "unrealized_pnl": round(pnl["unrealized"], 4),
                        "total_pnl": round(pnl["total"], 4),
                    })

            pnl = self.agent.get_pnl(tick.oracle_price)
            total = pnl["total"]
            pnl_curve.append(total)
            peak_pnl = max(peak_pnl, total)
            drawdown = peak_pnl - total
            max_drawdown = max(max_drawdown, drawdown)

        # Compute Sharpe ratio
        if len(pnl_curve) > 1:
            import numpy as np
            returns = np.diff(pnl_curve)
            sharpe = float(np.mean(returns) / np.std(returns)) if np.std(returns) > 0 else 0.0
        else:
            sharpe = 0.0

        final_pnl = self.agent.get_pnl(data[-1].oracle_price) if data else {"realized": 0, "unrealized": 0, "total": 0}

        return BacktestResult(
            total_ticks=len(data),
            total_fills=total_fills,
            realized_pnl=final_pnl["realized"],
            unrealized_pnl=final_pnl["unrealized"],
            total_pnl=final_pnl["total"],
            max_drawdown=max_drawdown,
            sharpe_ratio=sharpe,
            final_position=self.agent.position.base_balance,
            fill_rate=total_fills / len(data) if data else 0,
            fills=fill_records,
        )

    def export(self, result: BacktestResult, output_path: str, fmt: str = "csv") -> str:
        """Export backtest fill records to CSV or Parquet.

        Args:
            result: BacktestResult from run().
            output_path: File path for the output (extension is auto-adjusted).
            fmt: "csv" or "parquet".

        Returns:
            The resolved output file path.
        """
        if not result.fills:
            logger.warning("No fills to export.")
            return ""

        import pandas as pd

        df = pd.DataFrame(result.fills)
        path = Path(output_path)

        if fmt == "parquet":
            path = path.with_suffix(".parquet")
            path.parent.mkdir(parents=True, exist_ok=True)
            df.to_parquet(str(path), index=False)
        else:
            path = path.with_suffix(".csv")
            path.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(str(path), index=False)

        logger.info(f"Exported {len(df)} fills to {path}")
        return str(path)

    @staticmethod
    def generate_mock_data(base_price: float, ticks: int, volatility: float = 0.02) -> list[BacktestTick]:
        """Generate mock price data for testing."""
        import random
        data = []
        price = base_price
        for i in range(ticks):
            price *= (1 + random.gauss(0, volatility))
            data.append(BacktestTick(
                timestamp=f"2026-01-01T{i:05d}",
                oracle_price=price,
                volume_24h=random.uniform(10000, 1000000),
            ))
        return data


if __name__ == "__main__":
    from ..agents.rwa_market_maker import RWAMarketMaker

    config = {
        "initial_quote": 10000,
        "base_spread_bps": 200,
        "max_order_size_pct": 0.1,
        "max_exposure": 50,
    }
    agent = RWAMarketMaker("backtest-rwa", config)
    engine = BacktestEngine(agent, fill_probability=0.4)
    data = engine.generate_mock_data(100.0, 500)
    result = engine.run(data)

    print(f"Backtest Results:")
    print(f"  Ticks:        {result.total_ticks}")
    print(f"  Fills:        {result.total_fills} ({result.fill_rate:.1%} fill rate)")
    print(f"  Realized PnL: {result.realized_pnl:.2f}")
    print(f"  Unrealized:   {result.unrealized_pnl:.2f}")
    print(f"  Total PnL:    {result.total_pnl:.2f}")
    print(f"  Max Drawdown: {result.max_drawdown:.2f}")
    print(f"  Sharpe Ratio: {result.sharpe_ratio:.3f}")
    print(f"  Fills exported: {engine.export(result, 'backtest_output.csv')}")
    print(f"  Parquet export: {engine.export(result, 'backtest_output', fmt='parquet')}")
