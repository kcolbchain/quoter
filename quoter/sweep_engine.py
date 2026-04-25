"""
Parameter Sweep Engine for quoter backtester
---------------------------------------------
MVP implementation for kcolbchain/quoter#20
Builds a grid of parameters → DataFrame of metrics for MM tuning.

Usage:
    from sweep_engine import ParameterSweep, SweepConfig
    config = SweepConfig(...)
    sweep = ParameterSweep(config)
    results = sweep.run()
"""

import itertools
import warnings
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Union

import pandas as pd
import numpy as np


@dataclass
class ParameterRange:
    """Defines a parameter's sweep range."""
    name: str
    values: List[Any]
    label: Optional[str] = None

    def __post_init__(self):
        if self.label is None:
            self.label = self.name


@dataclass
class SweepConfig:
    """Configuration for a parameter sweep."""
    # Parameter ranges to sweep over
    parameters: List[ParameterRange]
    # Strategy function: (params_dict, data) -> metrics_dict
    strategy_fn: Callable
    # Optional: data to pass to strategy
    data: Optional[pd.DataFrame] = None
    # Optional: custom metric aggregators
    aggregators: Optional[Dict[str, str]] = None
    # Optional: parallel workers (0 = sequential)
    workers: int = 0
    # Optional: progress callback
    progress_fn: Optional[Callable[[int, int], None]] = None

    def validate(self):
        for p in self.parameters:
            if not p.values:
                raise ValueError(f"Parameter '{p.name}' has no values to sweep")


@dataclass
class SweepResult:
    """Result of a single parameter combination."""
    params: Dict[str, Any]
    metrics: Dict[str, float]

    @property
    def combined(self) -> Dict[str, Any]:
        return {**self.params, **self.metrics}


class ParameterSweep:
    """
    Engine for sweeping parameters across a backtesting strategy.

    Runs the same backtest across a grid of parameters, collecting
    metrics into a DataFrame for analysis and tuning.

    Example:
        >>> config = SweepConfig(
        ...     parameters=[
        ...         ParameterRange("spread_bps", [10, 20, 50, 100]),
        ...         ParameterRange("inventory_skew", [0.5, 1.0, 2.0]),
        ...         ParameterRange("max_inventory", [100, 500, 1000]),
        ...     ],
        ...     strategy_fn=my_mm_strategy,
        ...     data=price_data,
        ... )
        >>> sweep = ParameterSweep(config)
        >>> results = sweep.run()
        >>> print(results.head())
        >>> best = results.loc[results["sharpe_ratio"].idxmax()]
    """

    def __init__(self, config: SweepConfig):
        self.config = config
        self.config.validate()
        self._results: List[SweepResult] = []

    @property
    def total_combinations(self) -> int:
        """Total number of parameter combinations."""
        total = 1
        for p in self.config.parameters:
            total *= len(p.values)
        return total

    def _build_grid(self) -> List[Dict[str, Any]]:
        """Build the Cartesian product of all parameter values."""
        param_names = [p.name for p in self.config.parameters]
        param_values = [p.values for p in self.config.parameters]
        grid = []
        for combo in itertools.product(*param_values):
            grid.append(dict(zip(param_names, combo)))
        return grid

    def _run_single(self, params: Dict[str, Any]) -> SweepResult:
        """Run strategy with a single parameter combination."""
        metrics = self.config.strategy_fn(params, self.config.data)
        # Ensure metrics are numeric
        numeric_metrics = {}
        for k, v in metrics.items():
            try:
                numeric_metrics[k] = float(v)
            except (TypeError, ValueError):
                numeric_metrics[k] = v
        return SweepResult(params=params, metrics=numeric_metrics)

    def run(self) -> pd.DataFrame:
        """
        Execute the full parameter sweep.

        Returns:
            DataFrame with one row per parameter combination,
            columns for all parameters and all metrics.
        """
        grid = self._build_grid()
        total = len(grid)

        for i, params in enumerate(grid):
            result = self._run_single(params)
            self._results.append(result)

            if self.config.progress_fn and (i + 1) % max(1, total // 10) == 0:
                self.config.progress_fn(i + 1, total)

        return self._to_dataframe()

    def _to_dataframe(self) -> pd.DataFrame:
        """Convert results to a pandas DataFrame."""
        if not self._results:
            return pd.DataFrame()

        rows = [r.combined for r in self._results]
        df = pd.DataFrame(rows)

        # Sort by total trade count (if available) then by first metric
        metric_cols = self._get_metric_columns()
        if metric_cols:
            df = df.sort_values(by=metric_cols[0], ascending=False).reset_index(drop=True)

        return df

    def _get_metric_columns(self) -> List[str]:
        """Get column names that are metrics (not parameters)."""
        param_names = {p.name for p in self.config.parameters}
        if not self._results:
            return []
        all_cols = set(self._results[0].metrics.keys())
        return sorted(all_cols - param_names)

    def get_best(self, metric: str, ascending: bool = False) -> Optional[Dict[str, Any]]:
        """Get the parameter combination with the best metric value."""
        if not self._results:
            return None
        best = max(self._results, key=lambda r: r.metrics.get(metric, float('-inf')))
        return best.combined

    def get_summary_stats(self) -> pd.DataFrame:
        """Get summary statistics for all metrics."""
        df = self._to_dataframe()
        if df.empty:
            return df
        metric_cols = self._get_metric_columns()
        return df[metric_cols].describe()

    def get_correlation_matrix(self) -> Optional[pd.DataFrame]:
        """Get correlation between parameters and metrics."""
        df = self._to_dataframe()
        if df.empty:
            return None
        return df.corr(numeric_only=True)


# ============================================================
# Example: Market Making Strategy for demonstration
# ============================================================

def example_mm_strategy(params: Dict[str, Any], data: Optional[pd.DataFrame] = None) -> Dict[str, float]:
    """
    Example market making strategy for sweep demonstration.

    In production, this would call the actual quoter backtester.
    """
    spread = params.get("spread_bps", 20) / 10000
    skew = params.get("inventory_skew", 1.0)
    max_inv = params.get("max_inventory", 100)

    # Simulated metrics (replace with actual backtest results)
    if data is not None and len(data) > 0:
        volatility = data["price"].pct_change().std()
    else:
        volatility = 0.01

    # Simple MM model: wider spread → more profit per trade, fewer fills
    fill_rate = max(0.01, 1.0 - spread * 50)
    profit_per_trade = spread * fill_rate * 10000
    total_trades = int(fill_rate * 1000)
    inventory_turnover = min(total_trades / max(max_inv, 1), 10.0)

    # Sharpe-like metric
    sharpe = (profit_per_trade * total_trades) / max(volatility * 100, 0.001)

    return {
        "total_trades": total_trades,
        "profit_per_trade": round(profit_per_trade, 4),
        "fill_rate": round(fill_rate, 4),
        "inventory_turnover": round(inventory_turnover, 2),
        "sharpe_ratio": round(sharpe, 4),
    }


def demo():
    """Run a demo sweep with example data."""
    # Generate sample price data
    np.random.seed(42)
    n_points = 1000
    prices = pd.DataFrame({
        "price": 100 * np.cumprod(1 + np.random.normal(0, 0.001, n_points)),
        "volume": np.random.exponential(100, n_points),
    })

    config = SweepConfig(
        parameters=[
            ParameterRange("spread_bps", [5, 10, 20, 50, 100]),
            ParameterRange("inventory_skew", [0.5, 1.0, 2.0, 5.0]),
            ParameterRange("max_inventory", [50, 100, 200, 500]),
        ],
        strategy_fn=example_mm_strategy,
        data=prices,
    )

    sweep = ParameterSweep(config)
    print(f"Running sweep: {sweep.total_combinations} combinations...")

    results = sweep.run()
    print(f"\nCompleted! Generated {len(results)} results.")
    print(f"\nColumns: {list(results.columns)}")

    print("\n=== Top 5 Results by Sharpe Ratio ===")
    top5 = results.nlargest(5, "sharpe_ratio")
    print(top5.to_string(index=False))

    print("\n=== Summary Statistics ===")
    print(sweep.get_summary_stats().to_string())

    print("\n=== Best Parameters ===")
    best = sweep.get_best("sharpe_ratio")
    if best:
        print(f"  Sharpe: {best['sharpe_ratio']:.4f}")
        print(f"  Spread: {best['spread_bps']} bps")
        print(f"  Skew: {best['inventory_skew']}")
        print(f"  Max Inv: {best['max_inventory']}")

    return results


if __name__ == "__main__":
    demo()
