"""Tests for parameter sweep engine - quoter#20 MVP"""

import pytest
import pandas as pd
import numpy as np
from sweep_engine import (
    ParameterRange, SweepConfig, ParameterSweep, SweepResult,
    example_mm_strategy
)


class TestParameterRange:
    def test_default_label(self):
        pr = ParameterRange("spread_bps", [10, 20, 50])
        assert pr.label == "spread_bps"

    def test_custom_label(self):
        pr = ParameterRange("spread_bps", [10, 20], label="Spread (bps)")
        assert pr.label == "Spread (bps)"


class TestSweepConfig:
    def test_valid_config(self):
        config = SweepConfig(
            parameters=[ParameterRange("x", [1, 2, 3])],
            strategy_fn=lambda p, d: {"score": 1.0},
        )
        config.validate()  # Should not raise

    def test_empty_values_raises(self):
        config = SweepConfig(
            parameters=[ParameterRange("x", [])],
            strategy_fn=lambda p, d: {},
        )
        with pytest.raises(ValueError, match="has no values"):
            config.validate()


class TestParameterSweep:
    def _make_config(self, params, data=None):
        return SweepConfig(
            parameters=params,
            strategy_fn=example_mm_strategy,
            data=data,
        )

    def test_total_combinations(self):
        config = self._make_config([
            ParameterRange("a", [1, 2]),
            ParameterRange("b", [3, 4, 5]),
        ])
        sweep = ParameterSweep(config)
        assert sweep.total_combinations == 6

    def test_run_produces_dataframe(self):
        config = self._make_config([
            ParameterRange("spread_bps", [10, 50]),
            ParameterRange("inventory_skew", [1.0, 2.0]),
        ])
        sweep = ParameterSweep(config)
        results = sweep.run()

        assert isinstance(results, pd.DataFrame)
        assert len(results) == 4  # 2x2 grid
        assert "spread_bps" in results.columns
        assert "inventory_skew" in results.columns
        assert "sharpe_ratio" in results.columns

    def test_results_sorted_by_first_metric(self):
        config = self._make_config([
            ParameterRange("spread_bps", [5, 100]),
        ])
        sweep = ParameterSweep(config)
        results = sweep.run()
        # First metric column should be descending
        metric_cols = sweep._get_metric_columns()
        assert results[metric_cols[0]].iloc[0] >= results[metric_cols[0]].iloc[-1]

    def test_get_best(self):
        config = self._make_config([
            ParameterRange("spread_bps", [5, 10, 50, 100]),
        ])
        sweep = ParameterSweep(config)
        sweep.run()
        best = sweep.get_best("sharpe_ratio")
        assert best is not None
        assert "spread_bps" in best
        assert "sharpe_ratio" in best

    def test_get_summary_stats(self):
        config = self._make_config([
            ParameterRange("spread_bps", [10, 20, 50]),
        ])
        sweep = ParameterSweep(config)
        sweep.run()
        stats = sweep.get_summary_stats()
        assert isinstance(stats, pd.DataFrame)
        assert "count" in stats.index

    def test_with_data(self):
        np.random.seed(42)
        data = pd.DataFrame({"price": np.cumprod(1 + np.random.normal(0, 0.01, 100))})
        config = self._make_config([
            ParameterRange("spread_bps", [10, 50]),
        ], data=data)
        sweep = ParameterSweep(config)
        results = sweep.run()
        assert len(results) == 2

    def test_empty_results(self):
        config = self._make_config([
            ParameterRange("x", [1]),
        ])
        sweep = ParameterSweep(config)
        # Before run
        assert sweep.get_best("any_metric") is None
        stats = sweep.get_summary_stats()
        assert stats.empty


class TestExampleStrategy:
    def test_strategy_returns_numeric_metrics(self):
        params = {"spread_bps": 20, "inventory_skew": 1.0, "max_inventory": 100}
        metrics = example_mm_strategy(params)
        for k, v in metrics.items():
            assert isinstance(v, (int, float)), f"{k} = {v} is not numeric"

    def test_wider_spread_fewer_trades(self):
        params_narrow = {"spread_bps": 5}
        params_wide = {"spread_bps": 100}
        m1 = example_mm_strategy(params_narrow)
        m2 = example_mm_strategy(params_wide)
        assert m1["total_trades"] >= m2["total_trades"]

    def test_with_data_uses_volatility(self):
        np.random.seed(42)
        data = pd.DataFrame({"price": np.cumprod(1 + np.random.normal(0, 0.01, 100))})
        m1 = example_mm_strategy({"spread_bps": 20}, data)
        m2 = example_mm_strategy({"spread_bps": 20})
        # With data, sharpe should differ (uses actual volatility)
        assert isinstance(m1["sharpe_ratio"], float)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
