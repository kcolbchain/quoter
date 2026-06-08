"""Tests for simulation presets."""

import subprocess
import sys

from src.utils.config import list_presets, load_preset


def test_simple_amm_lp_preset_is_registered():
    assert "simple-amm-lp" in list_presets()


def test_simple_amm_lp_preset_has_reasonable_lp_parameters():
    preset = load_preset("simple-amm-lp")

    assert preset["pair"] == "ETH/USDC"
    assert preset["strategy"] == "adaptive"
    assert 0 < preset["spread"] < 1
    assert preset["agent"]["initial_base"] > 0
    assert preset["agent"]["initial_quote"] > 0
    assert preset["agent"]["max_inventory_pct"] <= 0.5


def test_simple_amm_lp_preset_runs_from_cli():
    result = subprocess.run(
        [
            sys.executable,
            "run.py",
            "--simulate",
            "--preset",
            "simple-amm-lp",
            "--ticks",
            "3",
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
