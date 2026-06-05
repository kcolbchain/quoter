"""Configuration management."""

import yaml
from pathlib import Path


def load_config(path: str = "config/default.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def merge_configs(base: dict, override: dict) -> dict:
    """Deep merge override into base."""
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_configs(result[key], value)
        else:
            result[key] = value
    return result

PRESETS = {
    "simple-amm-lp": "presets/simple-amm-lp.yaml",
}

def load_preset(name: str) -> dict:
    """Load a registered preset."""
    if name not in PRESETS:
        raise ValueError(f"Unknown preset '{name}'. Available presets: {list(PRESETS.keys())}")
    return load_config(PRESETS[name])
