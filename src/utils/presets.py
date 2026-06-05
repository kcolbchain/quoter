"""Preset management — discover and load simulation presets."""

import yaml
from pathlib import Path

PRESETS_DIR = Path(__file__).resolve().parent.parent.parent / "presets"


def list_presets() -> list[str]:
    """Return sorted list of available preset names (from filenames without extension)."""
    if not PRESETS_DIR.is_dir():
        return []
    return sorted(p.stem for p in PRESETS_DIR.glob("*.yaml"))


def load_preset(name: str) -> dict:
    """Load a preset YAML file by name. Raises FileNotFoundError if missing."""
    path = PRESETS_DIR / f"{name}.yaml"
    if not path.is_file():
        raise FileNotFoundError(f"Preset '{name}' not found. Available: {list_presets()}")
    with open(path) as f:
        return yaml.safe_load(f)
