from pathlib import Path

import yaml


PRESETS_DIR = Path("presets")


def load_config(path: str = "config/default.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f) or {}


def merge_configs(base: dict, override: dict) -> dict:
    """Deep merge override into base."""
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_configs(result[key], value)
        else:
            result[key] = value
    return result


def list_presets(presets_dir: Path = PRESETS_DIR) -> list[str]:
    """Return registered preset names from YAML files."""
    if not presets_dir.exists():
        return []
    return sorted(path.stem for path in presets_dir.glob("*.yaml"))


def load_preset(name: str, presets_dir: Path = PRESETS_DIR) -> dict:
    """Load a named simulation preset."""
    if name.endswith(".yaml") or "/" in name or "\\" in name:
        raise ValueError("Preset names should not include paths or extensions")

    preset_path = presets_dir / f"{name}.yaml"
    if not preset_path.exists():
        available = ", ".join(list_presets(presets_dir)) or "none"
        raise ValueError(f"Unknown preset '{name}'. Available presets: {available}")

    return load_config(str(preset_path))
