from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

from lunar_tools._optional import require_extra


def _load_yaml(text: str) -> Dict[str, Any]:
    try:
        import yaml
    except ImportError:  # pragma: no cover - optional dependency guard
        require_extra("YAML configuration parsing", extras="presentation")

    data = yaml.safe_load(text) or {}
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError("YAML configuration must be a mapping at the top level.")
    return dict(data)


def load_config_file(path: str | Path) -> Dict[str, Any]:
    """
    Load a JSON or YAML configuration file into a dictionary.
    """
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {file_path}")

    text = file_path.read_text(encoding="utf-8")
    suffix = file_path.suffix.lower()

    if suffix in {".yaml", ".yml"}:
        return _load_yaml(text)

    if suffix == ".json" or not suffix:
        if not text.strip():
            return {}
        data = json.loads(text)
        if not isinstance(data, dict):
            raise ValueError("JSON configuration must be a mapping at the top level.")
        return data

    raise ValueError(
        f"Unsupported configuration format '{file_path.suffix}'. Use JSON (.json) or YAML (.yaml/.yml)."
    )


__all__ = ["load_config_file"]
