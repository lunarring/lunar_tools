from __future__ import annotations

from importlib import import_module
from typing import Iterable, Sequence


class OptionalDependencyError(ImportError):
    """Raised when a feature is requested without its optional dependencies."""

    def __init__(self, feature: str, extras: Sequence[str]) -> None:
        extras_list = list(dict.fromkeys(extras))
        install_suggestions = " or ".join(
            f"`pip install lunar_tools[{extra}]`" for extra in extras_list
        )
        message = (
            f"{feature} requires optional dependencies that are not installed. "
            f"Install the feature extras via {install_suggestions}."
        )
        super().__init__(message)
        self.feature = feature
        self.extras = extras_list


def _coerce_extras(extras: str | Iterable[str]) -> list[str]:
    if isinstance(extras, str):
        return [extras]
    return list(extras)


def optional_import(module: str, *, feature: str, extras: str | Iterable[str]):
    """Import a module and raise a friendly message if dependencies are missing."""
    extras_list = _coerce_extras(extras)
    try:
        return import_module(module)
    except OptionalDependencyError:
        raise
    except ImportError as exc:
        raise OptionalDependencyError(feature, extras_list) from exc


def optional_import_attr(
    module: str,
    attribute: str,
    *,
    feature: str,
    extras: str | Iterable[str],
):
    """Import an attribute from a module guarded by optional dependency messaging."""
    mod = optional_import(module, feature=feature, extras=extras)
    try:
        return getattr(mod, attribute)
    except AttributeError as exc:
        raise AttributeError(f"Module '{module}' has no attribute '{attribute}'") from exc


def require_extra(feature: str, *, extras: str | Iterable[str]) -> None:
    """Raise a consistent error indicating that an extra must be installed."""
    extras_list = _coerce_extras(extras)
    raise OptionalDependencyError(feature, extras_list)
