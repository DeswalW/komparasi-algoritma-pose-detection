"""
backends/registry.py – Factory for creating BackendAdapter instances by name.
"""

from __future__ import annotations

from typing import Dict, Type

from .base import BackendAdapter


_REGISTRY: Dict[str, Type[BackendAdapter]] = {}


def register(name: str):
    """Decorator to register a BackendAdapter subclass under *name*."""
    def decorator(cls: Type[BackendAdapter]):
        _REGISTRY[name.lower()] = cls
        cls.NAME = name.lower()
        return cls
    return decorator


def get_backend(name: str) -> BackendAdapter:
    """Instantiate and return the adapter registered under *name*.

    Parameters
    ----------
    name : backend identifier, case-insensitive.
           Supported values (after importing the modules below):
                         'mediapipe', 'alphapose', 'movenet', 'openpose',
                         'posenet', 'blazepose', 'hrnet', 'yolopose',
                         'efficientpose'

    Raises
    ------
    KeyError if the name is unknown.
    """
    key = name.lower()
    if key not in _REGISTRY:
        available = ", ".join(sorted(_REGISTRY.keys()))
        raise KeyError(
            f"Unknown backend {name!r}. "
            f"Available backends: {available or '(none registered yet)'}"
        )
    return _REGISTRY[key]()


def list_backends() -> list[str]:
    """Return list of registered backend names."""
    return sorted(_REGISTRY.keys())


# ── Auto-import all backend modules so they register themselves ───────────────

def _import_all() -> None:
    """Import backend modules to trigger @register decorators."""
    from . import mediapipe_pose  # noqa: F401
    from . import alphapose       # noqa: F401
    from . import movenet         # noqa: F401
    from . import openpose        # noqa: F401
    from . import posenet         # noqa: F401
    from . import blazepose       # noqa: F401
    from . import hrnet           # noqa: F401
    from . import yolopose        # noqa: F401
    from . import efficientpose   # noqa: F401


_import_all()
