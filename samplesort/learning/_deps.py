"""Lazy imports for the optional torch + LeRobot stack.

The core sim, perception, planning, pipeline and benchmark paths install and run
without PyTorch or LeRobot. These helpers import them on demand and turn an
``ImportError`` into an actionable message rather than a traceback.
"""

from __future__ import annotations

import logging
import os
from typing import Any

logger = logging.getLogger(__name__)

INSTALL_HINT = (
    'The learning stack is an optional extra. Install it with:\n    pip install -e ".[learning]"'
)


class LearningDependencyError(RuntimeError):
    """Raised when the optional torch/LeRobot stack is needed but not installed."""


def _force_offline() -> None:
    """Keep Hugging Face libraries from reaching the network.

    Everything SampleSort records, trains and evaluates is local, and LeRobot
    otherwise tries to resolve a dataset's ``repo_id`` against the Hub — which
    fails on an air-gapped machine and in CI. An explicit setting from the
    environment always wins, so pushing to the Hub stays possible for anyone who
    opts in.
    """
    for name in ("HF_HUB_OFFLINE", "HF_DATASETS_OFFLINE"):
        os.environ.setdefault(name, "1")
    os.environ.setdefault("WANDB_MODE", "disabled")


def require_torch() -> Any:
    """Import and return the :mod:`torch` module.

    Raises:
        LearningDependencyError: If PyTorch is not installed.
    """
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - depends on the install
        raise LearningDependencyError(
            f"PyTorch is required for this command.\n{INSTALL_HINT}"
        ) from exc
    return torch


def require_lerobot() -> Any:
    """Import and return the :mod:`lerobot` package, forcing offline mode first.

    Raises:
        LearningDependencyError: If LeRobot is not installed.
    """
    _force_offline()
    try:
        import lerobot
    except ImportError as exc:  # pragma: no cover - depends on the install
        raise LearningDependencyError(
            f"LeRobot is required for this command.\n{INSTALL_HINT}"
        ) from exc
    return lerobot


def is_available() -> bool:
    """Whether both PyTorch and LeRobot can be imported."""
    try:
        require_torch()
        require_lerobot()
    except LearningDependencyError:
        return False
    return True


def resolve_device(requested: str) -> str:
    """Pick a usable torch device, falling back to CPU with a warning.

    Args:
        requested: The device named in the config, e.g. ``"cpu"`` or ``"cuda"``.

    Returns:
        A device string that torch can actually use here.
    """
    torch = require_torch()
    if requested.startswith("cuda") and not torch.cuda.is_available():
        logger.warning("CUDA was requested but is not available; falling back to CPU")
        return "cpu"
    if requested == "mps" and not torch.backends.mps.is_available():
        logger.warning("MPS was requested but is not available; falling back to CPU")
        return "cpu"
    return requested
