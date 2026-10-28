"""Central logging configuration for the CLI and library code."""

from __future__ import annotations

import logging
import os

_CONFIGURED = False


def setup_logging(level: str | int = "INFO", *, force: bool = False) -> None:
    """Install a single stderr handler for the ``samplesort`` logger hierarchy.

    Args:
        level: Logging level name or numeric value.
        force: Reconfigure even if logging was already set up in this process.
    """
    global _CONFIGURED
    if _CONFIGURED and not force:
        return

    env_level = os.environ.get("SAMPLESORT_LOG_LEVEL")
    resolved = env_level if env_level else level

    logging.basicConfig(
        level=resolved,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
        force=True,
    )
    # PyBullet and friends are chatty at DEBUG; keep third-party noise down.
    for noisy in ("matplotlib", "PIL", "urllib3", "huggingface_hub", "datasets"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    _CONFIGURED = True
