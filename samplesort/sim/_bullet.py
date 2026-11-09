"""Import PyBullet without its banner landing on stdout.

``pybullet`` prints a build banner from native code the moment it is imported,
and ``BulletClient`` adds an ``argv[0]=`` line when it connects. Both go straight
to file descriptor 1, so Python-level redirection does not catch them — and
SampleSort's CLI writes real results to stdout, where that noise does not belong.

Every module that needs PyBullet imports it from here, so the suppression happens
exactly once, at first import.
"""

from __future__ import annotations

import contextlib
import os
import sys
from collections.abc import Iterator


@contextlib.contextmanager
def quiet_stdout() -> Iterator[None]:
    """Silence writes to file descriptor 1 for the duration of the block.

    Falls through untouched when stdout is not a real file descriptor, which is
    the case under some test runners and notebook frontends.
    """
    try:
        original = os.dup(1)
    except (AttributeError, OSError):  # pragma: no cover - unusual environments
        yield
        return

    devnull = os.open(os.devnull, os.O_WRONLY)
    try:
        sys.stdout.flush()
        os.dup2(devnull, 1)
        yield
    finally:
        sys.stdout.flush()
        os.dup2(original, 1)
        os.close(devnull)
        os.close(original)


with quiet_stdout():
    import pybullet as pb
    import pybullet_utils.bullet_client as bc

__all__ = ["bc", "pb", "quiet_stdout"]
