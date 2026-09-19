"""Build the arm and camera backends selected by ``default.yaml``'s ``mode``.

Everything above the HAL asks for an :class:`~samplesort.hal.arm.ArmInterface` and
a :class:`~samplesort.hal.camera.CameraInterface` and gets whichever
implementation the config selected. Real-hardware modules are imported lazily so
a sim-only install never needs ``pyserial`` or a connected camera.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from samplesort.config import SampleSortConfig
from samplesort.hal.arm import ArmInterface
from samplesort.hal.camera import CameraInterface
from samplesort.sim.world import SimWorld

logger = logging.getLogger(__name__)


@dataclass
class Backend:
    """The arm, camera and (in sim) the world they live in.

    Attributes:
        arm: The selected arm implementation.
        camera: The selected camera implementation.
        world: The simulation world, or ``None`` in real mode.
    """

    arm: ArmInterface
    camera: CameraInterface
    world: SimWorld | None = None

    def connect(self) -> None:
        """Connect the camera and the arm."""
        self.arm.connect()
        self.camera.connect()

    def disconnect(self) -> None:
        """Disconnect everything, including the sim world when there is one."""
        self.camera.disconnect()
        self.arm.disconnect()
        if self.world is not None:
            self.world.disconnect()

    def __enter__(self) -> Backend:
        """Connect on entry to a ``with`` block."""
        self.connect()
        return self

    def __exit__(self, *exc_info: object) -> None:
        """Disconnect on exit from a ``with`` block."""
        self.disconnect()


def build_backend(
    config: SampleSortConfig, *, gui: bool = False, seed: int | None = None
) -> Backend:
    """Create the arm and camera implementations for the configured mode.

    Args:
        config: The validated configuration bundle.
        gui: In sim mode, open a PyBullet GUI window.
        seed: In sim mode, override the world's RNG seed.

    Returns:
        An unconnected :class:`Backend`.

    Raises:
        ValueError: If ``config.mode`` is not ``sim`` or ``real``.
    """
    if config.mode == "sim":
        from samplesort.hal.sim_arm import SimArm
        from samplesort.hal.sim_camera import SimCamera

        world = SimWorld(config, gui=gui, seed=seed)
        return Backend(
            arm=SimArm(config.arm, world),
            camera=SimCamera(config.camera, world),
            world=world,
        )

    if config.mode == "real":
        from samplesort.hal.real_arm import RealArm
        from samplesort.hal.real_camera import RealCamera

        logger.info("building real hardware backend on %s", config.arm.serial_port)
        return Backend(arm=RealArm(config.arm), camera=RealCamera(config.camera))

    raise ValueError(f"unknown mode '{config.mode}'; expected 'sim' or 'real'")
