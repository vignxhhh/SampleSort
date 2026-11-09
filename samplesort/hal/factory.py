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
from samplesort.hal.scanner import IdScannerInterface
from samplesort.sim.world import SimWorld

logger = logging.getLogger(__name__)


@dataclass
class Backend:
    """The arm, camera and (in sim) the world they live in.

    Attributes:
        arm: The selected arm implementation.
        camera: The selected camera implementation.
        world: The simulation world, or ``None`` in real mode.
        scanner: The wrist-camera sample-ID scanner, or ``None`` when none is
            configured.
    """

    arm: ArmInterface
    camera: CameraInterface
    world: SimWorld | None = None
    scanner: IdScannerInterface | None = None

    def connect(self) -> None:
        """Connect the camera, the arm and the scanner when there is one."""
        self.arm.connect()
        self.camera.connect()
        if self.scanner is not None:
            self.scanner.connect()

    def disconnect(self) -> None:
        """Disconnect everything, including the sim world when there is one."""
        if self.scanner is not None:
            self.scanner.disconnect()
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
        from samplesort.hal.sim_scanner import SimIdScanner

        world = SimWorld(config, gui=gui, seed=seed)
        wrist = config.camera.wrist
        return Backend(
            arm=SimArm(config.arm, world),
            camera=SimCamera(config.camera, world),
            world=world,
            scanner=SimIdScanner(wrist, world) if wrist is not None else None,
        )

    if config.mode == "real":
        from samplesort.hal.real_arm import RealArm
        from samplesort.hal.real_camera import RealCamera

        logger.info("building real hardware backend on %s", config.arm.serial_port)
        wrist = config.camera.wrist
        scanner: IdScannerInterface | None = None
        if wrist is not None and wrist.index is not None:
            from samplesort.hal.scanner import RealIdScanner

            scanner = RealIdScanner(wrist)
        return Backend(arm=RealArm(config.arm), camera=RealCamera(config.camera), scanner=scanner)

    raise ValueError(f"unknown mode '{config.mode}'; expected 'sim' or 'real'")
