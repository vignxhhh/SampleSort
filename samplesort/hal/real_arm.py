"""SO-101 arm driven over the Feetech serial bus via LeRobot.

Verified against LeRobot 0.4.4's ``lerobot.motors.feetech.FeetechMotorsBus``:

* motors are declared as ``{name: Motor(id, model, norm_mode)}``,
* body joints use :attr:`MotorNormMode.DEGREES`, the gripper uses ``RANGE_0_100``,
* positions move through ``sync_read("Present_Position")`` and
  ``sync_write("Goal_Position", ...)``,
* the bus refuses normalised reads and writes unless a calibration is registered,
  so :class:`RealArm` loads one and fails with an actionable message if it is absent.

Two conversions sit between the kinematic model and the hardware:

1. **Units** — the bus speaks degrees, the kinematics speak radians.
2. **Frame** — the bus reports degrees from each joint's *calibrated mid-position*,
   which is not SampleSort's kinematic zero. ``servo_offsets_deg`` and
   ``servo_signs`` in ``arm_so101.yaml`` align the two, and are measured during
   bring-up (see ``docs/hardware_setup.md``).

Joint limits from the config are enforced before every command, so a bad IK
solution or a wild policy action cannot drive the arm into itself.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

import numpy as np

from samplesort.config import ArmConfig
from samplesort.control.trajectory import interpolate
from samplesort.hal.arm import ArmError, ArmInterface, JointVector

logger = logging.getLogger(__name__)

#: Control-loop period used while streaming a trajectory, in seconds.
STREAM_PERIOD_S = 1.0 / 50.0

#: The bus normalises the gripper to 0-100; these are the two ends.
GRIPPER_OPEN = 100.0
GRIPPER_CLOSED = 0.0

#: Seconds allowed for the gripper to finish opening or closing.
GRIPPER_SETTLE_S = 0.5


class RealArm(ArmInterface):
    """Drives a physical SO-101 follower arm through LeRobot's Feetech bus.

    Args:
        config: Arm geometry, joint limits and serial wiring.
        bus: A preconstructed motors bus. Mainly for tests; normally left unset
            so the arm builds its own from the config.
    """

    def __init__(self, config: ArmConfig, bus: Any | None = None) -> None:
        """Prepare the arm without opening the serial port yet."""
        super().__init__(config)
        self._bus = bus
        self._gripper_open = True

    # ------------------------------------------------------------------ naming

    @property
    def gripper_name(self) -> str:
        """Bus name of the gripper motor."""
        return "gripper"

    def _build_bus(self) -> Any:
        """Construct the Feetech bus from the config and its calibration file.

        Raises:
            ArmError: If LeRobot is missing or the calibration file is absent.
        """
        try:
            from lerobot.motors import Motor, MotorNormMode
            from lerobot.motors.feetech import FeetechMotorsBus
        except ImportError as exc:
            raise ArmError(
                "Driving real SO-101 hardware needs LeRobot. Install it with:\n"
                '    pip install -e ".[learning]"'
            ) from exc

        ids = self.config.servo_ids or list(range(1, self.config.num_joints + 1))
        motors = {
            name: Motor(motor_id, self.config.servo_model, MotorNormMode.DEGREES)
            for name, motor_id in zip(self.config.joint_names, ids, strict=True)
        }
        motors[self.gripper_name] = Motor(
            self.config.gripper_servo_id, self.config.servo_model, MotorNormMode.RANGE_0_100
        )

        return FeetechMotorsBus(
            port=self.config.serial_port,
            motors=motors,
            calibration=self.load_calibration(),
        )

    def load_calibration(self) -> dict[str, Any]:
        """Read the servo calibration LeRobot's bus requires.

        Returns:
            A mapping from motor name to ``MotorCalibration``.

        Raises:
            ArmError: If the file is missing or malformed.
        """
        from lerobot.motors import MotorCalibration

        path = Path(self.config.servo_calibration_path)
        if not path.is_file():
            raise ArmError(
                f"no servo calibration at {path}. The Feetech bus cannot report "
                f"normalised positions without one. Calibrate the arm first — see "
                f"docs/hardware_setup.md — then point arm_so101.yaml's "
                f"servo_calibration_path at the file it writes."
            )
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
            return {name: MotorCalibration(**values) for name, values in raw.items()}
        except (json.JSONDecodeError, TypeError, ValueError) as exc:
            raise ArmError(f"could not read the servo calibration at {path}: {exc}") from exc

    # ---------------------------------------------------------------- lifecycle

    def connect(self) -> None:
        """Open the serial port, configure the motors and enable torque.

        Raises:
            ArmError: If the bus cannot be opened.
        """
        if self._connected:
            return
        if self._bus is None:
            self._bus = self._build_bus()

        try:
            self._bus.connect()
            self._bus.configure_motors()
            self._bus.enable_torque()
        except Exception as exc:  # noqa: BLE001 - the SDK raises many types
            raise ArmError(
                f"could not connect to the arm on {self.config.serial_port}: {exc}\n"
                f"Check the cable, that the port is right, and that you have "
                f"permission to open it (on Linux: add yourself to the 'dialout' group)."
            ) from exc

        self._connected = True
        logger.info("connected to the SO-101 arm on %s", self.config.serial_port)

    def disconnect(self) -> None:
        """Disable torque and close the serial port."""
        if self._bus is not None and self._connected:
            try:
                self._bus.disconnect(disable_torque=True)
            except Exception:  # noqa: BLE001 - never let teardown mask a real error
                logger.exception("error while disconnecting the arm")
        self._connected = False
        logger.info("disconnected from the SO-101 arm")

    @property
    def bus(self) -> Any:
        """The underlying motors bus.

        Raises:
            ArmError: If the arm has not been connected.
        """
        if self._bus is None:
            raise ArmError("the arm has not been connected; call connect() first")
        return self._bus

    # ------------------------------------------------------------- conversions

    def joints_to_degrees(self, q: np.ndarray) -> dict[str, float]:
        """Convert kinematic joint angles to per-motor degrees for the bus."""
        offsets = np.asarray(self.config.servo_offsets_deg, dtype=float)
        signs = np.asarray(self.config.servo_signs, dtype=float)
        degrees = signs * np.degrees(np.asarray(q, dtype=float)) + offsets
        return dict(zip(self.config.joint_names, (float(v) for v in degrees), strict=True))

    def degrees_to_joints(self, degrees: dict[str, float]) -> np.ndarray:
        """Convert per-motor degrees from the bus back to kinematic joint angles."""
        offsets = np.asarray(self.config.servo_offsets_deg, dtype=float)
        signs = np.asarray(self.config.servo_signs, dtype=float)
        raw = np.array([float(degrees[name]) for name in self.config.joint_names])
        return np.asarray(np.radians((raw - offsets) / signs), dtype=float)

    # ------------------------------------------------------------------ motion

    def get_joint_positions(self) -> np.ndarray:
        """Read the current joint angles in radians.

        Raises:
            ArmError: If the arm is not connected or the read failed.
        """
        self._require_connected()
        try:
            present = self.bus.sync_read("Present_Position", self.config.joint_names)
        except Exception as exc:  # noqa: BLE001 - the SDK raises many types
            raise ArmError(f"could not read joint positions: {exc}") from exc
        return self.degrees_to_joints(present)

    def move_to_joints(self, q: JointVector, duration: float | None = None) -> None:
        """Stream a smoothed trajectory to the servos.

        Setpoints are interpolated and written at :data:`STREAM_PERIOD_S` rather
        than sent as a single goal, so the arm follows a controlled profile instead
        of slewing at whatever rate the servo's internal controller picks.

        Args:
            q: Target joint angles in radians.
            duration: Seconds the motion should take, or ``None`` for the
                configured default.

        Raises:
            JointLimitError: If ``q`` violates the configured joint limits.
            ArmError: If the arm is not connected or a write failed.
        """
        self._require_connected()
        target = self.validate_joints(q)
        start = self.get_joint_positions()
        seconds = self.config.default_move_duration if duration is None else float(duration)
        seconds = max(seconds, STREAM_PERIOD_S)

        steps = max(1, int(round(seconds / STREAM_PERIOD_S)))
        for waypoint in interpolate(start, target, steps):
            self._write_goal(self.joints_to_degrees(waypoint))
            time.sleep(STREAM_PERIOD_S)

        self._write_goal(self.joints_to_degrees(target))

    def _write_goal(self, degrees: dict[str, float]) -> None:
        """Write one set of goal positions to the bus.

        Raises:
            ArmError: If the write failed.
        """
        try:
            self.bus.sync_write("Goal_Position", degrees)
        except Exception as exc:  # noqa: BLE001 - the SDK raises many types
            raise ArmError(f"could not write goal positions: {exc}") from exc

    # ----------------------------------------------------------------- gripper

    def open_gripper(self) -> None:
        """Open the gripper to its configured open width."""
        self._set_gripper(GRIPPER_OPEN)
        self._gripper_open = True

    def close_gripper(self) -> None:
        """Close the gripper onto whatever sits between the fingers."""
        self._set_gripper(GRIPPER_CLOSED)
        self._gripper_open = False

    def _set_gripper(self, value: float) -> None:
        """Command the gripper servo and wait for it to settle.

        Raises:
            ArmError: If the arm is not connected or the write failed.
        """
        self._require_connected()
        try:
            self.bus.sync_write("Goal_Position", {self.gripper_name: value})
        except Exception as exc:  # noqa: BLE001 - the SDK raises many types
            raise ArmError(f"could not command the gripper: {exc}") from exc
        time.sleep(GRIPPER_SETTLE_S)

    @property
    def gripper_is_open(self) -> bool:
        """Whether the last gripper command was an open."""
        return self._gripper_open

    def read_gripper(self) -> float:
        """Read the gripper's normalised position, 0 (closed) to 100 (open).

        Raises:
            ArmError: If the arm is not connected or the read failed.
        """
        self._require_connected()
        try:
            return float(
                self.bus.sync_read("Present_Position", [self.gripper_name])[self.gripper_name]
            )
        except Exception as exc:  # noqa: BLE001 - the SDK raises many types
            raise ArmError(f"could not read the gripper position: {exc}") from exc
