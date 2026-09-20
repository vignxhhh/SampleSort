"""Tests for the real-hardware HAL.

No physical arm or camera is present, so the serial bus and the capture device
are replaced with fakes. What these tests pin down is the logic that sits between
SampleSort and the hardware — unit and frame conversions, joint-limit
enforcement, calibration loading, and the error messages an operator will
actually see — which is where the bugs that matter live.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pytest

from samplesort.config import ArmConfig, SampleSortConfig
from samplesort.hal.arm import ArmError, JointLimitError
from samplesort.hal.camera import CameraError
from samplesort.hal.real_arm import GRIPPER_CLOSED, GRIPPER_OPEN, RealArm
from samplesort.hal.real_camera import RealCamera


class FakeBus:
    """Stands in for LeRobot's ``FeetechMotorsBus``, recording what it was told."""

    def __init__(self, joint_names: list[str], gripper_name: str = "gripper") -> None:
        """Start disconnected with every motor at zero."""
        self.joint_names = joint_names
        self.gripper_name = gripper_name
        self.positions: dict[str, float] = dict.fromkeys([*joint_names, gripper_name], 0.0)
        self.connected = False
        self.torque_enabled = False
        self.configured = False
        self.writes: list[dict[str, float]] = []

    def connect(self) -> None:
        self.connected = True

    def disconnect(self, disable_torque: bool = True) -> None:
        self.connected = False
        self.torque_enabled = not disable_torque

    def configure_motors(self) -> None:
        self.configured = True

    def enable_torque(self, motors: Any = None, num_retry: int = 0) -> None:
        self.torque_enabled = True

    def sync_read(self, data_name: str, motors: Any = None) -> dict[str, float]:
        assert data_name == "Present_Position"
        names = motors if motors is not None else list(self.positions)
        if isinstance(names, str):
            names = [names]
        return {name: self.positions[name] for name in names}

    def sync_write(self, data_name: str, values: dict[str, float]) -> None:
        assert data_name == "Goal_Position"
        self.writes.append(dict(values))
        self.positions.update(values)


class BrokenBus(FakeBus):
    """A bus whose every operation fails, as a dead serial link would."""

    def connect(self) -> None:
        raise OSError("could not open /dev/ttyACM0")


@pytest.fixture
def real_config(config: SampleSortConfig) -> SampleSortConfig:
    """The shipped config switched to real mode."""
    return config.model_copy(update={"mode": "real"})


@pytest.fixture
def fake_bus(config: SampleSortConfig) -> FakeBus:
    """A fake bus wired to the configured joint names."""
    return FakeBus(list(config.arm.joint_names))


@pytest.fixture
def arm(config: SampleSortConfig, fake_bus: FakeBus) -> RealArm:
    """A connected ``RealArm`` backed by the fake bus."""
    built = RealArm(config.arm, bus=fake_bus)
    built.connect()
    return built


# ------------------------------------------------------------------- lifecycle


def test_connect_configures_and_enables_torque(arm: RealArm, fake_bus: FakeBus) -> None:
    assert arm.is_connected
    assert fake_bus.connected
    assert fake_bus.configured
    assert fake_bus.torque_enabled


def test_connect_is_idempotent(arm: RealArm, fake_bus: FakeBus) -> None:
    arm.connect()
    assert fake_bus.connected


def test_disconnect_disables_torque(arm: RealArm, fake_bus: FakeBus) -> None:
    arm.disconnect()
    assert not arm.is_connected
    assert not fake_bus.connected
    assert not fake_bus.torque_enabled


def test_commands_before_connect_are_refused(config: SampleSortConfig, fake_bus: FakeBus) -> None:
    built = RealArm(config.arm, bus=fake_bus)
    with pytest.raises(ArmError, match="not connected"):
        built.get_joint_positions()


def test_connection_failure_explains_itself(config: SampleSortConfig) -> None:
    built = RealArm(config.arm, bus=BrokenBus(list(config.arm.joint_names)))
    with pytest.raises(ArmError) as excinfo:
        built.connect()
    message = str(excinfo.value)
    assert config.arm.serial_port in message
    assert "dialout" in message, "the message should name the usual Linux fix"


def test_bus_property_requires_connection(config: SampleSortConfig) -> None:
    built = RealArm(config.arm)
    with pytest.raises(ArmError, match="has not been connected"):
        _ = built.bus


# ----------------------------------------------------------------- conversions


def test_radians_to_degrees_round_trip(arm: RealArm) -> None:
    angles = np.array([0.1, -0.5, 1.2, -0.3, 0.7])
    degrees = arm.joints_to_degrees(angles)
    assert set(degrees) == set(arm.config.joint_names)
    np.testing.assert_allclose(arm.degrees_to_joints(degrees), angles, atol=1e-12)


def test_offsets_shift_the_zero(config: SampleSortConfig, fake_bus: FakeBus) -> None:
    shifted = config.arm.model_copy(update={"servo_offsets_deg": [10.0, 20.0, 0.0, 0.0, 0.0]})
    built = RealArm(shifted, bus=fake_bus)
    degrees = built.joints_to_degrees(np.zeros(5))
    assert degrees[shifted.joint_names[0]] == pytest.approx(10.0)
    assert degrees[shifted.joint_names[1]] == pytest.approx(20.0)
    np.testing.assert_allclose(built.degrees_to_joints(degrees), np.zeros(5), atol=1e-12)


def test_signs_flip_joint_direction(config: SampleSortConfig, fake_bus: FakeBus) -> None:
    flipped = config.arm.model_copy(update={"servo_signs": [-1, 1, 1, 1, 1]})
    built = RealArm(flipped, bus=fake_bus)
    degrees = built.joints_to_degrees(np.array([np.pi / 2, 0.0, 0.0, 0.0, 0.0]))
    assert degrees[flipped.joint_names[0]] == pytest.approx(-90.0)
    np.testing.assert_allclose(
        built.degrees_to_joints(degrees), [np.pi / 2, 0, 0, 0, 0], atol=1e-12
    )


def test_offsets_and_signs_compose(config: SampleSortConfig, fake_bus: FakeBus) -> None:
    tweaked = config.arm.model_copy(
        update={"servo_signs": [-1, 1, -1, 1, 1], "servo_offsets_deg": [5.0, -7.0, 3.0, 0.0, 2.0]}
    )
    built = RealArm(tweaked, bus=fake_bus)
    angles = np.array([0.2, -0.4, 0.6, -0.1, 0.3])
    np.testing.assert_allclose(
        built.degrees_to_joints(built.joints_to_degrees(angles)), angles, atol=1e-12
    )


def test_reads_are_converted_back_to_radians(arm: RealArm, fake_bus: FakeBus) -> None:
    fake_bus.positions[arm.config.joint_names[0]] = 90.0
    assert arm.get_joint_positions()[0] == pytest.approx(np.pi / 2)


def test_bad_sign_is_rejected_by_the_config(config: SampleSortConfig) -> None:
    data = config.arm.model_dump()
    data["servo_signs"] = [2, 1, 1, 1, 1]
    with pytest.raises(ValueError, match=r"must be \+1 or -1"):
        ArmConfig.model_validate(data)


def test_wrong_offset_count_is_rejected(config: SampleSortConfig) -> None:
    data = config.arm.model_dump()
    data["servo_offsets_deg"] = [0.0, 0.0]
    with pytest.raises(ValueError, match="servo_offsets_deg must have one entry per joint"):
        ArmConfig.model_validate(data)


# ---------------------------------------------------------------------- motion


def test_move_streams_setpoints_and_lands_on_target(arm: RealArm, fake_bus: FakeBus) -> None:
    target = np.array(arm.config.home_position)
    arm.move_to_joints(target, duration=0.06)

    assert len(fake_bus.writes) > 1, "the move should stream, not send one goal"
    np.testing.assert_allclose(arm.get_joint_positions(), target, atol=1e-9)


def test_streamed_path_is_monotonic(arm: RealArm, fake_bus: FakeBus) -> None:
    from tests.assertions import is_monotonic

    arm.move_to_joints(arm.config.home_position, duration=0.1)
    joint = arm.config.joint_names[1]
    path = np.array([[write[joint]] for write in fake_bus.writes if joint in write])
    assert is_monotonic(path)


def test_move_enforces_joint_limits(arm: RealArm, fake_bus: FakeBus) -> None:
    bad = list(arm.config.home_position)
    bad[1] = 99.0
    with pytest.raises(JointLimitError, match="outside its"):
        arm.move_to_joints(bad)
    assert fake_bus.writes == [], "nothing should reach the bus for an illegal target"


def test_move_rejects_the_wrong_joint_count(arm: RealArm, fake_bus: FakeBus) -> None:
    with pytest.raises(JointLimitError, match="expected 5 joint values"):
        arm.move_to_joints([0.0, 0.0])
    assert fake_bus.writes == []


def test_go_home_uses_the_configured_pose(arm: RealArm) -> None:
    arm.go_home(duration=0.06)
    np.testing.assert_allclose(arm.get_joint_positions(), arm.config.home_position, atol=1e-9)


# --------------------------------------------------------------------- gripper


def test_gripper_commands_write_the_expected_values(arm: RealArm, fake_bus: FakeBus) -> None:
    arm.close_gripper()
    assert fake_bus.positions["gripper"] == pytest.approx(GRIPPER_CLOSED)
    assert not arm.gripper_is_open

    arm.open_gripper()
    assert fake_bus.positions["gripper"] == pytest.approx(GRIPPER_OPEN)
    assert arm.gripper_is_open


def test_gripper_position_can_be_read_back(arm: RealArm) -> None:
    arm.open_gripper()
    assert arm.read_gripper() == pytest.approx(GRIPPER_OPEN)


# ----------------------------------------------------------------- calibration


def test_missing_calibration_says_how_to_fix_it(config: SampleSortConfig, tmp_path: Path) -> None:
    arm_config = config.arm.model_copy(update={"servo_calibration_path": tmp_path / "absent.json"})
    with pytest.raises(ArmError) as excinfo:
        RealArm(arm_config).read_calibration_file()
    message = str(excinfo.value)
    assert "no servo calibration" in message
    assert "hardware_setup" in message


def test_malformed_calibration_is_reported(config: SampleSortConfig, tmp_path: Path) -> None:
    path = tmp_path / "calibration.json"
    path.write_text("{not json", encoding="utf-8")
    arm_config = config.arm.model_copy(update={"servo_calibration_path": path})
    with pytest.raises(ArmError, match="could not read the servo calibration"):
        RealArm(arm_config).read_calibration_file()


def test_valid_calibration_loads(config: SampleSortConfig, tmp_path: Path) -> None:
    path = tmp_path / "calibration.json"
    payload = {
        name: {
            "id": index + 1,
            "drive_mode": 0,
            "homing_offset": 0,
            "range_min": 0,
            "range_max": 4095,
        }
        for index, name in enumerate([*config.arm.joint_names, "gripper"])
    }
    path.write_text(json.dumps(payload), encoding="utf-8")

    arm_config = config.arm.model_copy(update={"servo_calibration_path": path})
    calibration = RealArm(arm_config).read_calibration_file()
    assert set(calibration) == set(payload)
    assert calibration[config.arm.joint_names[0]]["range_max"] == 4095


def test_calibration_missing_fields_is_reported(config: SampleSortConfig, tmp_path: Path) -> None:
    path = tmp_path / "calibration.json"
    path.write_text(json.dumps({"base_yaw": {"id": 1}}), encoding="utf-8")
    arm_config = config.arm.model_copy(update={"servo_calibration_path": path})
    with pytest.raises(ArmError, match="is missing drive_mode"):
        RealArm(arm_config).read_calibration_file()


def test_empty_calibration_is_reported(config: SampleSortConfig, tmp_path: Path) -> None:
    path = tmp_path / "calibration.json"
    path.write_text("{}", encoding="utf-8")
    arm_config = config.arm.model_copy(update={"servo_calibration_path": path})
    with pytest.raises(ArmError, match="non-empty mapping"):
        RealArm(arm_config).read_calibration_file()


# ---------------------------------------------------------------- real backend


def test_factory_builds_the_real_backend(real_config: SampleSortConfig) -> None:
    from samplesort.hal import factory

    built = factory.build_backend(real_config)
    assert isinstance(built.arm, RealArm)
    assert isinstance(built.camera, RealCamera)
    assert built.world is None


# --------------------------------------------------------------------- camera


class FakeCapture:
    """Stands in for ``cv2.VideoCapture``."""

    def __init__(
        self, width: int = 640, height: int = 480, *, opened: bool = True, fail_reads: int = 0
    ) -> None:
        """Create a capture that yields a constant frame."""
        self.width = width
        self.height = height
        self._opened = opened
        self.fail_reads = fail_reads
        self.released = False
        self.reads = 0
        self.properties: dict[int, float] = {}

    def isOpened(self) -> bool:  # noqa: N802 - mirrors the OpenCV API
        return self._opened

    def set(self, prop: int, value: float) -> bool:  # noqa: A003
        self.properties[prop] = value
        if prop == cv2.CAP_PROP_FRAME_WIDTH:
            self.width = int(value)
        elif prop == cv2.CAP_PROP_FRAME_HEIGHT:
            self.height = int(value)
        return True

    def get(self, prop: int) -> float:
        if prop == cv2.CAP_PROP_FRAME_WIDTH:
            return float(self.width)
        if prop == cv2.CAP_PROP_FRAME_HEIGHT:
            return float(self.height)
        return self.properties.get(prop, 0.0)

    def read(self) -> tuple[bool, np.ndarray | None]:
        self.reads += 1
        if self.fail_reads > 0:
            self.fail_reads -= 1
            return False, None
        return True, np.full((self.height, self.width, 3), 128, dtype=np.uint8)

    def release(self) -> None:
        self.released = True


def _patch_capture(monkeypatch: pytest.MonkeyPatch, capture: FakeCapture) -> None:
    from samplesort.hal import real_camera

    monkeypatch.setattr(real_camera.cv2, "VideoCapture", lambda *args: capture)


def test_camera_opens_and_warms_up(
    config: SampleSortConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    from samplesort.hal.real_camera import WARMUP_FRAMES

    capture = FakeCapture(*config.camera.resolution)
    _patch_capture(monkeypatch, capture)

    camera = RealCamera(config.camera)
    camera.connect()
    assert camera.is_connected
    assert capture.reads == WARMUP_FRAMES, "warm-up frames should be discarded"
    camera.disconnect()
    assert capture.released


def test_camera_read_returns_a_valid_frame(
    config: SampleSortConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_capture(monkeypatch, FakeCapture(*config.camera.resolution))
    with RealCamera(config.camera) as camera:
        frame = camera.read()
    width, height = config.camera.resolution
    assert frame.shape == (height, width, 3)
    assert frame.dtype == np.uint8


def test_camera_that_will_not_open_explains_itself(
    config: SampleSortConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_capture(monkeypatch, FakeCapture(opened=False))
    with pytest.raises(CameraError, match="could not open camera index"):
        RealCamera(config.camera).connect()


def test_camera_refusing_the_resolution_is_rejected(
    config: SampleSortConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    class StubbornCapture(FakeCapture):
        """A camera that ignores the requested resolution, as many webcams do."""

        def set(self, prop: int, value: float) -> bool:  # noqa: A003
            return True

    _patch_capture(monkeypatch, StubbornCapture(320, 240))
    with pytest.raises(CameraError, match="refused"):
        RealCamera(config.camera).connect()


def test_camera_retries_a_dropped_frame(
    config: SampleSortConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    capture = FakeCapture(*config.camera.resolution)
    _patch_capture(monkeypatch, capture)
    camera = RealCamera(config.camera)
    camera.connect()

    capture.fail_reads = 2
    frame = camera.read()  # should succeed on the third attempt
    assert frame.shape == (config.camera.height, config.camera.width, 3)
    camera.disconnect()


def test_camera_gives_up_after_too_many_drops(
    config: SampleSortConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    from samplesort.hal.real_camera import READ_RETRIES

    capture = FakeCapture(*config.camera.resolution)
    _patch_capture(monkeypatch, capture)
    camera = RealCamera(config.camera)
    camera.connect()

    capture.fail_reads = READ_RETRIES + 1
    with pytest.raises(CameraError, match="returned no frame"):
        camera.read()
    camera.disconnect()


def test_camera_read_before_connect_is_refused(config: SampleSortConfig) -> None:
    with pytest.raises(CameraError, match="not connected"):
        RealCamera(config.camera).read()
