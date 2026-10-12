"""Tests for the HAL contracts and their PyBullet implementations."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest

from samplesort.config import SampleSortConfig
from samplesort.hal.arm import ArmError, JointLimitError
from samplesort.hal.camera import CameraError
from samplesort.hal.factory import Backend, build_backend
from samplesort.hal.sim_arm import SimArm
from samplesort.hal.sim_camera import SimCamera
from samplesort.sim.world import SimWorld


@pytest.fixture
def backend(config: SampleSortConfig) -> Iterator[Backend]:
    """A connected headless sim backend, torn down after the test."""
    built = build_backend(config, gui=False, seed=config.seed)
    with built:
        yield built


def test_factory_builds_sim_backend(config: SampleSortConfig) -> None:
    built = build_backend(config)
    assert isinstance(built.arm, SimArm)
    assert isinstance(built.camera, SimCamera)
    assert isinstance(built.world, SimWorld)


def test_factory_rejects_unknown_mode(config: SampleSortConfig) -> None:
    broken = config.model_copy(update={"mode": "holodeck"})
    with pytest.raises(ValueError, match="unknown mode"):
        build_backend(broken)


def test_arm_reports_connection_state(config: SampleSortConfig) -> None:
    world = SimWorld(config, seed=config.seed)
    arm = SimArm(config.arm, world)
    assert not arm.is_connected
    with pytest.raises(ArmError, match="not connected"):
        arm.get_joint_positions()
    arm.connect()
    assert arm.is_connected
    arm.disconnect()
    world.disconnect()
    assert not arm.is_connected


def test_camera_requires_connection(config: SampleSortConfig) -> None:
    world = SimWorld(config, seed=config.seed)
    camera = SimCamera(config.camera, world)
    with pytest.raises(CameraError, match="not connected"):
        camera.read()
    world.disconnect()


def test_arm_starts_at_home(backend: Backend) -> None:
    q = backend.arm.get_joint_positions()
    expected = np.asarray(backend.arm.config.home_position)
    assert np.allclose(q, expected, atol=0.05)


def test_move_to_joints_reaches_target(backend: Backend) -> None:
    target = np.asarray(backend.arm.config.observe_position)
    backend.arm.move_to_joints(target, duration=1.0)
    assert np.allclose(backend.arm.get_joint_positions(), target, atol=0.03)


def test_go_home_returns_to_home(backend: Backend) -> None:
    backend.arm.move_to_joints(backend.arm.config.observe_position, duration=0.6)
    backend.arm.go_home(duration=0.8)
    assert np.allclose(
        backend.arm.get_joint_positions(), backend.arm.config.home_position, atol=0.03
    )


def test_joint_limits_are_enforced(backend: Backend) -> None:
    bad = list(backend.arm.config.home_position)
    bad[1] = 99.0
    with pytest.raises(JointLimitError, match="outside its"):
        backend.arm.move_to_joints(bad)


def test_wrong_joint_count_is_rejected(backend: Backend) -> None:
    with pytest.raises(JointLimitError, match="expected 5 joint values"):
        backend.arm.move_to_joints([0.0, 0.0])


def test_non_finite_joints_are_rejected(backend: Backend) -> None:
    bad = list(backend.arm.config.home_position)
    bad[0] = float("nan")
    with pytest.raises(JointLimitError, match="non-finite"):
        backend.arm.move_to_joints(bad)


def test_clamp_to_limits(backend: Backend) -> None:
    clamped = backend.arm.clamp_to_limits([99.0, -99.0, 0.0, 0.0, 0.0])
    for value, (low, high) in zip(clamped, backend.arm.config.joint_limits, strict=True):
        assert low <= value <= high


def test_camera_frame_shape_and_dtype(backend: Backend) -> None:
    frame = backend.camera.read()
    width, height = backend.camera.resolution
    assert frame.shape == (height, width, 3)
    assert frame.dtype == np.uint8


def test_camera_projects_world_points_into_frame(backend: Backend) -> None:
    assert backend.world is not None
    world = backend.world
    camera = backend.camera
    assert isinstance(camera, SimCamera)
    width, height = camera.resolution

    tubes = world.spawn_tubes(4)
    for tube in tubes:
        u, v = camera.world_to_pixel(world.tube_cap_position(tube.body_id))
        assert 0 <= u < width, f"{tube.label} projected outside the frame horizontally"
        assert 0 <= v < height, f"{tube.label} projected outside the frame vertically"


def test_camera_up_axis_points_along_plus_x(backend: Backend) -> None:
    camera = backend.camera
    assert isinstance(camera, SimCamera)
    near = camera.world_to_pixel(np.array([0.15, 0.0, 0.0]))
    far = camera.world_to_pixel(np.array([0.25, 0.0, 0.0]))
    # Larger X must render higher up the image, i.e. at a smaller row index.
    assert far[1] < near[1]


def test_spawned_tubes_respect_min_separation(backend: Backend) -> None:
    assert backend.world is not None
    tubes = backend.world.spawn_tubes(8)
    points = [backend.world.tube_xy(t.body_id) for t in tubes]
    min_sep = backend.world.config.workspace.tube.min_separation
    for i, a in enumerate(points):
        for b in points[i + 1 :]:
            assert np.hypot(a[0] - b[0], a[1] - b[1]) >= min_sep - 1e-6


def test_spawned_tubes_land_in_the_pickup_zone(backend: Backend) -> None:
    assert backend.world is not None
    zone = backend.world.config.workspace.pickup_zone
    for tube in backend.world.spawn_tubes(6):
        x, y = backend.world.tube_xy(tube.body_id)
        assert zone.contains(x, y)


def test_spawned_tubes_stand_upright(backend: Backend) -> None:
    assert backend.world is not None
    for tube in backend.world.spawn_tubes(6):
        assert backend.world.tube_is_upright(tube.body_id)


def test_overfilling_the_pickup_zone_raises(backend: Backend) -> None:
    assert backend.world is not None
    with pytest.raises(RuntimeError, match="pickup zone holds at most"):
        backend.world.spawn_tubes(backend.world.pickup_capacity + 1)


def test_same_seed_gives_the_same_layout(config: SampleSortConfig) -> None:
    def layout(seed: int) -> list[tuple[str, tuple[float, float]]]:
        with SimWorld(config, seed=seed) as world:
            return [(t.label, world.tube_xy(t.body_id)) for t in world.spawn_tubes(6)]

    first = layout(7)
    assert layout(7) == first
    assert layout(8) != first


def test_labels_cover_every_class_when_count_allows(backend: Backend) -> None:
    assert backend.world is not None
    labels = {t.label for t in backend.world.spawn_tubes(8)}
    assert labels == set(backend.world.config.classes.labels)


def test_generated_urdf_matches_the_config(config: SampleSortConfig, tmp_path: Path) -> None:
    from samplesort.sim.assets.arm_builder import write_arm_urdf

    path = write_arm_urdf(config.arm, tmp_path)
    urdf = path.read_text()

    assert path.name == f"{config.arm.name}_generated.urdf"
    assert urdf.startswith("<?xml")
    # Every configured joint appears, with its configured limits.
    for name, (low, high) in zip(config.arm.joint_names, config.arm.joint_limits, strict=True):
        assert f'<joint name="{name}" type="revolute">' in urdf
        assert f'lower="{low:.6f}" upper="{high:.6f}"' in urdf
    # Link lengths come from the config, not from constants in the builder.
    for length in config.arm.link_lengths:
        assert f"{length:.6f}" in urdf


def test_urdf_falls_back_when_the_package_directory_is_read_only(
    config: SampleSortConfig, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A non-editable install has a read-only package directory.

    Permissions cannot be used to provoke this in the test suite, because the
    suite may run as root, so the failing write is simulated directly.
    """
    import tempfile

    from samplesort.sim.assets import arm_builder

    assets_dir = Path(arm_builder.__file__).resolve().parent
    real_write_text = Path.write_text

    def refuse_in_package(self: Path, *args: object, **kwargs: object) -> int:
        if self.parent == assets_dir:
            raise OSError(30, "Read-only file system")
        return real_write_text(self, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(Path, "write_text", refuse_in_package)
    monkeypatch.setattr(tempfile, "gettempdir", lambda: str(tmp_path))

    path = arm_builder.write_arm_urdf(config.arm)
    assert path.is_file()
    assert tmp_path in path.parents
    assert path.read_text().startswith("<?xml")


def test_sim_world_connects_with_a_read_only_package_directory(
    config: SampleSortConfig, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The whole simulation must still start from a read-only install."""
    import tempfile

    from samplesort.sim.assets import arm_builder

    assets_dir = Path(arm_builder.__file__).resolve().parent
    real_write_text = Path.write_text

    def refuse_in_package(self: Path, *args: object, **kwargs: object) -> int:
        if self.parent == assets_dir:
            raise OSError(30, "Read-only file system")
        return real_write_text(self, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(Path, "write_text", refuse_in_package)
    monkeypatch.setattr(tempfile, "gettempdir", lambda: str(tmp_path))

    with SimWorld(config, seed=1) as world:
        assert world.arm_id >= 0
        assert len(world.spawn_tubes(2)) == 2
