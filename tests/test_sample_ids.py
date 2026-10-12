"""QR sample IDs: label generation, the wrist scanner, and IDs reaching the log."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest
from pydantic import ValidationError

from samplesort.config import SampleSortConfig, TubeConfig, WristCameraConfig
from samplesort.control.scripted import ScriptedController
from samplesort.hal.factory import build_backend
from samplesort.hal.scanner import IdScannerInterface, RealIdScanner, ScannerError
from samplesort.logging_db.sort_log import SortLog
from samplesort.pipeline import SortPipeline
from samplesort.planning.kinematics import inverse_kinematics
from samplesort.planning.task_planner import PickPlaceJob
from samplesort.sim.assets.labels import qr_image, write_label_assets

# ----------------------------------------------------------------- label assets


def test_qr_image_round_trips_through_the_decoder() -> None:
    image = qr_image("SS-0042")
    canvas = np.full((image.shape[0] + 80, image.shape[1] + 80, 3), 255, dtype=np.uint8)
    canvas[40:-40, 40:-40] = image
    text, _, _ = cv2.QRCodeDetector().detectAndDecode(canvas)
    assert text == "SS-0042"


def test_qr_image_uses_the_smallest_code_for_a_short_id() -> None:
    # 21 modules plus a 2-module quiet zone each side, upscaled 12x.
    assert qr_image("SS-0001").shape[:2] == (25 * 12, 25 * 12)


def test_qr_image_rejects_an_empty_payload() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        qr_image("")


def test_label_assets_are_double_sided_and_textured(tmp_path: Path) -> None:
    mesh, texture = write_label_assets("SS-0007", tmp_path)
    faces = [line for line in mesh.read_text().splitlines() if line.startswith("f ")]
    # Two triangles, each present in both windings, so the quad is never culled.
    assert len(faces) == 4
    assert cv2.imread(str(texture)) is not None


def test_label_assets_sanitise_the_file_name(tmp_path: Path) -> None:
    _, texture = write_label_assets("A/B C", tmp_path)
    assert texture.parent == tmp_path
    assert texture.name == "label_A_B_C.png"


def test_a_label_too_large_for_the_cap_is_rejected() -> None:
    with pytest.raises(ValidationError, match="label_side"):
        TubeConfig(
            body_radius=0.008,
            body_height=0.04,
            cap_radius=0.0095,
            cap_height=0.012,
            min_separation=0.04,
            mass=0.015,
            label_side=0.014,
        )


# ------------------------------------------------------------------ sim world


def test_tubes_get_sequential_ids_when_qr_is_enabled(config: SampleSortConfig) -> None:
    backend = build_backend(config, seed=42)
    with backend:
        assert backend.world is not None
        ids = [tube.sample_id for tube in backend.world.spawn_tubes(4)]
    assert ids == ["SS-0001", "SS-0002", "SS-0003", "SS-0004"]


def test_tubes_have_no_id_when_qr_is_disabled(config: SampleSortConfig) -> None:
    config.classes.qr_enabled = False
    backend = build_backend(config, seed=42)
    with backend:
        assert backend.world is not None
        tubes = backend.world.spawn_tubes(2)
    assert [tube.sample_id for tube in tubes] == [None, None]


def test_the_id_counter_restarts_with_the_world(config: SampleSortConfig) -> None:
    backend = build_backend(config, seed=42)
    with backend:
        assert backend.world is not None
        backend.world.spawn_tubes(2)
        backend.world.reset()
        assert backend.world.spawn_tubes(1)[0].sample_id == "SS-0001"


def _pre_grasp_over(config: SampleSortConfig, xy: tuple[float, float]) -> np.ndarray:
    job = PickPlaceJob(
        class_label="red", pick_xy=xy, rack_id="rack_a", slot_index=0, place_xy=(0.1, 0.2)
    )
    ws = config.workspace
    return inverse_kinematics(
        config.arm, job.pick_pose(ws.table_grasp_height + ws.approach_clearance)
    )


def test_sim_scanner_reads_the_tube_under_the_gripper(config: SampleSortConfig) -> None:
    backend = build_backend(config, seed=42)
    with backend:
        world, scanner = backend.world, backend.scanner
        assert world is not None and scanner is not None
        tubes = world.spawn_tubes(3)
        for tube in tubes:
            world.reset_arm(_pre_grasp_over(config, world.tube_xy(tube.body_id)))
            world.step(1)
            assert scanner.scan() == tube.sample_id


def test_sim_scanner_reads_nothing_over_an_empty_table(config: SampleSortConfig) -> None:
    backend = build_backend(config, seed=42)
    with backend:
        assert backend.world is not None and backend.scanner is not None
        backend.world.reset_arm(_pre_grasp_over(config, (0.19, 0.0)))
        backend.world.step(1)
        assert backend.scanner.scan() is None


def test_a_disconnected_scanner_refuses_to_scan(config: SampleSortConfig) -> None:
    backend = build_backend(config, seed=42)
    assert backend.scanner is not None
    with pytest.raises(ScannerError, match="not connected"):
        backend.scanner.scan()


# ------------------------------------------------------------- choosing a code


class _FrameScanner(IdScannerInterface):
    """A scanner that returns a fixed frame, for testing the shared logic."""

    def __init__(self, frame: np.ndarray | None) -> None:
        super().__init__(WristCameraConfig())
        self.frame = frame

    def connect(self) -> None:
        self._connected = True

    def disconnect(self) -> None:
        self._connected = False

    def capture(self) -> np.ndarray:
        if self.frame is None:
            raise ScannerError("no frame")
        return self.frame


def _two_codes_frame() -> np.ndarray:
    canvas = np.full((480, 640), 255, dtype=np.uint8)
    for text, (top, left) in (("EDGE-1", (20, 20)), ("CENTRE-2", (170, 250))):
        code = cv2.QRCodeEncoder.create().encode(text)
        canvas[top : top + 140, left : left + 140] = cv2.resize(
            code, (140, 140), interpolation=cv2.INTER_NEAREST
        )
    return cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR)


def test_the_code_nearest_the_centre_wins() -> None:
    scanner = _FrameScanner(_two_codes_frame())
    scanner.connect()
    assert scanner.scan() == "CENTRE-2"


def test_a_failed_capture_is_recorded_as_no_id() -> None:
    scanner = _FrameScanner(None)
    scanner.connect()
    assert scanner.scan() is None


# --------------------------------------------------------------- real scanner


class _FakeCapture:
    def __init__(self, opened: bool = True, frame: np.ndarray | None = None) -> None:
        self.opened = opened
        self.frame = frame
        self.released = False
        self.props: dict[int, float] = {}

    def isOpened(self) -> bool:  # noqa: N802 - mirrors cv2.VideoCapture
        return self.opened

    def set(self, prop: int, value: float) -> bool:
        self.props[prop] = value
        return True

    def read(self) -> tuple[bool, np.ndarray | None]:
        return (self.frame is not None, self.frame)

    def release(self) -> None:
        self.released = True


def test_real_scanner_needs_a_camera_index() -> None:
    with pytest.raises(ScannerError, match="index"):
        RealIdScanner(WristCameraConfig(index=None))


def test_real_scanner_reports_a_camera_that_will_not_open() -> None:
    scanner = RealIdScanner(
        WristCameraConfig(index=1), capture_factory=lambda _: _FakeCapture(False)
    )
    with pytest.raises(ScannerError, match="could not open"):
        scanner.connect()


def test_real_scanner_decodes_frames_and_releases_the_device() -> None:
    fake = _FakeCapture(frame=_two_codes_frame())
    scanner = RealIdScanner(WristCameraConfig(index=1, width=640, height=480), lambda _: fake)
    with scanner:
        assert scanner.scan() == "CENTRE-2"
        assert fake.props[cv2.CAP_PROP_FRAME_WIDTH] == 640
    assert fake.released


def test_real_scanner_treats_a_dropped_frame_as_no_id() -> None:
    scanner = RealIdScanner(WristCameraConfig(index=1), lambda _: _FakeCapture(frame=None))
    with scanner:
        assert scanner.scan() is None


# ------------------------------------------------------ controller and pipeline


class _FixedScanner(_FrameScanner):
    def __init__(self, text: str | None) -> None:
        super().__init__(np.zeros((8, 8, 3), dtype=np.uint8))
        self.text = text
        self.calls = 0

    def scan(self) -> str | None:
        self.calls += 1
        return self.text


def test_controller_scans_only_when_qr_is_enabled(config: SampleSortConfig) -> None:
    backend = build_backend(config, seed=42)
    with backend:
        assert backend.world is not None
        tube = backend.world.spawn_tubes(1)[0]
        scanner = _FixedScanner("FROM-WRIST")
        scanner.connect()
        job = PickPlaceJob(
            class_label=tube.label,
            pick_xy=backend.world.tube_xy(tube.body_id),
            rack_id=config.classes.by_label(tube.label).rack_id,
            slot_index=0,
            place_xy=tuple(
                config.workspace.rack(config.classes.by_label(tube.label).rack_id).slots[0]
            ),
        )
        controller = ScriptedController(config, backend.arm, backend.world, scanner=scanner)
        assert controller.execute(job).sample_id == "FROM-WRIST"

        config.classes.qr_enabled = False
        assert controller.execute(job).sample_id is None
        assert scanner.calls == 1


def test_scanned_ids_reach_the_sort_log(config: SampleSortConfig, tmp_path: Path) -> None:
    backend = build_backend(config, seed=42)
    with backend, SortLog(tmp_path / "log.db") as log:
        assert backend.world is not None
        truth = {tube.sample_id: tube.label for tube in backend.world.spawn_tubes(3)}
        report = SortPipeline(config, backend, sort_log=log).run()
        records = log.query()
    assert report.succeeded == 3
    assert {record.sample_id for record in records} == set(truth)
    for record in records:
        assert truth[record.sample_id] == record.class_label
