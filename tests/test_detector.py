"""Tests for the tube detector, on synthetic frames and on the simulator."""

from __future__ import annotations

from collections.abc import Iterator

import cv2
import numpy as np
import pytest

from samplesort.config import SampleSortConfig
from samplesort.hal.factory import Backend, build_backend
from samplesort.hal.sim_camera import SimCamera
from samplesort.perception.calibration import calibration_from_camera_pose
from samplesort.perception.detector import TubeDetector
from samplesort.perception.types import Detection
from tests.synthetic import CAP_BGR, synth_calibration, synth_frame, table_to_synth_pixel


@pytest.fixture
def detector(config: SampleSortConfig) -> TubeDetector:
    """A detector wired to the synthetic-frame calibration."""
    return TubeDetector(config, synth_calibration())


@pytest.fixture
def backend(config: SampleSortConfig) -> Iterator[Backend]:
    """A connected headless sim backend with the arm folded out of view."""
    built = build_backend(config, gui=False, seed=config.seed)
    with built:
        built.arm.move_to_joints(config.arm.observe_position, duration=0.8)
        yield built


# --------------------------------------------------------------- synthetic frames


def test_detects_one_cap_of_each_class(detector: TubeDetector) -> None:
    caps = [
        ("red", (0.16, 0.08)),
        ("blue", (0.18, -0.04)),
        ("green", (0.22, 0.02)),
        ("yellow", (0.20, -0.10)),
    ]
    detections = detector.detect(synth_frame(caps))
    assert len(detections) == 4
    assert sorted(d.class_label for d in detections) == sorted(label for label, _ in caps)


def test_detected_positions_match_the_drawn_positions(detector: TubeDetector) -> None:
    caps = [("red", (0.16, 0.08)), ("blue", (0.22, -0.06))]
    detections = detector.detect(synth_frame(caps))
    for label, (x, y) in caps:
        match = next(d for d in detections if d.class_label == label)
        assert match.distance_to(x, y) < 0.002


def test_an_empty_frame_yields_no_detections(detector: TubeDetector) -> None:
    assert detector.detect(synth_frame([])) == []


def test_desaturated_background_objects_are_ignored(detector: TubeDetector) -> None:
    frame = synth_frame([("red", (0.18, 0.0))])
    # Paint a rack-grey plate over part of the frame; it must not be detected.
    cv2.rectangle(frame, (40, 40), (200, 300), (160, 160, 163), -1)
    detections = detector.detect(frame)
    assert len(detections) == 1
    assert detections[0].class_label == "red"


def test_blobs_below_the_area_threshold_are_dropped(
    config: SampleSortConfig, detector: TubeDetector
) -> None:
    frame = np.full((480, 640, 3), 215, dtype=np.uint8)
    cv2.circle(frame, (300, 200), 1, CAP_BGR["red"], -1)
    assert cv2.contourArea(np.array([[[299, 199]], [[301, 199]], [[301, 201]], [[299, 201]]])) < (
        config.classes.detector.min_area_px
    )
    assert detector.detect(frame) == []


def test_blobs_above_the_area_threshold_are_dropped(detector: TubeDetector) -> None:
    frame = np.full((480, 640, 3), 215, dtype=np.uint8)
    cv2.rectangle(frame, (20, 20), (620, 460), CAP_BGR["blue"], -1)
    assert detector.detect(frame) == []


def test_detections_are_sorted_by_confidence(detector: TubeDetector) -> None:
    caps = [("red", (0.16, 0.08)), ("blue", (0.18, -0.04)), ("green", (0.22, 0.02))]
    confidences = [d.confidence for d in detector.detect(synth_frame(caps))]
    assert confidences == sorted(confidences, reverse=True)


def test_pickup_zone_filter_excludes_outside_detections(
    config: SampleSortConfig, detector: TubeDetector
) -> None:
    zone = config.workspace.pickup_zone
    inside = ((zone.x_min + zone.x_max) / 2.0, 0.0)
    outside = (zone.x_min - 0.05, zone.y_max + 0.06)
    frame = synth_frame([("red", inside), ("blue", outside)])

    assert len(detector.detect(frame)) == 2
    filtered = detector.detect_in_pickup_zone(frame)
    assert len(filtered) == 1
    assert filtered[0].class_label == "red"


def test_detector_without_calibration_reports_nan_positions(config: SampleSortConfig) -> None:
    uncalibrated = TubeDetector(config, calibration=None)
    detections = uncalibrated.detect(synth_frame([("red", (0.18, 0.0))]))
    assert len(detections) == 1
    assert not np.isfinite(detections[0].table_xy).any()
    # And such detections must never survive the pickup-zone filter.
    assert uncalibrated.detect_in_pickup_zone(synth_frame([("red", (0.18, 0.0))])) == []


def test_preprocess_rejects_a_non_bgr_frame(detector: TubeDetector) -> None:
    with pytest.raises(ValueError, match="3-channel BGR frame"):
        detector.preprocess(np.zeros((10, 10), dtype=np.uint8))


def test_annotate_returns_a_new_image(detector: TubeDetector) -> None:
    frame = synth_frame([("red", (0.18, 0.0))])
    original = frame.copy()
    annotated = detector.annotate(frame, detector.detect(frame))
    assert annotated.shape == frame.shape
    np.testing.assert_array_equal(frame, original)
    assert not np.array_equal(annotated, frame)


def test_detection_helpers() -> None:
    detection = Detection(
        pixel_xy=(10.0, 20.0), table_xy=(0.2, 0.1), class_label="red", confidence=0.8
    )
    np.testing.assert_allclose(detection.position, [0.2, 0.1])
    assert detection.distance_to(0.2, 0.1) == pytest.approx(0.0)
    assert detection.distance_to(0.2, 0.2) == pytest.approx(0.1)
    assert "red" in detection.describe()


# ------------------------------------------------------------- simulated frames


def test_detects_every_spawned_tube_in_sim(config: SampleSortConfig, backend: Backend) -> None:
    assert backend.world is not None and isinstance(backend.camera, SimCamera)
    calibration = calibration_from_camera_pose(
        config.camera, config.workspace, backend.camera.world_to_pixel
    )
    detector = TubeDetector(config, calibration)
    tubes = backend.world.spawn_tubes(8)

    detections = detector.detect_in_pickup_zone(backend.camera.read())
    assert len(detections) == len(tubes)


def test_sim_detections_match_ground_truth(config: SampleSortConfig, backend: Backend) -> None:
    assert backend.world is not None and isinstance(backend.camera, SimCamera)
    calibration = calibration_from_camera_pose(
        config.camera, config.workspace, backend.camera.world_to_pixel
    )
    detector = TubeDetector(config, calibration)
    truth = {
        tube.body_id: (tube.label, backend.world.tube_xy(tube.body_id))
        for tube in backend.world.spawn_tubes(8)
    }

    for detection in detector.detect_in_pickup_zone(backend.camera.read()):
        x, y = detection.table_xy
        body_id = min(truth, key=lambda b: np.hypot(*(np.subtract(truth[b][1], (x, y)))))
        label, (true_x, true_y) = truth[body_id]
        assert detection.class_label == label
        # Comfortably inside the gripper's grasp tolerance.
        assert np.hypot(x - true_x, y - true_y) < config.arm.gripper.grasp_tolerance_xy / 2.0


def test_sim_detection_is_deterministic(config: SampleSortConfig, backend: Backend) -> None:
    assert backend.world is not None and isinstance(backend.camera, SimCamera)
    calibration = calibration_from_camera_pose(
        config.camera, config.workspace, backend.camera.world_to_pixel
    )
    detector = TubeDetector(config, calibration)
    backend.world.spawn_tubes(6)
    frame = backend.camera.read()

    first = detector.detect(frame)
    second = detector.detect(frame)
    assert [(d.class_label, d.pixel_xy) for d in first] == [
        (d.class_label, d.pixel_xy) for d in second
    ]


# ------------------------------------------------------------ fallen or not


def _oval_cap(frame: np.ndarray, centre: tuple[int, int]) -> None:
    """A slightly oval cap outline (aspect ~1.2), below the plain aspect threshold."""
    cv2.ellipse(frame, (centre, (22, 18), 30.0), CAP_BGR["red"], -1)


def test_a_squarish_cap_with_a_body_beside_it_is_fallen(detector: TubeDetector) -> None:
    # Measured in sim: a lying tube seen obliquely showed a 1.22 outline, under
    # the 1.25 threshold, and was taken for a standing one.
    frame = synth_frame([])
    centre = (300, 240)
    direction = np.array([np.cos(np.radians(30.0)), np.sin(np.radians(30.0))])
    start = np.array(centre) + direction * 12
    end = np.array(centre) + direction * 50
    cv2.line(frame, tuple(int(c) for c in start), tuple(int(c) for c in end), (165, 168, 175), 15)
    _oval_cap(frame, centre)
    assert len(detector.detect_fallen(frame)) == 1
    assert detector.detect(frame) == []


def test_a_squarish_cap_with_no_body_is_standing(detector: TubeDetector) -> None:
    frame = synth_frame([])
    _oval_cap(frame, (300, 240))
    assert detector.detect_fallen(frame) == []
    assert len(detector.detect(frame)) == 1


def test_an_oval_cap_standing_in_a_rack_is_not_fallen(
    config: SampleSortConfig, detector: TubeDetector
) -> None:
    # The grey, holed rack plate looks like tube body; it must not make a
    # standing tube at the edge of the view count as fallen.
    frame = synth_frame([])
    rack = config.workspace.rack("rack_a")
    x0, x1, y0, y1 = rack.footprint
    corners = [table_to_synth_pixel(x, y) for x, y in ((x0, y0), (x1, y0), (x1, y1), (x0, y1))]
    cv2.fillConvexPoly(frame, np.round(np.array(corners)).astype(np.int32), (158, 158, 160))
    for x, y in rack.slots[1:]:
        u, v = table_to_synth_pixel(x, y)
        cv2.circle(frame, (int(u), int(v)), 13, (60, 60, 62), -1)
    u, v = table_to_synth_pixel(*rack.slots[0])
    _oval_cap(frame, (int(u), int(v)))
    assert detector.detect_fallen(frame) == []
    assert len(detector.detect(frame)) == 1
