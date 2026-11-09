"""Lighting variation, the learned cap detector and the detector study."""

from __future__ import annotations

import importlib.util
import math
from pathlib import Path

import cv2
import numpy as np
import pytest

from samplesort.config import SampleSortConfig
from samplesort.hal.sim_camera import Lighting
from samplesort.perception import detector_study as study
from samplesort.perception.detector_study import (
    CONDITIONS,
    HELD_OUT,
    Cap,
    match,
    random_lighting,
    render_scenes,
)
from samplesort.perception.learned_detector import (
    OUTPUT_STRIDE,
    decode_peaks,
    encode_targets,
    gray_world,
    prepare_input,
)
from samplesort.perception.types import Detection

requires_torch = pytest.mark.skipif(
    importlib.util.find_spec("torch") is None, reason="PyTorch is not installed"
)

# ------------------------------------------------------------------- lighting


def test_camera_response_scales_and_tints() -> None:
    grey = np.full((4, 4, 3), 100, dtype=np.uint8)
    out = Lighting(exposure=1.5, white_balance=(1.0, 0.5, 2.0)).apply_camera_response(grey)
    assert out[0, 0].tolist() == [150, 75, 255]


def test_sensor_noise_is_reproducible_from_its_seed() -> None:
    grey = np.full((8, 8, 3), 100, dtype=np.uint8)
    a = Lighting(noise_std=5.0, seed=3).apply_camera_response(grey)
    b = Lighting(noise_std=5.0, seed=3).apply_camera_response(grey)
    c = Lighting(noise_std=5.0, seed=4).apply_camera_response(grey)
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)


def test_vignette_darkens_corners_but_not_the_centre() -> None:
    grey = np.full((41, 61, 3), 200, dtype=np.uint8)
    out = Lighting(vignette=0.5).apply_camera_response(grey)
    assert out[20, 30, 0] == 200
    # Pixel centres stop half a pixel short of the true corner.
    assert int(out[0, 0, 0]) == pytest.approx(100, abs=4)


def test_default_camera_render_is_unchanged_without_lighting(config: SampleSortConfig) -> None:
    from samplesort.hal.factory import build_backend

    backend = build_backend(config, seed=7)
    with backend:
        assert backend.camera.lighting is None  # type: ignore[attr-defined]
        first = backend.camera.read()
        backend.camera.lighting = Lighting(exposure=0.5)  # type: ignore[attr-defined]
        dim = backend.camera.read()
    assert dim.mean() < first.mean() * 0.6


def test_gray_world_removes_a_colour_cast() -> None:
    rng = np.random.default_rng(0)
    scene = rng.integers(90, 130, size=(32, 32, 3)).astype(np.uint8)
    cast = Lighting(white_balance=(0.6, 1.0, 1.3)).apply_camera_response(scene)
    corrected = gray_world(cast).reshape(-1, 3).mean(axis=0)
    assert np.allclose(corrected, 200.0, atol=3.0)


# ----------------------------------------------------------- study conditions


def test_training_lighting_stays_in_its_documented_ranges() -> None:
    rng = np.random.default_rng(1)
    for _ in range(200):
        light = random_lighting(rng)
        assert 0.5 <= light.exposure <= 1.6
        assert all(0.88 <= g <= 1.12 for g in light.white_balance)
        assert light.noise_std <= 4.0
        assert 0.3 <= light.ambient <= 0.7
        elevation = math.degrees(math.asin(light.direction[2] / np.linalg.norm(light.direction)))
        assert elevation >= 15.0 - 1e-9


def test_held_out_conditions_really_are_outside_training() -> None:
    def elevation(light: Lighting) -> float:
        return math.degrees(math.asin(light.direction[2] / np.linalg.norm(light.direction)))

    assert set(HELD_OUT) <= set(CONDITIONS)
    assert CONDITIONS["very_dim"].exposure < 0.5
    assert CONDITIONS["noisy_dark"].noise_std > 4.0
    assert any(not 0.88 <= g <= 1.12 for g in CONDITIONS["magenta_cast"].white_balance)
    raking = CONDITIONS["raking_warm"]
    assert elevation(raking) < 15.0 and raking.ambient < 0.3


def test_ground_truth_caps_sit_on_coloured_pixels(config: SampleSortConfig) -> None:
    scenes = list(render_scenes(config, [5000, 5001], lambda _s, _r: {"n": Lighting()}))
    for scene in scenes:
        hsv = cv2.cvtColor(scene.frames["n"], cv2.COLOR_BGR2HSV)
        assert scene.caps
        for cap in scene.caps:
            u, v = (int(round(c)) for c in cap.pixel)
            # The QR label covers the middle of the cap; its coloured rim
            # surrounds the projected centre.
            patch = hsv[v - 7 : v + 8, u - 7 : u + 8, 1]
            assert patch.max() > 100


def test_knocked_over_tubes_are_not_targets(config: SampleSortConfig) -> None:
    # Find a seed whose scene knocks a tube over, then check it is not a target.
    for seed in range(5000, 5040):
        rng = np.random.default_rng(seed)
        count = int(rng.integers(3, 9))
        if count > 1 and rng.random() < study.KNOCKED_FRACTION:
            (scene,) = render_scenes(config, [seed], lambda _s, _r: {})
            assert len(scene.caps) == count - 1
            return
    pytest.fail("no knocked-over scene in the seed range")


# ---------------------------------------------------------------- matching


def _det(xy: tuple[float, float], label: str) -> Detection:
    return Detection(pixel_xy=(0.0, 0.0), table_xy=xy, class_label=label, confidence=1.0)


def test_match_counts_found_tubes_and_correct_classes() -> None:
    labels = ["red", "blue"]
    caps = [Cap((0, 0), (0.10, 0.00), 0), Cap((0, 0), (0.20, 0.00), 1)]
    detections = [
        _det((0.104, 0.0), "red"),  # 4 mm off: found, right class
        _det((0.20, 0.003), "red"),  # found, wrong class
        _det((0.30, 0.0), "blue"),  # nothing there
    ]
    score = match(detections, caps, labels)
    assert (score.found, score.correct_class, score.detections) == (2, 1, 3)
    assert score.recall == 1.0 and score.sortable == 0.5
    assert score.precision == pytest.approx(2 / 3)


def test_match_pairs_each_tube_with_one_detection() -> None:
    caps = [Cap((0, 0), (0.1, 0.0), 0)]
    detections = [_det((0.1, 0.0), "red"), _det((0.101, 0.0), "red")]
    score = match(detections, caps, ["red"])
    assert score.found == 1 and score.precision == 0.5


def test_match_rejects_detections_beyond_the_radius() -> None:
    caps = [Cap((0, 0), (0.1, 0.0), 0)]
    far = study.MATCH_RADIUS_M + 0.001
    assert match([_det((0.1 + far, 0.0), "red")], caps, ["red"]).found == 0


# ---------------------------------------------------- targets and decoding


def test_targets_decode_back_to_the_cap_centres() -> None:
    caps = [(123.0, 211.0, 0), (402.5, 96.25, 2), (300.0, 300.0, 1)]
    heatmap, offset, mask = encode_targets(caps, 3, (60, 80))
    assert mask.sum() == 3
    found = decode_peaks(heatmap, offset, threshold=0.99)
    assert len(found) == 3
    for u, v, cls in caps:
        match_ = min(found, key=lambda f: math.dist(f[:2], (u, v)))
        assert math.dist(match_[:2], (u, v)) < 1e-3
        assert match_[2] == cls


def test_targets_ignore_caps_outside_the_frame() -> None:
    heatmap, _, mask = encode_targets([(-5.0, 10.0, 0), (10_000.0, 10.0, 0)], 1, (60, 80))
    assert mask.sum() == 0 and heatmap.max() == 0.0


def test_network_input_is_half_resolution_and_standardised() -> None:
    frame = np.random.default_rng(0).integers(0, 255, size=(480, 640, 3)).astype(np.uint8)
    x = prepare_input(frame)
    assert x.shape == (3, 240, 320)
    assert np.allclose(x.reshape(3, -1).mean(axis=1), 0.0, atol=1e-4)
    assert 480 // OUTPUT_STRIDE == 60


# ---------------------------------------------------- the learned detector


@pytest.mark.learning
@requires_torch
def test_model_output_has_the_documented_shape() -> None:
    import torch

    from samplesort.perception.learned_detector import build_model

    model = build_model(4)
    out = model(torch.zeros(2, 3, 240, 320))
    assert tuple(out["heatmap"].shape) == (2, 4, 60, 80)
    assert tuple(out["offset"].shape) == (2, 2, 60, 80)
    assert sum(p.numel() for p in model.parameters()) < 100_000


@pytest.mark.learning
@requires_torch
def test_training_reduces_the_loss_and_checkpoints_round_trip(
    config: SampleSortConfig, tmp_path: Path
) -> None:
    from samplesort.perception.learned_detector import (
        LearnedDetector,
        checkpoint_meta,
        save_checkpoint,
    )

    settings = study.TrainingSettings(scenes=6, lightings_per_scene=2, epochs=4, batch_size=4)
    frames, targets = study.build_training_set(config, settings)
    assert len(frames) == 12
    model, history = study.train_detector(config, frames, targets, settings, log_every=0)
    assert history[-1] < history[0]

    meta = {"train_frames": len(frames), "loss_history": history}
    path = save_checkpoint(model, config.classes.labels, tmp_path / "cap.pt", meta)
    assert checkpoint_meta(path) == meta
    loaded = LearnedDetector.load(config, path)
    assert loaded.predict(frames[0]) == LearnedDetector(config, model).predict(frames[0])

    other = config.model_copy(deep=True)
    other.classes.classes = other.classes.classes[:-1]
    with pytest.raises(ValueError, match="do not match"):
        LearnedDetector.load(other, path)


@pytest.mark.learning
@requires_torch
def test_learned_detections_are_mapped_onto_the_table(config: SampleSortConfig) -> None:
    from samplesort.perception.learned_detector import LearnedDetector

    class _Oracle:
        """Stands in for a perfect network: peaks exactly at the true caps."""

    calibration = study.shared_calibration(config)
    (scene,) = render_scenes(config, [5002], lambda _s, _r: {"n": Lighting()})
    detector = LearnedDetector(config, _Oracle(), calibration)
    peaks = [(c.pixel[0], c.pixel[1], c.class_index, 0.9) for c in scene.caps]
    detector.predict = lambda _frame: peaks  # type: ignore[method-assign]
    score = match(detector.detect(scene.frames["n"]), scene.caps, config.classes.labels)
    assert score.sortable == 1.0 and score.precision == 1.0


def test_cli_lists_the_detector_study() -> None:
    from typer.testing import CliRunner

    from samplesort.cli import app

    result = CliRunner().invoke(app, ["detector-study", "--help"])
    assert result.exit_code == 0
    assert "lighting" in result.output
