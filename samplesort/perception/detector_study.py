"""How well do the cap detectors hold up when the lighting changes?

Three detectors are compared on the same rendered frames:

* **HSV** — the shipped :class:`~samplesort.perception.detector.TubeDetector`,
  whose thresholds were tuned under the default sim lighting.
* **HSV + gray-world** — the same detector on frames first corrected with
  :func:`~samplesort.perception.learned_detector.gray_world`. The learned model
  normalises its input the same way, so without this arm a gain could not be
  credited to learning rather than to normalising.
* **Learned** — :class:`~samplesort.perception.learned_detector.LearnedDetector`,
  trained here on scenes rendered under randomised lighting.

Training and test scenes come from disjoint seed ranges. The test conditions
include two that lie outside the training distribution (``very_dim`` and
``magenta_cast``), so the study also says something about generalisation, not
only about interpolation between lightings the model has seen.

A detection counts as correct when it lands within :data:`MATCH_RADIUS_M` of a
standing tube on the table and names that tube's class. Knocked-over tubes are
in some scenes; they are not targets, so a detection on one is a false positive.
"""

from __future__ import annotations

import json
import logging
import math
import time
from collections.abc import Callable, Iterator
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from samplesort.config import SampleSortConfig
from samplesort.hal.factory import build_backend
from samplesort.hal.sim_camera import Lighting, SimCamera
from samplesort.perception.calibration import Calibration, calibration_from_camera_pose
from samplesort.perception.detector import TubeDetector
from samplesort.perception.learned_detector import (
    INPUT_SCALE,
    OUTPUT_STRIDE,
    LearnedDetector,
    build_model,
    checkpoint_meta,
    encode_targets,
    gray_world,
    prepare_input,
    save_checkpoint,
)
from samplesort.perception.types import Detection

logger = logging.getLogger(__name__)

#: A detection within this distance of a standing tube (on the table) matches it.
MATCH_RADIUS_M = 0.008

#: Training seeds start here; test seeds start at :data:`TEST_SEED_BASE`.
TRAIN_SEED_BASE = 1000
TEST_SEED_BASE = 5000

#: Fraction of scenes in which one tube is knocked over.
KNOCKED_FRACTION = 0.3

#: Default output filenames.
RESULTS_JSON = "detector_study.json"
RESULTS_MD = "detector_study.md"

#: Conditions inside the training distribution, then the held-out ones.
CONDITIONS: dict[str, Lighting] = {
    "nominal": Lighting(),
    "bright": Lighting(exposure=1.45),
    "dim": Lighting(exposure=0.55),
    "warm": Lighting(color=(1.0, 0.85, 0.62), white_balance=(1.0, 0.97, 0.9)),
    "cool": Lighting(color=(0.72, 0.86, 1.0), white_balance=(0.92, 1.0, 1.08)),
    "low_sun": Lighting(
        direction=(1.0, 0.3, 0.35), ambient=0.35, diffuse=0.65, shadow=True, exposure=0.9
    ),
    "vignette": Lighting(vignette=0.6, exposure=0.7),
    # Held out: each lies outside a training range of random_lighting().
    "very_dim": Lighting(exposure=0.35, noise_std=4.0),
    "noisy_dark": Lighting(exposure=0.45, noise_std=8.0),
    "magenta_cast": Lighting(white_balance=(1.25, 0.78, 1.2)),
    "raking_warm": Lighting(
        direction=(1.0, 0.3, 0.25),
        color=(1.0, 0.8, 0.55),
        ambient=0.25,
        diffuse=0.75,
        shadow=True,
    ),
}

#: Conditions deliberately outside the training ranges of :func:`random_lighting`:
#: darker, noisier, a stronger colour cast, and a lower, warmer, harsher light.
HELD_OUT = ("very_dim", "noisy_dark", "magenta_cast", "raking_warm")

DETECTORS = ("hsv", "hsv_gray_world", "learned")


def random_lighting(rng: np.random.Generator) -> Lighting:
    """Draw a training lighting condition.

    Ranges: exposure 0.5-1.6, light colour anywhere between warm and cool, white
    balance within ±12% per channel, light elevation of at least 15°, shadows
    half the time, vignetting up to 0.4, and up to 4 levels of sensor noise.
    """
    warmth = float(rng.uniform(-1.0, 1.0))
    if warmth >= 0.0:
        color = (1.0, 1.0 - 0.15 * warmth, 1.0 - 0.38 * warmth)
    else:
        color = (1.0 + 0.28 * warmth, 1.0 + 0.14 * warmth, 1.0)
    elevation = math.radians(float(rng.uniform(15.0, 90.0)))
    azimuth = float(rng.uniform(0.0, 2.0 * math.pi))
    direction = (
        math.cos(elevation) * math.cos(azimuth),
        math.cos(elevation) * math.sin(azimuth),
        math.sin(elevation),
    )
    return Lighting(
        direction=direction,
        color=color,
        ambient=float(rng.uniform(0.3, 0.7)),
        diffuse=float(rng.uniform(0.25, 0.65)),
        specular=float(rng.uniform(0.0, 0.1)),
        shadow=bool(rng.random() < 0.5),
        exposure=float(rng.uniform(0.5, 1.6)),
        white_balance=tuple(float(g) for g in rng.uniform(0.88, 1.12, size=3)),  # type: ignore[arg-type]
        noise_std=float(rng.uniform(0.0, 4.0)),
        vignette=float(rng.uniform(0.0, 0.4)),
        seed=int(rng.integers(0, 2**31 - 1)),
    )


@dataclass
class Cap:
    """Ground truth for one standing tube."""

    pixel: tuple[float, float]
    table_xy: tuple[float, float]
    class_index: int


@dataclass
class Scene:
    """One layout of tubes, viewed under one or more lightings."""

    seed: int
    caps: list[Cap]
    frames: dict[str, np.ndarray]


def _calibration(config: SampleSortConfig, camera: SimCamera) -> Calibration:
    return calibration_from_camera_pose(config.camera, config.workspace, camera.world_to_pixel)


def render_scenes(
    config: SampleSortConfig,
    seeds: list[int],
    lightings: Callable[[int, np.random.Generator], dict[str, Lighting]],
    *,
    min_tubes: int = 3,
    max_tubes: int = 8,
) -> Iterator[Scene]:
    """Render one scene per seed under the lightings ``lightings`` returns for it.

    Each scene spawns ``min_tubes``-``max_tubes`` tubes, knocks one of them over
    in :data:`KNOCKED_FRACTION` of scenes, parks the arm at its observe pose and
    renders the frames.
    """
    labels = config.classes.labels
    tube = config.workspace.tube
    backend = build_backend(config, seed=seeds[0] if seeds else 0)
    world = backend.world
    camera = backend.camera
    assert world is not None and isinstance(camera, SimCamera)
    with backend:
        for seed in seeds:
            world.reset(seed=seed)
            rng = np.random.default_rng(seed)
            tubes = world.spawn_tubes(int(rng.integers(min_tubes, max_tubes + 1)))
            fallen: set[int] = set()
            if len(tubes) > 1 and rng.random() < KNOCKED_FRACTION:
                victim = tubes[int(rng.integers(len(tubes)))]
                world.knock_over(victim.body_id, float(rng.uniform(0.0, 2.0 * math.pi)))
                fallen.add(victim.body_id)
            world.reset_arm(config.arm.observe_position)
            world.step(config.pipeline.settle_steps)

            caps: list[Cap] = []
            for state in tubes:
                if state.body_id in fallen or not world.tube_is_upright(state.body_id):
                    continue
                top = world.tube_cap_position(state.body_id) + np.array([0, 0, tube.cap_height / 2])
                caps.append(
                    Cap(
                        pixel=camera.world_to_pixel(top),
                        table_xy=world.tube_xy(state.body_id),
                        class_index=labels.index(state.label),
                    )
                )
            frames: dict[str, np.ndarray] = {}
            for name, lighting in lightings(seed, rng).items():
                camera.lighting = lighting
                frames[name] = camera.read()
            camera.lighting = None
            yield Scene(seed=seed, caps=caps, frames=frames)


# ------------------------------------------------------------------- training


@dataclass
class TrainingSettings:
    """Knobs for :func:`train_detector`."""

    scenes: int = 300
    lightings_per_scene: int = 3
    epochs: int = 24
    batch_size: int = 16
    learning_rate: float = 3e-3
    seed: int = 0


def build_training_set(
    config: SampleSortConfig, settings: TrainingSettings
) -> tuple[list[np.ndarray], list[list[tuple[float, float, int]]]]:
    """Render training frames under random lighting, with their cap targets."""
    seeds = list(range(TRAIN_SEED_BASE, TRAIN_SEED_BASE + settings.scenes))

    def lightings(_: int, rng: np.random.Generator) -> dict[str, Lighting]:
        return {f"l{i}": random_lighting(rng) for i in range(settings.lightings_per_scene)}

    frames: list[np.ndarray] = []
    targets: list[list[tuple[float, float, int]]] = []
    for scene in render_scenes(config, seeds, lightings):
        caps = [(c.pixel[0], c.pixel[1], c.class_index) for c in scene.caps]
        for frame in scene.frames.values():
            frames.append(frame)
            targets.append(caps)
    return frames, targets


def _focal_loss(logits: Any, target: Any, torch: Any) -> Any:
    """CenterNet's penalty-reduced focal loss on Gaussian heatmaps."""
    prob = torch.sigmoid(logits).clamp(1e-4, 1 - 1e-4)
    positive = target.eq(1.0).float()
    negative = 1.0 - positive
    pos_loss = torch.log(prob) * (1 - prob) ** 2 * positive
    neg_loss = torch.log(1 - prob) * prob**2 * (1 - target) ** 4 * negative
    count = positive.sum().clamp(min=1.0)
    return -(pos_loss.sum() + neg_loss.sum()) / count


def train_detector(
    config: SampleSortConfig,
    frames: list[np.ndarray],
    targets: list[list[tuple[float, float, int]]],
    settings: TrainingSettings,
    *,
    log_every: int = 1,
) -> tuple[Any, list[float]]:
    """Train a :func:`build_model` network on CPU.

    Returns:
        The trained model (in eval mode) and the mean loss of every epoch.
    """
    from samplesort.learning._deps import require_torch

    torch = require_torch()
    torch.manual_seed(settings.seed)
    num_classes = len(config.classes.labels)
    inputs = np.stack([prepare_input(frame) for frame in frames])
    _, _, in_h, in_w = inputs.shape
    out_shape = (in_h * 2 // OUTPUT_STRIDE, in_w * 2 // OUTPUT_STRIDE)
    encoded = [encode_targets(caps, num_classes, out_shape) for caps in targets]
    heatmaps = torch.from_numpy(np.stack([e[0] for e in encoded]))
    offsets = torch.from_numpy(np.stack([e[1] for e in encoded]))
    masks = torch.from_numpy(np.stack([e[2] for e in encoded]))
    x_all = torch.from_numpy(inputs)

    model = build_model(num_classes)
    optimiser = torch.optim.AdamW(model.parameters(), lr=settings.learning_rate, weight_decay=1e-4)
    steps_per_epoch = math.ceil(len(frames) / settings.batch_size)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimiser,
        max_lr=settings.learning_rate,
        total_steps=max(1, settings.epochs * steps_per_epoch),
    )
    generator = torch.Generator().manual_seed(settings.seed)
    history: list[float] = []
    model.train()
    for epoch in range(settings.epochs):
        order = torch.randperm(len(frames), generator=generator)
        total = 0.0
        for start in range(0, len(frames), settings.batch_size):
            index = order[start : start + settings.batch_size]
            out = model(x_all[index])
            heat_loss = _focal_loss(out["heatmap"], heatmaps[index], torch)
            mask = masks[index].unsqueeze(1)
            off_loss = (torch.abs(out["offset"] - offsets[index]) * mask).sum() / mask.sum().clamp(
                min=1.0
            )
            loss = heat_loss + off_loss
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
            scheduler.step()
            total += loss.item() * len(index)
        history.append(total / len(frames))
        if log_every and (epoch + 1) % log_every == 0:
            logger.info("epoch %d/%d  loss %.4f", epoch + 1, settings.epochs, history[-1])
    model.eval()
    return model, history


# ----------------------------------------------------------------- evaluation


@dataclass
class Score:
    """Detection counts for one detector under one condition."""

    targets: int = 0
    detections: int = 0
    found: int = 0
    correct_class: int = 0

    @property
    def recall(self) -> float:
        """Standing tubes found, whatever class was named."""
        return self.found / self.targets if self.targets else 1.0

    @property
    def precision(self) -> float:
        """Detections that landed on a standing tube."""
        return self.found / self.detections if self.detections else 1.0

    @property
    def sortable(self) -> float:
        """Standing tubes found *and* named with the right class."""
        return self.correct_class / self.targets if self.targets else 1.0

    def add(self, other: Score) -> None:
        """Accumulate another score into this one."""
        self.targets += other.targets
        self.detections += other.detections
        self.found += other.found
        self.correct_class += other.correct_class


def match(detections: list[Detection], caps: list[Cap], labels: list[str]) -> Score:
    """Greedily match detections to standing tubes, nearest pairs first."""
    pairs: list[tuple[float, int, int]] = []
    for i, detection in enumerate(detections):
        for j, cap in enumerate(caps):
            distance = math.dist(detection.table_xy, cap.table_xy)
            if distance <= MATCH_RADIUS_M:
                pairs.append((distance, i, j))
    used_d: set[int] = set()
    used_c: set[int] = set()
    score = Score(targets=len(caps), detections=len(detections))
    for _, i, j in sorted(pairs):
        if i in used_d or j in used_c:
            continue
        used_d.add(i)
        used_c.add(j)
        score.found += 1
        score.correct_class += detections[i].class_label == labels[caps[j].class_index]
    return score


def _in_zone(config: SampleSortConfig, detections: list[Detection]) -> list[Detection]:
    zone = config.workspace.pickup_zone
    return [d for d in detections if all(np.isfinite(d.table_xy)) and zone.contains(*d.table_xy)]


@dataclass
class StudyResult:
    """Everything the study measured."""

    conditions: list[str]
    held_out: list[str]
    test_scenes: int
    train_frames: int
    train_seconds: float
    loss_history: list[float]
    scores: dict[str, dict[str, dict[str, float]]] = field(default_factory=dict)

    def to_markdown(self) -> str:
        """Render a table of sortable-tube rates (and recall/precision) per condition."""
        lines = [
            "# Detector lighting study",
            "",
            f"{self.test_scenes} test scenes per condition; learned model trained on "
            f"{self.train_frames} frames in {self.train_seconds / 60:.1f} min on CPU. "
            f"Held-out conditions (outside the training ranges): {', '.join(self.held_out)}.",
            "",
            "Each cell: tubes found with the right class / recall / precision.",
            "",
            "| Condition | " + " | ".join(DETECTORS) + " |",
            "|---|" + "---|" * len(DETECTORS),
        ]
        for condition in [*self.conditions, "overall"]:
            name = f"{condition} *(held out)*" if condition in self.held_out else condition
            cells = []
            for detector in DETECTORS:
                s = self.scores[detector][condition]
                cells.append(f"{s['sortable']:.0%} / {s['recall']:.0%} / {s['precision']:.0%}")
            lines.append(f"| {name} | " + " | ".join(cells) + " |")
        return "\n".join(lines) + "\n"


def evaluate(
    config: SampleSortConfig,
    learned: LearnedDetector | None,
    *,
    scenes: int = 40,
    conditions: dict[str, Lighting] | None = None,
) -> dict[str, dict[str, Score]]:
    """Score every detector under every condition on fresh test scenes."""
    conditions = conditions if conditions is not None else CONDITIONS
    labels = config.classes.labels
    seeds = list(range(TEST_SEED_BASE, TEST_SEED_BASE + scenes))
    scores = {
        detector: {name: Score() for name in conditions}
        for detector in DETECTORS
        if detector != "learned" or learned is not None
    }
    hsv: TubeDetector | None = None
    for scene in render_scenes(config, seeds, lambda _seed, _rng: dict(conditions)):
        if hsv is None:
            hsv = _hsv_detector(config)
        for name, frame in scene.frames.items():
            results = {
                "hsv": hsv.detect_in_pickup_zone(frame),
                "hsv_gray_world": hsv.detect_in_pickup_zone(gray_world(frame)),
            }
            if learned is not None:
                results["learned"] = _in_zone(config, learned.detect(frame))
            for detector, detections in results.items():
                scores[detector][name].add(match(detections, scene.caps, labels))
    return scores


def _hsv_detector(config: SampleSortConfig) -> TubeDetector:
    calibration = shared_calibration(config)
    detector = TubeDetector(config, calibration)
    # Sample IDs are read by the wrist scanner; skip the overhead QR pass.
    detector.qr_reader.enabled = False
    return detector


def shared_calibration(config: SampleSortConfig) -> Calibration:
    """The analytic sim calibration both detectors map pixels through."""
    backend = build_backend(config, seed=0)
    camera = backend.camera
    assert isinstance(camera, SimCamera)
    with backend:
        return _calibration(config, camera)


def run_study(
    config: SampleSortConfig,
    output_dir: Path,
    *,
    settings: TrainingSettings | None = None,
    test_scenes: int = 40,
    checkpoint: Path | None = None,
) -> StudyResult:
    """Train (or load) the learned detector, evaluate all three, write the results.

    Args:
        config: The configuration bundle.
        output_dir: Where ``detector_study.json`` and ``.md`` are written.
        settings: Training settings; ignored when ``checkpoint`` exists.
        test_scenes: Test layouts, each rendered under every condition.
        checkpoint: Load this checkpoint if it exists; otherwise train and save to it.
    """
    settings = settings or TrainingSettings()
    calibration = shared_calibration(config)
    history: list[float] = []
    train_frames = 0
    train_seconds = 0.0
    if checkpoint is not None and checkpoint.is_file():
        logger.info("loading cap detector from %s", checkpoint)
        learned = LearnedDetector.load(config, checkpoint, calibration)
        meta = checkpoint_meta(checkpoint)
        train_frames = int(meta.get("train_frames", 0))
        train_seconds = float(meta.get("train_seconds", 0.0))
        history = [float(v) for v in meta.get("loss_history", [])]
    else:
        started = time.perf_counter()
        frames, targets = build_training_set(config, settings)
        train_frames = len(frames)
        logger.info("rendered %d training frames; training", train_frames)
        model, history = train_detector(config, frames, targets, settings)
        train_seconds = time.perf_counter() - started
        learned = LearnedDetector(config=config, model=model, calibration=calibration)
        if checkpoint is not None:
            meta = {
                "train_frames": train_frames,
                "train_seconds": train_seconds,
                "loss_history": history,
            }
            save_checkpoint(model, config.classes.labels, checkpoint, meta)

    scores = evaluate(config, learned, scenes=test_scenes)
    result = StudyResult(
        conditions=list(CONDITIONS),
        held_out=list(HELD_OUT),
        test_scenes=test_scenes,
        train_frames=train_frames,
        train_seconds=train_seconds,
        loss_history=history,
    )
    for detector, by_condition in scores.items():
        overall = Score()
        result.scores[detector] = {}
        for name, score in [*by_condition.items(), ("overall", overall)]:
            if name != "overall":
                overall.add(score)
            result.scores[detector][name] = {
                **asdict(score),
                "recall": score.recall,
                "precision": score.precision,
                "sortable": score.sortable,
            }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / RESULTS_JSON).write_text(json.dumps(asdict(result), indent=2))
    (output_dir / RESULTS_MD).write_text(result.to_markdown())
    return result


def contact_sheet(
    config: SampleSortConfig, seed: int = TEST_SEED_BASE, *, labels: bool = True
) -> np.ndarray:
    """One test layout under every condition, tiled (optionally labelled), for the docs."""
    tiles: list[np.ndarray] = []
    for scene in render_scenes(config, [seed], lambda _s, _r: dict(CONDITIONS)):
        for name, frame in scene.frames.items():
            tile = cv2.resize(frame, None, fx=INPUT_SCALE, fy=INPUT_SCALE)
            if labels:
                cv2.putText(tile, name, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3)
                cv2.putText(tile, name, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            tiles.append(tile)
    while len(tiles) % 4:
        tiles.append(np.full_like(tiles[0], 255))
    rows = [np.hstack(tiles[i : i + 4]) for i in range(0, len(tiles), 4)]
    return np.vstack(rows)
