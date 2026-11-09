"""Record demonstration episodes into a LeRobot-format dataset.

In simulation the demonstrations are generated automatically: the scripted
controller solves a pick-and-place job, and this module replays its joint
waypoints at the dataset frame rate while capturing an observation before every
step. On real hardware the same loop records leader-arm teleoperation instead.

Observation and action spaces:

* ``observation.images.top`` — the overhead frame, downscaled to
  ``learning.image_size``, RGB ``uint8``.
* ``observation.state`` — the five arm joints plus a gripper-open fraction.
* ``action`` — the commanded next state, same six values.
"""

from __future__ import annotations

import logging
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from samplesort.config import SampleSortConfig
from samplesort.control.scripted import ScriptedController
from samplesort.control.trajectory import interpolate
from samplesort.hal.factory import Backend, build_backend
from samplesort.learning._deps import require_lerobot
from samplesort.pipeline import SortPipeline
from samplesort.planning.kinematics import UnreachableError
from samplesort.planning.task_planner import PickPlaceJob

logger = logging.getLogger(__name__)

#: Gripper-open fraction recorded alongside the joint angles.
GRIPPER_OPEN = 1.0
GRIPPER_CLOSED = 0.0

#: Stages during which the gripper should be closed.
_CLOSED_STAGES = frozenset({"lift", "pre_place", "place"})


@dataclass
class RecordingStats:
    """What a recording session produced.

    Attributes:
        episodes: Number of episodes written.
        frames: Total frames across every episode.
        dataset_dir: Where the dataset was written.
        successful_episodes: Episodes whose demonstration actually succeeded.
    """

    episodes: int
    frames: int
    dataset_dir: Path
    successful_episodes: int

    @property
    def is_empty(self) -> bool:
        """Whether nothing was recorded."""
        return self.episodes == 0


def build_features(config: SampleSortConfig) -> dict[str, dict[str, Any]]:
    """Describe the dataset's observation and action spaces for LeRobot.

    Args:
        config: The validated configuration bundle.

    Returns:
        A LeRobot ``features`` dict.
    """
    width, height = config.learning.image_size
    joint_names = [*config.arm.joint_names, "gripper"]
    dimension = len(joint_names)
    return {
        "observation.images.top": {
            "dtype": "image",
            "shape": (height, width, 3),
            "names": ["height", "width", "channels"],
        },
        "observation.state": {"dtype": "float32", "shape": (dimension,), "names": joint_names},
        "action": {"dtype": "float32", "shape": (dimension,), "names": joint_names},
    }


def observation_image(config: SampleSortConfig, frame_bgr: np.ndarray) -> np.ndarray:
    """Downscale a BGR camera frame to the dataset's RGB observation format.

    Args:
        config: Supplies ``learning.image_size``.
        frame_bgr: A BGR ``uint8`` frame from the camera.

    Returns:
        An RGB ``uint8`` array of shape ``(height, width, 3)``.
    """
    width, height = config.learning.image_size
    resized = cv2.resize(frame_bgr, (width, height), interpolation=cv2.INTER_AREA)
    return np.ascontiguousarray(cv2.cvtColor(resized, cv2.COLOR_BGR2RGB))


def observation_state(joints: np.ndarray, gripper: float) -> np.ndarray:
    """Concatenate joint angles and the gripper fraction into a state vector."""
    return np.concatenate([np.asarray(joints, dtype=np.float32), [np.float32(gripper)]])


class DemonstrationRecorder:
    """Drives the scripted controller while capturing frames into a dataset.

    Args:
        config: The validated configuration bundle.
        backend: A connected arm and camera.
        controller: The controller whose waypoints are replayed.
    """

    def __init__(
        self, config: SampleSortConfig, backend: Backend, controller: ScriptedController
    ) -> None:
        """Bind the recorder (see the class docstring for args)."""
        self.config = config
        self.backend = backend
        self.controller = controller
        self.frames: list[dict[str, Any]] = []

    def _capture(self, target: np.ndarray, gripper_now: float, gripper_next: float) -> None:
        """Record one (observation, action) pair at the current arm state."""
        joints = self.backend.arm.get_joint_positions()
        self.frames.append(
            {
                "observation.images.top": observation_image(
                    self.config, self.backend.camera.read()
                ),
                "observation.state": observation_state(joints, gripper_now),
                "action": observation_state(target, gripper_next),
                "task": self.config.learning.task_description,
            }
        )

    def demonstrate(self, job: PickPlaceJob) -> bool:
        """Replay one pick-and-place demonstration, capturing every frame.

        Args:
            job: The job to demonstrate.

        Returns:
            Whether the demonstration completed successfully.
        """
        try:
            stages = self.controller.solve_waypoints(job)
        except UnreachableError:
            logger.warning("skipping unreachable demonstration: %s", job.describe())
            return False

        steps_per_stage = max(2, self.config.learning.fps // 2)
        self.backend.arm.open_gripper()
        gripper = GRIPPER_OPEN

        for stage, target in stages:
            next_gripper = GRIPPER_CLOSED if stage in _CLOSED_STAGES else GRIPPER_OPEN
            current = self.backend.arm.get_joint_positions()
            for waypoint in interpolate(current, target, steps_per_stage):
                self._capture(waypoint, gripper, next_gripper)
                self.backend.arm.move_to_joints(waypoint, 1.0 / self.config.learning.fps)

            # The gripper changes state once the stage's pose has been reached.
            if stage == "grasp":
                self.backend.arm.close_gripper()
                gripper = GRIPPER_CLOSED
            elif stage == "place":
                self.backend.arm.open_gripper()
                gripper = GRIPPER_OPEN
                if self.backend.world is not None:
                    self.backend.world.step(self.config.pipeline.settle_steps)

        # One final frame so the episode ends on a settled observation.
        final = self.backend.arm.get_joint_positions()
        self._capture(final, gripper, gripper)

        if job.body_id is not None and self.backend.world is not None:
            x, y = self.backend.world.tube_xy(job.body_id)
            error = float(np.hypot(x - job.place_xy[0], y - job.place_xy[1]))
            return error <= ScriptedController.PLACE_TOLERANCE_M
        return True


def record_episodes(
    config: SampleSortConfig,
    *,
    episodes: int = 5,
    num_tubes: int = 3,
    dataset_dir: Path | str | None = None,
    overwrite: bool = False,
    seed: int | None = None,
    keep_failures: bool = False,
) -> RecordingStats:
    """Record demonstration episodes into a LeRobot dataset.

    One episode is one tube sorted from the pickup zone into its rack. Each
    episode uses its own seeded world, so the dataset covers a spread of layouts.

    Args:
        config: The validated configuration bundle.
        episodes: How many episodes to record.
        num_tubes: Tubes spawned per episode; the nearest one is demonstrated.
        dataset_dir: Where to write. Defaults to ``config.learning.dataset_dir``.
        overwrite: Delete an existing dataset at that path first.
        seed: First world seed; episode *i* uses ``seed + i``.
        keep_failures: Keep episodes whose demonstration failed. Off by default,
            since imitation learning from failed demonstrations teaches failure.

    Returns:
        Statistics about what was written.

    Raises:
        ValueError: If ``episodes`` or ``num_tubes`` is not positive.
        FileExistsError: If the destination exists and ``overwrite`` is ``False``.
        LearningDependencyError: If LeRobot is not installed.
    """
    if episodes < 1:
        raise ValueError(f"episodes must be at least 1, got {episodes}")
    if num_tubes < 1:
        raise ValueError(f"num_tubes must be at least 1, got {num_tubes}")

    require_lerobot()
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    destination = Path(dataset_dir or config.learning.dataset_dir)
    if destination.exists():
        if not overwrite:
            raise FileExistsError(
                f"a dataset already exists at {destination}; pass --overwrite to replace it"
            )
        shutil.rmtree(destination)

    dataset = LeRobotDataset.create(
        repo_id=config.learning.dataset_repo_id,
        fps=config.learning.fps,
        features=build_features(config),
        root=destination,
        robot_type=config.arm.name,
        # PNG frames rather than encoded video: no ffmpeg/codec dependency, and
        # these datasets are small enough that the size difference does not matter.
        use_videos=False,
    )

    base_seed = config.seed if seed is None else seed
    total_frames = 0
    written = 0
    successes = 0

    for index in range(episodes):
        episode_seed = base_seed + index
        backend = build_backend(config, gui=False, seed=episode_seed)
        with backend:
            assert backend.world is not None
            backend.world.spawn_tubes(num_tubes)

            pipeline = SortPipeline(config, backend, sort_log=_null_log())
            detections = pipeline.perceive()
            plan = pipeline.plan(detections)
            if plan.is_empty:
                logger.warning("episode %d: nothing detected, skipping", index)
                continue

            controller = ScriptedController(config, backend.arm, backend.world)
            recorder = DemonstrationRecorder(config, backend, controller)
            ok = recorder.demonstrate(plan.jobs[0])
            captured = recorder.frames

        if not ok and not keep_failures:
            logger.warning("episode %d demonstration failed; discarding it", index)
            continue

        for frame in captured:
            dataset.add_frame(frame)
        dataset.save_episode()
        written += 1
        successes += int(ok)
        total_frames += len(captured)
        logger.info("recorded episode %d/%d (%d frames)", written, episodes, len(captured))

    dataset.finalize()
    logger.info("wrote %d episode(s), %d frames to %s", written, total_frames, destination)
    return RecordingStats(
        episodes=written,
        frames=total_frames,
        dataset_dir=destination,
        successful_episodes=successes,
    )


def _null_log() -> Any:
    """An in-memory sort log; recording does not belong in the chain-of-custody record."""
    from samplesort.logging_db.sort_log import SortLog

    return SortLog(":memory:")
