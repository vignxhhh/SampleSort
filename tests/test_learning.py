"""Tests for the record → train → evaluate pipeline and the learned controller.

Per spec section 6 the goal is a *runnable* pipeline, not a competitive policy,
so these tests assert that each stage produces well-formed artefacts and that the
stages compose — not that the policy performs well.

Everything here is skipped cleanly when the optional torch/LeRobot extra is not
installed.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest

from samplesort.config import LearningConfig, SampleSortConfig
from samplesort.learning._deps import is_available, resolve_device

pytestmark = pytest.mark.learning

requires_learning = pytest.mark.skipif(
    not is_available(), reason="the optional torch/LeRobot extra is not installed"
)

#: Tiny settings so the whole pipeline runs in seconds on CPU.
TINY = {
    "fps": 10,
    "image_size": (64, 64),
    "chunk_size": 8,
    "n_action_steps": 8,
    "dim_model": 32,
    "n_heads": 2,
    "dim_feedforward": 64,
    "n_encoder_layers": 1,
    "n_decoder_layers": 1,
    "use_vae": False,
    "steps": 4,
    "batch_size": 2,
    "log_every": 2,
    "save_every": 0,
    "rollout_max_steps": 5,
}


@pytest.fixture
def tiny_config(config: SampleSortConfig) -> SampleSortConfig:
    """The shipped config with a deliberately tiny learning section."""
    smaller = config.model_copy(deep=True)
    smaller.learning = config.learning.model_copy(update=TINY)
    return smaller


@pytest.fixture(scope="module")
def recorded(request: pytest.FixtureRequest) -> Iterator[tuple[SampleSortConfig, Path]]:
    """Record a small dataset once and share it across the module's tests."""
    if not is_available():
        pytest.skip("the optional torch/LeRobot extra is not installed")

    from samplesort.config import load_config
    from samplesort.learning.record import record_episodes

    config = load_config(mode="sim", seed=42)
    config.learning = config.learning.model_copy(update=TINY)
    directory = Path(request.config.rootpath) / ".pytest_cache" / "samplesort-dataset"

    record_episodes(config, episodes=2, num_tubes=2, dataset_dir=directory, overwrite=True, seed=42)
    yield config, directory


# --------------------------------------------------------------------- config


def test_learning_config_defaults(config: SampleSortConfig) -> None:
    assert config.learning.fps > 0
    assert config.learning.n_action_steps <= config.learning.chunk_size
    assert config.learning.dim_model % config.learning.n_heads == 0


def test_action_steps_cannot_exceed_the_chunk() -> None:
    with pytest.raises(ValueError, match="cannot exceed"):
        LearningConfig(chunk_size=8, n_action_steps=16)


def test_dim_model_must_divide_by_heads() -> None:
    with pytest.raises(ValueError, match="divisible by"):
        LearningConfig(dim_model=30, n_heads=8)


# ------------------------------------------------------------ pure helpers


def test_feature_schema_matches_the_config(config: SampleSortConfig) -> None:
    from samplesort.learning.record import build_features

    features = build_features(config)
    assert set(features) == {"observation.images.top", "observation.state", "action"}

    width, height = config.learning.image_size
    assert features["observation.images.top"]["shape"] == (height, width, 3)

    dimension = config.arm.num_joints + 1  # joints plus the gripper
    assert features["observation.state"]["shape"] == (dimension,)
    assert features["action"]["shape"] == (dimension,)
    assert features["observation.state"]["names"][-1] == "gripper"


def test_observation_image_is_rgb_and_downscaled(config: SampleSortConfig) -> None:
    from samplesort.learning.record import observation_image

    # A BGR frame that is unambiguously blue; RGB output must put it last.
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    frame[:, :, 0] = 255

    image = observation_image(config, frame)
    width, height = config.learning.image_size
    assert image.shape == (height, width, 3)
    assert image.dtype == np.uint8
    assert image[0, 0, 2] == 255 and image[0, 0, 0] == 0


def test_observation_state_appends_the_gripper() -> None:
    from samplesort.learning.record import observation_state

    state = observation_state(np.array([0.1, 0.2, 0.3, 0.4, 0.5]), 1.0)
    assert state.shape == (6,)
    assert state.dtype == np.float32
    assert state[-1] == pytest.approx(1.0)


def test_resolve_device_falls_back_to_cpu() -> None:
    if not is_available():
        pytest.skip("the optional torch/LeRobot extra is not installed")
    assert resolve_device("cpu") == "cpu"
    # On a CPU-only machine this must degrade rather than raise.
    assert resolve_device("cuda") in ("cpu", "cuda")


def test_record_rejects_bad_arguments(tiny_config: SampleSortConfig) -> None:
    from samplesort.learning.record import record_episodes

    with pytest.raises(ValueError, match="episodes must be at least 1"):
        record_episodes(tiny_config, episodes=0)
    with pytest.raises(ValueError, match="num_tubes must be at least 1"):
        record_episodes(tiny_config, episodes=1, num_tubes=0)


# ------------------------------------------------------------------- record


@requires_learning
@pytest.mark.slow
def test_recording_produces_a_loadable_dataset(
    recorded: tuple[SampleSortConfig, Path],
) -> None:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    config, directory = recorded
    assert (directory / "meta" / "info.json").is_file()

    dataset = LeRobotDataset(repo_id=config.learning.dataset_repo_id, root=directory)
    assert dataset.num_episodes == 2
    assert dataset.num_frames > 0

    sample = dataset[0]
    assert set(sample) >= {"observation.images.top", "observation.state", "action"}

    width, height = config.learning.image_size
    assert tuple(sample["observation.images.top"].shape) == (3, height, width)
    assert sample["observation.state"].shape == (config.arm.num_joints + 1,)
    assert sample["action"].shape == (config.arm.num_joints + 1,)


@requires_learning
@pytest.mark.slow
def test_recorded_actions_stay_inside_the_joint_limits(
    recorded: tuple[SampleSortConfig, Path],
) -> None:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    config, directory = recorded
    dataset = LeRobotDataset(repo_id=config.learning.dataset_repo_id, root=directory)

    for index in range(min(20, dataset.num_frames)):
        action = dataset[index]["action"].numpy()
        for value, (low, high) in zip(action[:-1], config.arm.joint_limits, strict=True):
            assert low - 1e-3 <= value <= high + 1e-3
        assert 0.0 <= action[-1] <= 1.0


@requires_learning
@pytest.mark.slow
def test_recording_refuses_to_clobber_without_overwrite(
    recorded: tuple[SampleSortConfig, Path],
) -> None:
    from samplesort.learning.record import record_episodes

    config, directory = recorded
    with pytest.raises(FileExistsError, match="--overwrite"):
        record_episodes(config, episodes=1, dataset_dir=directory, overwrite=False)


# -------------------------------------------------------------------- train


@requires_learning
def test_training_on_a_missing_dataset_says_how_to_fix_it(
    tiny_config: SampleSortConfig, tmp_path: Path
) -> None:
    from samplesort.learning.train import TrainingError, load_dataset

    with pytest.raises(TrainingError, match="samplesort record"):
        load_dataset(tiny_config, tmp_path / "absent")


@requires_learning
@pytest.mark.slow
def test_policy_config_matches_the_dataset(recorded: tuple[SampleSortConfig, Path]) -> None:
    from samplesort.learning.train import build_policy_config, load_dataset

    config, directory = recorded
    dataset = load_dataset(config, directory)
    policy_config = build_policy_config(config, dataset)

    assert "action" in policy_config.output_features
    assert "observation.state" in policy_config.input_features
    assert "observation.images.top" in policy_config.input_features
    assert policy_config.chunk_size == config.learning.chunk_size
    # Must not reach the network for ImageNet weights.
    assert policy_config.pretrained_backbone_weights is None


@requires_learning
@pytest.mark.slow
def test_training_runs_and_saves_a_checkpoint(
    recorded: tuple[SampleSortConfig, Path], tmp_path: Path
) -> None:
    from samplesort.learning.train import TRAINING_SUMMARY, train

    config, directory = recorded
    checkpoint = tmp_path / "ckpt"
    result = train(config, dataset_dir=directory, output_dir=checkpoint, steps=3, batch_size=2)

    assert result.steps == 3
    assert np.isfinite(result.final_loss)
    assert result.num_frames > 0
    assert result.num_episodes == 2
    assert (checkpoint / "config.json").is_file()
    assert (checkpoint / "model.safetensors").is_file()

    summary = json.loads((checkpoint / TRAINING_SUMMARY).read_text())
    assert summary["steps"] == 3
    assert summary["device"] == "cpu"


@requires_learning
@pytest.mark.slow
def test_a_saved_checkpoint_reloads_and_acts(
    recorded: tuple[SampleSortConfig, Path], tmp_path: Path
) -> None:
    import torch

    from samplesort.learning.train import load_policy, train

    config, directory = recorded
    checkpoint = tmp_path / "ckpt"
    train(config, dataset_dir=directory, output_dir=checkpoint, steps=2, batch_size=2)

    policy = load_policy(config, checkpoint)
    width, height = config.learning.image_size
    action = policy.select_action(
        {
            "observation.images.top": torch.rand(1, 3, height, width),
            "observation.state": torch.rand(1, config.arm.num_joints + 1),
        }
    )
    assert tuple(action.shape) == (1, config.arm.num_joints + 1)
    assert torch.isfinite(action).all()


@requires_learning
def test_loading_a_missing_checkpoint_says_how_to_fix_it(
    tiny_config: SampleSortConfig, tmp_path: Path
) -> None:
    from samplesort.learning.train import TrainingError, load_policy

    with pytest.raises(TrainingError, match="samplesort train"):
        load_policy(tiny_config, tmp_path / "absent")


# ------------------------------------------------------- learned controller


@requires_learning
def test_learned_controller_needs_a_policy(tiny_config: SampleSortConfig) -> None:
    from samplesort.control.learned import LearnedController
    from samplesort.hal.factory import build_backend

    backend = build_backend(tiny_config, gui=False, seed=1)
    with backend, pytest.raises(ValueError, match="needs either a policy_path or a policy"):
        LearnedController(tiny_config, backend.arm, backend.camera, world=backend.world)


@requires_learning
@pytest.mark.slow
def test_learned_controller_shares_the_scripted_interface(
    recorded: tuple[SampleSortConfig, Path], tmp_path: Path
) -> None:
    from samplesort.control.learned import LearnedController
    from samplesort.control.scripted import ScriptedController
    from samplesort.hal.factory import build_backend
    from samplesort.learning.train import load_policy, train
    from samplesort.planning.task_planner import PickPlaceJob

    config, directory = recorded
    checkpoint = tmp_path / "ckpt"
    train(config, dataset_dir=directory, output_dir=checkpoint, steps=2, batch_size=2)
    policy = load_policy(config, checkpoint)

    backend = build_backend(config, gui=False, seed=3)
    with backend:
        controller = LearnedController(
            config, backend.arm, backend.camera, world=backend.world, policy=policy
        )
        assert isinstance(controller, ScriptedController)

        job = PickPlaceJob(
            class_label="red",
            pick_xy=(0.19, 0.0),
            rack_id="rack_a",
            slot_index=0,
            place_xy=config.workspace.racks[0].slots[0],
        )
        # Waypoint planning and feasibility are inherited unchanged.
        assert controller.is_feasible(job)
        assert [name for name, _ in controller.waypoint_poses(job)] == [
            "pre_grasp",
            "grasp",
            "lift",
            "pre_place",
            "place",
            "retreat",
        ]


@requires_learning
@pytest.mark.slow
def test_learned_rollout_completes_and_reports_a_result(
    recorded: tuple[SampleSortConfig, Path], tmp_path: Path
) -> None:
    """A barely-trained policy will fail, but it must fail cleanly and be scored."""
    from samplesort.control.learned import LearnedController
    from samplesort.control.scripted import FailureReason
    from samplesort.hal.factory import build_backend
    from samplesort.learning.train import load_policy, train
    from samplesort.planning.task_planner import PickPlaceJob

    config, directory = recorded
    checkpoint = tmp_path / "ckpt"
    train(config, dataset_dir=directory, output_dir=checkpoint, steps=2, batch_size=2)
    policy = load_policy(config, checkpoint)

    backend = build_backend(config, gui=False, seed=5)
    with backend:
        assert backend.world is not None
        tubes = backend.world.spawn_tubes(1)
        x, y = backend.world.tube_xy(tubes[0].body_id)
        rack_id = config.rack_for_class(tubes[0].label)

        controller = LearnedController(
            config, backend.arm, backend.camera, world=backend.world, policy=policy
        )
        result = controller.execute(
            PickPlaceJob(
                class_label=tubes[0].label,
                pick_xy=(x, y),
                rack_id=rack_id,
                slot_index=0,
                place_xy=config.workspace.rack(rack_id).slots[0],
                body_id=tubes[0].body_id,
            )
        )

    assert isinstance(result.success, bool)
    assert isinstance(result.reason, FailureReason)
    assert result.duration_s > 0.0


@requires_learning
def test_unreachable_job_is_rejected_without_running_the_policy(
    tiny_config: SampleSortConfig,
) -> None:
    from samplesort.control.learned import LearnedController
    from samplesort.control.scripted import FailureReason
    from samplesort.hal.factory import build_backend
    from samplesort.planning.task_planner import PickPlaceJob

    class RefusingPolicy:
        """A stand-in that fails the test if the rollout ever calls it."""

        def parameters(self) -> Iterator[object]:
            import torch

            yield torch.zeros(1)

        def reset(self) -> None:
            pass

        def select_action(self, _batch: dict[str, object]) -> object:
            raise AssertionError("the policy must not run for an unreachable job")

    backend = build_backend(tiny_config, gui=False, seed=1)
    with backend:
        controller = LearnedController(
            tiny_config,
            backend.arm,
            backend.camera,
            world=backend.world,
            policy=RefusingPolicy(),
        )
        result = controller.execute(
            PickPlaceJob("red", (1.5, 0.0), "rack_a", 0, tiny_config.workspace.racks[0].slots[0])
        )
    assert result.reason is FailureReason.UNREACHABLE_PICK


# ----------------------------------------------------------------- evaluate


def test_evaluate_rejects_bad_arguments(tiny_config: SampleSortConfig, tmp_path: Path) -> None:
    from samplesort.learning.evaluate import evaluate

    with pytest.raises(ValueError, match="episodes must be at least 1"):
        evaluate(tiny_config, tmp_path, episodes=0)
    with pytest.raises(ValueError, match="num_tubes must be at least 1"):
        evaluate(tiny_config, tmp_path, episodes=1, num_tubes=0)


def test_evaluation_report_aggregates() -> None:
    from samplesort.learning.evaluate import EvaluationReport, RolloutResult

    report = EvaluationReport(policy_path=Path("ckpt"))
    assert report.success_rate == 0.0
    assert report.mean_duration_s == 0.0

    report.rollouts = [
        RolloutResult(seed=1, success=True, reason="none", duration_s=1.0),
        RolloutResult(seed=2, success=False, reason="grasp_missed", duration_s=3.0),
        RolloutResult(seed=3, success=False, reason="grasp_missed", duration_s=2.0),
    ]
    assert report.episodes == 3
    assert report.successes == 1
    assert report.success_rate == pytest.approx(1 / 3)
    assert report.mean_duration_s == pytest.approx(2.0)
    assert report.failures_by_reason() == {"grasp_missed": 2}
    assert "successes" in "\n".join(report.summary_lines())
    assert json.loads(json.dumps(report.to_dict()))["episodes"] == 3


@requires_learning
@pytest.mark.slow
def test_evaluate_runs_rollouts_and_writes_a_report(
    recorded: tuple[SampleSortConfig, Path], tmp_path: Path
) -> None:
    from samplesort.learning.evaluate import EVALUATION_FILENAME, evaluate
    from samplesort.learning.train import train

    config, directory = recorded
    checkpoint = tmp_path / "ckpt"
    train(config, dataset_dir=directory, output_dir=checkpoint, steps=2, batch_size=2)

    report = evaluate(config, checkpoint, episodes=2, num_tubes=1, seed=11)

    assert report.episodes == 2
    assert 0.0 <= report.success_rate <= 1.0
    assert [r.seed for r in report.rollouts] == [11, 12]

    written = json.loads((checkpoint / EVALUATION_FILENAME).read_text())
    assert written["episodes"] == 2
    assert "failures_by_reason" in written
