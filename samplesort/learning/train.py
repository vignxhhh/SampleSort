"""Train an ACT policy on a recorded SampleSort dataset.

A thin wrapper around LeRobot's :class:`~lerobot.policies.act.modeling_act.ACTPolicy`:
it builds the policy config from ``learning:`` in ``default.yaml``, wires the
dataset's action chunking through ``delta_timestamps``, and runs a plain PyTorch
training loop. Per spec section 6 the goal is a *runnable* pipeline, not a
competitive policy.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from samplesort.config import SampleSortConfig
from samplesort.learning._deps import require_lerobot, require_torch, resolve_device

logger = logging.getLogger(__name__)

#: Written next to the checkpoint so a run can be traced back to its settings.
TRAINING_SUMMARY = "training_summary.json"


class TrainingError(RuntimeError):
    """Raised when training cannot start — usually a missing or unusable dataset."""


@dataclass
class TrainingResult:
    """What a training run produced.

    Attributes:
        checkpoint_dir: Where the trained policy was saved.
        steps: How many optimiser steps ran.
        final_loss: Loss at the last step.
        losses: Loss at each logged step, as ``(step, loss)`` pairs.
        duration_s: Wall-clock seconds spent training.
        device: The device training actually ran on.
        dataset_dir: The dataset that was trained on.
        num_frames: Frames in that dataset.
        num_episodes: Episodes in that dataset.
    """

    checkpoint_dir: Path
    steps: int
    final_loss: float
    losses: list[tuple[int, float]] = field(default_factory=list)
    duration_s: float = 0.0
    device: str = "cpu"
    dataset_dir: Path = Path()
    num_frames: int = 0
    num_episodes: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Serialise the result for ``training_summary.json``."""
        return {
            "checkpoint_dir": str(self.checkpoint_dir),
            "dataset_dir": str(self.dataset_dir),
            "steps": self.steps,
            "final_loss": self.final_loss,
            "losses": [[step, loss] for step, loss in self.losses],
            "duration_s": round(self.duration_s, 3),
            "device": self.device,
            "num_frames": self.num_frames,
            "num_episodes": self.num_episodes,
        }


def load_dataset(config: SampleSortConfig, dataset_dir: Path | str | None = None) -> Any:
    """Open a recorded dataset with the action-chunk delta timestamps ACT needs.

    Args:
        config: The validated configuration bundle.
        dataset_dir: Dataset root. Defaults to ``config.learning.dataset_dir``.

    Returns:
        A ``LeRobotDataset`` whose ``action`` field yields ``chunk_size`` future
        actions per sample.

    Raises:
        TrainingError: If the dataset is missing or holds no frames.
    """
    require_lerobot()
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    root = Path(dataset_dir or config.learning.dataset_dir)
    if not (root / "meta" / "info.json").is_file():
        raise TrainingError(
            f"no LeRobot dataset found at {root}. Record one first:\n"
            f"    samplesort record --episodes 20"
        )

    fps = config.learning.fps
    delta = {"action": [i / fps for i in range(config.learning.chunk_size)]}
    dataset = LeRobotDataset(
        repo_id=config.learning.dataset_repo_id, root=root, delta_timestamps=delta
    )
    if dataset.num_frames == 0:
        raise TrainingError(f"the dataset at {root} contains no frames")
    return dataset


def build_policy_config(config: SampleSortConfig, dataset: Any) -> Any:
    """Build the ACT config for a dataset's observation and action spaces.

    Args:
        config: The validated configuration bundle.
        dataset: The dataset the policy will be trained on.

    Returns:
        A configured ``ACTConfig``.
    """
    require_lerobot()
    from lerobot.configs.types import FeatureType
    from lerobot.datasets.utils import dataset_to_policy_features
    from lerobot.policies.act.configuration_act import ACTConfig

    learning = config.learning
    features = dataset_to_policy_features(dataset.meta.features)
    output_features = {k: v for k, v in features.items() if v.type is FeatureType.ACTION}
    input_features = {k: v for k, v in features.items() if k not in output_features}

    return ACTConfig(
        chunk_size=learning.chunk_size,
        n_action_steps=learning.n_action_steps,
        dim_model=learning.dim_model,
        n_heads=learning.n_heads,
        dim_feedforward=learning.dim_feedforward,
        n_encoder_layers=learning.n_encoder_layers,
        n_decoder_layers=learning.n_decoder_layers,
        vision_backbone=learning.vision_backbone,
        # Torchvision would otherwise download ImageNet weights, which breaks
        # offline machines and CI. Backbone weights are learned from scratch.
        pretrained_backbone_weights=None,
        use_vae=learning.use_vae,
        input_features=input_features,
        output_features=output_features,
        device=resolve_device(learning.device),
    )


def train(
    config: SampleSortConfig,
    *,
    dataset_dir: Path | str | None = None,
    output_dir: Path | str | None = None,
    steps: int | None = None,
    batch_size: int | None = None,
    learning_rate: float | None = None,
    device: str | None = None,
) -> TrainingResult:
    """Train an ACT policy and save it.

    Args:
        config: The validated configuration bundle.
        dataset_dir: Dataset to train on. Defaults to the configured one.
        output_dir: Where to save the checkpoint. Defaults to the configured one.
        steps: Optimiser steps. Defaults to ``learning.steps``.
        batch_size: Batch size. Defaults to ``learning.batch_size``.
        learning_rate: Adam learning rate. Defaults to ``learning.learning_rate``.
        device: Torch device. Defaults to ``learning.device``.

    Returns:
        A summary of the run, which is also written to ``training_summary.json``.

    Raises:
        TrainingError: If the dataset is missing or empty.
        LearningDependencyError: If torch or LeRobot is not installed.
    """
    torch = require_torch()
    require_lerobot()
    from lerobot.policies.act.modeling_act import ACTPolicy

    learning = config.learning
    total_steps = steps if steps is not None else learning.steps
    batch = batch_size if batch_size is not None else learning.batch_size
    rate = learning_rate if learning_rate is not None else learning.learning_rate
    target_device = resolve_device(device if device is not None else learning.device)
    checkpoint_dir = Path(output_dir or learning.checkpoint_dir)

    dataset = load_dataset(config, dataset_dir)
    logger.info(
        "training on %d frames from %d episode(s)", dataset.num_frames, dataset.num_episodes
    )

    policy_config = build_policy_config(config, dataset)
    policy = ACTPolicy(policy_config, dataset_stats=dataset.meta.stats)
    policy.to(target_device)
    policy.train()

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=min(batch, dataset.num_frames),
        shuffle=True,
        num_workers=learning.num_workers,
        drop_last=False,
    )
    optimiser = torch.optim.Adam(policy.parameters(), lr=rate)

    started = time.perf_counter()
    losses: list[tuple[int, float]] = []
    step = 0
    last_loss = float("nan")

    while step < total_steps:
        for raw_batch in loader:
            if step >= total_steps:
                break
            tensors = {
                key: value.to(target_device) if hasattr(value, "to") else value
                for key, value in raw_batch.items()
            }
            output = policy.forward(tensors)
            loss = output[0] if isinstance(output, tuple) else output["loss"]

            optimiser.zero_grad()
            loss.backward()
            optimiser.step()

            step += 1
            last_loss = float(loss.detach())
            if step == 1 or step % learning.log_every == 0 or step == total_steps:
                losses.append((step, last_loss))
                logger.info("step %d/%d  loss %.4f", step, total_steps, last_loss)
            if learning.save_every and step % learning.save_every == 0:
                _save(policy, checkpoint_dir)

    _save(policy, checkpoint_dir)

    result = TrainingResult(
        checkpoint_dir=checkpoint_dir,
        steps=step,
        final_loss=last_loss,
        losses=losses,
        duration_s=time.perf_counter() - started,
        device=target_device,
        dataset_dir=Path(dataset_dir or learning.dataset_dir),
        num_frames=int(dataset.num_frames),
        num_episodes=int(dataset.num_episodes),
    )
    (checkpoint_dir / TRAINING_SUMMARY).write_text(
        json.dumps(result.to_dict(), indent=2) + "\n", encoding="utf-8"
    )
    logger.info("saved checkpoint to %s after %d step(s)", checkpoint_dir, step)
    return result


def _save(policy: Any, checkpoint_dir: Path) -> None:
    """Write the policy's weights and config to ``checkpoint_dir``."""
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    policy.save_pretrained(checkpoint_dir)


def load_policy(
    config: SampleSortConfig, checkpoint_dir: Path | str, device: str | None = None
) -> Any:
    """Load a trained ACT policy from a checkpoint directory.

    Args:
        config: The validated configuration bundle.
        checkpoint_dir: Directory written by :func:`train`.
        device: Torch device. Defaults to ``learning.device``.

    Returns:
        The policy in eval mode, with its action queue reset.

    Raises:
        TrainingError: If the directory does not hold a checkpoint.
    """
    require_torch()
    require_lerobot()
    from lerobot.policies.act.modeling_act import ACTPolicy

    path = Path(checkpoint_dir)
    if not (path / "config.json").is_file():
        raise TrainingError(
            f"no ACT checkpoint found at {path}. Train one first:\n"
            f"    samplesort train --dataset {config.learning.dataset_dir}"
        )

    target_device = resolve_device(device if device is not None else config.learning.device)
    policy = ACTPolicy.from_pretrained(path)
    policy.to(target_device)
    policy.eval()
    policy.reset()
    return policy
