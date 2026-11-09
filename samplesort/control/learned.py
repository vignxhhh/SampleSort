"""Closed-loop control from a trained ACT policy.

:class:`LearnedController` presents exactly the same interface as
:class:`~samplesort.control.scripted.ScriptedController`, so the pipeline and the
benchmark switch between them with a flag and nothing else changes. It subclasses
the scripted controller deliberately: waypoint planning, feasibility checking and
placement verification are shared, and only :meth:`execute` differs — the policy
drives the arm from camera and joint observations instead of following a
precomputed path.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

import numpy as np

from samplesort.config import SampleSortConfig
from samplesort.control.scripted import (
    ExecutionResult,
    FailureReason,
    PlacementVerifier,
    ScriptedController,
)
from samplesort.hal.arm import ArmError, ArmInterface, JointLimitError
from samplesort.hal.camera import CameraInterface
from samplesort.planning.task_planner import PickPlaceJob

logger = logging.getLogger(__name__)

#: Gripper action values at or below this count as "close the gripper".
GRIPPER_CLOSE_THRESHOLD = 0.5


class LearnedController(ScriptedController):
    """Runs a trained ACT policy in closed loop on camera and joint observations.

    Args:
        config: The validated configuration bundle.
        arm: The arm to drive.
        camera: The camera supplying visual observations.
        world: Optional simulation world used to verify grasps and placements.
        policy_path: Directory holding the trained checkpoint.
        policy: A preloaded policy, used instead of loading from ``policy_path``.
    """

    def __init__(
        self,
        config: SampleSortConfig,
        arm: ArmInterface,
        camera: CameraInterface,
        *,
        world: PlacementVerifier | None = None,
        policy_path: Path | str | None = None,
        policy: Any | None = None,
    ) -> None:
        """Load the policy and bind it to an arm and camera."""
        super().__init__(config, arm, world)
        self.camera = camera
        self.policy_path = Path(policy_path) if policy_path is not None else None

        if policy is not None:
            self.policy = policy
        elif self.policy_path is not None:
            from samplesort.learning.train import load_policy

            self.policy = load_policy(config, self.policy_path)
        else:
            raise ValueError("LearnedController needs either a policy_path or a policy")

        self._torch = _require_torch()
        self.device = str(next(self.policy.parameters()).device)

    # ------------------------------------------------------------- observations

    def _observation(self) -> dict[str, Any]:
        """Build one batched observation for the policy from the live sensors."""
        from samplesort.learning.record import observation_image, observation_state

        torch = self._torch
        image = observation_image(self.config, self.camera.read())
        # LeRobot policies expect float CHW images in [0, 1].
        image_tensor = (
            torch.from_numpy(image).permute(2, 0, 1).to(torch.float32).div(255.0).unsqueeze(0)
        )

        gripper = 1.0 if getattr(self.arm, "gripper_is_open", True) else 0.0
        state = observation_state(self.arm.get_joint_positions(), gripper)
        state_tensor = torch.from_numpy(state).to(torch.float32).unsqueeze(0)

        return {
            "observation.images.top": image_tensor.to(self.device),
            "observation.state": state_tensor.to(self.device),
        }

    def _apply(self, action: np.ndarray) -> None:
        """Send one policy action to the arm, clamped into the joint limits."""
        joints = self.arm.clamp_to_limits(action[: self.config.arm.num_joints])
        self.arm.move_to_joints(joints, 1.0 / self.config.learning.fps)

        wants_closed = float(action[-1]) <= GRIPPER_CLOSE_THRESHOLD
        currently_open = bool(getattr(self.arm, "gripper_is_open", True))
        if wants_closed and currently_open:
            self.arm.close_gripper()
        elif not wants_closed and not currently_open:
            self.arm.open_gripper()

    # ----------------------------------------------------------------- rollout

    def execute(self, job: PickPlaceJob) -> ExecutionResult:
        """Run the policy in closed loop until it finishes or hits the step cap.

        Args:
            job: The job to attempt. Used for placement verification and for the
                feasibility pre-check; the policy itself is not told the plan.

        Returns:
            An :class:`ExecutionResult` scored the same way the scripted
            controller's is, so the two modes are directly comparable.
        """
        started = time.perf_counter()
        result = ExecutionResult(success=False)

        if not self.is_feasible(job):
            result.reason = self._unreachable_stage(job)
            result.duration_s = time.perf_counter() - started
            return result

        self.policy.reset()
        try:
            self.arm.open_gripper()
            for step in range(self.config.learning.rollout_max_steps):
                with self._torch.no_grad():
                    action = self.policy.select_action(self._observation())
                self._apply(np.asarray(action.squeeze(0).cpu(), dtype=float))

                if self.has_finished(job, step):
                    break
        except JointLimitError as exc:
            # An untrained policy can emit nonsense; that is a failed rollout,
            # not a crash.
            logger.info("policy produced an invalid joint target: %s", exc)
            result.reason = FailureReason.ARM_ERROR
            result.duration_s = time.perf_counter() - started
            return result
        except ArmError as exc:
            logger.error("arm error during policy rollout: %s", exc)
            result.reason = FailureReason.ARM_ERROR
            result.duration_s = time.perf_counter() - started
            return result

        result.grasped = self._holding()
        self._settle()

        error = self._placement_error(job)
        result.place_error_m = error
        if error is None:
            # No ground truth to check against; trust that a released tube landed.
            result.success = not result.grasped
            result.reason = (
                FailureReason.NONE if result.success else FailureReason.DROPPED_IN_TRANSIT
            )
        elif error <= self.PLACE_TOLERANCE_M:
            result.success = True
            result.reason = FailureReason.NONE
        else:
            result.reason = (
                FailureReason.GRASP_MISSED if not result.grasped else FailureReason.MISPLACED
            )

        result.duration_s = time.perf_counter() - started
        return result

    def has_finished(self, job: PickPlaceJob, step: int) -> bool:
        """Whether the rollout can stop early.

        The episode ends once the tube is within placement tolerance of its slot
        and the gripper has let go. Without ground truth the rollout always runs
        to the step cap.

        Args:
            job: The job being attempted.
            step: The rollout step just completed.

        Returns:
            Whether to stop the rollout.
        """
        error = self._placement_error(job)
        if error is None:
            return False
        if error <= self.PLACE_TOLERANCE_M and not self._holding():
            logger.debug("policy finished at step %d (%.3f m from the slot)", step, error)
            return True
        return False


def _require_torch() -> Any:
    """Import torch, raising a friendly error when the extra is not installed."""
    from samplesort.learning._deps import require_torch

    return require_torch()
