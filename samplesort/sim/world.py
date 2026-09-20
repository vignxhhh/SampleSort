"""PyBullet scene holding the table, arm, sample tubes and racks.

The world owns the physics client and every body in it. Randomisation is driven by
an explicitly seeded :class:`numpy.random.Generator`, so a given seed always
produces the same tube layout — which is what makes the benchmark reproducible.
"""

from __future__ import annotations

import contextlib
import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from samplesort.config import SampleSortConfig
from samplesort.sim._bullet import bc, pb, quiet_stdout
from samplesort.sim.assets.arm_builder import write_arm_urdf

logger = logging.getLogger(__name__)

#: Fixed physics timestep. Everything in the sim is stepped at this rate.
SIM_TIMESTEP = 1.0 / 240.0

#: BGR-ordered display colours are derived from the class config; these are the
#: RGBA values PyBullet wants for each known cap label, filled in from `classes.yaml`.
_DEFAULT_CAP_RGBA = (0.5, 0.5, 0.5, 1.0)


@dataclass
class TubeState:
    """A sample tube in the scene and the ground truth about it."""

    body_id: int
    label: str
    spawn_xy: tuple[float, float]
    #: Rack the tube has been placed into, or ``None`` while it is unsorted.
    placed_rack: str | None = None
    #: Slot index within :attr:`placed_rack`, or ``None`` while unsorted.
    placed_slot: int | None = None

    @property
    def is_sorted(self) -> bool:
        """Whether the tube has been placed into a rack."""
        return self.placed_rack is not None


@dataclass
class RackBodies:
    """The PyBullet bodies that make up one rack."""

    rack_id: str
    base_id: int
    slot_ids: list[int] = field(default_factory=list)


def _hex_to_rgba(value: str) -> tuple[float, float, float, float]:
    """Convert a ``#rrggbb`` string to a PyBullet RGBA tuple."""
    text = value.lstrip("#")
    if len(text) != 6:
        return _DEFAULT_CAP_RGBA
    r, g, b = (int(text[i : i + 2], 16) / 255.0 for i in (0, 2, 4))
    return (r, g, b, 1.0)


class SimWorld:
    """A seeded PyBullet scene containing the arm, table, racks and sample tubes.

    Args:
        config: The full validated configuration bundle.
        gui: Open a PyBullet GUI window instead of running headless.
        seed: RNG seed for tube placement. Defaults to ``config.seed``.
    """

    def __init__(
        self, config: SampleSortConfig, *, gui: bool = False, seed: int | None = None
    ) -> None:
        """Create an unconnected world (see the class docstring for args)."""
        self.config = config
        self.gui = gui
        self.seed = config.seed if seed is None else seed
        self.rng = np.random.default_rng(self.seed)

        self.client: bc.BulletClient | None = None
        self.arm_id: int = -1
        self.table_id: int = -1
        self.tubes: list[TubeState] = []
        self.racks: list[RackBodies] = []

        self.joint_indices: list[int] = []
        self.finger_indices: list[int] = []
        self.tool_link_index: int = -1

        self._urdf_path: Path | None = None
        self._connected = False

    # ---------------------------------------------------------------- lifecycle

    def connect(self) -> None:
        """Start the physics server and build the static parts of the scene."""
        if self._connected:
            return
        mode = pb.GUI if self.gui else pb.DIRECT
        with quiet_stdout():
            self.client = bc.BulletClient(connection_mode=mode)
        self.client.setGravity(0.0, 0.0, -9.81)
        self.client.setPhysicsEngineParameter(
            fixedTimeStep=SIM_TIMESTEP, numSolverIterations=120, deterministicOverlappingPairs=1
        )
        if self.gui:
            self.client.configureDebugVisualizer(pb.COV_ENABLE_GUI, 0)
            self.client.resetDebugVisualizerCamera(
                cameraDistance=0.85,
                cameraYaw=50,
                cameraPitch=-40,
                cameraTargetPosition=[0.2, 0.0, 0.0],
            )
        self._connected = True

        self._build_table()
        self._build_arm()
        self._build_racks()
        logger.debug("sim world connected (gui=%s, seed=%d)", self.gui, self.seed)

    def disconnect(self) -> None:
        """Shut the physics server down and drop every body handle."""
        if self.client is not None:
            # A world torn down twice, or after the server died, is not an error.
            with contextlib.suppress(pb.error):
                self.client.disconnect()
        self.client = None
        self._connected = False
        self.tubes = []
        self.racks = []

    @property
    def is_connected(self) -> bool:
        """Whether the physics client is live."""
        return self._connected

    @property
    def bullet(self) -> bc.BulletClient:
        """The live physics client.

        Raises:
            RuntimeError: If the world has not been connected.
        """
        if self.client is None:
            raise RuntimeError("SimWorld is not connected; call connect() first")
        return self.client

    def __enter__(self) -> SimWorld:
        """Connect on entry to a ``with`` block."""
        self.connect()
        return self

    def __exit__(self, *exc_info: object) -> None:
        """Disconnect on exit from a ``with`` block."""
        self.disconnect()

    # ------------------------------------------------------------ scene building

    def _build_table(self) -> None:
        """Create the table slab with its top surface at ``workspace.table_height``."""
        client = self.bullet
        ws = self.config.workspace
        half_x, half_y = ws.table_size[0] / 2.0, ws.table_size[1] / 2.0
        thickness = 0.02
        collision = client.createCollisionShape(
            pb.GEOM_BOX, halfExtents=[half_x, half_y, thickness / 2.0]
        )
        visual = client.createVisualShape(
            pb.GEOM_BOX,
            halfExtents=[half_x, half_y, thickness / 2.0],
            rgbaColor=[0.88, 0.88, 0.90, 1.0],
        )
        # Offset in +X so the arm sits at the near edge with the workspace in front.
        self.table_id = client.createMultiBody(
            baseMass=0.0,
            baseCollisionShapeIndex=collision,
            baseVisualShapeIndex=visual,
            basePosition=[half_x - 0.1, 0.0, ws.table_height - thickness / 2.0],
        )
        client.changeDynamics(self.table_id, -1, lateralFriction=1.0, restitution=0.0)

    def _build_arm(self) -> None:
        """Generate the arm URDF, load it and cache its joint indices."""
        client = self.bullet
        self._urdf_path = write_arm_urdf(self.config.arm)
        self.arm_id = client.loadURDF(
            str(self._urdf_path),
            basePosition=[0.0, 0.0, self.config.workspace.table_height],
            useFixedBase=True,
            flags=pb.URDF_USE_INERTIA_FROM_FILE,
        )

        name_to_index: dict[str, int] = {}
        for index in range(client.getNumJoints(self.arm_id)):
            info = client.getJointInfo(self.arm_id, index)
            name_to_index[info[1].decode()] = index
            if info[12].decode() == "tool_link":
                self.tool_link_index = index

        self.joint_indices = [name_to_index[name] for name in self.config.arm.joint_names]
        self.finger_indices = [
            name_to_index["left_finger_joint"],
            name_to_index["right_finger_joint"],
        ]

        for index in self.joint_indices:
            client.changeDynamics(self.arm_id, index, linearDamping=0.04, angularDamping=0.04)
        for index in self.finger_indices:
            client.changeDynamics(self.arm_id, index, lateralFriction=1.2)

        self.reset_arm(self.config.arm.home_position)

    def _build_racks(self) -> None:
        """Draw every rack as a neutral-grey plate with a recessed slot per position.

        Racks are deliberately desaturated so the HSV cap detector ignores them.
        """
        client = self.bullet
        ws = self.config.workspace
        tube = ws.tube
        for rack_cfg in ws.racks:
            xs = [x for x, _ in rack_cfg.slots]
            ys = [y for _, y in rack_cfg.slots]
            pad = 0.030
            half_x = (max(xs) - min(xs)) / 2.0 + pad
            half_y = (max(ys) - min(ys)) / 2.0 + pad
            center = ((max(xs) + min(xs)) / 2.0, (max(ys) + min(ys)) / 2.0)

            plate_h = ws.rack_plate_height
            collision = client.createCollisionShape(
                pb.GEOM_BOX, halfExtents=[half_x, half_y, plate_h / 2.0]
            )
            visual = client.createVisualShape(
                pb.GEOM_BOX,
                halfExtents=[half_x, half_y, plate_h / 2.0],
                rgbaColor=[0.62, 0.62, 0.63, 1.0],
            )
            base_id = client.createMultiBody(
                baseMass=0.0,
                baseCollisionShapeIndex=collision,
                baseVisualShapeIndex=visual,
                basePosition=[center[0], center[1], ws.table_height + plate_h / 2.0],
            )
            rack = RackBodies(rack_id=rack_cfg.id, base_id=base_id)

            for slot_x, slot_y in rack_cfg.slots:
                marker = client.createVisualShape(
                    pb.GEOM_CYLINDER,
                    radius=tube.cap_radius * 1.6,
                    length=0.002,
                    rgbaColor=[0.32, 0.32, 0.33, 1.0],
                )
                slot_id = client.createMultiBody(
                    baseMass=0.0,
                    baseVisualShapeIndex=marker,
                    basePosition=[slot_x, slot_y, ws.table_height + plate_h + 0.001],
                )
                rack.slot_ids.append(slot_id)
            self.racks.append(rack)

    # -------------------------------------------------------------------- tubes

    def spawn_tubes(self, count: int, labels: list[str] | None = None) -> list[TubeState]:
        """Place ``count`` tubes at random, non-overlapping spots in the pickup zone.

        Args:
            count: How many tubes to create.
            labels: Explicit class labels, one per tube. When omitted, labels are
                drawn from the configured classes so that every class appears at
                least once where the count allows.

        Returns:
            The newly created tube states, which are also appended to :attr:`tubes`.

        Raises:
            RuntimeError: If a non-overlapping layout could not be found.
        """
        chosen = self._choose_labels(count) if labels is None else list(labels)
        if len(chosen) != count:
            raise ValueError(f"expected {count} labels, got {len(chosen)}")

        positions = self._sample_positions(count)
        created: list[TubeState] = []
        for label, (x, y) in zip(chosen, positions, strict=True):
            body_id = self._create_tube_body(x, y, label)
            state = TubeState(body_id=body_id, label=label, spawn_xy=(x, y))
            self.tubes.append(state)
            created.append(state)
            logger.debug("spawned %s tube %d at (%.3f, %.3f)", label, body_id, x, y)

        self.step(self.config.pipeline.settle_steps)
        return created

    def _choose_labels(self, count: int) -> list[str]:
        """Pick ``count`` class labels, covering every class before repeating any."""
        labels = self.config.classes.labels
        chosen: list[str] = []
        while len(chosen) < count:
            block = list(labels)
            self.rng.shuffle(block)
            chosen.extend(block)
        chosen = chosen[:count]
        self.rng.shuffle(chosen)
        return chosen

    def _sample_positions(self, count: int) -> list[tuple[float, float]]:
        """Pick ``count`` XY positions in the pickup zone, guaranteed non-overlapping.

        A jittered grid is used rather than rejection sampling: the zone is split
        into cells at least ``min_separation`` wide, ``count`` distinct cells are
        drawn at random, and each tube is jittered within its cell by at most half
        the slack. Neighbouring tubes therefore always stay at least
        ``min_separation`` apart, and placement never fails for a count the zone
        can actually hold.

        Args:
            count: Number of positions to generate.

        Returns:
            One ``(x, y)`` pair per tube.

        Raises:
            RuntimeError: If the pickup zone cannot hold ``count`` tubes.
        """
        zone = self.config.workspace.pickup_zone
        tube = self.config.workspace.tube
        min_sep = tube.min_separation
        margin = tube.cap_radius + 0.005

        usable_w = (zone.x_max - margin) - (zone.x_min + margin)
        usable_h = (zone.y_max - margin) - (zone.y_min + margin)
        cols = max(1, int(usable_w // min_sep))
        rows = max(1, int(usable_h // min_sep))
        capacity = cols * rows
        if count > capacity:
            raise RuntimeError(
                f"pickup zone holds at most {capacity} tubes at {min_sep:.3f} m spacing, "
                f"but {count} were requested; lower --num-tubes or enlarge pickup_zone"
            )
        if count < 0:
            raise ValueError(f"count must be non-negative, got {count}")

        cell_w = usable_w / cols
        cell_h = usable_h / rows
        jitter_x = max(0.0, (cell_w - min_sep) / 2.0)
        jitter_y = max(0.0, (cell_h - min_sep) / 2.0)

        cells = self.rng.permutation(capacity)[:count]
        positions: list[tuple[float, float]] = []
        for cell in cells:
            col, row = int(cell) % cols, int(cell) // cols
            cx = zone.x_min + margin + (col + 0.5) * cell_w
            cy = zone.y_min + margin + (row + 0.5) * cell_h
            x = cx + float(self.rng.uniform(-jitter_x, jitter_x))
            y = cy + float(self.rng.uniform(-jitter_y, jitter_y))
            positions.append((x, y))
        return positions

    @property
    def pickup_capacity(self) -> int:
        """Maximum number of tubes the pickup zone can hold at the configured spacing."""
        zone = self.config.workspace.pickup_zone
        tube = self.config.workspace.tube
        margin = tube.cap_radius + 0.005
        usable_w = (zone.x_max - margin) - (zone.x_min + margin)
        usable_h = (zone.y_max - margin) - (zone.y_min + margin)
        cols = max(1, int(usable_w // tube.min_separation))
        rows = max(1, int(usable_h // tube.min_separation))
        return cols * rows

    def _create_tube_body(self, x: float, y: float, label: str) -> int:
        """Create one tube: a translucent body cylinder with a coloured cap on top."""
        client = self.bullet
        ws = self.config.workspace
        t = ws.tube
        rgba = _hex_to_rgba(self.config.classes.by_label(label).display_color)

        body_col = client.createCollisionShape(
            pb.GEOM_CYLINDER, radius=t.body_radius, height=t.body_height
        )
        body_vis = client.createVisualShape(
            pb.GEOM_CYLINDER,
            radius=t.body_radius,
            length=t.body_height,
            rgbaColor=[0.86, 0.88, 0.92, 1.0],
        )
        cap_col = client.createCollisionShape(
            pb.GEOM_CYLINDER, radius=t.cap_radius, height=t.cap_height
        )
        cap_vis = client.createVisualShape(
            pb.GEOM_CYLINDER, radius=t.cap_radius, length=t.cap_height, rgbaColor=list(rgba)
        )

        body_id: int = client.createMultiBody(
            baseMass=t.mass,
            baseCollisionShapeIndex=body_col,
            baseVisualShapeIndex=body_vis,
            basePosition=[x, y, ws.table_height + t.body_height / 2.0],
            # Sink the centre of mass towards the base so the tube stands upright,
            # which is how a real tube sits in a weighted holder.
            baseInertialFramePosition=[0.0, 0.0, -t.body_height * 0.35],
            linkMasses=[t.mass * 0.2],
            linkCollisionShapeIndices=[cap_col],
            linkVisualShapeIndices=[cap_vis],
            linkPositions=[[0.0, 0.0, (t.body_height + t.cap_height) / 2.0]],
            linkOrientations=[[0.0, 0.0, 0.0, 1.0]],
            linkInertialFramePositions=[[0.0, 0.0, 0.0]],
            linkInertialFrameOrientations=[[0.0, 0.0, 0.0, 1.0]],
            linkParentIndices=[0],
            linkJointTypes=[pb.JOINT_FIXED],
            linkJointAxis=[[0.0, 0.0, 1.0]],
        )
        client.changeDynamics(body_id, -1, lateralFriction=1.1, rollingFriction=0.002)
        client.changeDynamics(body_id, 0, lateralFriction=1.1)
        return body_id

    def tube_position(self, body_id: int) -> np.ndarray:
        """Return a tube's base-frame XYZ position in world coordinates.

        The base frame is PyBullet's centre-of-mass frame, which sits below the
        tube's geometric centre because the tube is bottom-weighted. Use
        :meth:`tube_cap_position` for the point the gripper actually targets.
        """
        position, _ = self.bullet.getBasePositionAndOrientation(body_id)
        return np.asarray(position, dtype=float)

    def tube_xy(self, body_id: int) -> tuple[float, float]:
        """Return a tube's XY position on the table."""
        position = self.tube_position(body_id)
        return (float(position[0]), float(position[1]))

    def tube_is_upright(self, body_id: int, *, max_tilt_deg: float = 35.0) -> bool:
        """Whether a tube is still standing rather than knocked over.

        Args:
            body_id: The tube to check.
            max_tilt_deg: Maximum tilt from vertical still counted as upright.
        """
        _, quat = self.bullet.getBasePositionAndOrientation(body_id)
        rotation = np.asarray(self.bullet.getMatrixFromQuaternion(quat)).reshape(3, 3)
        tube_axis = rotation @ np.array([0.0, 0.0, 1.0])
        tilt = np.degrees(np.arccos(np.clip(abs(float(tube_axis[2])), -1.0, 1.0)))
        return bool(tilt <= max_tilt_deg)

    def tube_cap_position(self, body_id: int) -> np.ndarray:
        """Return the XYZ centre of a tube's coloured cap in the world frame."""
        state = self.bullet.getLinkState(body_id, 0)
        return np.asarray(state[0], dtype=float)

    def tube_by_id(self, body_id: int) -> TubeState:
        """Look up a tube state by PyBullet body id.

        Raises:
            KeyError: If no tube with that id exists in this world.
        """
        for tube in self.tubes:
            if tube.body_id == body_id:
                return tube
        raise KeyError(f"no tube with body id {body_id} in this world")

    def remaining_tubes(self) -> list[TubeState]:
        """Tubes that are still waiting to be sorted."""
        return [tube for tube in self.tubes if not tube.is_sorted]

    # ------------------------------------------------------------------ stepping

    def step(self, steps: int = 1) -> None:
        """Advance the simulation.

        Args:
            steps: Number of fixed-timestep physics steps to run.
        """
        client = self.bullet
        for _ in range(max(0, steps)):
            client.stepSimulation()

    def reset_arm(self, q: list[float] | np.ndarray) -> None:
        """Teleport the arm joints to a configuration and hold them there.

        Position-control targets are set alongside the teleport so the arm does not
        immediately sag under gravity on the next :meth:`step`.
        """
        client = self.bullet
        targets = np.asarray(q, dtype=float)
        for index, value in zip(self.joint_indices, targets, strict=True):
            client.resetJointState(self.arm_id, index, float(value))
            client.setJointMotorControl2(
                self.arm_id, index, pb.POSITION_CONTROL, targetPosition=float(value), force=40.0
            )
        for index in self.finger_indices:
            client.resetJointState(self.arm_id, index, 0.0)
            client.setJointMotorControl2(
                self.arm_id, index, pb.POSITION_CONTROL, targetPosition=0.0, force=20.0
            )

    def tool_pose(self) -> tuple[np.ndarray, np.ndarray]:
        """Return the tool centre point as ``(position_xyz, orientation_quat)``."""
        state = self.bullet.getLinkState(
            self.arm_id, self.tool_link_index, computeForwardKinematics=1
        )
        return np.asarray(state[4], dtype=float), np.asarray(state[5], dtype=float)

    def reset(self, *, seed: int | None = None) -> None:
        """Tear the scene down and rebuild it, optionally with a new seed.

        Args:
            seed: New RNG seed. ``None`` keeps the current one.
        """
        if seed is not None:
            self.seed = seed
        self.rng = np.random.default_rng(self.seed)
        was_connected = self._connected
        self.disconnect()
        if was_connected:
            self.connect()
