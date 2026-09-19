"""Pydantic models that load, validate and bundle the YAML configuration files.

Every hardware- or workspace-specific number in SampleSort lives in ``configs/``
and reaches the rest of the codebase through :class:`SampleSortConfig`.  Loading
fails loudly with a :class:`ConfigError` that names the offending file.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

logger = logging.getLogger(__name__)

DEFAULT_CONFIG_DIR = Path(__file__).resolve().parent.parent / "configs"

Mode = Literal["sim", "real"]
ControlMode = Literal["scripted", "learned"]


class ConfigError(RuntimeError):
    """Raised when a configuration file is missing, malformed or inconsistent."""


class _Base(BaseModel):
    """Base model that rejects unknown keys so typos surface immediately."""

    model_config = ConfigDict(extra="forbid")


# --------------------------------------------------------------------------- arm


class GripperConfig(_Base):
    """Parallel-gripper geometry and the tolerances that define a valid grasp."""

    open_width: float = Field(gt=0.0)
    closed_width: float = Field(gt=0.0)
    finger_length: float = Field(gt=0.0)
    grasp_tolerance_xy: float = Field(gt=0.0)
    grasp_tolerance_z: float = Field(gt=0.0)

    @model_validator(mode="after")
    def _check_widths(self) -> GripperConfig:
        if self.closed_width >= self.open_width:
            raise ValueError("gripper.closed_width must be smaller than gripper.open_width")
        return self


class ArmConfig(_Base):
    """Kinematic model, joint limits and servo wiring for the arm."""

    name: str
    base_height: float = Field(gt=0.0)
    link_lengths: list[float]
    joint_names: list[str]
    joint_limits: list[tuple[float, float]]
    home_position: list[float]
    observe_position: list[float]
    max_joint_velocity: float = Field(gt=0.0)
    default_move_duration: float = Field(gt=0.0)
    gripper: GripperConfig

    serial_port: str = "/dev/ttyACM0"
    baudrate: int = 1_000_000
    servo_model: str = "sts3215"
    servo_ids: list[int] = Field(default_factory=list)
    gripper_servo_id: int = 6

    @field_validator("link_lengths")
    @classmethod
    def _check_links(cls, value: list[float]) -> list[float]:
        if len(value) != 3:
            raise ValueError("link_lengths must hold exactly 3 entries (upper arm, forearm, tool)")
        if any(length <= 0.0 for length in value):
            raise ValueError("link_lengths must all be positive")
        return value

    @model_validator(mode="after")
    def _check_consistency(self) -> ArmConfig:
        n = len(self.joint_names)
        if n != 5:
            raise ValueError("SampleSort models a 5-DOF arm; joint_names must hold 5 entries")
        if len(self.joint_limits) != n:
            raise ValueError("joint_limits must have one entry per joint")
        for field_name, pose in (
            ("home_position", self.home_position),
            ("observe_position", self.observe_position),
        ):
            if len(pose) != n:
                raise ValueError(f"{field_name} must have one entry per joint")
        for name, (low, high) in zip(self.joint_names, self.joint_limits, strict=True):
            if low >= high:
                raise ValueError(f"joint_limits for '{name}' are inverted: [{low}, {high}]")
        for field_name, pose in (
            ("home_position", self.home_position),
            ("observe_position", self.observe_position),
        ):
            for name, (low, high), q in zip(self.joint_names, self.joint_limits, pose, strict=True):
                if not low <= q <= high:
                    raise ValueError(f"{field_name} for '{name}' ({q}) violates its joint limits")
        if self.servo_ids and len(self.servo_ids) != n:
            raise ValueError("servo_ids must have one entry per joint when provided")
        return self

    @property
    def num_joints(self) -> int:
        """Number of controllable arm joints, excluding the gripper."""
        return len(self.joint_names)

    @property
    def max_reach(self) -> float:
        """Distance from the shoulder that the fully extended arm can reach."""
        return float(sum(self.link_lengths))


# ------------------------------------------------------------------------ camera


class CameraConfig(_Base):
    """Overhead camera hardware settings plus its nominal pose in the base frame."""

    index: int = 0
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    fps: int = Field(gt=0)
    intrinsics_path: Path

    position: tuple[float, float, float]
    look_at: tuple[float, float, float]
    up_axis: tuple[float, float, float]
    vertical_fov_deg: float = Field(gt=0.0, lt=180.0)
    near: float = Field(gt=0.0)
    far: float = Field(gt=0.0)

    @model_validator(mode="after")
    def _check_clip(self) -> CameraConfig:
        if self.far <= self.near:
            raise ValueError("camera.far must be greater than camera.near")
        if self.position == self.look_at:
            raise ValueError("camera.position and camera.look_at must differ")
        return self

    @property
    def resolution(self) -> tuple[int, int]:
        """Image size as ``(width, height)``."""
        return (self.width, self.height)


# --------------------------------------------------------------------- workspace


class PickupZone(_Base):
    """Axis-aligned region of the table where unsorted tubes appear."""

    x_min: float
    x_max: float
    y_min: float
    y_max: float

    @model_validator(mode="after")
    def _check_bounds(self) -> PickupZone:
        if self.x_min >= self.x_max or self.y_min >= self.y_max:
            raise ValueError("pickup_zone bounds are inverted or degenerate")
        return self

    def contains(self, x: float, y: float) -> bool:
        """Return whether ``(x, y)`` lies inside the zone."""
        return self.x_min <= x <= self.x_max and self.y_min <= y <= self.y_max

    @property
    def center(self) -> tuple[float, float]:
        """Centre of the zone."""
        return ((self.x_min + self.x_max) / 2.0, (self.y_min + self.y_max) / 2.0)


class TubeConfig(_Base):
    """Physical dimensions of a sample tube."""

    body_radius: float = Field(gt=0.0)
    body_height: float = Field(gt=0.0)
    cap_radius: float = Field(gt=0.0)
    cap_height: float = Field(gt=0.0)
    min_separation: float = Field(gt=0.0)
    mass: float = Field(gt=0.0)

    @property
    def total_height(self) -> float:
        """Overall height of body plus cap."""
        return self.body_height + self.cap_height


class RackConfig(_Base):
    """A rack and the table-frame XY coordinates of each of its slots."""

    id: str
    label: str
    slots: list[tuple[float, float]]

    @field_validator("slots")
    @classmethod
    def _check_slots(cls, value: list[tuple[float, float]]) -> list[tuple[float, float]]:
        if not value:
            raise ValueError("a rack must define at least one slot")
        return value

    @property
    def num_slots(self) -> int:
        """Total number of slots in this rack."""
        return len(self.slots)


class MarkerBoardConfig(_Base):
    """Printed ArUco/ChArUco board used to calibrate the camera to the table plane."""

    dictionary: str
    squares_x: int = Field(gt=1)
    squares_y: int = Field(gt=1)
    square_length: float = Field(gt=0.0)
    marker_length: float = Field(gt=0.0)
    origin: tuple[float, float]

    @model_validator(mode="after")
    def _check_marker(self) -> MarkerBoardConfig:
        if self.marker_length >= self.square_length:
            raise ValueError("marker_board.marker_length must be smaller than square_length")
        return self


class WorkspaceConfig(_Base):
    """Table, pickup zone, tube geometry and rack layout."""

    table_height: float
    table_size: tuple[float, float]
    pickup_zone: PickupZone
    tube: TubeConfig
    grasp_height: float = Field(gt=0.0)
    approach_clearance: float = Field(gt=0.0)
    racks: list[RackConfig]
    marker_board: MarkerBoardConfig

    @model_validator(mode="after")
    def _check_racks(self) -> WorkspaceConfig:
        if not self.racks:
            raise ValueError("workspace must define at least one rack")
        ids = [rack.id for rack in self.racks]
        duplicates = {i for i in ids if ids.count(i) > 1}
        if duplicates:
            raise ValueError(f"duplicate rack ids in workspace config: {sorted(duplicates)}")
        return self

    def rack(self, rack_id: str) -> RackConfig:
        """Look up a rack by id.

        Raises:
            ConfigError: if no rack with that id exists.
        """
        for rack in self.racks:
            if rack.id == rack_id:
                return rack
        known = ", ".join(r.id for r in self.racks)
        raise ConfigError(f"unknown rack id '{rack_id}'; workspace defines: {known}")

    @property
    def total_slots(self) -> int:
        """Number of slots across every rack."""
        return sum(rack.num_slots for rack in self.racks)


# ----------------------------------------------------------------------- classes


class ClassConfig(_Base):
    """One sample category: its cap colour ranges and its destination rack."""

    label: str
    rack_id: str
    hue_ranges: list[tuple[int, int]]
    saturation_range: tuple[int, int]
    value_range: tuple[int, int]
    display_color: str = "#888888"

    @field_validator("hue_ranges")
    @classmethod
    def _check_hue(cls, value: list[tuple[int, int]]) -> list[tuple[int, int]]:
        if not value:
            raise ValueError("each class needs at least one hue range")
        for low, high in value:
            if not (0 <= low <= 179 and 0 <= high <= 179):
                raise ValueError(f"hue range [{low}, {high}] outside OpenCV's 0-179 range")
            if low > high:
                raise ValueError(f"hue range [{low}, {high}] is inverted; split wrap-around ranges")
        return value

    @model_validator(mode="after")
    def _check_sv(self) -> ClassConfig:
        for name, (low, high) in (
            ("saturation_range", self.saturation_range),
            ("value_range", self.value_range),
        ):
            if not (0 <= low <= 255 and 0 <= high <= 255):
                raise ValueError(f"{name} must lie within 0-255")
            if low > high:
                raise ValueError(f"{name} is inverted: [{low}, {high}]")
        return self


class DetectorConfig(_Base):
    """Blob-filtering thresholds for the HSV cap detector."""

    min_area_px: int = Field(gt=0)
    max_area_px: int = Field(gt=0)
    blur_kernel: int = Field(ge=1)
    morph_kernel: int = Field(ge=1)
    min_confidence: float = Field(ge=0.0, le=1.0)

    @model_validator(mode="after")
    def _check_area_and_kernels(self) -> DetectorConfig:
        if self.min_area_px >= self.max_area_px:
            raise ValueError("detector.min_area_px must be smaller than max_area_px")
        kernels = (("blur_kernel", self.blur_kernel), ("morph_kernel", self.morph_kernel))
        for name, kernel in kernels:
            if kernel % 2 == 0:
                raise ValueError(f"detector.{name} must be odd, got {kernel}")
        return self


class ClassesConfig(_Base):
    """The full sample taxonomy plus detector tuning."""

    classes: list[ClassConfig]
    detector: DetectorConfig
    qr_enabled: bool = False

    @model_validator(mode="after")
    def _check_labels(self) -> ClassesConfig:
        if not self.classes:
            raise ValueError("at least one sample class must be defined")
        labels = [c.label for c in self.classes]
        duplicates = {label for label in labels if labels.count(label) > 1}
        if duplicates:
            raise ValueError(f"duplicate class labels: {sorted(duplicates)}")
        return self

    @property
    def labels(self) -> list[str]:
        """All class labels in declaration order."""
        return [c.label for c in self.classes]

    def by_label(self, label: str) -> ClassConfig:
        """Look up a class by label.

        Raises:
            ConfigError: if the label is unknown.
        """
        for cls_cfg in self.classes:
            if cls_cfg.label == label:
                return cls_cfg
        raise ConfigError(f"unknown class label '{label}'; known: {', '.join(self.labels)}")


# ---------------------------------------------------------------------- pipeline


class PipelineConfig(_Base):
    """Runtime behaviour of the main sort loop."""

    control_mode: ControlMode = "scripted"
    max_iterations: int = Field(default=24, gt=0)
    grasp_retries: int = Field(default=1, ge=0)
    settle_steps: int = Field(default=60, ge=0)
    policy_path: Path | None = None


# ------------------------------------------------------------------------ bundle


class SampleSortConfig(_Base):
    """Every validated config file, plus the directory they were loaded from."""

    mode: Mode = "sim"
    seed: int = 42
    log_level: str = "INFO"
    database_path: Path
    output_dir: Path
    pipeline: PipelineConfig = Field(default_factory=PipelineConfig)

    arm: ArmConfig
    camera: CameraConfig
    workspace: WorkspaceConfig
    classes: ClassesConfig

    config_dir: Path

    @model_validator(mode="after")
    def _check_cross_references(self) -> SampleSortConfig:
        rack_ids = {rack.id for rack in self.workspace.racks}
        for cls_cfg in self.classes.classes:
            if cls_cfg.rack_id not in rack_ids:
                raise ValueError(
                    f"class '{cls_cfg.label}' targets unknown rack '{cls_cfg.rack_id}'; "
                    f"workspace defines: {', '.join(sorted(rack_ids))}"
                )
        zone = self.workspace.pickup_zone
        reach = self.arm.max_reach
        for x, y in ((zone.x_min, zone.y_min), (zone.x_max, zone.y_max)):
            if (x**2 + y**2) ** 0.5 > reach:
                raise ValueError(
                    f"pickup_zone corner ({x}, {y}) is outside the arm's {reach:.3f} m reach"
                )
        return self

    def rack_for_class(self, label: str) -> str:
        """Return the rack id that samples of ``label`` belong in."""
        return self.classes.by_label(label).rack_id


def _read_yaml(path: Path) -> dict[str, Any]:
    """Parse a YAML file into a dict, raising :class:`ConfigError` on any problem."""
    if not path.is_file():
        raise ConfigError(f"config file not found: {path}")
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ConfigError(f"could not parse YAML in {path}: {exc}") from exc
    if raw is None:
        raise ConfigError(f"config file is empty: {path}")
    if not isinstance(raw, dict):
        raise ConfigError(f"expected a YAML mapping at the top level of {path}, got {type(raw)}")
    return raw


def _build(model: type[BaseModel], data: dict[str, Any], source: Path) -> Any:
    """Validate ``data`` against ``model``, reporting the source file on failure."""
    try:
        return model.model_validate(data)
    except ValidationError as exc:
        raise ConfigError(f"invalid configuration in {source}:\n{exc}") from exc


def load_config(
    config_dir: Path | str | None = None,
    *,
    mode: Mode | None = None,
    seed: int | None = None,
) -> SampleSortConfig:
    """Load and validate every SampleSort config file.

    Args:
        config_dir: Directory holding ``default.yaml`` and friends. Defaults to the
            repository's ``configs/`` directory.
        mode: Optional override for the ``sim``/``real`` backend selection.
        seed: Optional override for the global RNG seed.

    Returns:
        The fully validated configuration bundle.

    Raises:
        ConfigError: if any file is missing, malformed or internally inconsistent.
    """
    directory = Path(config_dir) if config_dir is not None else DEFAULT_CONFIG_DIR
    directory = directory.resolve()
    if not directory.is_dir():
        raise ConfigError(f"config directory not found: {directory}")

    defaults = _read_yaml(directory / "default.yaml")

    arm_file = directory / defaults.pop("arm_config", "arm_so101.yaml")
    camera_file = directory / defaults.pop("camera_config", "camera.yaml")
    workspace_file = directory / defaults.pop("workspace_config", "workspace.yaml")
    classes_file = directory / defaults.pop("classes_config", "classes.yaml")

    bundle: dict[str, Any] = dict(defaults)
    bundle["arm"] = _build(ArmConfig, _read_yaml(arm_file), arm_file)
    bundle["camera"] = _build(CameraConfig, _read_yaml(camera_file), camera_file)
    bundle["workspace"] = _build(WorkspaceConfig, _read_yaml(workspace_file), workspace_file)
    bundle["classes"] = _build(ClassesConfig, _read_yaml(classes_file), classes_file)
    bundle["config_dir"] = directory

    if mode is not None:
        bundle["mode"] = mode
    if seed is not None:
        bundle["seed"] = seed

    config: SampleSortConfig = _build(SampleSortConfig, bundle, directory / "default.yaml")
    logger.debug("loaded configuration from %s (mode=%s)", directory, config.mode)
    return config
