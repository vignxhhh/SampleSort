"""Generate the simulated arm's URDF from :class:`~samplesort.config.ArmConfig`.

The SO-101 is approximated by a 5-DOF chain built from primitives: a yaw joint at
the base followed by three coplanar pitch joints and a roll joint at the tool, plus
two prismatic gripper fingers.

Generating the URDF from the same numbers the analytic kinematics use guarantees
that :mod:`samplesort.planning.kinematics` and the PyBullet model agree. The joint
frames are chosen so that:

* pitch joints use axis ``(0, -1, 0)``, making a positive joint angle raise the link,
* the elevation of link *i* is the running sum ``q1 + ... + qi``,
* the tool frame origin sits exactly at the tool centre point (TCP) between the fingers.
"""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path

from samplesort.config import ArmConfig

logger = logging.getLogger(__name__)

#: Density-free link masses (kg). Real values would come from the CAD model; these
#: only need to be plausible enough for a stable position-controlled simulation.
_LINK_MASS = 0.25
_FINGER_MASS = 0.03
_BASE_MASS = 1.0


def _inertial(mass: float, ixx: float, iyy: float, izz: float, origin: str = "0 0 0") -> str:
    """Render a URDF ``<inertial>`` block."""
    return (
        f"      <inertial>\n"
        f'        <origin xyz="{origin}" rpy="0 0 0"/>\n'
        f'        <mass value="{mass:.6f}"/>\n'
        f'        <inertia ixx="{ixx:.8f}" ixy="0" ixz="0" '
        f'iyy="{iyy:.8f}" iyz="0" izz="{izz:.8f}"/>\n'
        f"      </inertial>\n"
    )


def _box_link(
    name: str,
    size: tuple[float, float, float],
    origin: str,
    rgba: str,
    mass: float,
) -> str:
    """Render a URDF link whose visual and collision geometry is a single box."""
    sx, sy, sz = size
    ixx = mass * (sy**2 + sz**2) / 12.0
    iyy = mass * (sx**2 + sz**2) / 12.0
    izz = mass * (sx**2 + sy**2) / 12.0
    geom = f'<box size="{sx:.6f} {sy:.6f} {sz:.6f}"/>'
    return (
        f'  <link name="{name}">\n'
        f"{_inertial(mass, ixx, iyy, izz, origin)}"
        f"      <visual>\n"
        f'        <origin xyz="{origin}" rpy="0 0 0"/>\n'
        f"        <geometry>{geom}</geometry>\n"
        f'        <material name="{name}_mat"><color rgba="{rgba}"/></material>\n'
        f"      </visual>\n"
        f"      <collision>\n"
        f'        <origin xyz="{origin}" rpy="0 0 0"/>\n'
        f"        <geometry>{geom}</geometry>\n"
        f"      </collision>\n"
        f"  </link>\n"
    )


def _revolute_joint(
    name: str, parent: str, child: str, origin: str, axis: str, low: float, high: float
) -> str:
    """Render a URDF revolute joint with explicit limits."""
    return (
        f'  <joint name="{name}" type="revolute">\n'
        f'    <parent link="{parent}"/>\n'
        f'    <child link="{child}"/>\n'
        f'    <origin xyz="{origin}" rpy="0 0 0"/>\n'
        f'    <axis xyz="{axis}"/>\n'
        f'    <limit lower="{low:.6f}" upper="{high:.6f}" effort="30.0" velocity="3.0"/>\n'
        f'    <dynamics damping="0.6" friction="0.1"/>\n'
        f"  </joint>\n"
    )


def build_arm_urdf(config: ArmConfig) -> str:
    """Build the URDF XML for the simulated arm.

    Args:
        config: Validated arm configuration supplying link lengths, joint limits
            and gripper geometry.

    Returns:
        The complete URDF document as a string.
    """
    l1, l2, l3 = config.link_lengths
    h0 = config.base_height
    grip = config.gripper
    half_open = grip.open_width / 2.0

    limits = dict(zip(config.joint_names, config.joint_limits, strict=True))
    yaw, shoulder, elbow, wrist_pitch, wrist_roll = config.joint_names

    parts: list[str] = [f'<?xml version="1.0"?>\n<robot name="{config.name}">\n']

    # Fixed mounting plate flush with the table. The base frame origin sits at its
    # centre, so the shoulder pitch axis ends up at exactly z = base_height.
    parts.append(
        _box_link("base_link", (0.09, 0.09, 0.015), "0 0 0.0075", "0.25 0.27 0.30 1.0", _BASE_MASS)
    )

    # Yaw column: rotates the whole arm about +Z and carries the shoulder up to h0.
    parts.append(
        _box_link("yaw_link", (0.05, 0.05, h0), f"0 0 {h0 / 2.0:.6f}", "0.35 0.38 0.42 1.0", 0.3)
    )
    parts.append(_revolute_joint(yaw, "base_link", "yaw_link", "0 0 0", "0 0 1", *limits[yaw]))

    # Three coplanar pitch links, each drawn from its own origin along +X.
    for child, length, parent, joint, origin, rgba in (
        ("upper_arm", l1, "yaw_link", shoulder, f"0 0 {h0:.6f}", "0.85 0.55 0.20 1.0"),
        ("forearm", l2, "upper_arm", elbow, f"{l1:.6f} 0 0", "0.80 0.62 0.32 1.0"),
        ("wrist_link", l3, "forearm", wrist_pitch, f"{l2:.6f} 0 0", "0.55 0.58 0.62 1.0"),
    ):
        parts.append(
            _box_link(child, (length, 0.032, 0.032), f"{length / 2.0:.6f} 0 0", rgba, _LINK_MASS)
        )
        parts.append(_revolute_joint(joint, parent, child, origin, "0 -1 0", *limits[joint]))

    # Tool frame: origin at the TCP, rotating about the tool axis.
    parts.append(_box_link("tool_link", (0.012, 0.030, 0.030), "0 0 0", "0.20 0.22 0.25 1.0", 0.05))
    parts.append(
        _revolute_joint(
            wrist_roll, "wrist_link", "tool_link", f"{l3:.6f} 0 0", "1 0 0", *limits[wrist_roll]
        )
    )

    # Two prismatic fingers, drawn backwards from the TCP so the grasp point is the
    # tool frame origin. They translate along the tool frame's ±Y.
    for finger, sign in (("left_finger", 1.0), ("right_finger", -1.0)):
        parts.append(
            _box_link(
                finger,
                (grip.finger_length, 0.008, 0.020),
                f"{-grip.finger_length / 2.0:.6f} 0 0",
                "0.15 0.16 0.18 1.0",
                _FINGER_MASS,
            )
        )
        parts.append(
            f'  <joint name="{finger}_joint" type="prismatic">\n'
            f'    <parent link="tool_link"/>\n'
            f'    <child link="{finger}"/>\n'
            f'    <origin xyz="0 {sign * half_open:.6f} 0" rpy="0 0 0"/>\n'
            f'    <axis xyz="0 {-sign:.1f} 0"/>\n'
            f'    <limit lower="0.0" upper="{half_open - grip.closed_width / 2.0:.6f}" '
            f'effort="20.0" velocity="0.5"/>\n'
            f'    <dynamics damping="1.0" friction="0.5"/>\n'
            f"  </joint>\n"
        )

    parts.append("</robot>\n")
    return "".join(parts)


def write_arm_urdf(config: ArmConfig, directory: Path | str | None = None) -> Path:
    """Write the generated URDF where PyBullet can load it from.

    The preferred home is next to the other simulation assets, which keeps the
    file easy to inspect during development. That directory is read-only under a
    system-wide (non-editable) install, so this falls back to a temporary
    directory rather than failing — the file is regenerated from config on every
    connect, so its location carries no state.

    Args:
        config: Arm configuration to generate from.
        directory: Destination directory. Defaults to ``samplesort/sim/assets``,
            with a temp-directory fallback.

    Returns:
        Path to the written ``.urdf`` file.

    Raises:
        OSError: If an explicitly requested ``directory`` cannot be written to.
    """
    filename = f"{config.name}_generated.urdf"
    urdf = build_arm_urdf(config)

    if directory is not None:
        target_dir = Path(directory)
        target_dir.mkdir(parents=True, exist_ok=True)
        path = target_dir / filename
        path.write_text(urdf, encoding="utf-8")
        return path

    for candidate in (Path(__file__).resolve().parent, Path(tempfile.gettempdir()) / "samplesort"):
        try:
            candidate.mkdir(parents=True, exist_ok=True)
            path = candidate / filename
            path.write_text(urdf, encoding="utf-8")
        except OSError as exc:
            logger.debug("cannot write the arm URDF to %s (%s); trying elsewhere", candidate, exc)
            continue
        logger.debug("wrote generated arm URDF to %s", path)
        return path

    raise OSError(
        f"could not write {filename} to the package assets directory or to "
        f"{tempfile.gettempdir()}; pass an explicit directory instead"
    )
