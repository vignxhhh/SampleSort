"""Export the exact simulation trajectory the web viewer replays.

Nothing here is re-authored for the browser: the joint angles are what the
controller commanded, the tube poses are what the physics solver produced, and
`tcp` is the analytic forward-kinematics result the page checks itself against.

Two scenarios are exported:

* ``nominal`` (``data/run.json``) — eight upright tubes, clean perception.
* ``recovery`` (``data/run_recovery.json``) — localisation noise injected so
  grasps can miss and be retried, and tubes knocked over so the arm has to
  stand them back up before sorting them.

Run from the repository root: ``python docs/web/export_run.py [nominal|recovery]``.
"""

import json
import math
import pathlib
import sys

from samplesort.config import load_config
from samplesort.hal.factory import build_backend
from samplesort.logging_db.sort_log import SortLog
from samplesort.perception.calibration import calibration_from_camera_pose
from samplesort.pipeline import SortPipeline
from samplesort.planning.kinematics import forward_kinematics
from samplesort.planning.task_planner import PickPlaceJob
from samplesort.sim.world import SimWorld

EVERY = 20  # physics steps between samples (240 Hz -> 12 Hz)

SCENARIOS = {
    # seed, tubes, tubes knocked over, localisation noise (m)
    "nominal": {"seed": 42, "tubes": 8, "knocked": 0, "noise": 0.0, "out": "run.json"},
    "recovery": {"seed": 3, "tubes": 6, "knocked": 2, "noise": 0.006, "out": "run_recovery.json"},
}
scenario_name = sys.argv[1] if len(sys.argv) > 1 else "nominal"
scenario = SCENARIOS[scenario_name]
if len(sys.argv) > 2:  # seed override, for searching
    scenario["seed"] = int(sys.argv[2])
OUT = str(pathlib.Path(__file__).resolve().parent / "data" / scenario["out"])

cfg = load_config()
cfg.seed = scenario["seed"]
cfg.pipeline.perception_noise_m = scenario["noise"]
backend = build_backend(cfg, gui=False, seed=scenario["seed"])

frames, events = [], []
counter = {"n": 0}
state = {"arm": None, "world": None, "order": []}


def sample() -> None:
    """Record one frame of joint angles, tube poses and the analytic TCP."""
    world, arm = state["world"], state["arm"]
    if world is None or arm is None:
        return
    q = arm.get_joint_positions()
    fingers = world.bullet.getJointStates(world.arm_id, world.finger_indices)
    travel = float(fingers[0][0])
    span = (cfg.arm.gripper.open_width - cfg.arm.gripper.closed_width) / 2.0
    # One entry per spawned tube, in spawn order; null once a tube the pipeline
    # gave up on has been retired from the scene.
    present = {t.body_id for t in world.tubes}
    tubes = []
    for body_id in state["order"]:
        if body_id not in present:
            tubes.append(None)
            continue
        p, o = world.bullet.getBasePositionAndOrientation(body_id)
        tubes.append([round(float(v), 5) for v in p] + [round(float(v), 4) for v in o])
    tcp = forward_kinematics(cfg.arm, q)
    frames.append(
        {
            "q": [round(float(v), 12) for v in q],
            "g": round(1.0 - travel / span if span else 1.0, 4),
            "t": tubes,
            "held": arm.grasped_body if arm.grasped_body is not None else -1,
            "tcp": [round(tcp.x, 12), round(tcp.y, 12), round(tcp.z, 12)],
        }
    )


original_step = SimWorld.step


def recording_step(self: SimWorld, steps: int = 1) -> None:
    """Step the world as usual, sampling a frame every ``EVERY`` steps."""
    for _ in range(max(0, steps)):
        original_step(self, 1)
        counter["n"] += 1
        if counter["n"] % EVERY == 0:
            sample()


SimWorld.step = recording_step

with backend, SortLog(":memory:") as log:
    world = backend.world
    state["world"], state["arm"] = world, backend.arm
    backend.arm.move_to_joints(cfg.arm.observe_position, 0.8)
    tubes = world.spawn_tubes(scenario["tubes"])
    for i, tube in enumerate(tubes[: scenario["knocked"]]):
        world.knock_over(tube.body_id, math.radians(37 + 111 * i + 13 * scenario["seed"]))
    order = [t.body_id for t in world.tubes]
    state["order"] = order
    labels = {t.body_id: t.label for t in world.tubes}
    sample_ids = {t.body_id: t.sample_id for t in world.tubes}

    for _ in range(3):
        sample()

    # The overhead frame and its detections, for the perception panel.
    pipeline = SortPipeline(cfg, backend, sort_log=log)
    frame_bgr = backend.camera.read()
    dets = pipeline.detector.detect_in_pickup_zone(frame_bgr)
    cal = calibration_from_camera_pose(cfg.camera, cfg.workspace, backend.camera.world_to_pixel)

    controller = pipeline.controller
    original_exec = controller.execute
    original_right = controller.right_tube

    def tracked(job: PickPlaceJob):  # noqa: ANN202 - mirrors execute()
        """Note the frames each attempt spans and how it ended, then run it unchanged."""
        start = len(frames)
        result = original_exec(job)
        # A retry re-measures the tube, so it is a new job object: count the
        # failed attempts on the same tube that came straight before it.
        attempt = 1
        for previous in reversed(events):
            if previous["kind"] != "pick" or previous["body"] != job.body_id or previous["ok"]:
                break
            attempt += 1
        events.append(
            {
                "f": start,
                "end": len(frames),
                "kind": "pick",
                "attempt": attempt,
                "label": job.class_label,
                "rack": job.rack_id,
                "slot": job.slot_index,
                "ok": result.success,
                "reason": result.reason.value,
                "id": result.sample_id,
                "body": job.body_id,
            }
        )
        return result

    def tracked_right(fallen, probe):  # noqa: ANN001, ANN202 - mirrors right_tube()
        """Note the frames a righting manoeuvre spans and how it ended."""
        start = len(frames)
        result = original_right(fallen, probe)
        events.append(
            {
                "f": start,
                "end": len(frames),
                "kind": "right",
                "label": fallen.class_label,
                "ok": result.success,
                "reason": result.reason.value,
                "side": result.body_side,
                "xy": [round(v, 4) for v in fallen.table_xy],
            }
        )
        return result

    controller.execute = tracked
    controller.right_tube = tracked_right

    report = pipeline.run()
    for _ in range(14):
        sample()

SimWorld.step = original_step

ws, arm, grip = cfg.workspace, cfg.arm, cfg.arm.gripper
payload = {
    "meta": {
        "scenario": scenario_name,
        "seed": cfg.seed,
        "noise_mm": round(scenario["noise"] * 1000, 1),
        "knocked": scenario["knocked"],
        "righted": report.righted,
        "retries": sum(o.attempts - 1 for o in report.outcomes),
        "fps": 240 / EVERY,
        "frames": len(frames),
        "sorted": report.succeeded,
        "attempted": report.attempted,
        "mean_s": round(report.mean_duration_s, 3),
    },
    "arm": {
        "base_height": arm.base_height,
        "links": arm.link_lengths,
        "limits": [list(lim) for lim in arm.joint_limits],
        "names": arm.joint_names,
        "home": arm.home_position,
        "observe": arm.observe_position,
        "gripper": {
            "open": grip.open_width,
            "closed": grip.closed_width,
            "finger": grip.finger_length,
            "tol_xy": grip.grasp_tolerance_xy,
            "tol_z": grip.grasp_tolerance_z,
        },
    },
    "workspace": {
        "table": list(ws.table_size),
        "table_h": ws.table_height,
        "zone": [
            ws.pickup_zone.x_min,
            ws.pickup_zone.x_max,
            ws.pickup_zone.y_min,
            ws.pickup_zone.y_max,
        ],
        "tube": {
            "br": ws.tube.body_radius,
            "bh": ws.tube.body_height,
            "cr": ws.tube.cap_radius,
            "ch": ws.tube.cap_height,
        },
        "grasp_h": ws.grasp_height,
        "clear": ws.approach_clearance,
        "plate_h": ws.rack_plate_height,
        "station": list(ws.righting_station) if ws.righting_station else None,
        "racks": [
            {"id": r.id, "label": r.label, "slots": [list(s) for s in r.slots]} for r in ws.racks
        ],
    },
    "classes": [
        {
            "label": c.label,
            "rack": c.rack_id,
            "color": c.display_color,
            "hue": [list(h) for h in c.hue_ranges],
            "sat": list(c.saturation_range),
            "val": list(c.value_range),
        }
        for c in cfg.classes.classes
    ],
    "tubes": [{"id": bid, "label": labels[bid], "sid": sample_ids[bid]} for bid in order],
    "events": events,
    "frames": frames,
    "detections": [
        {
            "u": round(d.pixel_xy[0], 2),
            "v": round(d.pixel_xy[1], 2),
            "label": d.class_label,
            "conf": round(d.confidence, 3),
            "area": round(d.area_px, 1),
            "xy": [round(d.table_xy[0], 4), round(d.table_xy[1], 4)],
        }
        for d in dets
    ],
    "calibration": {
        "rms_px": round(cal.rms_error_px, 4),
        "rms_m": round(cal.rms_error_m, 6),
        "cam_h": cal.camera_height,
        "nadir": list(cal.camera_nadir_xy),
    },
}
with pathlib.Path(OUT).open("w", encoding="utf-8") as fh:
    json.dump(payload, fh, separators=(",", ":"))
print(
    "frames:",
    len(frames),
    "| events:",
    len(events),
    "| sorted:",
    report.succeeded,
    "/",
    report.attempted,
    "| retries:",
    payload["meta"]["retries"],
    "| righted:",
    report.righted,
    "/",
    scenario["knocked"],
    "| left on table:",
    len(report.fallen_left),
    "| failures:",
    [(e["kind"], e["reason"]) for e in events if not e["ok"]],
)
