"""Render the gallery stills the web page shows, from real simulation runs.

Each image is a moment from an actual pipeline run, captured as it happens:
the arm, tubes and racks are wherever the physics put them. Frames are rendered
by PyBullet's software renderer at twice the output size, with shadows, and
downsampled, which gives clean edges without a GPU.

Run from the repository root: ``python docs/web/export_gallery.py``.
Writes ``docs/web/data/gallery/*.jpg`` and ``gallery.json`` (captions).
"""

import json
import math
import pathlib

import cv2
import numpy as np

from samplesort.config import load_config
from samplesort.hal.factory import build_backend
from samplesort.logging_db.sort_log import SortLog
from samplesort.pipeline import SortPipeline
from samplesort.sim._bullet import pb

OUT = pathlib.Path(__file__).resolve().parent / "data" / "gallery"
OUT.mkdir(parents=True, exist_ok=True)
W, H, SS = 1600, 1000, 2
captions: dict[str, str] = {}


def render(world, eye, target, *, fov=36.0, w=W, h=H):
    """One supersampled, shadowed render of the world from ``eye`` towards ``target``."""
    view = pb.computeViewMatrix(list(eye), list(target), [0, 0, 1])
    proj = pb.computeProjectionMatrixFOV(fov, w / h, 0.01, 5.0)
    _, _, rgba, _, _ = world.bullet.getCameraImage(
        w * SS,
        h * SS,
        view,
        proj,
        renderer=pb.ER_TINY_RENDERER,
        shadow=1,
        lightDirection=[0.45, -0.75, 1.3],
        lightColor=[1.0, 0.97, 0.93],
        lightAmbientCoeff=0.40,
        lightDiffuseCoeff=0.40,
        lightSpecularCoeff=0.22,
        flags=pb.ER_NO_SEGMENTATION_MASK,
    )
    rgb = np.asarray(rgba, np.uint8).reshape(h * SS, w * SS, 4)[:, :, :3]
    img = cv2.resize(rgb[:, :, ::-1], (w, h), interpolation=cv2.INTER_AREA)
    # A gentle vignette, the way a real lens frames a bench.
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
    r2 = ((xs - w / 2) / (w / 2)) ** 2 + ((ys - h / 2) / (h / 2)) ** 2
    shaded = img.astype(np.float32) * (1.0 - 0.13 * r2)[:, :, None]
    return np.clip(shaded, 0, 255).astype(np.uint8)


def save(name, img, caption):
    """Write one still and remember its caption."""
    cv2.imwrite(str(OUT / f"{name}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 88])
    captions[name] = caption
    print("wrote", name, img.shape)


def tcp(world):
    """The tool centre point's position."""
    return np.asarray(world.tool_pose()[0], dtype=float)


# Rendering mid-run perturbs PyBullet's collision bookkeeping enough to change
# how a run unfolds (measured: a clean 8/8 run became 6/8 with a tube knocked
# over). So moments are snapshotted with saveState while the run goes on, and
# rendered from the snapshots once it has finished.
pending: list = []


def snapshot(world, name, eye, target, fov, caption):
    """Remember the simulator state now, to render once the run is over."""
    pending.append((name, world.bullet.saveState(), eye, target, fov, caption))


def render_pending(world):
    """Render every snapshot taken so far."""
    for name, state, eye, target, fov, caption in pending:
        world.bullet.restoreState(stateId=state)
        save(name, render(world, eye, target, fov=fov), caption)
    pending.clear()


cfg = load_config()

# ----------------------------------------------------------------- clean run
backend = build_backend(cfg, seed=42)
with backend, SortLog(":memory:") as log:
    world, arm = backend.world, backend.arm
    world.spawn_tubes(8)
    world.reset_arm(cfg.arm.observe_position)
    world.step(60)
    snapshot(
        world,
        "overview",
        (0.58, -0.30, 0.30),
        (0.19, 0.0, 0.03),
        34,
        "The cell before a run: eight capped tubes in the pickup zone, three racks, the arm "
        "folded back so the overhead camera has a clear view.",
    )

    pipeline = SortPipeline(cfg, backend, sort_log=log)
    frame = backend.camera.read()
    dets = pipeline.detector.detect_in_pickup_zone(frame)
    annotated = frame.copy()
    for d in dets:
        u, v = (int(round(c)) for c in d.pixel_xy)
        cv2.circle(annotated, (u, v), 15, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.circle(annotated, (u, v), 2, (255, 255, 255), -1, cv2.LINE_AA)
    us = [d.pixel_xy[0] for d in dets]
    vs = [d.pixel_xy[1] for d in dets]
    u0, u1 = int(min(us)) - 45, int(max(us)) + 45
    v0, v1 = int(min(vs)) - 40, int(max(vs)) + 40
    crop = annotated[max(v0, 0) : v1, max(u0, 0) : u1]
    save(
        "overhead",
        cv2.resize(crop, (crop.shape[1] * 4, crop.shape[0] * 4), interpolation=cv2.INTER_CUBIC),
        "What the overhead camera sees, with the detector's output: every cap found, "
        "classified by colour and located on the bench to about a millimetre.",
    )

    shots = {"grasp": False, "carry": False}
    original_close = arm.close_gripper
    original_move = arm.move_to_joints

    def close_and_capture():
        """Close the gripper, and snapshot the first successful grasp."""
        original_close()
        if not shots["grasp"] and arm.grasped_body is not None:
            shots["grasp"] = True
            p = tcp(world)
            eye = p + np.array([0.22, -0.26, 0.10])
            snapshot(
                world,
                "grasp",
                eye,
                p + np.array([0, 0, -0.03]),
                34,
                "The gripper closing on a cap. The wrist camera read this tube's QR label a "
                "moment earlier, from 7 cm above it.",
            )

    def move_and_capture(q, duration=None):
        """Move, and snapshot the first carry towards a side rack."""
        original_move(q, duration)
        if not shots["carry"] and arm.grasped_body is not None:
            p = tcp(world)
            if p[2] > 0.10 and abs(p[1]) > 0.12:
                shots["carry"] = True
                snapshot(
                    world,
                    "carry",
                    (p[0] + 0.30, p[1] + 0.32 * np.sign(p[1]), 0.30),
                    (p[0], p[1], 0.06),
                    42,
                    "Carrying a tube to its rack. Each move is a waypoint solved exactly by "
                    "closed-form inverse kinematics.",
                )

    arm.close_gripper = close_and_capture
    arm.move_to_joints = move_and_capture
    report = pipeline.run()
    arm.close_gripper = original_close
    arm.move_to_joints = original_move
    world.step(60)
    lying = [t.label for t in world.tubes if not world.tube_is_upright(t.body_id)]
    print("clean run:", report.succeeded, "/", report.attempted, "sorted; lying:", lying)
    assert report.succeeded == 8 and not lying, "the clean run did not finish cleanly"
    final = world.bullet.saveState()
    render_pending(world)
    world.bullet.restoreState(stateId=final)
    save(
        "sorted",
        render(world, (0.56, -0.34, 0.36), (0.20, 0.0, 0.02), fov=38),
        "The same bench after the run: every tube in the rack its cap colour belongs to, "
        "and one row per tube in the sort log.",
    )

# -------------------------------------------------------------- recovery run
cfg_r = load_config()
cfg_r.pipeline.perception_noise_m = 0.006
backend = build_backend(cfg_r, seed=3)
with backend, SortLog(":memory:") as log:
    world, arm = backend.world, backend.arm
    tubes = world.spawn_tubes(6)
    for i, tube in enumerate(tubes[:2]):
        world.knock_over(tube.body_id, math.radians(37 + 111 * i + 13 * 3))
    world.reset_arm(cfg_r.arm.observe_position)
    world.step(60)
    snapshot(
        world,
        "fallen",
        (0.52, -0.28, 0.24),
        (0.19, 0.0, 0.02),
        34,
        "A harder start: two tubes knocked over. The overhead camera tells them apart from "
        "standing ones by the outline of the cap.",
    )
    pipeline = SortPipeline(cfg_r, backend, sort_log=log)
    station = cfg_r.workspace.righting_station
    done = {"inspect": False}
    original_right = pipeline.controller.right_tube

    def right_and_capture(fallen, probe):
        """Right a tube, snapshotting it while it is held level for inspection."""

        def probe_and_capture(cap, axis):
            if not done["inspect"]:
                done["inspect"] = True
                # Film the held tube side-on: from across its axis.
                side = np.array([-axis[1], axis[0], 0.0])
                if side[0] < 0:
                    side = -side
                snapshot(
                    world,
                    "inspect",
                    tuple(np.asarray(cap) + side * 0.24 + np.array([0, 0, 0.07])),
                    (cap[0], cap[1], cap[2] - 0.01),
                    40,
                    "A fallen tube held level over clear table, so the overhead camera can see "
                    "which end is the body before it is stood up.",
                )
            return probe(cap, axis)

        return original_right(fallen, probe_and_capture)

    pipeline.controller.right_tube = right_and_capture
    original_open = arm.open_gripper
    stood = {"done": False}

    def open_and_capture():
        """Release, and snapshot the first tube left standing at the station."""
        held = arm.grasped_body
        original_open()
        if held is not None and not stood["done"] and station is not None:
            p = np.asarray(world.tube_position(held))
            if math.dist(p[:2], station) < 0.04 and world.tube_is_upright(held):
                world.step(30)
                stood["done"] = True
                snapshot(
                    world,
                    "righted",
                    (station[0] + 0.16, station[1] - 0.22, 0.17),
                    (station[0], station[1], 0.03),
                    44,
                    "Stood back up at the righting station, cap on top. From here it is "
                    "sorted like any other tube.",
                )

    arm.open_gripper = open_and_capture
    report = pipeline.run()
    print("recovery run:", report.succeeded, "/", report.attempted, "sorted;", end=" ")
    print(report.righted, "righted")
    render_pending(world)

(OUT / "gallery.json").write_text(json.dumps(captions, indent=1))
print("captions:", list(captions))
