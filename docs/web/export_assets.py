"""Export the still images and study results the web page shows.

* ``data/wrist_scan.png`` / ``.json`` — what the wrist camera sees from the
  pre-grasp pose, and the ID it decodes there.
* ``data/lighting.jpg`` — one test layout under every lighting condition of the
  detector study.
* ``data/detector_study.json`` — copied from a ``samplesort detector-study`` run
  (pass its output directory; defaults to ``outputs/``).

Run from the repository root: ``python docs/web/export_assets.py [study_dir]``.
"""

import json
import pathlib
import shutil
import sys

import cv2

from samplesort.config import load_config
from samplesort.hal.factory import build_backend
from samplesort.perception.detector_study import RESULTS_JSON, contact_sheet
from samplesort.planning.kinematics import inverse_kinematics
from samplesort.planning.task_planner import PickPlaceJob

DATA = pathlib.Path(__file__).resolve().parent / "data"
cfg = load_config()

# --- the wrist camera's view of one cap -------------------------------------
backend = build_backend(cfg, seed=42)
with backend:
    world, scanner = backend.world, backend.scanner
    tube = world.spawn_tubes(8)[2]
    job = PickPlaceJob(
        class_label=tube.label,
        pick_xy=world.tube_xy(tube.body_id),
        rack_id="rack_a",
        slot_index=0,
        place_xy=(0.1, 0.2),
    )
    ws = cfg.workspace
    world.reset_arm(
        inverse_kinematics(cfg.arm, job.pick_pose(ws.table_grasp_height + ws.approach_clearance))
    )
    world.step(1)
    frame = scanner.capture()
    decoded = scanner.decode_nearest_centre(frame)
    assert decoded == tube.sample_id, (decoded, tube.sample_id)
    cv2.imwrite(str(DATA / "wrist_scan.png"), frame)
    (DATA / "wrist_scan.json").write_text(json.dumps({"decoded": decoded}))
    print("wrist scan:", decoded, frame.shape)

# --- the lighting conditions -------------------------------------------------
sheet = contact_sheet(cfg, labels=False)  # the page captions each tile itself
cv2.imwrite(str(DATA / "lighting.jpg"), sheet, [cv2.IMWRITE_JPEG_QUALITY, 86])
print("lighting sheet:", sheet.shape)

# --- the study's numbers -----------------------------------------------------
study_dir = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else pathlib.Path("outputs")
source = study_dir / RESULTS_JSON
if source.is_file():
    shutil.copy(source, DATA / RESULTS_JSON)
    result = json.loads(source.read_text())
    print("detector study:", {d: s["overall"]["sortable"] for d, s in result["scores"].items()})
else:
    print(f"no {source}; run 'samplesort detector-study' first")
