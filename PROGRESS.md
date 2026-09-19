# SampleSort — Build Progress

Tracks the Build Plan from section 8 of the specification. Every phase is
implemented, tested, linted, type-checked and committed before the next begins.

Resume rule: continue from the first unchecked item below.

## Build Plan

- [x] **Phase 1 — Scaffold.** `pyproject.toml`, package layout, Pydantic config
      models + YAML configs, CLI skeleton, CI workflow, ruff/mypy/pytest config.
- [x] **Phase 2 — HAL + Simulation.** Abstract interfaces, PyBullet world, sim arm,
      sim camera, `samplesort sim-demo`.
- [x] **Phase 3 — Kinematics + Scripted control.** FK/IK, trajectory interpolation,
      scripted pick/place of a single tube in sim.
- [x] **Phase 4 — Perception.** Calibration, detector, colour classifier, synthetic-image tests.
- [ ] **Phase 5 — Planning + Pipeline.** Rack state, task planner, full sort loop, SQLite logging.
- [ ] **Phase 6 — Benchmark + Dashboard.** Metrics JSON/Markdown, Streamlit dashboard.
- [ ] **Phase 7 — Learning.** Demo recording, ACT training wrapper, evaluation, learned control.
- [ ] **Phase 8 — Real hardware path.** Real arm/camera HAL, ArUco generator, calibration script, docs.
- [ ] **Phase 9 — Documentation.** README, architecture diagram, results.

## Post-build steps

- [ ] Step 2 — Full system testing (clean install, headless suite, every CLI path).
- [ ] Step 3 — Acceptance verification against spec section 9.
- [ ] Step 4 — Final polish (README, dead code, `.gitignore`, LICENSE).

---

## Phase 1 — Scaffold ✅

**Delivered**

- `pyproject.toml` with pinned-floor dependencies, console script `samplesort`,
  and ruff / mypy / pytest configuration in one place.
- `samplesort/config.py`: Pydantic v2 models for arm, camera, workspace, classes
  and pipeline settings, loaded from `configs/*.yaml`.
- Five config files under `configs/`.
- `samplesort/cli.py` Typer skeleton with `--version` and `config`.
- `.github/workflows/ci.yml`: install, lint, format-check, type-check, pytest, CLI smoke test.
- `tests/test_config.py` — 20 tests covering happy path and every validation failure mode.

**Design decisions**

- `extra="forbid"` on every config model, so a typo'd YAML key is an error rather
  than a silently ignored setting.
- Cross-file validation lives on `SampleSortConfig`: classes must target racks that
  exist, and the pickup zone must sit inside the arm's reach. These are the two
  mistakes that would otherwise only surface deep inside a sort run.
- Arm geometry (`base_height`, `link_lengths`) is config, not code. Both the analytic
  kinematics and the generated PyBullet model read the same numbers, so they cannot drift.
- The learning stack (torch + LeRobot) is an optional extra rather than a hard
  dependency: the core sim, perception, planning and benchmark paths must install
  and run without it.

**Known issues**

- None.

## Phase 2 — HAL + Simulation ✅

**Delivered**

- `hal/arm.py`, `hal/camera.py`: abstract interfaces with shared joint-limit
  validation, connection-state guards and context-manager support.
- `sim/assets/arm_builder.py`: generates the arm URDF from `arm_so101.yaml`.
- `sim/world.py`: seeded PyBullet scene (table, arm, three racks, sample tubes).
- `hal/sim_arm.py`, `hal/sim_camera.py`: the PyBullet implementations.
- `hal/factory.py`: `build_backend()` picks sim vs real from `config.mode`.
- `samplesort sim-demo [--gui] [--seed] [--num-tubes]`.
- `tests/test_hal_sim.py` — 20 tests. Suite total: 40 passing.

**Design decisions**

- *Arm geometry retuned.* The spec's rack layout was unreachable with the first
  guess at link lengths. Links are now 175/175/90 mm on an 80 mm shoulder (350 mm
  planar reach); the pickup zone and racks were moved inside that envelope, with
  every grasp and approach pose verified to have IK margin (`|D| <= 0.82`).
- *Tube placement uses a jittered grid, not rejection sampling.* Rejection
  sampling failed to fit 8 tubes in the zone. The grid splits the zone into cells
  at least `min_separation` wide and jitters within each, so separation is
  guaranteed and placement never fails below the zone's stated capacity.
- *Grasping is constraint-based, not friction-based.* `close_gripper()` attaches a
  fixed constraint when a tube's cap is inside the configured tolerance box around
  the TCP. Friction grasps on a 8 mm cylinder are numerically fragile and would
  make benchmark numbers noise rather than signal. Positioning errors still cause
  genuine failures, and results stay reproducible for a given seed.
- *New config key `observe_position`.* At the home pose the arm occludes the pickup
  zone from the overhead camera. The pipeline retracts to this folded-back pose
  before every perception step.
- *Tubes are bottom-weighted* (inertial frame sunk below the geometric centre) so
  they stand like a tube in a weighted holder instead of toppling on contact.
- *Racks are rendered desaturated grey* so the HSV cap detector ignores them —
  which also exercises the detector's rejection path in the end-to-end test.

**Known issues**

- `SimCamera` uses PyBullet's `ER_TINY_RENDERER`; it is headless-safe and
  deterministic but has no shadows or anti-aliasing. Good enough for HSV blob
  detection, and it keeps CI free of any GPU requirement.

## Phase 3 — Kinematics + Scripted control ✅

**Delivered**

- `planning/kinematics.py`: `Pose`, analytic `forward_kinematics`,
  `inverse_kinematics`, `link_positions`, `is_reachable`.
- `control/trajectory.py`: smoothstep easing, `interpolate`, `resample`,
  velocity-aware `duration_for`, plus path-inspection helpers.
- `planning/task_planner.py`: the `PickPlaceJob` contract.
- `control/scripted.py`: `ScriptedController` with a six-stage waypoint sequence,
  per-stage verification and a `FailureReason` taxonomy.
- `tests/test_kinematics.py` (17), `tests/test_trajectory.py` (16),
  `tests/test_scripted_control.py` (11). Suite total: 84 passing.

**Design decisions**

- *IK is analytic, not numerical.* The spec allows numerical IK, but the arm's
  yaw + three-coplanar-pitch + roll structure has an exact closed form. It is
  faster, has no convergence failures, exposes both elbow branches explicitly, and
  round-trips against FK to machine precision (measured worst case: 3e-16 m).
- *Both elbow branches are tried.* `inverse_kinematics` prefers elbow-up, falls
  back to elbow-down, and only then reports `UnreachableError` — distinguishing
  "outside the workspace" from "reachable but violates joint limits".
- *`PickPlaceJob` lives in `planning/task_planner.py` from phase 3.* The control
  layer needs the data contract before the planner that produces it exists. Only
  the dataclass landed here; `TaskPlanner` itself arrives in phase 5.
- *Grasp verification goes through a `Protocol`.* `ScriptedController` checks for
  a `has_object()` method rather than importing `SimArm`, so `control/` stays
  independent of the simulation backend. A real arm without a grasp sensor is
  assumed to have succeeded (wrist-camera verification is a listed stretch goal).
- *New config key `rack_plate_height`* — placing onto a rack needs a TCP height
  8 mm above the table grasp height. It was previously hard-coded inside
  `sim/world.py`, which violated the no-hard-coded-hardware-values standard.

**Known issues**

- Placement lands within ~5 mm of the slot centre, well inside the 30 mm
  tolerance. Tighter placement would need closed-loop visual servoing, which is
  out of scope for the scripted baseline.

## Phase 4 — Perception ✅

**Delivered**

- `perception/types.py`: the `Detection` dataclass.
- `perception/calibration.py`: `Calibration` (save/load, pixel↔table mapping,
  parallax correction), `calibration_from_camera_pose` for sim,
  `calibrate_from_aruco` + `board_corner_table_positions` for real hardware.
- `perception/classifier.py`: `ColorClassifier` (HSV masks, per-pixel and
  per-region classification) and `QRReader` for the sample-ID stretch goal.
- `perception/detector.py`: `TubeDetector` with mask → morphology → contour →
  area/circularity/confidence filtering, plus `annotate()` for debugging.
- `tests/synthetic.py`: shared synthetic-image renderers.
- `tests/test_classifier.py` (21), `tests/test_calibration.py` (21),
  `tests/test_detector.py` (15). Suite total: 141 passing.

**Design decisions**

- *Parallax is corrected explicitly.* The homography is fitted on the table plane
  because that is where a printed ArUco board lies, but tube caps sit 64 mm above
  it, which produced 4–10 mm of localisation error. `Calibration` now carries the
  camera height and nadir and scales a detection back towards the nadir by
  `(H - h) / H`. Measured error drops from ~10 mm to ~0.01 mm in sim, and the same
  correction applies unchanged on real hardware.
- *Hue ranges are merged across the 0/179 wrap.* `classes.yaml` must express red
  as two intervals, which put pure red at a range edge and scored it 0.5.
  Rejoining them into a single `[172, 188]` interval evaluated modulo 180 scores
  pure red 1.0, as it should.
- *Saturation and value are scored by headroom, not centrality.* The upper bound
  on those channels is a don't-care ceiling, so a fully saturated cap is the best
  case and must not be penalised for sitting at the top of its range. Only hue is
  scored by centrality, where drifting either way really does mean a different class.
- *Confidence blends colour and shape.* Caps are circular from overhead, so the
  contour's isoperimetric ratio multiplies the colour score. This is what lets the
  detector reject the elongated rack plates and the arm's own links.
- *A blob is dropped when its dominant colour disagrees with the mask that found
  it*, leaving it to the other class's own pass. That keeps overlapping HSV ranges
  from producing duplicate detections of the same cap.
- *The ArUco board's table orientation is a documented convention* (table +X up
  the image, +Y left, matching `camera.yaml`). A board laid down rotated fails
  loudly with a large residual rather than silently yielding a rotated frame.
- *`rms_error_px` and `rms_error_m` are both reported.* The first is the
  reprojection residual a calibration technician reads; the second is the
  table-plane residual that bounds downstream placement error.

**Measured**

- Sim detection: 8/8 tubes found and correctly labelled, ~1.0–1.4 mm localisation
  error — roughly a tenth of the 18 mm grasp tolerance.
- Synthetic ArUco board: sub-pixel fit residual.

**Known issues**

- The HSV detector needs stable lighting on real hardware; `samplesort calibrate`
  (phase 8) exists partly to re-derive thresholds per rig. Swapping in a trained
  detector is a listed stretch goal.
