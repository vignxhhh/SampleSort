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
- [x] **Phase 5 — Planning + Pipeline.** Rack state, task planner, full sort loop, SQLite logging.
- [x] **Phase 6 — Benchmark + Dashboard.** Metrics JSON/Markdown, Streamlit dashboard.
- [x] **Phase 7 — Learning.** Demo recording, ACT training wrapper, evaluation, learned control.
- [x] **Phase 8 — Real hardware path.** Real arm/camera HAL, ArUco generator, calibration script, docs.
- [x] **Phase 9 — Documentation.** README, architecture diagram, results.

## Post-build steps

- [x] Step 2 — Full system testing (clean install, headless suite, every CLI path).
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

## Phase 5 — Planning + Pipeline ✅

**Delivered**

- `planning/rack_state.py`: `RackState` occupancy tracking, `SlotAssignment`,
  `RackFullError` with a message naming the rack and its capacity.
- `planning/task_planner.py`: `TaskPlanner`, `PlanResult`, `SkippedDetection`.
- `logging_db/sort_log.py`: `SortLog` and `SortRecord` over SQLite, with filtered
  queries and the aggregates the dashboard and benchmark need.
- `pipeline.py`: `SortPipeline` running perceive → plan → execute → log, plus
  `PipelineReport` with success rate, grasp rate, timing and failure breakdown.
- CLI: `samplesort sim-demo` now runs a full sort; new `samplesort run
  [--mode] [--dry-run] [--policy]`.
- `tests/test_rack_state.py` (15), `tests/test_task_planner.py` (13),
  `tests/test_sort_log.py` (18), `tests/test_pipeline_sim.py` (20).
  Suite total: 207 passing.

**Design decisions**

- *Tube geometry retuned to fix a real collision.* With 50 mm tubes, a carried
  tube hung 57 mm below the TCP and its base passed *below* the tops of tubes
  still standing on the table, knocking them over mid-transit — two grasp misses
  per 8-tube run. Tubes are now 40 + 12 mm with a 46 mm grasp height and a 70 mm
  approach clearance, giving 18 mm of clearance over a standing tube. Result: 8/8
  on seeds 42, 7 and 1234.
- *Jobs are ordered nearest-first.* Short reaches are faster and more accurate,
  and clearing near tubes first reduces the chance of brushing one while reaching
  past it.
- *Slots are reserved optimistically and released on failure*, so a failed place
  does not permanently burn a slot.
- *Sim ground truth is attached to jobs explicitly*, in `SortPipeline.plan()`,
  and only in sim. It is instrumentation — it drives grasp/placement verification
  and lets the benchmark score "did this tube reach the rack its *true* class
  belongs in". Perception and planning never read it, and it is simply absent on
  real hardware.
- *Unsortable tubes are retired.* A tube that fails its grasp and its retry would
  otherwise be re-detected forever. In sim it is removed from the scene; on real
  hardware the operator is asked to clear it by hand.
- *`run_id` groups every row from one run*, which is what makes the benchmark's
  per-trial and scripted-vs-learned comparisons possible from the log alone.
- *The sort log is append-only.* Nothing updates or deletes a row in normal
  operation, which is the right shape for a chain-of-custody record and removes
  any need for schema migrations.

**Measured**

- `samplesort sim-demo --seed 42 --num-tubes 8`: 8/8 sorted, ~0.5 s per job,
  24 rows written to `outputs/sortlog.db` across three seeds, every row in the
  rack its class maps to.

**Known issues**

- Jobs are planned in a batch from one frame, so if the arm did disturb a tube
  mid-batch the remaining pick positions would be stale. The clearance fix removes
  the cause; the outer loop re-perceives and recovers if it ever happens anyway.

## Phase 6 — Benchmark + Dashboard ✅

**Delivered**

- `benchmark.py`: `run_benchmark`, `TrialResult`, `BenchmarkReport` (JSON +
  Markdown rendering, optional scripted-vs-learned comparison column),
  `write_report`.
- `dashboard/app.py`: Streamlit app over the sort log — headline metrics,
  per-class and per-rack charts, time-per-sample trend, failure breakdown,
  recent-sorts table, with mode/class/run filters.
- CLI: `samplesort benchmark [--trials] [--num-tubes] [--output-dir] [--policy]
  [--perception-noise]` and `samplesort dashboard [--port] [--host] [--headless]`.
- `docs/results.md` filled in with real measured numbers.
- `tests/test_benchmark.py` (22), `tests/test_dashboard.py` (6).
  Suite total: 235 passing.

**Design decisions**

- *Sorting accuracy is scored against ground truth, not against the controller's
  own verdict.* A misclassified tube is placed perfectly, just in the wrong rack —
  only the simulator's true label catches that. This is what makes "sorting
  accuracy" and "grasp success rate" two genuinely different numbers.
- *Added `--perception-noise`.* Without it the benchmark reported 100% on
  everything and carried no information: sim perception is ~1 mm accurate against
  an 18 mm grasp tolerance, so there was no way for the scripted baseline to fail.
  Injecting localisation error turns the benchmark into a sensitivity measurement
  and, as a side effect, exercises the whole failure-reporting path end to end.
  Default is 0.0, so the headline numbers stay honest.
- *Every trial is its own seeded world* (`seed + i`), so trial-to-trial variance
  is real scene variation rather than RNG drift, and any run is reproducible.
- *Benchmark rows go into the same sort log* under a `bench-<mode>-<seed>` run id,
  so the dashboard can show benchmark data without a second storage path.
- *The dashboard's `main()` is guarded by `__name__ == "__main__"`.* Streamlit
  runs the file as `__main__`, so it renders when served and stays importable for
  the data-layer tests.
- *`_environment()` uses `importlib.metadata`* rather than `__version__`
  attributes, because PyBullet does not expose one.

**Measured** (see `docs/results.md` for the full tables)

- Scripted, 20 trials × 6 tubes = 120 tubes: **100% sorting accuracy, 100% grasp
  success, 100% completion, 0.38 s ± 0.03 per sample**, 55.4 s wall clock, zero failures.
- Noise sweep: fully robust to 5 mm localisation error; grasp success falls to
  73.9% at 10 mm, 50.0% at 15 mm and 33.3% at 20 mm. Sorting accuracy holds at
  100% until 20 mm, since position noise does not affect classification.
- `samplesort dashboard` serves HTTP 200 and `/_stcore/health` returns `ok`.

**Known issues**

- Without injected noise the scripted baseline saturates, so the headline table
  cannot distinguish a good controller from a perfect simulator. The noise sweep
  exists to give the results page something falsifiable to say.

## Phase 7 — Learning ✅

**Delivered**

- `learning/_deps.py`: lazy torch/LeRobot imports with an actionable install
  message, plus `resolve_device` fallback.
- `learning/record.py`: `DemonstrationRecorder` and `record_episodes` writing
  LeRobot-format datasets from scripted demonstrations.
- `learning/train.py`: `load_dataset`, `build_policy_config`, `train`,
  `load_policy`, and a `training_summary.json` artefact.
- `learning/evaluate.py`: `evaluate` running K rollouts, `EvaluationReport`,
  and an `evaluation.json` artefact.
- `control/learned.py`: `LearnedController` running the ACT policy closed-loop.
- New `learning:` config section; CLI `record`, `train`, `evaluate`.
- `tests/test_learning.py` (23). Suite total: 258 passing.

**Design decisions**

- *`LearnedController` subclasses `ScriptedController`.* Waypoint planning,
  feasibility checking, grasp sensing and placement verification are identical
  between the two modes; only `execute()` differs. Subclassing means the pipeline,
  the benchmark and the CLI accept either with no branching, and the two modes are
  scored by exactly the same rules — which is what makes the comparison column meaningful.
- *Demonstrations are recorded by replaying the scripted controller's solved
  waypoints*, interpolated at the dataset frame rate, rather than by hooking into
  `move_to_joints`. That keeps the HAL free of recording concerns and gives exact
  control over the capture rate.
- *Failed demonstrations are discarded by default.* Imitation learning from a
  failed demonstration teaches the failure; `--keep-failures` is there for anyone
  who wants them anyway.
- *Datasets store PNG frames (`use_videos=False`).* Video encoding pulls in an
  ffmpeg/codec dependency that would break CI and offline machines, and these
  datasets are small enough that the size difference does not matter.
- *`pretrained_backbone_weights=None`.* The ACT default downloads ImageNet
  weights from the network on first use, which fails offline and in CI.
- *Hugging Face libraries are forced offline* in `_deps._force_offline()`, because
  LeRobot otherwise resolves a dataset's `repo_id` against the Hub even for a
  purely local dataset. An explicit environment setting always wins, so opting in
  to the Hub is still possible.
- *An untrained policy emitting out-of-limit joint targets is a failed rollout,
  not a crash.* `LearnedController.execute` catches `JointLimitError` and scores it
  as `arm_error`.
- *Observation space is the overhead image (downscaled to 96×96) plus five joints
  and a gripper-open fraction*; the action space is the same six values. Small
  images keep CPU training tractable, which is the whole point of the defaults.

**Measured** (real runs, not estimates)

- `samplesort record --episodes 3 --num-tubes 3`: 3 episodes, 93 frames,
  3/3 successful demonstrations.
- `samplesort train --steps 30 --batch-size 4`: loss 75.22 → 4.29 on CPU in ~4 s;
  checkpoint written with `config.json`, `model.safetensors`, `training_summary.json`.
- `samplesort evaluate --episodes 3 --num-tubes 1`: ran 3 rollouts, 0/3 successes,
  all scored `grasp_missed`, `evaluation.json` written.

**Known issues**

- The evaluated policy performs badly (0% success). This is expected and is not a
  pipeline defect: spec section 6 explicitly says not to train the policy as part
  of the build. 30 optimiser steps on 93 frames cannot produce competent
  manipulation. The pipeline demonstrably runs end to end, which is what was asked
  for. A real run would want hundreds of episodes and tens of thousands of steps.

## Phase 8 — Real hardware path ✅

**Delivered**

- `hal/real_arm.py`: `RealArm` over LeRobot's `FeetechMotorsBus`, with unit and
  frame conversion, streamed trajectories, joint-limit enforcement and
  calibration loading.
- `hal/real_camera.py`: `RealCamera` over `cv2.VideoCapture`, with resolution
  verification, warm-up frames and dropped-frame retries.
- `scripts/generate_aruco_board.py` and `scripts/calibrate_camera.py`.
- CLI: `samplesort calibrate [--generate-board] [--image] [--output]`.
- `docs/hardware_setup.md`: BOM, geometry, bring-up, calibration, safety,
  troubleshooting.
- New arm config keys: `servo_calibration_path`, `servo_offsets_deg`, `servo_signs`.
- `tests/test_real_hal.py` (31) plus a board round-trip test. Suite total: 290 passing.

**Design decisions**

- *Written against the installed LeRobot 0.4.4 source, not against assumptions.*
  Verified by reading `lerobot/motors/feetech/feetech.py` and
  `lerobot/robots/so_follower/`: motors are declared as
  `{name: Motor(id, model, norm_mode)}`, body joints use `MotorNormMode.DEGREES`
  and the gripper `RANGE_0_100`, and motion goes through
  `sync_read("Present_Position")` / `sync_write("Goal_Position", ...)`.
- *Calibration is required, and says so.* `_normalize` in LeRobot's bus raises
  `"has no calibration registered"` without one. `RealArm` loads a calibration
  file up front and fails with a message naming the fix and the doc, rather than
  letting an opaque SDK error surface mid-run.
- *Two explicit conversions, both configurable.* The bus reports degrees from each
  joint's calibrated mid-position; the kinematics use radians from a different
  zero. `servo_offsets_deg` and `servo_signs` bridge them, and §3.3 of the
  hardware doc is a measured procedure rather than a guess.
- *Moves are streamed, not sent as a single goal.* Writing one `Goal_Position`
  would let each servo slew at its own internal rate; interpolating at 50 Hz gives
  the same smoothstep profile the simulator uses.
- *The camera verifies the resolution it actually got.* Many webcams silently
  ignore a `set()` and deliver a different size — and a homography fitted at one
  resolution is wrong at another. This is caught at `connect()` rather than
  becoming mysterious grasp errors.
- *Warm-up frames are discarded.* Auto-exposure needs a moment, and a dark first
  frame would fail every HSV threshold.
- *The board generator and the ArUco solver share
  `board_corner_table_positions`*, so the printed target and the solver cannot
  disagree about marker ids, corner ordering or orientation. A test renders the
  real script's output and fits it: 0.96 px residual, 0.08 mm on the table plane.

**Measured**

- Generated board → `calibrate_from_aruco`: 10 markers detected, 0.96 px
  reprojection residual, 0.08 mm table-plane residual, exact corner round-trip.

**Known issues**

- **Not validated on physical hardware.** No SO-101 was available. The logic
  between SampleSort and the hardware — conversions, limit enforcement,
  calibration handling, error paths — is covered by 31 tests against fakes, but
  the serial link itself has never been exercised. `docs/hardware_setup.md` and
  `docs/results.md` both say so plainly.
- The HSV thresholds in `classes.yaml` are tuned for the simulator's rendering and
  will need retuning under real lighting. §4.4 of the hardware doc covers it.

## Phase 9 — Documentation ✅

**Delivered**

- `README.md`: overview, what-it-does table, sim quickstart with real output,
  Mermaid architecture diagram, measured benchmark tables, hardware summary
  linking `docs/hardware_setup.md`, project structure, development commands,
  roadmap with the spec's stretch goals, and a clearly marked demo-GIF placeholder.
- `docs/architecture.md`: a full system flowchart, a sequence diagram of the sort
  loop, a layer-contract table, and fifteen recorded design decisions (D1–D15).
- `LICENSE`: MIT.

**Design decisions**

- *Every number in the README is real output*, pulled from
  `outputs/benchmark.json` when the file was generated — not an estimate.
- *The layer-contract table is the load-bearing piece of the architecture doc.*
  The invariant that nothing above `hal/` imports a concrete backend is what makes
  the sim and real paths interchangeable, so it is stated explicitly rather than
  left implicit in a diagram.
- *Design decisions are recorded with the problem that motivated them*, not just
  the choice. Several (D6 parallax, D7 hue scoring, D15 tube geometry) exist
  because a measurement came back wrong first.
- *Mermaid diagrams are structurally validated* by a parser check on block
  openers and `end` keywords, so a malformed diagram cannot ship silently.

**Known issues**

- The demo GIF is a placeholder. Recording one needs a display for
  `sim-demo --gui`, which this build environment does not have. The README
  documents the exact command to record it.

## Step 2 — Full system testing ✅

| Check | Result |
| --- | --- |
| Clean install: fresh venv, `pip install -e .` | **PASS** — resolved and installed, console script works |
| Suite headless, no hardware, no GPU | **PASS** — 292 passed in the dev env; 278 passed / 14 skipped in the torch-free env |
| `sim-demo --seed 42 --num-tubes 8` sorts correctly and logs | **PASS** — 8/8, all 8 rows in the correct rack |
| `run --mode scripted --dry-run` | **PASS** — exit 0, perceives and plans without moving |
| `benchmark --trials 20` produces JSON + Markdown | **PASS** — 100% across 120 tubes, artefacts written |
| `record` → `train` → `evaluate` end to end | **PASS** — 3 eps/93 frames → loss 4.39 → 3 rollouts scored |
| `dashboard` launches and serves | **PASS** — HTTP 200, `/_stcore/health` = `ok`, 128 rows available |
| CI workflow valid, same steps as local | **PASS** — ruff check, ruff format --check, mypy, pytest + CLI smoke, on 3.10 and 3.11 |
| Every command's `--help` | **PASS** — all 8 commands plus the root |

**Bugs this step found and fixed**

1. **`RealArm.load_calibration()` imported LeRobot unconditionally.** Running the
   clean-install suite (no torch, no LeRobot) surfaced `ModuleNotFoundError` where
   an actionable `ArmError` was intended — exactly the path a user without the
   optional extra would hit. Split into `read_calibration_file()` (pure, validates
   presence, JSON and required fields) and `load_calibration()` (constructs
   LeRobot objects, with its own install hint). Two new tests.
2. **Dangling grasp constraint.** Retiring an unsortable tube removed its
   PyBullet body while the gripper's fixed constraint still referenced it,
   producing `removeConstraint failed` on stdout during `evaluate`. The pipeline
   now releases the gripper before removing the body.
3. **PyBullet banner polluted stdout.** `argv[0]=` was landing in the CLI's own
   output. PyBullet writes it from native code to file descriptor 1, so Python
   redirection does not catch it; `sim/_bullet.py` now imports PyBullet once
   behind an fd-level redirect. stdout is now clean and pipeable; the build
   banner correctly stays on stderr.

**Dependency pinning**

`pyproject.toml` now carries verified upper bounds, and `constraints.txt` records
the exact versions everything was measured with. The core install was
additionally verified against **opencv-python-headless 5.0.0.93** — newer than
the 4.11 the dev environment uses — confirming the `<6` bound is real rather
than assumed.
