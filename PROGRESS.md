# SampleSort — Build Progress

Tracks the Build Plan from section 8 of the specification. Every phase is
implemented, tested, linted, type-checked and committed before the next begins.

Resume rule: continue from the first unchecked item below.

## Build Plan

- [x] **Phase 1 — Scaffold.** `pyproject.toml`, package layout, Pydantic config
      models + YAML configs, CLI skeleton, CI workflow, ruff/mypy/pytest config.
- [x] **Phase 2 — HAL + Simulation.** Abstract interfaces, PyBullet world, sim arm,
      sim camera, `samplesort sim-demo`.
- [ ] **Phase 3 — Kinematics + Scripted control.** FK/IK, trajectory interpolation,
      scripted pick/place of a single tube in sim.
- [ ] **Phase 4 — Perception.** Calibration, detector, colour classifier, synthetic-image tests.
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
