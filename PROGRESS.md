# SampleSort — Build Progress

Tracks the Build Plan from section 8 of the specification. Every phase is
implemented, tested, linted, type-checked and committed before the next begins.

Resume rule: continue from the first unchecked item below.

## Build Plan

- [x] **Phase 1 — Scaffold.** `pyproject.toml`, package layout, Pydantic config
      models + YAML configs, CLI skeleton, CI workflow, ruff/mypy/pytest config.
- [ ] **Phase 2 — HAL + Simulation.** Abstract interfaces, PyBullet world, sim arm,
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
