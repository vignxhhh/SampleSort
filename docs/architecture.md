# SampleSort Architecture

*(Filled in during Phase 9. Design decisions are recorded below as they are made.)*

## Design Decisions

### D1 — Arm geometry is generated from config, not hard-coded

The analytic kinematics and the PyBullet URDF are both derived from
`configs/arm_so101.yaml`. This makes it impossible for the simulated arm and the
kinematic model to disagree, and satisfies the "no hard-coded hardware values"
standard from section 7 of the spec.

### D2 — The learning stack is an optional extra

`torch` and `lerobot` are declared under the `learning` extra rather than as core
dependencies. The sim, perception, planning, pipeline, benchmark and dashboard
paths all install and run without them, which keeps `pip install -e .` light and
keeps CI fast.
