# Interactive run viewer

A single-page operations console that replays a real simulation run in the
browser. It is not a video and not a re-creation: `data/run.json` is the
trajectory the physics engine actually produced, and the page rebuilds the arm
from the same joint frames the generated URDF declares.

```
docs/web/
├── index.html               the console (no build step, no dependencies)
├── data/run.json            460 frames of joint angles, tube poses and TCP
├── data/overhead.png        the overhead camera frame the detector panel uses
├── export_run.py            regenerates run.json from a live simulation
└── verify_kinematics.mjs    checks the JS joint hierarchy against Python's FK
```

## Viewing it

The page fetches `data/run.json`, so it needs to be served over HTTP rather
than opened as a `file://` URL:

```bash
python -m http.server 8000 --directory docs/web
# then open http://localhost:8000
```

## What it shows

| View | Contents |
| --- | --- |
| Overview | Live WebGL replay with orbit, scrub and per-frame joint/TCP readout |
| Sort log | One row per job, with grasp and release times read out of the trajectory |
| Perception | The HSV → open → contour → area chain re-run live against the real frame |
| Workspace | Top-down bench map in millimetres, with the tool position tracked live |
| Robustness | Benchmark results as injected localisation noise rises to 20 mm |
| Status | What has been measured, and what still needs hardware |

## The self-check

The console's headline claim is that its kinematics agree with the
simulation's. That is not an assertion — the page walks every frame on load,
compares its own tool-centre position against the analytic forward kinematics
Python exported, and displays the worst-case disagreement. The same check runs
standalone:

```bash
node docs/web/verify_kinematics.mjs
```

```
frames checked : 460
worst TCP error: 8.727e-13 m  (frame 289)
               = 0.873 picometres
PASS — hierarchy matches the Python kinematics
```

That figure is floating-point noise, which is the point: the browser model and
the simulation are the same model.

## Regenerating the data

```bash
python docs/web/export_run.py
```

This runs a full headless sort at `seed 42` and rewrites `data/run.json`,
sampling every 20th physics step (240 Hz → 12 Hz). Re-run
`verify_kinematics.mjs` afterwards — a regression in the kinematics or a change
to the URDF generator will show up there first.
