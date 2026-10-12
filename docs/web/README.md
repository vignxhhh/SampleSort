# Interactive run viewer

A single page that replays real simulation runs in the browser. It is not a
video and not a re-creation: the run files are the trajectories the physics
engine actually produced, and the page rebuilds the arm from the same joint
frames the generated URDF declares.

```
docs/web/
├── index.html                 the page (no build step, no dependencies)
├── data/run.json              clean run: 8 tubes, 460 frames
├── data/run_recovery.json     recovery run: 6 mm noise, 2 tubes knocked over
├── data/overhead.png          the overhead frame the HSV lab runs on
├── data/wrist_scan.png/.json  the wrist camera's view of a QR-labelled cap
├── data/lighting.jpg          one test layout under the 11 lighting conditions
├── data/detector_study.json   output of `samplesort detector-study`
├── export_run.py              regenerates the run files from a live simulation
├── export_assets.py           regenerates the images and copies the study
├── verify_kinematics.mjs      checks the JS joint hierarchy against Python's FK
├── record_gif.mjs             records ../demo.gif from the page, headlessly
└── make_gif.py                assembles the recorded frames into the GIF
```

## Viewing it

The page fetches its data, so serve it over HTTP rather than opening it as a
`file://` URL:

```bash
python -m http.server 8000 --directory docs/web
# then open http://localhost:8000
```

## What it shows

| Section | Contents |
| --- | --- |
| Live run | WebGL replay with orbit and scrub; switch between the clean and the recovery run; event log with sample IDs, missed grasps, retries and righting |
| Capabilities | The wrist camera's QR read, righting, retries, the lighting result and the live kinematics check, each linked to the moment in the replay |
| Lighting study | Every condition's frame, and per-detector scores (HSV, HSV + gray-world, learned CNN), with held-out conditions marked |
| Perception lab | The HSV → open → contour → area chain re-run live against the real overhead frame |
| Robustness | Benchmark results as injected localisation noise rises to 20 mm |
| How it works | The six stages, and a top-down bench map with the tool tracked live |
| Limits | What has not been measured or does not work yet |

## The self-check

The page's headline claim is that its kinematics agree with the simulation's.
That is not an assertion — on load it walks every frame of the selected run,
compares its own tool-centre position against the analytic forward kinematics
Python exported, and displays the worst-case disagreement. The same check runs
standalone:

```bash
node docs/web/verify_kinematics.mjs                          # clean run
node docs/web/verify_kinematics.mjs ./data/run_recovery.json  # recovery run
```

```
frames checked : 460
worst TCP error: 8.758e-13 m  (frame 280)
               = 0.876 picometres
PASS — hierarchy matches the Python kinematics
```

That figure is floating-point noise, which is the point: the browser model and
the simulation are the same model.

## Regenerating the data

```bash
python docs/web/export_run.py nominal     # data/run.json, seed 42
python docs/web/export_run.py recovery    # data/run_recovery.json, seed 3
samplesort detector-study                 # needs the learning extra
python docs/web/export_assets.py outputs  # images + study results
```

Runs are sampled every 20th physics step (240 Hz → 12 Hz). Re-run
`verify_kinematics.mjs` afterwards — a regression in the kinematics or a change
to the URDF generator will show up there first.

To re-record the README GIF (needs Node with `playwright` installed):

```bash
python -m http.server 8899 --directory docs/web &
node docs/web/record_gif.mjs http://localhost:8899/index.html /tmp/frames nominal 4
python docs/web/make_gif.py /tmp/frames docs/demo.gif 640 12
```
