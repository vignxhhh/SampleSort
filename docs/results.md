# SampleSort — Benchmark Results

All numbers on this page are real output from `samplesort benchmark`, not
estimates. Regenerate them with:

```bash
samplesort benchmark --trials 20 --num-tubes 6
```

which writes `outputs/benchmark.json` and `outputs/benchmark.md`.

Every trial is a fresh, seeded PyBullet world (trial *i* uses `seed + i`), so a
given `--trials` and seed always reproduce the same layouts and the same numbers.

---

## Headline result — scripted mode

Run on 2026-10-12T07:51:54+00:00, 20 trials × 6 tubes
= 120 tubes, at commit `34156e6`, with QR sample-ID scanning on.

| Metric | scripted mode |
| --- | ---: |
| Trials | 20 |
| Tubes per trial | 6 |
| Perception noise (σ) | none |
| Tubes attempted | 120 |
| Sorting accuracy (correct rack) | 100.0% |
| Grasp success rate | 100.0% |
| Completion rate | 100.0% |
| Mean time per sample | 1.16 s ± 0.12 |
| Total wall-clock | 162.6 s |

**About the time per sample.** It is wall-clock time of the simulation on
whatever machine runs it, so it is only comparable within one machine. On the
machine above: 1.16 s with QR scanning, 0.92 s without (the wrist-camera render
and decode cost ~0.24 s per tube), and the code as of the previous published
result (`b6a8979`, 0.39 s on its own host) took 0.95 s. The earlier figure was a
faster host, not faster code.

**Failures by type**

No failures were recorded.

### What the metrics mean

| Metric | Definition |
| --- | --- |
| **Sorting accuracy** | Of the tubes that were placed, the fraction that landed in the rack their *true* class maps to. Measured against simulator ground truth, so misclassification is caught even when the motion was perfect. |
| **Grasp success rate** | Attempts whose grasp stage actually picked a tube up, counting retries in the denominator. |
| **Completion rate** | Of all tubes spawned, the fraction successfully sorted. |
| **Mean time per sample** | Wall-clock seconds per successfully sorted tube, averaged over trials. This is simulated-scene wall time on CPU, not real-robot cycle time. |

---

## Robustness: localisation-noise sweep

The simulator's perception is near-exact (≈1 mm localisation error), which is far
better than a real overhead rig achieves. With no noise injected, the scripted
baseline saturates at 100% and the headline table says nothing about how much
error the system tolerates.

`--perception-noise` injects zero-mean Gaussian error into each detection's table
position, so the benchmark can measure the envelope directly. The gripper's
configured horizontal grasp tolerance is **18 mm**
(`arm_so101.yaml: gripper.grasp_tolerance_xy`).

20 trials × 6 tubes = 120 tubes at each noise level, each trial with its own
noise draw. A missed grasp is retried once, after looking at the tube again:

| Localisation noise (σ) | Sorting accuracy | Grasp success | Completion | Jobs failed after retry |
| ---: | ---: | ---: | ---: | --- |
| 0 mm | 100.0% | 100.0% | 100.0% | — |
| 5 mm | 100.0% | 97.6% | 100.0% | — |
| 10 mm | 100.0% | 80.6% | 96.7% | `grasp_missed`×5 |
| 15 mm | 97.9% | 52.1% | 80.8% | `grasp_missed`×32, `misplaced`×1 |
| 20 mm | 96.2% | 33.6% | 65.8% | `grasp_missed`×66 |

Grasp success counts every attempt, retries included; completion is the
fraction of all tubes that ended up sorted.

Reproduce a row with:

```bash
samplesort benchmark --trials 20 --num-tubes 6 --perception-noise 10
```

**Two changes since the previous version of this table.** First, it was
produced with 10 trials per level, but every trial drew the *same* noise
sequence (the generator was seeded from the shared config), so each row was
really one noise draw; changing only `--seed` moved grasp success at 20 mm
between 5% and 54%. Noise is now seeded per trial and each level uses 120
tubes. Second, a retry used to re-run the job at the same, badly measured
position and usually missed again; it now looks at the tube first. With
independent noise but the old retry, completion was 97.5 / 86.7 / 62.5 /
47.5% at 5 / 10 / 15 / 20 mm; the fresh look raises that to 100 / 96.7 /
80.8 / 65.8%.

**Reading the curve.** Single grasps start to miss at 5 mm, but a second look
recovers them; from 10 mm some tubes miss twice, and by 20 mm a third end up
unsorted. Sorting accuracy stays at or above 96% throughout because
*classification* is unaffected by position noise — a tube that is picked up
goes to the right rack; a missed grasp is detected and logged rather than
placed wrongly. At 15–20 mm, noise starts pushing detections outside the
pickup zone, which shows up as `outside_pickup_zone` skips. A real rig's error
is partly systematic (calibration bias), which a second look does not average
away, so treat these as the best case for random error.

**Practical read.** An overhead rig needs its calibration residual well under
5 mm — ideally ~2–3 mm — for the scripted controller to run at full
reliability. `samplesort calibrate` reports its RMS reprojection error so this
can be checked before a run.

---

## Lighting: HSV vs normalisation vs a learned detector

The HSV thresholds were tuned under the simulator's default light. This study
asks how they hold up when the light changes, and whether a trained detector
does better. Run it with:

```bash
pip install -e ".[learning]"
samplesort detector-study --scenes 300 --epochs 24 --test-scenes 40
```

**Setup.** Training: 300 layouts (seeds 1000+) × 3 random lightings = 900 frames,
each with 3–8 standing tubes and, in 30% of layouts, one knocked over. Random
lighting covers exposure 0.5–1.6, warm↔cool light, white balance ±12%, light
elevation ≥ 15°, shadows half the time, vignetting ≤ 0.4 and sensor noise ≤ 4
levels. Test: 40 different layouts (seeds 5000+), each rendered under all 11
conditions below. Four conditions lie **outside every training range**:
darker (`very_dim`), noisier (`noisy_dark`), a stronger colour cast
(`magenta_cast`), and lower, warmer, harsher light (`raking_warm`).

**Detectors.** *HSV* is the shipped detector. *HSV + gray-world* is the same
detector on frames whose channels are each rescaled to a common mean — the
learned model normalises its input the same way, so without this column a gain
could not be credited to learning rather than to normalising. *Learned* is a
CenterNet-style CNN (55k parameters, 13 min on a 4-core CPU).

**Scoring.** A detection counts when it lands within 8 mm of a standing tube on
the table and names its class. Cells are tubes found *and* correctly classified;
precision was 98–100% for every detector and condition except where HSV found
nothing at all.

| Condition | HSV | HSV + gray-world | Learned |
| --- | ---: | ---: | ---: |
| nominal | 100% | 100% | 100% |
| bright (exposure ×1.45) | 100% | 100% | 100% |
| dim (×0.55) | 76% | 100% | 100% |
| warm | 100% | 100% | 100% |
| cool | 100% | 100% | 99% |
| low sun, shadows | 49% | 100% | 100% |
| vignette | 76% | 100% | 100% |
| very dim *(held out)* | 0% | 100% | 99% |
| noisy dark *(held out)* | 22% | 97% | 99% |
| magenta cast *(held out)* | 53% | 100% | **88%** |
| raking warm *(held out)* | 0% | **85%** | 99% |
| **overall** | **61%** | **98%** | **99%** |

**Reading it.** Almost all of the HSV detector's fragility is a global
exposure / colour-cast problem, and a one-line gray-world correction fixes it.
The learned model adds little on top: it is clearly better only under low,
warm, raking light (99% vs 85%), where shadows and a coloured light vary across
the frame and no single global correction suffices — and it is clearly *worse*
under a colour cast stronger than any it was trained on (88% vs 100%). Given
that, the shipped pipeline keeps the HSV detector; the practical recommendation
for a real rig is to apply gray-world correction (or fix white balance and
exposure in the camera) before thresholding.

---

## Standing knocked-over tubes back up

A fallen tube is detected from its cap outline, gripped by the cap, carried to
the righting station, held level while the overhead camera judges which end is
the body, then pitched upright and released. Measured on single knocked-over
tubes, 10 seeds × 3 sub-millimetre perturbations of the detected position, on
both OpenCV 4.11 and 5.0:

| | OpenCV 4.11 | OpenCV 5.0 |
| --- | ---: | ---: |
| Stood up, cap first, on the first attempt | 27 / 30 | 28 / 30 |
| Reported "can't tell which end is the body" (put back down and retried by the pipeline) | 3 | 2 |
| Ended in the wrong state without reporting it | **0** | **0** |

The perturbations matter: an earlier version stood only 6 of 15 tubes up, and a
1 µm change in the detected position could flip the outcome. The causes and
fixes — lowering in Cartesian steps, allowing for the arm's ~6 mm sag under
load, and ignoring racks when judging the body side — are recorded as design
decision D19 in [architecture.md](architecture.md).

**Known limit.** In one earlier multi-tube run, a tube knocked over by a
neighbour rolled 75 mm out of the work area, under the arm's base, where nothing
can reach it. The run reports it as left on the table.

---

## Sample IDs

With `qr_enabled: true` each cap carries an 8.5 mm QR label. The overhead camera
cannot resolve codes that small; a wrist camera reads them from the pre-grasp
pose, 7 cm above the cap. Three seeded 8-tube runs: **24 / 24** IDs
read, each matching the tube actually picked. Under 9 mm of injected
localisation error the wrist camera sometimes misses the code (the row is
logged with no ID rather than a wrong one).

---

## Scripted vs learned

The learned (ACT) column is produced by passing a trained checkpoint:

```bash
samplesort benchmark --trials 20 --policy checkpoints/act/
```

This renders both modes side by side in the same table. Per spec section 6, this
project does **not** train a competitive policy as part of the build — it makes
the record → train → evaluate pipeline runnable. A checkpoint from a handful of
scripted demonstrations and a few hundred steps will score far below the scripted
baseline; that is expected and is not a defect in the pipeline.

---

## Environment

| Component | Version |
| --- | --- |
| Python | 3.11.15 |
| Platform | `Linux-6.18.44-fc-v114-x86_64-with-glibc2.39` |
| NumPy | 2.4.6 |
| SciPy | 1.17.1 |
| OpenCV | 4.11.0.86 |
| PyBullet | 3.2.7 |
| SampleSort | 1.0.0 |

CI additionally runs the full suite on Python 3.10 with opencv 5.0, so both
combinations in `pyproject.toml`'s declared range are exercised.

The benchmark runs headless (`DIRECT` mode) on CPU. No GPU and no hardware are
involved, which is what lets CI run the same numbers.

---

## Not yet validated on hardware

Every number on this page comes from simulation. The following need a physical
SO-101 rig to measure and are explicitly **unvalidated**:

- Real grasp success rate on physical tubes, where friction, cap tolerance and
  servo backlash all matter and the constraint-based sim grasp models none of them.
- Real cycle time, which is bounded by servo speed rather than by CPU.
- HSV classifier accuracy under real lighting, including specular highlights on
  glossy caps and shadows cast by the arm.
- ArUco calibration residual on a printed board, versus the sub-pixel fit the
  synthetic board achieves.

See `docs/hardware_setup.md` for the rig build and the bring-up procedure.
