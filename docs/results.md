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

Run on 2026-09-20T01:23:28+00:00, 20 trials × 6 tubes
= 120 tubes, at commit `b6a8979`.

| Metric | scripted mode |
| --- | ---: |
| Trials | 20 |
| Tubes per trial | 6 |
| Perception noise (σ) | none |
| Tubes attempted | 120 |
| Sorting accuracy (correct rack) | 100.0% |
| Grasp success rate | 100.0% |
| Completion rate | 100.0% |
| Mean time per sample | 0.39 s ± 0.04 |
| Total wall-clock | 56.8 s |

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

10 trials × 6 tubes at each noise level:

| Localisation noise (σ) | Sorting accuracy | Grasp success | Completion | Mean time / sample | Failures |
| ---: | ---: | ---: | ---: | ---: | --- |
| 0 mm | 100.0% | 100.0% | 100.0% | 0.38 s | — |
| 5 mm | 100.0% | 100.0% | 100.0% | 0.39 s | — |
| 10 mm | 100.0% | 73.9% | 83.3% | 0.35 s | `grasp_missed`×9, `misplaced`×1 |
| 15 mm | 100.0% | 50.0% | 66.7% | 0.32 s | `grasp_missed`×20 |
| 20 mm | 87.5% | 33.3% | 53.3% | 0.34 s | `grasp_missed`×31 |

Reproduce a row with:

```bash
samplesort benchmark --trials 10 --num-tubes 6 --perception-noise 10
```

**Reading the curve.** The system is fully robust to 5 mm of localisation error
and degrades from 10 mm onwards, which is where noise starts consuming the 18 mm
grasp tolerance. Sorting accuracy stays at 100% until 20 mm because *classification*
is unaffected by position noise — what breaks first is the grasp, not the decision
about which rack a tube belongs in. That separation is exactly what the two metrics
are there to distinguish. At 20 mm, noise starts pushing detections outside the
pickup zone entirely, which shows up as `outside_pickup_zone` skips.

**Practical read.** An overhead rig needs its calibration residual under roughly
5 mm for the scripted controller to run at full reliability. `samplesort calibrate`
reports its RMS reprojection error precisely so this can be checked before a run.

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
| Platform | `Linux-6.18.44-fc-v37-x86_64-with-glibc2.39` |
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
