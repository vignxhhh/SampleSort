# SampleSort — Hardware Setup

Everything in SampleSort runs in simulation with no hardware. This page covers
the physical rig for `mode: real`.

> **Status.** The real-hardware path is implemented and type-checked against
> LeRobot 0.4.4's actual `FeetechMotorsBus` API, but it has **not been validated
> on a physical arm**. Treat this as a bring-up procedure to follow carefully,
> not a tested recipe. `docs/results.md` lists exactly which numbers still need
> hardware to confirm.

---

## 1. Bill of materials

| Item | Notes |
| --- | --- |
| SO-101 (or SO-100) follower arm | 5 DOF + parallel gripper, Feetech STS3215 serial-bus servos |
| SO-101 leader arm | Only needed for teleoperated recording |
| Feetech USB-to-TTL bus adapter | Usually enumerates as `/dev/ttyACM0` |
| 12 V / 5 A power supply | For the servo bus — **not** USB-powered |
| USB webcam | 640×480 or better, fixed focus preferred |
| Overhead camera mount | Rigid; any drift invalidates the calibration |
| Flat matte table surface | Light, non-glossy; gloss creates specular highlights that break HSV thresholds |
| Printed ArUco board | See §4 |
| Sample tubes with coloured caps | Red, blue, green, yellow by default |
| Tube racks | Slot pitch must match `workspace.yaml` |

---

## 2. Workspace geometry

Every dimension lives in `configs/workspace.yaml` and `configs/arm_so101.yaml`.
Nothing is hard-coded, so adapt the config to your table rather than the reverse.

The frame is the **arm base frame**, in metres:

- Origin at the centre of the arm's mounting plate, on the table surface.
- **+X** points away from the arm, across the workspace.
- **+Y** points to the arm's left.
- **+Z** points up.

Shipped defaults:

| Element | Value |
| --- | --- |
| Arm reach (planar) | 350 mm (175 + 175 mm links, 80 mm shoulder height) |
| Pickup zone | X ∈ [130, 250] mm, Y ∈ [−135, 135] mm |
| Rack A (red) | Y = +200 mm, X = 100 / 145 / 190 / 235 mm |
| Rack B (blue) | Y = −200 mm, same X positions |
| Rack C (green, yellow) | X = 295 mm, Y = +75 / +25 / −25 / −75 mm |
| Tube | 8 mm radius, 40 mm body, 12 mm cap |
| Grasp height | 46 mm (cap centre) |
| Approach clearance | 60 mm |

**If you change the arm's link lengths or the rack positions, re-check reach.**
Run `samplesort config` to validate, then:

```bash
python -c "
from samplesort.config import load_config
from samplesort.planning.kinematics import Pose, is_reachable
c = load_config()
ws = c.workspace
for rack in ws.racks:
    for i, (x, y) in enumerate(rack.slots):
        for z in (ws.rack_grasp_height, ws.rack_grasp_height + ws.approach_clearance):
            ok = is_reachable(c.arm, Pose(x, y, z))
            print(f'{rack.id} slot {i} z={z:.3f}: {\"OK\" if ok else \"UNREACHABLE\"}')
"
```

The test suite checks this for the shipped config
(`tests/test_kinematics.py::test_every_pickup_and_rack_pose_is_reachable`), so
adapting the config and re-running `pytest` will also catch an unreachable layout.

---

## 3. Arm bring-up

### 3.1 Wiring and permissions

1. Daisy-chain the servos, IDs **1–5** for the joints and **6** for the gripper,
   base outwards. These are `servo_ids` and `gripper_servo_id` in `arm_so101.yaml`.
2. Connect the bus adapter and the 12 V supply. Power the servos **before**
   opening the port.
3. Find the port:
   ```bash
   ls /dev/ttyACM* /dev/ttyUSB*
   ```
   and set `serial_port` in `arm_so101.yaml`.
4. On Linux, grant yourself access to the port:
   ```bash
   sudo usermod -aG dialout "$USER"   # then log out and back in
   ```

### 3.2 Servo calibration

The Feetech bus **cannot report normalised positions without a calibration** —
`RealArm` will refuse to connect and say so. Calibration records each joint's
homing offset and travel range. Use LeRobot's own tool, which is what wrote the
format `RealArm` reads:

```bash
python -m lerobot.calibrate --robot.type=so101_follower \
    --robot.port=/dev/ttyACM0 --robot.id=samplesort
```

Point `servo_calibration_path` in `arm_so101.yaml` at the JSON it writes. The
file maps each motor name to `{id, drive_mode, homing_offset, range_min, range_max}`.

### 3.3 Aligning the kinematic zero

This is the step most likely to bite, so do it deliberately.

LeRobot reports each joint in **degrees measured from its calibrated
mid-position**. SampleSort's kinematics use a different zero: **all joints at 0
means the arm is extended horizontally along +X with the tool also pointing
along +X**. The two are reconciled by `servo_offsets_deg` and `servo_signs`.

1. With torque **disabled**, move the arm by hand to the kinematic zero pose —
   fully extended, horizontal, tool pointing straight out along +X. A straight
   edge or a printed side-profile template helps.
2. Read the raw servo angles:
   ```bash
   python -c "
   from samplesort.config import load_config
   from samplesort.hal.real_arm import RealArm
   c = load_config(mode='real')
   arm = RealArm(c.arm); arm.connect()
   print(arm.bus.sync_read('Present_Position', c.arm.joint_names))
   arm.disconnect()
   "
   ```
3. Put those readings into `servo_offsets_deg`, in `joint_names` order.
4. Check each joint's direction: command a small positive change and confirm the
   joint moves the way the kinematics expect (a positive pitch angle **raises**
   the link; positive base yaw turns towards **+Y**). Flip the corresponding
   entry in `servo_signs` to `-1` for any joint that moves the wrong way.
5. Verify the round trip — commanding the home pose should land the tool where
   forward kinematics says it will:
   ```bash
   python -c "
   from samplesort.config import load_config
   from samplesort.planning.kinematics import forward_kinematics
   c = load_config(mode='real')
   print('home TCP should be at:', forward_kinematics(c.arm, c.arm.home_position))
   "
   ```
   Measure the real tool position with a ruler. Disagreement here means the
   offsets or signs are wrong — **fix it before going further**, because every
   grasp depends on it.

### 3.4 First motion — carefully

Keep a hand on the power switch.

```bash
samplesort config                        # validate everything first
samplesort run --mode scripted --dry-run # perceive and plan only, no motion
```

`--dry-run` exercises the camera, calibration, detector and planner with the arm
stationary. Only once that looks right should you allow motion.

---

## 4. Camera and table calibration

### 4.1 Mount the camera

Mount it looking **straight down** at the workspace, rigidly. Set `position` and
`look_at` in `camera.yaml` to the measured mounting pose — these feed the
parallax correction, which matters because tube caps sit 52 mm above the table
plane the homography is fitted on.

**Orientation convention (important).** The camera must be rotated so that:

- table **+X points up** the image,
- table **+Y points left**.

This matches `camera.yaml`'s `up_axis: [1, 0, 0]`. The ArUco solver assumes it. A
board laid down rotated relative to this will fail loudly with a large residual
rather than silently giving you a rotated frame — but it is much easier to mount
it correctly than to debug it later.

### 4.2 Print the board

```bash
samplesort calibrate --generate-board --board-output docs/aruco_board.png --dpi 300
```

Print at **100% scale** — no "fit to page", no "shrink to fit". Then **measure a
marker with a ruler**: it must be exactly the `marker_length` from
`workspace.yaml` (30 mm by default). A board printed at 96% produces a
calibration that is wrong by 4% everywhere, and nothing will tell you.

Mount the board flat on the table with its origin corner at the `origin`
coordinates in `workspace.yaml` (50 mm, −150 mm by default). Tape it down flat —
a curled board bends the homography.

### 4.3 Fit the homography

```bash
samplesort calibrate --save-frame outputs/calibration_frame.png
```

This writes `configs/homography.npz`, which the pipeline loads automatically. It
reports two residuals:

- **reprojection error (px)** — under ~1 px is good, over 3 px triggers a warning.
- **table-plane error (mm)** — this bounds your grasp accuracy.

Per `docs/results.md`, the scripted controller is fully robust up to about 5 mm of
localisation error and degrades past 10 mm. **Aim for a table-plane residual
under 3 mm.** If it is worse, the usual causes are, in order: the board is not
flat, it was printed at the wrong scale, the camera is not looking straight down,
or the mount flexes.

Remove the board before sorting.

### 4.4 Tune the colour thresholds

The HSV ranges in `classes.yaml` are tuned for the simulator's rendering. Real
lighting is different — this will need adjusting, and it is the single most
likely thing to need work on a new rig.

```bash
python -c "
import cv2
from samplesort.config import load_config
from samplesort.hal.real_camera import RealCamera
c = load_config(mode='real')
with RealCamera(c.camera) as cam:
    frame = cam.read()
cv2.imwrite('outputs/frame.png', frame)
hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
print('click a cap in outputs/frame.png, then read its HSV here')
print('centre pixel HSV:', hsv[hsv.shape[0]//2, hsv.shape[1]//2])
"
```

Sample several pixels from each cap colour, widen the ranges to cover them, and
verify with:

```bash
samplesort run --mode scripted --dry-run
```

which prints what it detected and where it planned to put it. Practical advice:

- **Diffuse, even lighting.** A ring light or a diffused overhead panel. Avoid a
  single point source — specular highlights on glossy caps read as white and
  fall outside every saturation range.
- **Keep the arm out of shot** during perception. The pipeline already retracts
  to `observe_position` before each frame, but check that pose really does clear
  the pickup zone on your rig.
- **Saturation floor is the important threshold.** Raising `saturation_range`'s
  lower bound is usually what rejects the table, the racks and the arm.

---

## 5. Running on hardware

Set `mode: real` in `configs/default.yaml`, then:

```bash
samplesort run --mode scripted --dry-run   # perceive and plan, no motion
samplesort run --mode scripted             # the real thing
samplesort dashboard                       # watch the sort log
```

### Teleoperated recording

With both leader and follower arms connected, `samplesort record` captures
teleoperation instead of generating scripted demonstrations. Wire the leader arm
through LeRobot's `so101_leader` teleoperator and point the recorder at it; the
dataset format, the training wrapper and the evaluation loop are all unchanged
from the simulated path.

---

## 6. Safety

- **Keep the workspace clear of hands.** These servos are small but they do not
  stop for you.
- **Power-cycle to recover, don't grab the arm.** Reaching into a moving arm is
  how people get pinched.
- Torque is disabled on `disconnect()`, so the arm goes limp when the process
  exits cleanly — **support it** if it is holding a raised pose.
- **Always `--dry-run` first** after changing calibration, config or thresholds.
- Joint limits from `arm_so101.yaml` are enforced on every command, but they only
  protect against bad *commands* — they cannot stop a collision with an object
  the planner does not know about.

---

## 7. Troubleshooting

| Symptom | Likely cause |
| --- | --- |
| `could not connect to the arm` | Wrong port, no permission (`dialout` group), or servos unpowered |
| `no servo calibration at ...` | Run LeRobot's calibration (§3.2) and set `servo_calibration_path` |
| Arm moves to the wrong place entirely | `servo_offsets_deg` / `servo_signs` not aligned (§3.3) |
| One joint moves backwards | Flip that joint's entry in `servo_signs` |
| `no ArUco markers found` | Board out of frame, out of focus, poorly lit, or a different dictionary |
| Large calibration residual | Board not flat, printed at the wrong scale, or camera not perpendicular |
| Nothing detected, but the caps are clearly visible | HSV ranges not tuned for your lighting (§4.4) |
| Detections drift a few mm and grasps miss | Camera mount moved since calibration — re-run `samplesort calibrate` |
| `camera ... refused WxH` | Set `camera.yaml`'s width/height to a format the device supports |
| Grasps miss consistently in one direction | Calibration residual, or the parallax `position`/`look_at` values are wrong |
