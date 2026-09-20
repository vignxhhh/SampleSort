/* Rebuild the arm's joint hierarchy exactly as the generated URDF declares it,
   then check the resulting tool position against the analytic forward
   kinematics Python exported for every frame. */
import { readFileSync } from "node:fs";
const D = JSON.parse(readFileSync(new URL("./data/run.json", import.meta.url), "utf8"));
const [l1, l2, l3] = D.arm.links, h0 = D.arm.base_height;

const I = () => [1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1];
function mul(a, b) {                       // column-major, a*b
  const o = new Array(16);
  for (let c = 0; c < 4; c++) for (let r = 0; r < 4; r++) {
    let s = 0;
    for (let k = 0; k < 4; k++) s += a[k*4+r] * b[c*4+k];
    o[c*4+r] = s;
  }
  return o;
}
const T = (x,y,z) => [1,0,0,0, 0,1,0,0, 0,0,1,0, x,y,z,1];
function axisRot(ax, ay, az, t) {          // Rodrigues, column-major
  const n = Math.hypot(ax,ay,az); ax/=n; ay/=n; az/=n;
  const c = Math.cos(t), s = Math.sin(t), k = 1-c;
  return [
    c+ax*ax*k,      ay*ax*k+az*s,  az*ax*k-ay*s, 0,
    ax*ay*k-az*s,   c+ay*ay*k,     az*ay*k+ax*s, 0,
    ax*az*k+ay*s,   ay*az*k-ax*s,  c+az*az*k,    0,
    0,0,0,1,
  ];
}

function tcpOf(q) {
  let m = I();
  m = mul(m, T(0, 0, 0));            m = mul(m, axisRot(0, 0, 1, q[0]));  // base_yaw
  m = mul(m, T(0, 0, h0));           m = mul(m, axisRot(0,-1, 0, q[1]));  // shoulder
  m = mul(m, T(l1, 0, 0));           m = mul(m, axisRot(0,-1, 0, q[2]));  // elbow
  m = mul(m, T(l2, 0, 0));           m = mul(m, axisRot(0,-1, 0, q[3]));  // wrist_pitch
  m = mul(m, T(l3, 0, 0));           m = mul(m, axisRot(1, 0, 0, q[4]));  // wrist_roll
  return [m[12], m[13], m[14]];                                           // tool_link origin
}

let worst = 0, worstFrame = -1;
D.frames.forEach((f, i) => {
  const p = tcpOf(f.q);
  const e = Math.hypot(p[0]-f.tcp[0], p[1]-f.tcp[1], p[2]-f.tcp[2]);
  if (e > worst) { worst = e; worstFrame = i; }
});
console.log("frames checked :", D.frames.length);
console.log("worst TCP error:", worst.toExponential(3), "m  (frame", worstFrame + ")");
console.log("               =", (worst * 1e12).toFixed(3), "picometres");
console.log(worst < 1e-9 ? "PASS — hierarchy matches the Python kinematics"
                         : "FAIL — the JS hierarchy disagrees");
