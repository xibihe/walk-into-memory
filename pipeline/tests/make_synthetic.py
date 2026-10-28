"""Build a small synthetic 'photo splat' (no real photo involved) to test the pipeline.

Writes OUT/synthetic.ply (ML Sharp-like layout: OpenCV camera frame, with
intrinsic and image_size elements) and OUT/synthetic_mask.png.
Scene: a slightly tilted floor 1.5 m below the camera, a back wall, a green
bush, a 'person' (an ellipsoid stack) and some points outside the frame.
"""
import pathlib
import sys

import cv2
import numpy as np
from plyfile import PlyData, PlyElement

C0 = 0.28209479177387814
out = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "synthetic_out")
out.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(7)
W, H, F = 320, 240, 260.0
K = np.array([[F, 0, W / 2], [0, F, H / 2], [0, 0, 1]])
parts = []


def add(xyz, rgb, size, opacity=0.9):
    parts.append((xyz, np.broadcast_to(rgb, xyz.shape) + rng.normal(0, 0.03, xyz.shape), size, opacity))


n = 30000   # floor, tilted 4 degrees around x
fx_, fz_ = rng.uniform(-6, 6, n), rng.uniform(0.8, 9, n)
tilt = np.radians(4)
add(np.column_stack([fx_, 1.5 + fz_ * np.sin(tilt), fz_ * np.cos(tilt)]), [0.45, 0.40, 0.33], 0.03)
n = 25000   # back wall
add(np.column_stack([rng.uniform(-7, 7, n), rng.uniform(-4, 1.9, n), np.full(n, 9.0) + rng.normal(0, 0.02, n)]),
    [0.55, 0.6, 0.7], 0.05)
n = 8000    # bush
c = rng.normal(0, 1, (n, 3))
add(c / np.linalg.norm(c, axis=1, keepdims=True) * rng.uniform(0, 0.8, (n, 1)) * [1.2, 0.8, 0.8] + [-1.8, 0.8, 5.0],
    [0.2, 0.55, 0.2], 0.04)
n = 9000    # person: body + head
body = rng.normal(0, 1, (n, 3)) * [0.18, 0.55, 0.12] + [0.4, 0.75, 3.0]
head = rng.normal(0, 1, (n // 4, 3)) * [0.09, 0.11, 0.09] + [0.4, -0.05, 3.0]
add(np.vstack([body, head]), [0.7, 0.35, 0.3], 0.02, 0.95)
n = 4000    # outside the picture (mirrored extrapolation)
add(np.column_stack([rng.uniform(5, 9, n), rng.uniform(-2, 1.4, n), rng.uniform(2, 6, n)]), [0.3, 0.3, 0.3], 0.05)
n = 2000    # needles along the view ray, to exercise anti-streak
needle = np.column_stack([rng.uniform(-2, 2, n), rng.uniform(-1, 1, n), rng.uniform(4, 8, n)])
parts.append((needle, np.full((n, 3), 0.5), None, 0.8))

xyz = np.vstack([p[0] for p in parts])
rgb = np.vstack([p[1] for p in parts]).clip(0, 1)
N = len(xyz)
names = ["x", "y", "z", "f_dc_0", "f_dc_1", "f_dc_2", "opacity", "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3"]
v = np.zeros(N, dtype=[(k, "<f4") for k in names])
for i, k in enumerate("xyz"):
    v[k] = xyz[:, i]
for i in range(3):
    v[f"f_dc_{i}"] = (rgb[:, i] - 0.5) / C0
op = np.concatenate([np.full(len(p[0]), p[3]) for p in parts])
v["opacity"] = np.log(op / (1 - op))
sizes = np.concatenate([np.full(len(p[0]), p[2] if p[2] else 0.01) for p in parts])
for i in range(3):
    v[f"scale_{i}"] = np.log(sizes * rng.uniform(0.7, 1.3, N))
q = rng.normal(0, 1, (N, 4))
q /= np.linalg.norm(q, axis=1, keepdims=True)
# needles: long axis (x) rotated onto the view ray
ray = needle / np.linalg.norm(needle, axis=1, keepdims=True)
from scipy.spatial.transform import Rotation
rq = np.array([Rotation.align_vectors([r], [[1, 0, 0]])[0].as_quat() for r in ray])  # x,y,z,w
q[-n:] = rq[:, [3, 0, 1, 2]]
v["scale_0"][-n:] = np.log(0.25)
for i in range(4):
    v[f"rot_{i}"] = q[:, i]

intr = np.array([(float(x),) for x in K.reshape(-1)], dtype=[("intrinsic", "<f4")])
size = np.array([(W,), (H,)], dtype=[("image_size", "<u4")])
PlyData([PlyElement.describe(v, "vertex"), PlyElement.describe(intr, "intrinsic"),
         PlyElement.describe(size, "image_size")], byte_order="<").write(str(out / "synthetic.ply"))

# person mask: project the person points and close the holes
pp = np.vstack([body, head])
u = (K[0, 0] * pp[:, 0] / pp[:, 2] + K[0, 2]).astype(int)
vv = (K[1, 1] * pp[:, 1] / pp[:, 2] + K[1, 2]).astype(int)
ok = (u >= 0) & (u < W) & (vv >= 0) & (vv < H)
mask = np.zeros((H, W), np.uint8)
mask[vv[ok], u[ok]] = 255
mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
cv2.imwrite(str(out / "synthetic_mask.png"), mask)
print("wrote", out / "synthetic.ply", N, "points;", out / "synthetic_mask.png", int((mask > 0).sum()), "mask px")
