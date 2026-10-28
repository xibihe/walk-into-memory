"""Read and write 3DGS PLY files while keeping the camera metadata elements."""
from dataclasses import dataclass, field

import numpy as np
from plyfile import PlyData, PlyElement

C0 = 0.28209479177387814  # SH degree-0 constant

REQUIRED = ("x", "y", "z", "f_dc_0", "f_dc_1", "f_dc_2", "opacity",
            "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3")


@dataclass
class SplatScene:
    vertex: np.ndarray                 # structured array, OpenCV camera frame (x right, y down, z forward)
    K: np.ndarray                      # 3x3 intrinsics in pixels
    size: tuple                        # (width, height) in pixels
    extras: list = field(default_factory=list)   # other PLY elements, passed through untouched

    @property
    def count(self):
        return len(self.vertex)


def read(path, focal_px=None, image_size=None):
    ply = PlyData.read(str(path))
    vertex = ply["vertex"].data.copy()
    missing = [name for name in REQUIRED if name not in vertex.dtype.names]
    if missing:
        raise ValueError(f"{path}: not a 3DGS PLY, missing properties {missing}")
    names = {e.name for e in ply.elements}
    if "intrinsic" in names and "image_size" in names:
        K = np.array([e[0] for e in ply["intrinsic"].data], np.float64).reshape(3, 3)
        size = tuple(int(e[0]) for e in ply["image_size"].data)
    else:
        if focal_px is None or image_size is None:
            raise ValueError(f"{path}: no 'intrinsic'/'image_size' elements; pass --focal-px and --image-size")
        w, h = image_size
        K = np.array([[focal_px, 0, w / 2], [0, focal_px, h / 2], [0, 0, 1]], np.float64)
        size = (int(w), int(h))
    extras = [e for e in ply.elements if e.name != "vertex"]
    # A few generators write NaN opacities for fully opaque points.
    op = vertex["opacity"]
    vertex["opacity"] = np.where(np.isfinite(op), op, 9.0).astype(op.dtype)
    return SplatScene(vertex, K, size, extras)


def write(path, vertex, extras=()):
    PlyData([PlyElement.describe(vertex, "vertex"), *extras], byte_order="<").write(str(path))


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


def logit(a):
    a = np.clip(a, 1e-6, 1 - 1e-6)
    return np.log(a / (1 - a)).astype(np.float32)


def positions(v):
    return np.stack([v["x"], v["y"], v["z"]], 1).astype(np.float64)


def colors(v):
    return (0.5 + C0 * np.stack([v["f_dc_0"], v["f_dc_1"], v["f_dc_2"]], 1)).clip(0, 1)


def scales(v):
    return np.exp(np.stack([v["scale_0"], v["scale_1"], v["scale_2"]], 1).astype(np.float64))


def quats(v):
    """(w, x, y, z), normalised."""
    q = np.stack([v["rot_0"], v["rot_1"], v["rot_2"], v["rot_3"]], 1).astype(np.float64)
    return q / np.maximum(np.linalg.norm(q, axis=1, keepdims=True), 1e-12)


def quat_to_mat(q):
    w, x, y, z = q.T
    return np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
                     2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
                     2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], 1).reshape(-1, 3, 3)


def mat_to_quat(R):
    """Single 3x3 rotation -> (w, x, y, z)."""
    from scipy.spatial.transform import Rotation
    x, y, z, w = Rotation.from_matrix(R).as_quat()
    return np.array([w, x, y, z])


def quat_mul(a, b):
    """Hamilton product a*b for (w, x, y, z); a may be a single quaternion."""
    a = np.broadcast_to(a, b.shape)
    aw, ax, ay, az = a.T
    bw, bx, by, bz = b.T
    return np.stack([aw * bw - ax * bx - ay * by - az * bz,
                     aw * bx + ax * bw + ay * bz - az * by,
                     aw * by - ax * bz + ay * bw + az * bx,
                     aw * bz + ax * by - ay * bx + az * bw], 1)
