"""Find the floor: RANSAC plane through the lower part of the picture.

Returns a rotation that levels the camera frame (floor normal -> +y, still
y-down like OpenCV) and the camera height above the floor.
"""
import numpy as np


def rotation_between(a, b):
    """Smallest rotation taking unit vector a onto unit vector b."""
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    v = np.cross(a, b)
    c = float(np.dot(a, b))
    if np.linalg.norm(v) < 1e-9:
        return np.eye(3) if c > 0 else np.diag([1.0, -1.0, -1.0])
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx * (1 / (1 + c))


def fit_floor(xyz, candidates, max_tilt_deg=35, iters=600, threshold=0.04, seed=0):
    """xyz: camera-frame points; candidates: bool mask of points that may be floor.

    Returns (normal pointing camera->floor, height) or None.
    """
    pts = xyz[candidates]
    if len(pts) < 300:
        return None
    rng = np.random.default_rng(seed)
    if len(pts) > 80000:
        pts = pts[rng.choice(len(pts), 80000, replace=False)]
    cos_tilt = np.cos(np.radians(max_tilt_deg))
    best, best_count = None, 0
    for _ in range(iters):
        p = pts[rng.choice(len(pts), 3, replace=False)]
        n = np.cross(p[1] - p[0], p[2] - p[0])
        norm = np.linalg.norm(n)
        if norm < 1e-9:
            continue
        n /= norm
        if n[1] < 0:
            n = -n
        if n[1] < cos_tilt:
            continue
        h = float(n @ p[0])
        if h <= 0.2:
            continue
        count = int((np.abs(pts @ n - h) < threshold * max(1.0, h / 1.5)).sum())
        if count > best_count:
            best, best_count = (n, h), count
    if best is None or best_count < max(300, 0.15 * len(pts)):
        return None
    n, h = best
    inl = pts[np.abs(pts @ n - h) < threshold * max(1.0, h / 1.5)]
    centre = inl.mean(0)
    n = np.linalg.svd(inl - centre, full_matrices=False)[2][2]
    if n[1] < 0:
        n = -n
    return n, float(n @ centre)
