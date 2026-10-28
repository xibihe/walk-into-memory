"""Minimal SPZ (version 2, SH degree 0) writer and reader.

SPZ is Niantic's compressed Gaussian splat format. Positions are stored in a
right-up-back frame; the scenes here are y-down / z-forward (OpenCV-like), so
the writer flips y and z, and the reader flips them back.
"""
import gzip
import struct

import numpy as np

MAGIC = 0x5053474E
FRACTIONAL_BITS = 12


def _u8(x):
    return np.clip(np.round(x), 0, 255).astype(np.uint8)


def write(path, v):
    n = len(v)
    f = lambda k: v[k].astype(np.float64)
    pos = np.stack([f("x"), -f("y"), -f("z")], 1)
    limit = 2 ** 23 / 2 ** FRACTIONAL_BITS
    if np.abs(pos).max(initial=0) >= limit:
        raise ValueError(f"scene extends past +-{limit:.0f} m; SPZ cannot store it at {FRACTIONAL_BITS} fractional bits")
    p = np.round(pos * 2 ** FRACTIONAL_BITS).astype(np.int64) & 0xFFFFFF
    pb = np.stack([p & 255, (p >> 8) & 255, (p >> 16) & 255], -1).astype(np.uint8).reshape(-1)
    alpha = _u8(1 / (1 + np.exp(-np.clip(f("opacity"), -30, 30))) * 255)
    rgb = _u8(np.stack([f(f"f_dc_{i}") for i in range(3)], 1) * 0.15 * 255 + 127.5).reshape(-1)
    sc = _u8((np.stack([f(f"scale_{i}") for i in range(3)], 1) + 10) * 16).reshape(-1)
    w, x, y, z = f("rot_0"), f("rot_1"), f("rot_2"), f("rot_3")
    q = np.stack([x, -y, -z, w], 1)
    q /= np.maximum(np.linalg.norm(q, axis=1, keepdims=True), 1e-12)
    q *= np.where(q[:, 3:4] < 0, -1, 1)
    rot = _u8(q[:, :3] * 127.5 + 127.5).reshape(-1)
    header = struct.pack("<IIIBBBB", MAGIC, 2, n, 0, FRACTIONAL_BITS, 0, 0)
    with gzip.open(path, "wb", compresslevel=9) as g:
        for b in (header, pb.tobytes(), alpha.tobytes(), rgb.tobytes(), sc.tobytes(), rot.tobytes()):
            g.write(b)


def read(path):
    """Returns dict of numpy arrays in the y-down frame (xyz, opacity logit-free alpha, rgb_dc, log_scale, quat wxyz)."""
    data = gzip.open(path, "rb").read()
    magic, version, n, sh, fb, flags, _ = struct.unpack_from("<IIIBBBB", data, 0)
    if magic != MAGIC:
        raise ValueError("not an SPZ file")
    if version != 2:
        raise ValueError(f"unsupported SPZ version {version}")
    off = 16
    pb = np.frombuffer(data, np.uint8, n * 9, off).reshape(n, 3, 3).astype(np.int64)
    off += n * 9
    p = pb[..., 0] | (pb[..., 1] << 8) | (pb[..., 2] << 16)
    p = np.where(p & 0x800000, p - 0x1000000, p) / 2 ** fb
    alpha = np.frombuffer(data, np.uint8, n, off) / 255.0
    off += n
    dc = (np.frombuffer(data, np.uint8, n * 3, off).reshape(n, 3) / 255.0 - 0.5) / 0.15
    off += n * 3
    log_scale = np.frombuffer(data, np.uint8, n * 3, off).reshape(n, 3) / 16.0 - 10
    off += n * 3
    r = np.frombuffer(data, np.uint8, n * 3, off).reshape(n, 3) / 127.5 - 1
    off += n * 3
    sh_coeffs = {0: 0, 1: 3, 2: 8, 3: 15}[sh]
    expected = off + n * sh_coeffs * 3
    w = np.sqrt(np.clip(1 - (r ** 2).sum(1), 0, 1))
    xyz = np.stack([p[:, 0], -p[:, 1], -p[:, 2]], 1)
    quat = np.stack([w, r[:, 0], -r[:, 1], -r[:, 2]], 1)
    return dict(count=n, sh_degree=sh, xyz=xyz, alpha=alpha, dc=dc, log_scale=log_scale, quat=quat,
                size_ok=len(data) == expected)
