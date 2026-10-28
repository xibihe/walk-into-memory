"""photo splat (camera frame) -> cleaned, pruned, levelled scene for the visionOS app.

Steps
 1. keep only Gaussians that project inside the original picture
 2. person protection: Gaussians on the person (mask + depth band) are never pruned or edited
 3. anti-streak: needle-like Gaussians pointing along the view ray are shortened
 4. backplate: a thin layer of inpainted Gaussians behind the person, so walking
    sideways does not reveal a black hole
 5. occluders that were already behind the person are kept
 6. frame-edge target opacity (hard guard by default, optional baked soft edge)
 7. multi-view contribution pruning down to --target Gaussians
 8. edge calibration, edge darkening and size cap
 9. level the scene on the floor and write the sidecar used by the app
"""
from dataclasses import dataclass, asdict
import math
import time

import cv2
import numpy as np
import torch
from scipy import ndimage

from . import edges, level
from .plyio import (C0, SplatScene, colors, logit, mat_to_quat, positions, quat_mul,
                    quat_to_mat, quats, scales, sigmoid)
from .raster import render, to_tensors, yaw_matrix


@dataclass
class Options:
    target: int = 560000             # final number of Gaussians
    edge: str = "hard"               # 'hard' or 'soft'
    seed: int = 0                    # outline seed for the soft edge
    backplate: bool = True
    level: bool = True
    camera_height: float = None      # metres; measured from the floor when None
    fallback_camera_height: float = 1.5
    scale: float = None              # multiply the scene size (ML Sharp output is already metric)
    person_height: float = None      # rescale so the (standing) person is this tall
    view_scale: float = 0.30         # resolution factor of the pruning renders
    calibration_iterations: int = 7
    still_box: tuple = None          # wind-free screen box, normalised [-1,1]: x0,x1,y0,y1
    water_box: tuple = None          # optional water shimmer box, same format


def _smooth(x):
    x = np.clip(x, 0, 1)
    return x * x * (3 - 2 * x)


class _Projector:
    def __init__(self, K, size):
        self.K, (self.W, self.H) = K, size

    def __call__(self, xyz):
        z = xyz[:, 2]
        zs = np.where(z > 0.05, z, np.inf)
        u = self.K[0, 0] * xyz[:, 0] / zs + self.K[0, 2]
        v = self.K[1, 1] * xyz[:, 1] / zs + self.K[1, 2]
        return z, u, v


def splat_colour_image(xyz, rgb, alpha, proj, W, H, factor=2):
    """Front-most opaque splat colour per (downsampled) pixel; holes filled by inpainting."""
    z, u, v = proj(xyz)
    ok = (z > 0.05) & (alpha > 0.5) & (u >= 0) & (u < W) & (v >= 0) & (v < H)
    w2, h2 = W // factor, H // factor
    pix = np.clip((v[ok] // factor).astype(int), 0, h2 - 1) * w2 + np.clip((u[ok] // factor).astype(int), 0, w2 - 1)
    order = np.argsort(-z[ok])  # far first, near overwrites
    img = np.zeros((h2 * w2, 3), np.float32)
    img[pix[order]] = rgb[ok][order]
    hole = np.ones(h2 * w2, bool)
    hole[pix] = False
    img8 = (img.reshape(h2, w2, 3) * 255).astype(np.uint8)
    img8 = cv2.inpaint(img8, hole.reshape(h2, w2).astype(np.uint8), 3, cv2.INPAINT_TELEA)
    return cv2.resize(img8, (W, H), interpolation=cv2.INTER_LINEAR).astype(np.float32) / 255


def process(scene: SplatScene, person=None, photo=None, o: Options = Options(), log=print):
    t0 = time.time()
    say = lambda *a: log(f"[{time.time() - t0:6.1f}s]", *a)
    v = scene.vertex.copy()
    K, (W, H) = scene.K, scene.size
    proj = _Projector(K, (W, H))
    N0 = len(v)
    xyz = positions(v)
    z, u, vv = proj(xyz)
    a0 = sigmoid(v["opacity"].astype(np.float64))

    # 1) in-frame only (drops the mirrored / extrapolated border some generators add)
    inimg = (z > 0.05) & (u >= 0) & (u < W) & (vv >= 0) & (vv < H)
    px = np.clip(u, 0, W - 1).astype(int)
    py = np.clip(vv, 0, H - 1).astype(int)

    # 2) person protection
    if person is None:
        person = np.zeros((H, W), bool)
    core = inimg & person[py, px] & (a0 > 0.3)
    has_person = core.sum() >= 50
    if has_person:
        pm = cv2.dilate(person.astype(np.uint8), np.ones((11, 11), np.uint8)) > 0
        sd = float(np.median(z[core]))
        band_lo = float(np.percentile(z[core], 2)) - 0.4
        band_hi = float(np.percentile(z[core], 90)) + 0.6
        prot = inimg & pm[py, px] & (z > band_lo) & (z < band_hi)
    else:
        sd = float(np.median(z[inimg & (a0 > 0.3)])) if (inimg & (a0 > 0.3)).any() else 3.0
        band_hi = np.inf
        prot = np.zeros(N0, bool)
    say("input", N0, "in frame", int(inimg.sum()), "person depth m", round(sd, 2), "protected", int(prot.sum()))

    # 3) anti-streak
    S = scales(v)
    Rq = quat_to_mat(quats(v))
    li = np.argmax(S, 1)
    axis = Rq[np.arange(len(S)), :, li]
    ray = xyz / np.maximum(np.linalg.norm(xyz, axis=1, keepdims=True), 1e-9)
    cosr = np.abs((axis * ray).sum(1))
    Ss = np.sort(S, 1)
    ratio = Ss[:, 2] / np.maximum(Ss[:, 1], 1e-9)
    cap = np.where(prot, 6.0, 3.0)
    hit = inimg & (cosr > 0.8) & (ratio > cap)
    for k in range(3):
        m = hit & (li == k)
        v[f"scale_{k}"][m] = np.log(Ss[m, 1] * cap[m]).astype(np.float32)
    say("anti-streak: shortened", int(hit.sum()))

    # 4) backplate behind the person
    nb = np.zeros(0, dtype=v.dtype)
    behind = np.zeros(N0, bool)
    if has_person and o.backplate:
        grow = int(0.02 * W) | 1
        fill_mask = cv2.dilate(person.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (grow, grow)))
        src = photo if photo is not None else splat_colour_image(xyz[inimg & ~prot], colors(v)[inimg & ~prot],
                                                                  a0[inimg & ~prot], proj, W, H)
        inp = cv2.inpaint((src * 255).astype(np.uint8), fill_mask, 9, cv2.INPAINT_TELEA).astype(np.float32) / 255
        band = cv2.erode(person.astype(np.uint8), np.ones((7, 7), np.uint8)) > 0
        exk = int(0.07 * W) | 1
        excl = cv2.dilate(person.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (exk, exk))) > 0
        h2, w2 = H // 2, W // 2
        okb = inimg & ~prot & (a0 > 0.3)
        pix = np.clip(vv[okb] // 2, 0, h2 - 1).astype(int) * w2 + np.clip(u[okb] // 2, 0, w2 - 1).astype(int)
        front = np.full(h2 * w2, np.inf, np.float32)
        np.minimum.at(front, pix, z[okb].astype(np.float32))
        front = front.reshape(h2, w2)
        unknown = ~np.isfinite(front) | (cv2.resize(excl.astype(np.uint8), (w2, h2), interpolation=cv2.INTER_NEAREST) > 0)
        if (~unknown).any():
            _, ind = ndimage.distance_transform_edt(unknown, return_indices=True)
            bg = front[ind[0], ind[1]]
        else:
            bg = np.full((h2, w2), sd + 1.0, np.float32)
        bg = ndimage.maximum_filter(ndimage.median_filter(bg, 5), 11)  # farthest nearby: stays behind real background
        bg = cv2.resize(bg, (W, H), interpolation=cv2.INTER_LINEAR)
        sp = 4
        yy, xx = np.mgrid[sp // 2:H:sp, sp // 2:W:sp]
        sel = band[yy, xx]
        xs, ys = xx[sel], yy[sel]
        dep = np.maximum(bg[ys, xs] * 1.03, sd + 0.5)
        rays = np.column_stack([(xs - K[0, 2]) / K[0, 0], (ys - K[1, 2]) / K[1, 1], np.ones(len(xs))])
        pts = rays * dep[:, None]
        nb = np.zeros(len(pts), dtype=v.dtype)
        for i, name in enumerate("xyz"):
            nb[name] = pts[:, i].astype(np.float32)
        col = inp[ys, xs]
        for i in range(3):
            nb[f"f_dc_{i}"] = ((col[:, i] - 0.5) / C0).astype(np.float32)
        nb["opacity"] = logit(np.full(len(nb), 0.92))
        std = 0.62 * sp * dep / K[0, 0]
        nb["scale_0"] = np.log(std).astype(np.float32)
        nb["scale_1"] = np.log(std).astype(np.float32)
        nb["scale_2"] = np.log(std * 0.22).astype(np.float32)   # thin, facing the camera
        nb["rot_0"] = 1.0
        # 5) keep what was already behind the person
        bk = int(0.03 * W) | 1
        bdil = cv2.dilate(person.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (bk, bk))) > 0
        behind = inimg & bdil[py, px] & (z >= band_hi) & ~prot
        say("backplate", len(nb), "kept behind person", int(behind.sum()))

    v = np.concatenate([v[inimg], nb])
    prot = np.concatenate([prot[inimg], np.zeros(len(nb), bool)])
    backp = np.concatenate([behind[inimg], np.ones(len(nb), bool)])
    xyz = positions(v)
    z, u, vv = proj(xyz)
    N = len(v)

    # 9a) level on the floor (needed now: pruning views orbit around gravity)
    L, floor = np.eye(3), None
    if o.level:
        cand = (vv > 0.55 * H) & (sigmoid(v["opacity"].astype(np.float64)) > 0.5) & (z < 15) & ~prot & ~backp
        if has_person:
            cand &= ~person[np.clip(vv, 0, H - 1).astype(int), np.clip(u, 0, W - 1).astype(int)]
        floor = level.fit_floor(xyz, cand)
    if floor is not None:
        n, measured = floor
        L = level.rotation_between(n, np.array([0.0, 1.0, 0.0]))
        say("floor found: tilt deg", round(math.degrees(math.acos(min(1, n[1]))), 1), "camera height m", round(measured, 3))
    else:
        measured = None
        say("no floor found; scene is not levelled")
    cam_h = o.camera_height or measured or o.fallback_camera_height
    qL = mat_to_quat(L)
    xyz_l = xyz @ L.T
    q_l = quat_mul(qL, quats(v))
    subj = np.median(xyz_l[prot], 0) if prot.any() else L @ np.array([0, 0, sd])

    # 6) edge target
    gy, gx = np.mgrid[0:H, 0:W]
    fxg, fyg = (gx + 0.5) / W, (gy + 0.5) / H
    if o.edge == "soft":
        ps = cv2.GaussianBlur(cv2.dilate(person.astype(np.uint8) * 255, np.ones((15, 15), np.uint8)), (0, 0), W * 0.01) / 255.0
        TF = np.maximum(edges.soft(fxg, fyg, o.seed), ps * edges.soft_subject(fxg, fyg, o.seed)).astype(np.float32)
    else:
        TF = edges.hard(fxg, fyg).astype(np.float32)
    del gy, gx, fxg, fyg

    def target(fx, fy):
        return TF[np.clip((np.asarray(fy) * H).astype(int), 0, H - 1), np.clip((np.asarray(fx) * W).astype(int), 0, W - 1)]

    # 7) contribution pruning
    a = sigmoid(v["opacity"].astype(np.float64))
    T0 = target(u / W, vv / H)
    w = np.where(T0 >= 0.999, 1.0, np.maximum(T0, 0) ** 1.6)
    w[prot] = 1.0
    G0 = to_tensors(xyz_l, colors(v), a * w, scales(v), q_l)
    Kt = K.astype(np.float32)

    def view(yaw=0.0, dx=0.0, push=0.0):
        R = (yaw_matrix(yaw) @ L).T
        right = L[:, 0].copy()
        right[1] = 0
        right /= np.linalg.norm(right)
        tow = subj.copy()
        tow[1] = 0
        tow /= max(np.linalg.norm(tow), 1e-9)
        eye = right * dx + tow * push
        return torch.tensor(R, dtype=torch.float32), torch.tensor(-R @ eye, dtype=torch.float32)

    views = [view(y) for y in (0, -10, 10, -20, 20, -25, 25)] + \
            [view(0, dx, p) for dx, p in ((-0.3, 0), (0.3, 0), (0, 1.0), (0, 1.5), (0.3, 1.5), (-0.3, 1.5), (0, -0.8))]
    alive = ((a * w) > 1 / 400) | prot | backp
    rounds = [r for r in (int(N * 0.75), int(N * 0.6)) if r > o.target] + [o.target]
    for goal in rounds:
        ids = np.where(alive)[0]
        if len(ids) <= goal:
            continue
        Gs = [g[torch.from_numpy(ids)] for g in G0]
        total = np.zeros(len(ids))
        for R, t in views:
            C = torch.zeros(len(ids), dtype=torch.float64)
            render(Gs, Kt, (W, H), R, t, scale=o.view_scale, bg=(0, 0, 0), rmax=24, contrib=C)
            total += C.numpy()
        total[prot[ids] | backp[ids]] = 1e18
        keep = np.argsort(-total)[:goal]
        alive = np.zeros(N, bool)
        alive[ids[keep]] = True
        say("prune round ->", int(alive.sum()))
    idx = np.where(alive)[0]
    v, prot, backp, a = v[idx], prot[idx], backp[idx], a[idx]
    xyz, xyz_l, q_l = xyz[idx], xyz_l[idx], q_l[idx]
    z, u, vv = proj(xyz)

    # 8) edge calibration: render coverage from the front and adjust opacity until the edge matches the target
    Sc = 0.25
    rw, rh = int(W * Sc), int(H * Sc)
    yy, xx = np.mgrid[0:rh, 0:rw]
    T = target((xx + 0.5) / rw, (yy + 0.5) / rh)
    BIN = 4
    gh, gw = max(1, rh // BIN), max(1, rw // BIN)
    Tb = cv2.resize(T, (gw, gh), interpolation=cv2.INTER_AREA)
    bu = np.clip((u * Sc // BIN).astype(int), 0, gw - 1)
    bv = np.clip((vv * Sc // BIN).astype(int), 0, gh - 1)
    k = np.ones((gh, gw))
    m = Tb < 0.999
    k[m] = np.maximum(Tb[m], 1e-3) ** 1.6
    zone = m[bv, bu]
    G = [g[torch.from_numpy(idx)] for g in G0]
    white = torch.ones_like(G[1])
    Rf, tf = view(0)
    for it in range(o.calibration_iterations):
        aa = np.where(zone, a * np.minimum(1.0, k[bv, bu]), a)
        G[2] = torch.from_numpy(aa.astype(np.float32))
        img, _ = render([G[0], white, G[2], G[3], G[4]], Kt, (W, H), Rf, tf, scale=Sc, bg=(0, 0, 0), rmax=24)
        Ab = cv2.resize(img[..., 0].astype(np.float64) / 255, (gw, gh), interpolation=cv2.INTER_AREA)
        if m.any():
            say("edge calibration", it, "mean error", round(float(np.abs(Ab - Tb)[m].mean()), 4))
        r = np.log(np.clip(1 - Tb, 0.003, 1)) / np.log(np.clip(1 - Ab, 0.003, 0.9999))
        k[m] = np.clip(k[m] * np.clip(r[m], 0.3, 3), 1e-5, 1.0)
        k[m] = np.minimum(k[m], Tb[m] ** 0.7)
    aa = np.where(zone, a * np.minimum(1.0, k[bv, bu]), a)
    eg = edges.border(u / W, vv / H)
    fin = ((aa > 1 / 400) & (eg > 0.01)) | prot | (backp & (aa > 1 / 400))
    v["opacity"][zone] = logit(aa)[zone]
    # fade colour to black in the edge band and stop edge splats from smearing past the frame
    Tp = target(u / W, vv / H)
    cz = zone & (Tp < 0.999)
    cf = np.maximum(Tp[cz], 0) ** 0.4
    for i in range(3):
        c = 0.5 + C0 * v[f"f_dc_{i}"][cz]
        v[f"f_dc_{i}"][cz] = ((c * cf - 0.5) / C0).astype(np.float32)
    capm = (0.012 * W) * np.maximum(z, 1e-3) / K[0, 0]
    near_edge = cz & (eg < 0.06)
    for i in range(3):
        sc = v[f"scale_{i}"]
        big = near_edge & (np.exp(sc) > capm)
        sc[big] = np.log(capm[big]).astype(np.float32)
    v, prot, xyz_l, q_l = v[fin], prot[fin], xyz_l[fin], q_l[fin]
    say("final", len(v), "protected kept", int(prot.sum()))

    # 9b) scale and write the levelled frame
    s = o.scale or 1.0
    if o.person_height and prot.any():
        top = float(np.percentile(xyz_l[prot][:, 1], 1))
        if cam_h - top > 0.3:
            s = o.person_height / (cam_h - top)
    for i, name in enumerate("xyz"):
        v[name] = (s * xyz_l[:, i]).astype(np.float32)
    for i in range(3):
        v[f"scale_{i}"] = (v[f"scale_{i}"] + math.log(s)).astype(np.float32)
    for i in range(4):
        v[f"rot_{i}"] = q_l[:, i].astype(np.float32)

    sidecar = make_sidecar(K, (W, H), L, s, cam_h * s, person if has_person else None,
                           (sd * s + 1.0) if has_person else 0.0, o)
    stats = dict(input=N0, in_frame=int(inimg.sum()), backplate=int(len(nb)), anti_streak=int(hit.sum()),
                 protected=int(prot.sum()), final=int(len(v)), person_found=bool(has_person),
                 person_depth_m=round(sd * s, 3), floor_found=floor is not None,
                 camera_height_m=round(cam_h * s, 3), scale=s, options=asdict(o),
                 seconds=round(time.time() - t0, 1))
    return v, sidecar, stats


def _box(person, W, H, pad, clip):
    ys, xs = np.nonzero(person)
    x0, x1 = 2 * xs.min() / W - 1 - pad, 2 * (xs.max() + 1) / W - 1 + pad
    y0, y1 = 2 * ys.min() / H - 1 - pad, 2 * (ys.max() + 1) / H - 1 + pad
    if clip:
        x0, x1, y0, y1 = (float(np.clip(c, -clip, clip)) for c in (x0, x1, y0, y1))
    return [round(float(c), 4) for c in (x0, x1, y0, y1)]


NO_BOX = [2.0, 2.0, 2.0, 2.0]


def make_sidecar(K, size, L, s, camera_height, person, still_depth, o):
    """Everything the shaders need to find the original picture frame in the scene.

    The scene is stored in a levelled, y-down frame:  p_scene = s * L * p_camera.
    The edge shader maps scene points back into the original camera:
      p_camera = M * p_scene + t,  with M = L^T / s and t = 0.
    """
    W, H = size
    M = (L.T / s).reshape(-1)
    ax, bx = 2 * K[0, 0] / W, 2 * K[0, 2] / W - 1
    ay, by = 2 * K[1, 1] / H, 2 * K[1, 2] / H - 1
    box = _box(person, W, H, 0.03, 0.78) if person is not None else NO_BOX
    edge_frame = [*map(float, M), 0.0, 0.0, 0.0, ax, bx, ay, by, *box, 1.02, 0.0, 0.0, 0.0]
    still = list(o.still_box) if o.still_box else (_box(person, W, H, 0.03, None) if person is not None else NO_BOX)
    wind = dict(ground=round(camera_height, 4), up=-1.0, stillBox=still, stillDepth=round(still_depth, 3))
    if o.water_box:
        wind["waterBox"] = list(o.water_box)
    return dict(format="walk-into-memory.scene", version=1, cameraHeight=round(camera_height, 4),
                imageSize=[int(W), int(H)], edgeFrame=[round(x, 6) for x in edge_frame], wind=wind)
