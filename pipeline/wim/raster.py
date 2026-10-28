"""A small, exact-ish 3D Gaussian splat rasterizer on the CPU (PyTorch).

Per-pixel depth-sorted alpha compositing. It is slow compared with a GPU
renderer but has no CUDA dependency, which is all the pipeline needs: it
renders a handful of low-resolution views to measure how much every Gaussian
actually contributes to the image.
"""
import math

import numpy as np
import torch


def quat2mat(q):
    q = q / q.norm(dim=1, keepdim=True)
    w, x, y, z = q.unbind(1)
    return torch.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
                        2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
                        2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], 1).view(-1, 3, 3)


def to_tensors(xyz, rgb, alpha, scale, quat):
    """numpy arrays -> list of float32 tensors in the order render() expects."""
    return [torch.from_numpy(np.ascontiguousarray(a, dtype=np.float32)) for a in (xyz, rgb, alpha, scale, quat)]


def render(G, K, size, R, t, scale=0.375, bg=(0.06, 0.06, 0.07), rmax=14, contrib=None, chunk=200000):
    """Render Gaussians G = [xyz, rgb, alpha(linear), scale(linear), quat(wxyz)] (world frame).

    R, t: world -> OpenCV camera (x right, y down, z forward).
    contrib: optional float64 tensor of len(G[0]); every Gaussian's total
             compositing weight in this view is added to it.
    Returns (uint8 HxWx3 image, number of fragments).
    """
    xyz, rgb, op, sc, q = G
    W, H = int(size[0] * scale), int(size[1] * scale)
    fx, fy, cx, cy = K[0, 0] * scale, K[1, 1] * scale, K[0, 2] * scale, K[1, 2] * scale
    pc = xyz @ R.T + t
    z = pc[:, 2]
    keep = z > 0.05
    kept = torch.nonzero(keep).squeeze(1)
    pc, z, rgb_, op_, sc_, q_ = pc[keep], z[keep], rgb[keep], op[keep], sc[keep], q[keep]
    u = fx * pc[:, 0] / z + cx
    v = fy * pc[:, 1] / z + cy
    M = quat2mat(q_) * sc_[:, None, :]
    cov = M @ M.transpose(1, 2)
    covc = R[None] @ cov @ R.T[None]
    J = torch.zeros(len(z), 2, 3)
    J[:, 0, 0] = fx / z
    J[:, 0, 2] = -fx * pc[:, 0] / z ** 2
    J[:, 1, 1] = fy / z
    J[:, 1, 2] = -fy * pc[:, 1] / z ** 2
    c2 = J @ covc @ J.transpose(1, 2)
    a = c2[:, 0, 0] + 0.3
    b = c2[:, 0, 1]
    c = c2[:, 1, 1] + 0.3
    det = a * c - b * b
    mid = 0.5 * (a + c)
    lam = mid + torch.sqrt((mid * mid - det).clamp(min=0.1))
    rad = torch.ceil(3 * torch.sqrt(lam))
    ok = (det > 0) & (u + rad >= 0) & (u - rad < W) & (v + rad >= 0) & (v - rad < H)
    rad = rad.clamp(max=rmax)
    ia, ib, ic = c / det, -b / det, a / det
    idx = torch.nonzero(ok).squeeze(1)
    if len(idx) == 0:
        img = torch.tensor(bg).expand(H, W, 3)
        return (img.clamp(0, 1).numpy() * 255).astype(np.uint8), 0
    idx = idx[torch.argsort(z[idx])]  # global front-to-back
    frags_pix, frags_a, frags_g = [], [], []
    rad_i = rad[idx].long()
    for r in sorted(set(rad_i.tolist())):
        sel = idx[rad_i == r]
        oy, ox = torch.meshgrid(torch.arange(-r, r + 1), torch.arange(-r, r + 1), indexing="ij")
        ox = ox.reshape(-1).float()
        oy = oy.reshape(-1).float()
        for s in range(0, len(sel), chunk):
            g = sel[s:s + chunk]
            px = torch.floor(u[g])[:, None] + ox[None]
            py = torch.floor(v[g])[:, None] + oy[None]
            dx = px + 0.5 - u[g][:, None]
            dy = py + 0.5 - v[g][:, None]
            pw = 0.5 * (ia[g][:, None] * dx * dx + 2 * ib[g][:, None] * dx * dy + ic[g][:, None] * dy * dy)
            al = (op_[g][:, None] * torch.exp(-pw)).clamp(max=0.99)
            m = (al > 1 / 255) & (px >= 0) & (px < W) & (py >= 0) & (py < H)
            frags_pix.append((py * W + px)[m].long())
            frags_a.append(al[m])
            frags_g.append(g[:, None].expand_as(m)[m])
    pix = torch.cat(frags_pix)
    al = torch.cat(frags_a).double()
    gi = torch.cat(frags_g)
    if len(pix) == 0:
        img = torch.tensor(bg).expand(H, W, 3)
        return (img.clamp(0, 1).numpy() * 255).astype(np.uint8), 0
    # Stable sort by pixel keeps the depth order inside each pixel.
    order_rank = torch.empty(len(z), dtype=torch.long)
    order_rank[idx] = torch.arange(len(idx))
    key = pix * (len(idx) + 1) + order_rank[gi]
    o = torch.argsort(key)
    pix, al, gi = pix[o], al[o], gi[o]
    lt = torch.log1p(-al)
    cs = torch.cumsum(lt, 0)
    _, counts = torch.unique_consecutive(pix, return_counts=True)
    starts = torch.cumsum(counts, 0) - counts
    base = (cs - lt)[starts]
    excl = cs - lt - torch.repeat_interleave(base, counts)
    w = (al * torch.exp(excl)).float()
    if contrib is not None:
        contrib.index_add_(0, kept[gi], w.double())
    img = torch.zeros(H * W, 3)
    img.index_add_(0, pix, w[:, None] * rgb_[gi])
    acc = torch.zeros(H * W)
    acc.index_add_(0, pix, w)
    img = img + (1 - acc)[:, None] * torch.tensor(bg)
    return (img.view(H, W, 3).clamp(0, 1).numpy() * 255).astype(np.uint8), len(pix)


def yaw_matrix(degrees):
    a = math.radians(degrees)
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0, -s], [0, 1, 0], [s, 0, c]])
