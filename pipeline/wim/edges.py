"""Target opacity across the picture frame (fx, fy are normalised 0..1 image coordinates).

'hard' (default): only a very narrow 1.2 % anti-aliased guard at the picture
border. The visionOS app draws the soft, breathing edge in its shader, so the
data itself keeps the original frame.
'soft': bake an organic edge into the data (superellipse with a noisy outline
and a wide feather). Useful for other viewers that have no edge shader.
"""
import numpy as np

P_EXP, R0, AMP, FEATHER, SUBJECT_EXTRA = 2.4, 1.15, 0.06, 0.62, 0.18


def smoothstep(x):
    x = np.clip(x, 0, 1)
    return x * x * (3 - 2 * x)


def border(fx, fy):
    fx, fy = np.asarray(fx), np.asarray(fy)
    return np.minimum.reduce([fx, 1 - fx, fy, 1 - fy])


def hard(fx, fy):
    return smoothstep(border(fx, fy) / 0.012)


def _radius(theta, seed):
    rng = np.random.default_rng(seed)
    r = np.zeros_like(theta)
    for k, a in ((2, 0.45), (3, 0.35), (5, 0.25), (7, 0.15)):
        r += a * np.sin(k * theta + rng.uniform(0, 2 * np.pi))
    return R0 + AMP * r / 1.2


def _polar(fx, fy):
    X = (np.asarray(fx) - 0.5) * 2
    Y = (np.asarray(fy) - 0.5) * 2
    return (np.abs(X) ** P_EXP + np.abs(Y) ** P_EXP) ** (1 / P_EXP), np.arctan2(Y, X)


def soft(fx, fy, seed=0):
    r, th = _polar(fx, fy)
    return smoothstep((_radius(th, seed) - r) / FEATHER) * smoothstep(border(fx, fy) / 0.035)


def soft_subject(fx, fy, seed=0):
    r, th = _polar(fx, fy)
    return smoothstep((_radius(th, seed) + SUBJECT_EXTRA - r) / 0.30) * smoothstep(border(fx, fy) / 0.035)
