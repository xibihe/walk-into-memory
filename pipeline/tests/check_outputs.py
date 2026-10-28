"""Sanity checks on one pipeline output (scene.spz + scene.json + scene.stats.json)."""
import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from wim import spz  # noqa: E402

base = pathlib.Path(sys.argv[1])
target = int(sys.argv[2])
d = spz.read(base.with_suffix(".spz"))
side = json.loads(base.with_suffix(".json").read_text())
stats = json.loads(base.with_name(base.name + ".stats.json").read_text())
assert d["size_ok"], "SPZ payload size mismatch"
assert d["sh_degree"] == 0
assert 0 < d["count"] <= target + stats["protected"] + stats["backplate"], d["count"]
assert d["count"] == stats["final"]
assert len(side["edgeFrame"]) == 24
assert np.isfinite(d["xyz"]).all()
assert stats["spz_roundtrip_max_error_m"] < 1e-3
q = d["quat"]
assert np.allclose(np.linalg.norm(q, axis=1), 1, atol=0.05)
M = np.array(side["edgeFrame"][:9]).reshape(3, 3)
assert np.allclose(M @ M.T, np.eye(3), atol=1e-4), "edge frame rotation must be orthonormal"
# Floor check: levelled scene -> floor points sit near y = cameraHeight
if stats["floor_found"]:
    y = d["xyz"][:, 1]
    near = np.abs(y - side["cameraHeight"]) < 0.08
    assert near.mean() > 0.2, near.mean()
print(base.name, "ok:", d["count"], "Gaussians, camera height", side["cameraHeight"], "m,",
      "floor found" if stats["floor_found"] else "no floor")
