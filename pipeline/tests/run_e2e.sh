#!/usr/bin/env bash
# End-to-end smoke test on a synthetic scene (no real photos).
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PYTHON:-python3}
OUT=$(mktemp -d)
"$PY" tests/make_synthetic.py "$OUT"
"$PY" -m wim run "$OUT/synthetic.ply" --mask "$OUT/synthetic_mask.png" --target 30000 -o "$OUT/scene"
"$PY" -m wim run "$OUT/synthetic.ply" --no-person --edge soft --target 25000 -o "$OUT/scene_soft"
"$PY" -m wim inspect "$OUT/scene.spz"
"$PY" tests/check_outputs.py "$OUT/scene" 30000
"$PY" tests/check_outputs.py "$OUT/scene_soft" 25000
echo "OK: outputs in $OUT"
