#!/usr/bin/env python3
"""One command: photo -> Apple ML Sharp (3DGS PLY) -> wim clean/prune/level -> SPZ.

    python photo_to_splat.py photo.jpg -o out/scene.spz --points 600000
    python photo_to_splat.py --check            # can this machine run ML Sharp locally?

Steps (same as the hand-run pipeline used for the Walk Into Memory scenes):
  1. `sharp predict` turns the single photo into a metric 3DGS .ply (~1.18M Gaussians)
  2. `wim run` keeps only Gaussians inside the photo frame (drops floaters / mirrored
     border junk), shortens needle streaks, adds a backplate, prunes by multi-view
     contribution to --points, levels the scene on the floor and writes .spz + .json
ML Sharp is NOT bundled: install it from https://github.com/apple/ml-sharp.
Its model weights are for non-commercial research use only.
"""
import argparse
import platform
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent


def check(verbose=True):
    """Return True if ML Sharp can run here. Prediction works on CPU, CUDA and Apple MPS."""
    ok = True
    msg = []
    py = sys.version_info
    msg.append(f"python {py.major}.{py.minor} (ML Sharp recommends 3.13 in its own env)")
    try:
        import torch
        dev = "cuda" if torch.cuda.is_available() else (
            "mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available() else "cpu")
        msg.append(f"torch {torch.__version__}, best device: {dev}")
        if dev == "cpu":
            msg.append("  no CUDA GPU / Apple Silicon found: prediction still works on CPU but is slow")
    except ImportError:
        ok = False
        msg.append("torch missing: pip install -r requirements.txt")
    sharp = shutil.which("sharp")
    if sharp:
        msg.append(f"sharp CLI: {sharp}")
    else:
        ok = False
        msg.append("sharp CLI not found: git clone https://github.com/apple/ml-sharp && "
                   "pip install -r requirements.txt (inside that repo)")
    msg.append(f"platform: {platform.system()} {platform.machine()}")
    msg.append("first run downloads a ~300 MB checkpoint to ~/.cache/torch/hub/checkpoints/")
    if not ok:
        msg.append("=> cannot run locally. Use Path B in the README (PhotoSplat3D, free, on-device).")
    else:
        msg.append("=> OK, this machine can run ML Sharp locally.")
    if verbose:
        print("\n".join(msg))
    return ok


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("photo", nargs="?", help="input photo (jpg/png)")
    ap.add_argument("-o", "--output", help="output .spz path (a .json sidecar is written next to it)")
    ap.add_argument("--points", type=int, default=560000, help="final Gaussian count (default 560000)")
    ap.add_argument("--ply", help="skip ML Sharp and use an existing 3DGS PLY (e.g. from PhotoSplat3D)")
    m = ap.add_mutually_exclusive_group()
    m.add_argument("--auto-mask", action="store_true", help="protect people found in the photo (needs torchvision)")
    m.add_argument("--mask", help="person mask image, white = person")
    ap.add_argument("--edge", choices=("hard", "soft"), default="hard")
    ap.add_argument("--keep-ply", action="store_true")
    ap.add_argument("--check", action="store_true", help="only check whether ML Sharp can run here")
    a = ap.parse_args(argv)

    if a.check:
        return 0 if check() else 1
    if not a.photo or not a.output:
        ap.error("photo and -o are required (or use --check)")
    photo = Path(a.photo).resolve()
    out = Path(a.output).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp:
        if a.ply:
            ply = Path(a.ply).resolve()
        else:
            if not check(verbose=False):
                check()
                return 1
            src, dst = Path(tmp, "in"), Path(tmp, "out")
            src.mkdir()
            shutil.copy(photo, src / photo.name)
            print("[1/2] ML Sharp: photo -> 3D Gaussians")
            subprocess.run(["sharp", "predict", "-i", str(src), "-o", str(dst)], check=True)
            plys = sorted(dst.rglob("*.ply"))
            if not plys:
                sys.exit("ML Sharp wrote no .ply")
            ply = plys[0]
        print("[2/2] clean, prune, level, SPZ")
        cmd = [sys.executable, "-m", "wim", "run", str(ply), "--photo", str(photo),
               "--target", str(a.points), "--edge", a.edge, "-o", str(out.with_suffix(""))]
        cmd += ["--auto-mask"] if a.auto_mask else (["--mask", a.mask] if a.mask else ["--no-person"])
        if a.keep_ply:
            cmd.append("--keep-ply")
        subprocess.run(cmd, check=True, cwd=HERE)
    print(f"done: {out.with_suffix('.spz')}  -> AirDrop / Files into the Vision Pro app")
    return 0


if __name__ == "__main__":
    sys.exit(main())
