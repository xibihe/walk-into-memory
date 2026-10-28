"""Command line: python -m wim run INPUT.ply -o OUT/scene [options]"""
import argparse
import json
import pathlib
import sys

import numpy as np

from . import __version__, plyio, spz
from .mask import auto_person_mask, load_mask, load_photo
from .pipeline import Options, process


def _box(text):
    vals = [float(x) for x in text.split(",")]
    if len(vals) != 4:
        raise argparse.ArgumentTypeError("expected x0,x1,y0,y1")
    return tuple(vals)


def _size(text):
    w, h = (int(x) for x in text.lower().split("x"))
    return w, h


def main(argv=None):
    ap = argparse.ArgumentParser(prog="wim", description="Walk Into Memory asset pipeline")
    ap.add_argument("--version", action="version", version=__version__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="clean + prune a photo splat and export .spz + .json")
    r.add_argument("input", help="3DGS PLY in camera coordinates (e.g. ML Sharp output)")
    r.add_argument("-o", "--output", required=True, help="output path without extension, e.g. out/garden")
    r.add_argument("--photo", help="the source photo (used for the backplate colours and --auto-mask)")
    g = r.add_mutually_exclusive_group()
    g.add_argument("--mask", help="person mask image, white = person (any size, resized to the splat image size)")
    g.add_argument("--auto-mask", action="store_true", help="detect people in --photo with torchvision Mask R-CNN")
    g.add_argument("--no-person", action="store_true", help="no person protection (landscapes, rooms)")
    r.add_argument("--target", type=int, default=560000, help="final Gaussian count (default 560000)")
    r.add_argument("--edge", choices=("hard", "soft"), default="hard",
                   help="hard: the app draws the soft edge (default); soft: bake an organic edge into the data")
    r.add_argument("--seed", type=int, default=0, help="soft edge outline seed")
    r.add_argument("--camera-height", type=float, help="camera height above the floor in metres (default: measured)")
    r.add_argument("--fallback-camera-height", type=float, default=1.5)
    r.add_argument("--no-level", action="store_true", help="do not level the scene on the floor")
    r.add_argument("--scale", type=float, help="uniform scale factor (ML Sharp output is already metric)")
    r.add_argument("--person-height", type=float, help="rescale so the standing person is this tall (metres)")
    r.add_argument("--no-backplate", action="store_true")
    r.add_argument("--view-scale", type=float, default=0.30, help="resolution factor of the pruning renders")
    r.add_argument("--still-box", type=_box, help="wind-free screen box x0,x1,y0,y1 in [-1,1] (default: person box)")
    r.add_argument("--water-box", type=_box, help="water shimmer box x0,x1,y0,y1 in [-1,1]")
    r.add_argument("--focal-px", type=float, help="focal length in pixels if the PLY has no intrinsics")
    r.add_argument("--image-size", type=_size, help="WxH if the PLY has no image_size element")
    r.add_argument("--keep-ply", action="store_true", help="also write the processed scene as PLY")

    i = sub.add_parser("inspect", help="print a summary of a .spz file")
    i.add_argument("spz")

    a = ap.parse_args(argv)
    if a.cmd == "inspect":
        d = spz.read(a.spz)
        lo, hi = d["xyz"].min(0), d["xyz"].max(0)
        print(json.dumps(dict(count=d["count"], sh_degree=d["sh_degree"], size_ok=d["size_ok"],
                              bounds_min=lo.round(3).tolist(), bounds_max=hi.round(3).tolist(),
                              mean_alpha=round(float(d["alpha"].mean()), 3)), indent=1))
        return 0

    scene = plyio.read(a.input, a.focal_px, a.image_size)
    W, H = scene.size
    photo = load_photo(a.photo, W, H) if a.photo else None
    if a.mask:
        person = load_mask(a.mask, W, H)
    elif a.auto_mask:
        if photo is None:
            ap.error("--auto-mask needs --photo")
        person = auto_person_mask(photo)
    elif a.no_person:
        person = None
    else:
        ap.error("give --mask, --auto-mask or --no-person")
    o = Options(target=a.target, edge=a.edge, seed=a.seed, backplate=not a.no_backplate, level=not a.no_level,
                camera_height=a.camera_height, fallback_camera_height=a.fallback_camera_height, scale=a.scale,
                person_height=a.person_height, view_scale=a.view_scale, still_box=a.still_box, water_box=a.water_box)
    v, sidecar, stats = process(scene, person, photo, o)
    out = pathlib.Path(a.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    spz.write(out.with_suffix(".spz"), v)
    out.with_suffix(".json").write_text(json.dumps(sidecar, indent=1))
    if a.keep_ply:
        plyio.write(out.with_suffix(".ply"), v)
    check = spz.read(out.with_suffix(".spz"))
    stats["spz_bytes"] = out.with_suffix(".spz").stat().st_size
    stats["spz_roundtrip_max_error_m"] = float(np.abs(check["xyz"] - plyio.positions(v)).max(initial=0))
    out.with_name(out.name + ".stats.json").write_text(json.dumps(stats, indent=1))
    print(f"wrote {out.with_suffix('.spz')} ({len(v)} Gaussians), {out.with_suffix('.json')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
