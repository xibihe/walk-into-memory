---
name: photo-to-splat
description: Turn one photo into a Walk Into Memory scene (.spz) for the Apple Vision Pro app. Use when a user gives a photo and wants a walkable Gaussian splat, or has a .ply/.spz/.sog from another tool.
---
# Photo → Gaussian splat scene

Pipeline: photo → Apple ML Sharp (`sharp predict`, metric 3DGS .ply, ~1.18M Gaussians)
→ `wim run` (keep in-frame Gaussians = floater/border junk removal, anti-streak, backplate,
multi-view contribution pruning to ~500–600k, floor levelling) → `.spz` + `.json` sidecar.

## Steps
1. `cd pipeline && pip install -r requirements.txt`
2. `python photo_to_splat.py --check`. Exit 0 = ML Sharp can run here (CPU, CUDA or Apple MPS;
   CPU is slow). Exit 1 = go to step 5.
3. Pick person handling: photo has people → `--auto-mask` (needs torchvision) or `--mask m.png`;
   landscape/room → nothing (defaults to `--no-person`).
4. `python photo_to_splat.py photo.jpg -o out/scene.spz --points 560000`
   Verify with `python -m wim inspect out/scene.spz` (count ≈ --points, size_ok true).
5. Cannot run locally: tell the user to use PhotoSplat3D (free, on-device: Mac app on Apple
   Silicon M2+, or the Vision Pro App Store app), export the .ply, then either import it in the
   app directly, or clean it here with `python photo_to_splat.py photo.jpg --ply that.ply -o out/scene.spz`.
6. Send the .spz to the headset (AirDrop / Files) and import it in the app. The app reads
   spz/sog/ply/splat and trims files above 3M points automatically.

## Rules
- Do not upload private photos to online services without the user's consent.
- ML Sharp weights are non-commercial research only; tell the user before commercial use.
- Never commit user photos or generated scenes to the repo.
