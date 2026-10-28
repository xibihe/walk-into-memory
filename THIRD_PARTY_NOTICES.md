# Third-party notices

## MetalSplatter — MIT, © 2026 Sean Cier
Vendored and modified in `app/Vendor/MetalSplatter` (based on upstream commit
`464eb37`). Full licence: `app/Vendor/MetalSplatter/LICENSE`, also bundled as
`SceneAssets/MetalSplatter-LICENSE.txt`.

## Spark — MIT, © 2025 World Labs Technologies, Inc.
The reveal styles (Spread, Magic, Unroll, Twister) and the value noise in
`SplatProcessing.metal` are adapted from Spark's splat reveal effects.
Full licence: `app/WalkIntoMemory/SceneAssets/SparkRevealReference-LICENSE.txt`.

## SPZ — MIT, © 2024 Niantic Labs (Swift port by Sean Cier)
Used through the `spz-swift` Swift package. The pipeline's `wim/spz.py` writes
the same open format. Full licence: `app/WalkIntoMemory/SceneAssets/SPZ-LICENSE.txt`.

## ZIPFoundation — MIT, © Thomas Zoechling
Swift package dependency of SplatIO (used to read SOG archives).

## Python packages used by the pipeline
numpy, SciPy, plyfile, OpenCV, PyTorch (and optionally torchvision) are
installed by the user from their own distributions under their own licences;
none of them are vendored here.

## Apple ML Sharp — NOT included
The pipeline expects the Gaussian PLY written by Apple's ML Sharp, but this
repository contains no ML Sharp code or weights. Install it yourself from
Apple's `apple/ml-sharp` repository and read its licence files:

- the code is under Apple's sample-code style licence;
- the model weights are under the Apple Machine Learning Research Model
  License, which allows research use only (non-commercial).

Scenes you generate with the ML Sharp weights may therefore be subject to
those terms (the licence also covers "Model Derivatives"); check before any
commercial use. Any other source of a camera-frame 3DGS PLY works with the pipeline too.
