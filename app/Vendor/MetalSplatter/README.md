# MetalSplatter (vendored, modified)

A modified copy of Sean Cier's MetalSplatter (MIT, see `LICENSE`), based on
upstream commit `464eb37`.

Main changes for Walk Into Memory (not exhaustive):

- reveal effect adapted from Spark's reveal shader (World Labs, MIT)
- soft breathing edge, reach dissolve and wind in `SplatProcessing.metal`
- re-sort only after 1 cm of head movement
- unused upstream parts (SampleApp, SampleBoxRenderer, SplatConverter) removed
- unused portal / artwork effect code paths removed from the shaders and the renderer
