Optional bundled scene.

Drop one scene here before building, for example:
  garden.spz    the .spz written by the pipeline
  garden.json   its sidecar (same base name)

The app opens the first .spz it finds in this folder when nothing has been
imported yet. Without a sidecar the scene still renders, but reach dissolve,
wind, immersion edges and photo placement are off.

The *-LICENSE.txt files are third-party notices shipped inside the app bundle.
Do not commit personal scenes: *.spz and *.ply are git-ignored.
