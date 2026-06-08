# ll_clouds

Point-cloud processing and analysis for the LatticeLabs CAD toolkit.

A standalone, dependency-light library (NumPy + SciPy; trimesh for mesh I/O)
covering the core point-cloud workflow:

- **I/O** — read/write PLY, PCD, XYZ; sample point clouds from meshes.
- **Preprocessing** — normalize, voxel downsample, farthest-point downsample,
  statistical outlier removal.
- **Features** — per-point normals (k-NN PCA), curvature, geometry statistics.
- **Registration** — point-to-point ICP.
- **Segmentation** — RANSAC plane fitting, Euclidean clustering.

Optional bridges convert documents/inputs from the sibling packages
[`cadling`](../cadling/) (CAD document processing) and
[`ll_ocadr`](../ll_ocadr/) (optical CAD recognition) into a `PointCloud`.
These package names are intentional — they are other members of the LatticeLabs
monorepo — and are imported lazily, so the core library has no hard dependency
on them.

See the per-module API and usage examples (added as the library is built out).
