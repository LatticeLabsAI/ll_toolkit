"""ll_clouds — point-cloud processing and analysis for the LatticeLabs CAD toolkit.

A standalone, NumPy/SciPy-based point-cloud library: I/O (PLY/PCD/XYZ + mesh
sampling), preprocessing (normalize, voxel/FPS downsample, outlier removal),
features (normals, curvature, statistics), registration (ICP), and segmentation
(RANSAC plane, Euclidean clustering). Heavy/optional integrations (cadling,
ll_ocadr) are exposed via lazily-imported bridges so the core library installs
and imports with only numpy + scipy (trimesh for mesh I/O).
"""
from __future__ import annotations

__version__ = "0.1.0"
