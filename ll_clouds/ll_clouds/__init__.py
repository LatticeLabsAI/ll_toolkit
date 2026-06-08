"""ll_clouds — point-cloud processing and analysis for the LatticeLabs CAD toolkit.

A standalone, NumPy/SciPy-based point-cloud library: I/O (PLY/PCD/XYZ + mesh
sampling), preprocessing (normalize, voxel/FPS downsample, outlier removal),
features (normals, curvature, statistics), registration (ICP), and segmentation
(RANSAC plane, Euclidean clustering). Heavy/optional integrations (cadling,
ll_ocadr) are exposed via lazily-imported bridges so the core library installs
and imports with only numpy + scipy (trimesh for mesh I/O).

Example:
    >>> from ll_clouds import sample_from_mesh, estimate_normals, ransac_plane
    >>> pc = sample_from_mesh("part.stl", n=4096, with_normals=True)
    >>> pc = estimate_normals(pc, k=24, as_cloud=True)
    >>> seg, plane = ransac_plane(pc, distance_threshold=0.02)
"""

from __future__ import annotations

from .datamodel import PointCloud, RegistrationResult, SegmentationResult
from .features import (
    bounding_box,
    centroid,
    estimate_curvature,
    estimate_normals,
    extent,
)
from .io import read_point_cloud, sample_from_mesh, write_point_cloud
from .preprocess import (
    farthest_point_downsample,
    normalize,
    remove_statistical_outliers,
    voxel_downsample,
)
from .registration import icp
from .segmentation import euclidean_cluster, ransac_plane

__version__ = "0.1.0"

__all__ = [
    "__version__",
    # data models
    "PointCloud",
    "RegistrationResult",
    "SegmentationResult",
    # io
    "read_point_cloud",
    "write_point_cloud",
    "sample_from_mesh",
    # preprocessing
    "normalize",
    "voxel_downsample",
    "farthest_point_downsample",
    "remove_statistical_outliers",
    # features
    "estimate_normals",
    "estimate_curvature",
    "bounding_box",
    "centroid",
    "extent",
    # registration
    "icp",
    # segmentation
    "ransac_plane",
    "euclidean_cluster",
]
