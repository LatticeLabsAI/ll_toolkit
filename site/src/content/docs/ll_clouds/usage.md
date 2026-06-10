---
title: ll_clouds — Usage
description: I/O, preprocessing, features, ICP registration, and segmentation over point clouds — with the real ll_clouds API.
sidebar:
  label: Usage
  order: 3
---

ll_clouds covers the core point-cloud workflow: I/O, preprocessing, features,
registration, and segmentation. The central type is the Pydantic `PointCloud`.

## Data model

```python
import numpy as np
from ll_clouds import PointCloud

pc = PointCloud(
    points=np.random.rand(1000, 3).astype(np.float32),  # [N, 3] required
    normals=None,                                        # optional [N, 3]
    colors=None,                                         # optional [N, 3] in [0, 1]
    labels=None,                                         # optional [N] int
    metadata={"source": "example"},
)
```

## I/O

```python
from ll_clouds import read_point_cloud, write_point_cloud, sample_from_mesh

pc = read_point_cloud("scan.ply")        # PLY / PCD / XYZ
write_point_cloud(pc, "scan_out.pcd")

# Sample N points from a mesh (trimesh object or file path)
pc = sample_from_mesh("part.stl", n=4096, with_normals=True, method="surface")
```

## Preprocessing

```python
from ll_clouds import (
    normalize, voxel_downsample, farthest_point_downsample,
    remove_statistical_outliers,
)

pc = normalize(pc)                              # center + unit-scale
pc = voxel_downsample(pc, voxel_size=0.01)
pc = farthest_point_downsample(pc, k=2048)      # FPS to exactly k points
pc = remove_statistical_outliers(pc)
```

## Features

```python
from ll_clouds import (
    estimate_normals, estimate_curvature, bounding_box, centroid, extent,
)

normals = estimate_normals(pc, k=16)            # [N, 3] via k-NN PCA
pc_with_normals = estimate_normals(pc, k=16, as_cloud=True)
curvature = estimate_curvature(pc, k=16)        # [N]

bb_min, bb_max = bounding_box(pc)
c = centroid(pc)
ext = extent(pc)
```

## Registration (ICP)

```python
from ll_clouds import icp

result = icp(source, target, max_iterations=50, tolerance=1e-8)
print(result.transformation)   # [4, 4] cumulative source→target transform
print(result.inlier_rmse, result.iterations, result.converged)
```

## Segmentation

```python
from ll_clouds import ransac_plane, euclidean_cluster

# RANSAC plane: labels are 1 (plane inliers) / 0 (rest), plus (a, b, c, d)
seg, coeffs = ransac_plane(pc, distance_threshold=0.05, num_iterations=200, seed=0)
print(seg.num_segments, coeffs)

# DBSCAN Euclidean clustering: labels 0..K-1, noise = -1
clusters = euclidean_cluster(pc, eps=0.5, min_points=10)
print(clusters.num_segments)
```

## Bridges (lazy)

```python
from ll_clouds.bridges import from_mesh, from_cadling_document, from_ll_ocadr_mesh

pc = from_mesh(trimesh_mesh, n=2048, with_normals=True)
pc = from_cadling_document(cadling_doc, include_normals=True)
pc = from_ll_ocadr_mesh(mesh_data)
```

These imports are lazy, so depending on cadling or ll_ocadr is optional.

## Related

- [Overview](/ll_toolkit/ll_clouds/overview/) · [Installation](/ll_toolkit/ll_clouds/installation/)
- Sample clouds from [cadling](/ll_toolkit/cadling/overview/) geometry or feed [ll_ocadr](/ll_toolkit/ll_ocadr/overview/).
