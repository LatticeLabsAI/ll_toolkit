"""Vertex operations subpackage for CAD mesh generation.

Provides three complementary modules for post-processing predicted
vertex positions:

- **vertex_validation**: Check bounds, collisions, manifold property,
  face degeneracy, winding consistency, and Euler characteristic.
- **vertex_clustering**: Group near-duplicate vertices, merge them,
  and remap face indices.
- **vertex_refinement**: Iteratively refine coarse predictions toward
  a target surface using gradient descent or scipy optimization.

Typical workflow::

    from geotoken.vertex import VertexValidator, VertexClusterer, VertexMerger
    from geotoken.vertex import CoarseToFineRefiner

    # 1. Validate raw predictions
    validator = VertexValidator(coord_bounds=(-1.0, 1.0))
    report = validator.validate(vertices, faces)

    # 2. Cluster and merge near-duplicate vertices
    clusterer = VertexClusterer(merge_distance=0.005)
    clustering = clusterer.cluster(vertices)
    merged_verts, clean_faces = VertexMerger.merge(vertices, faces, clustering)

    # 3. Refine toward target surface
    refiner = CoarseToFineRefiner(max_iterations=20)
    result = refiner.refine(merged_verts, target_points=reference_samples)
"""
from __future__ import annotations

from geotoken.vertex.vertex_clustering import (
    ClusteringResult,
    VertexClusterer,
    VertexMerger,
)
from geotoken.vertex.vertex_refinement import (
    CoarseToFineRefiner,
    OptimizationResult,
    RefinementResult,
    VertexPositionOptimizer,
)
from geotoken.vertex.vertex_validation import (
    BoundsCheckResult,
    CollisionCheckResult,
    DegeneracyCheckResult,
    EulerCheckResult,
    ManifoldCheckResult,
    TopologyValidator,
    VertexValidationReport,
    VertexValidator,
    WindingCheckResult,
)

__all__ = [
    # Validation
    "VertexValidator",
    "TopologyValidator",
    "VertexValidationReport",
    "BoundsCheckResult",
    "CollisionCheckResult",
    "DegeneracyCheckResult",
    "ManifoldCheckResult",
    "WindingCheckResult",
    "EulerCheckResult",
    # Clustering
    "VertexClusterer",
    "VertexMerger",
    "ClusteringResult",
    # Refinement
    "CoarseToFineRefiner",
    "VertexPositionOptimizer",
    "RefinementResult",
    "OptimizationResult",
]
