"""Graph construction utilities for CAD data.

This module converts mesh and B-Rep CAD data to PyTorch Geometric graphs
for use in graph neural networks (GNNs). Provides real geometric feature
extraction to replace placeholder random data in training pipelines.

Main Functions:
    mesh_to_pyg_graph: Convert trimesh to PyTorch Geometric Data
    brep_to_pyg_graph: Convert STEP B-Rep entities to PyTorch Geometric Data
    compute_mesh_features: Extract features from mesh faces
    compute_brep_features: Extract features from B-Rep faces

Feature Utilities:
    compute_face_centroid: Compute centroid of a triangular face
    compute_face_normal: Compute normal vector of a face
    compute_face_area: Compute area of a face
    compute_dihedral_angle: Compute angle between two faces

Usage Example:
    ```python
    import trimesh
    from cadling.lib.graph import mesh_to_pyg_graph

    # Load mesh
    mesh = trimesh.load("model.stl")

    # Convert to PyG graph
    graph = mesh_to_pyg_graph(mesh, use_face_graph=True)

    # graph.x contains real geometric features, not random data!
    # graph.edge_index contains real face adjacency
    ```

    ```python
    from cadling.backend.step import STEPParser
    from cadling.lib.graph import brep_to_pyg_graph

    # Parse STEP file
    parser = STEPParser()
    entities = parser.parse(step_content)

    # Convert to PyG graph
    graph = brep_to_pyg_graph(entities)

    # graph.x contains [N, 24] real surface features
    # graph.edge_attr contains [E, 8] real edge features
    ```
"""

from .mesh_graph import mesh_to_pyg_graph, compute_mesh_features
from .brep_graph import brep_to_pyg_graph, compute_brep_features, SURFACE_TYPES
from .features import (
    compute_face_centroid,
    compute_face_normal,
    compute_face_normal_normalized,
    compute_face_area,
    compute_dihedral_angle,
    compute_edge_length,
    compute_edge_midpoint,
    compute_vertex_curvature,
    compute_bounding_box,
    compute_face_bounding_box,
    normalize_features,
    standardize_features,
)

__all__ = [
    # Main conversion functions
    "mesh_to_pyg_graph",
    "brep_to_pyg_graph",
    # Feature computation
    "compute_mesh_features",
    "compute_brep_features",
    # Geometric utilities
    "compute_face_centroid",
    "compute_face_normal",
    "compute_face_normal_normalized",
    "compute_face_area",
    "compute_dihedral_angle",
    "compute_edge_length",
    "compute_edge_midpoint",
    "compute_vertex_curvature",
    "compute_bounding_box",
    "compute_face_bounding_box",
    # Feature normalization
    "normalize_features",
    "standardize_features",
    # Constants
    "SURFACE_TYPES",
]
