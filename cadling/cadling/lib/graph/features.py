"""Geometric feature computation utilities for CAD graph construction.

This module provides low-level geometric calculations for computing face features,
edge features, and topological properties used in graph neural networks.
"""

from __future__ import annotations

from typing import Optional, Tuple
import numpy as np


def compute_face_centroid(vertices: np.ndarray, face: np.ndarray) -> np.ndarray:
    """Compute centroid of a triangular face.

    Args:
        vertices: Array of vertex positions [N, 3]
        face: Array of 3 vertex indices for the triangle

    Returns:
        Centroid position [3] as mean of the three vertices
    """
    return vertices[face].mean(axis=0)


def compute_face_normal(vertices: np.ndarray, face: np.ndarray) -> np.ndarray:
    """Compute normal vector of a triangular face.

    Uses the cross product of two edge vectors to compute the face normal.
    The normal is NOT normalized to unit length.

    Args:
        vertices: Array of vertex positions [N, 3]
        face: Array of 3 vertex indices for the triangle

    Returns:
        Normal vector [3] (not normalized)
    """
    v0, v1, v2 = vertices[face]
    edge1 = v1 - v0
    edge2 = v2 - v0
    normal = np.cross(edge1, edge2)
    return normal


def compute_face_normal_normalized(vertices: np.ndarray, face: np.ndarray) -> np.ndarray:
    """Compute normalized normal vector of a triangular face.

    Args:
        vertices: Array of vertex positions [N, 3]
        face: Array of 3 vertex indices for the triangle

    Returns:
        Unit normal vector [3]
    """
    normal = compute_face_normal(vertices, face)
    norm = np.linalg.norm(normal)
    if norm < 1e-10:
        return np.array([0.0, 0.0, 1.0])  # Default to z-axis for degenerate faces
    return normal / norm


def compute_face_area(vertices: np.ndarray, face: np.ndarray) -> float:
    """Compute area of a triangular face.

    Area = 0.5 * ||(v1 - v0) × (v2 - v0)||

    Args:
        vertices: Array of vertex positions [N, 3]
        face: Array of 3 vertex indices for the triangle

    Returns:
        Face area as a scalar
    """
    normal = compute_face_normal(vertices, face)
    return 0.5 * np.linalg.norm(normal)


def compute_dihedral_angle(
    normal1: np.ndarray,
    normal2: np.ndarray
) -> float:
    """Compute dihedral angle between two faces in radians.

    The dihedral angle is the angle between the two face normals.
    Result is in the range [0, π].

    Args:
        normal1: Normal vector of first face [3]
        normal2: Normal vector of second face [3]

    Returns:
        Dihedral angle in radians [0, π]
    """
    # Normalize the normals
    n1_norm = np.linalg.norm(normal1)
    n2_norm = np.linalg.norm(normal2)

    if n1_norm < 1e-10 or n2_norm < 1e-10:
        return 0.0  # Degenerate case

    n1 = normal1 / n1_norm
    n2 = normal2 / n2_norm

    # Compute angle
    cos_angle = np.dot(n1, n2)
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    angle = np.arccos(cos_angle)

    return float(angle)


def compute_edge_length(
    vertices: np.ndarray,
    edge: Tuple[int, int]
) -> float:
    """Compute length of an edge.

    Args:
        vertices: Array of vertex positions [N, 3]
        edge: Tuple of two vertex indices

    Returns:
        Edge length as a scalar
    """
    v0, v1 = vertices[edge[0]], vertices[edge[1]]
    return float(np.linalg.norm(v1 - v0))


def compute_edge_midpoint(
    vertices: np.ndarray,
    edge: Tuple[int, int]
) -> np.ndarray:
    """Compute midpoint of an edge.

    Args:
        vertices: Array of vertex positions [N, 3]
        edge: Tuple of two vertex indices

    Returns:
        Midpoint position [3]
    """
    v0, v1 = vertices[edge[0]], vertices[edge[1]]
    return (v0 + v1) / 2.0


def compute_vertex_curvature(
    mesh: "trimesh.Trimesh",  # type: ignore
    vertex_idx: int,
    method: str = "gaussian"
) -> float:
    """Estimate curvature at a vertex using discrete differential geometry.

    This is a simplified curvature estimation. For production use, consider
    more sophisticated methods from geometry processing libraries.

    Args:
        mesh: Trimesh object
        vertex_idx: Index of the vertex
        method: Curvature type - "gaussian" or "mean"

    Returns:
        Estimated curvature value
    """
    try:
        import trimesh
    except ImportError:
        return 0.0

    # Get neighboring faces
    vertex_faces = mesh.vertex_faces[vertex_idx]
    vertex_faces = vertex_faces[vertex_faces != -1]  # Remove invalid faces

    if len(vertex_faces) < 3:
        return 0.0  # Not enough faces to estimate curvature

    # Simple curvature estimation based on normal variation
    normals = mesh.face_normals[vertex_faces]

    if method == "gaussian":
        # Gaussian curvature approximation: variance of normals
        normal_variance = np.var(normals, axis=0).sum()
        return float(normal_variance)

    elif method == "mean":
        # Mean curvature approximation: average normal deviation
        mean_normal = normals.mean(axis=0)
        mean_normal_norm = np.linalg.norm(mean_normal)
        if mean_normal_norm < 1e-10:
            return 0.0
        mean_normal = mean_normal / mean_normal_norm

        deviations = np.array([
            np.linalg.norm(n - mean_normal)
            for n in normals
        ])
        return float(deviations.mean())

    else:
        raise ValueError(f"Unknown curvature method: {method}")


def compute_bounding_box(points: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute axis-aligned bounding box for a set of points.

    Args:
        points: Array of 3D points [N, 3]

    Returns:
        Tuple of (min_corner [3], max_corner [3], dimensions [3])
    """
    min_corner = points.min(axis=0)
    max_corner = points.max(axis=0)
    dimensions = max_corner - min_corner
    return min_corner, max_corner, dimensions


def compute_face_bounding_box(
    vertices: np.ndarray,
    face: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute bounding box for a single face.

    Args:
        vertices: Array of vertex positions [N, 3]
        face: Array of 3 vertex indices for the triangle

    Returns:
        Tuple of (min_corner [3], max_corner [3], dimensions [3])
    """
    face_vertices = vertices[face]
    return compute_bounding_box(face_vertices)


def normalize_features(features: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Normalize features to zero mean and unit variance.

    Args:
        features: Feature array [N, D]
        eps: Small epsilon to avoid division by zero

    Returns:
        Normalized features [N, D]
    """
    mean = features.mean(axis=0, keepdims=True)
    std = features.std(axis=0, keepdims=True)
    std = np.maximum(std, eps)  # Avoid division by zero
    return (features - mean) / std


def standardize_features(
    features: np.ndarray,
    feature_min: Optional[np.ndarray] = None,
    feature_max: Optional[np.ndarray] = None,
    eps: float = 1e-8
) -> np.ndarray:
    """Standardize features to [0, 1] range.

    Args:
        features: Feature array [N, D]
        feature_min: Pre-computed minimum values [D] (computed from data if None)
        feature_max: Pre-computed maximum values [D] (computed from data if None)
        eps: Small epsilon to avoid division by zero

    Returns:
        Standardized features [N, D] in range [0, 1]
    """
    if feature_min is None:
        feature_min = features.min(axis=0, keepdims=True)
    if feature_max is None:
        feature_max = features.max(axis=0, keepdims=True)

    feature_range = feature_max - feature_min
    feature_range = np.maximum(feature_range, eps)  # Avoid division by zero

    return (features - feature_min) / feature_range
