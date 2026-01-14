"""Core CADling functionality.

This module provides core CADling functionality including geometric algorithms,
topology processing, and feature extraction utilities.

Functions:
    compute_bounding_box: Compute 3D bounding box from points
    compute_centroid: Compute centroid of points
    normalize_vector: Normalize a 3D vector
    cross_product: Compute cross product of two vectors
    dot_product: Compute dot product of two vectors
"""

from __future__ import annotations

import math
from typing import List, Tuple


def compute_bounding_box(
    points: List[List[float]],
) -> Tuple[List[float], List[float]]:
    """Compute 3D axis-aligned bounding box from points.

    Args:
        points: List of 3D points [[x, y, z], ...]

    Returns:
        Tuple of (min_coords, max_coords)
    """
    if not points:
        return ([0.0, 0.0, 0.0], [0.0, 0.0, 0.0])

    min_coords = [float("inf")] * 3
    max_coords = [float("-inf")] * 3

    for point in points:
        for i in range(3):
            min_coords[i] = min(min_coords[i], point[i])
            max_coords[i] = max(max_coords[i], point[i])

    return (min_coords, max_coords)


def compute_centroid(points: List[List[float]]) -> List[float]:
    """Compute centroid (center of mass) of points.

    Args:
        points: List of 3D points [[x, y, z], ...]

    Returns:
        Centroid coordinates [x, y, z]
    """
    if not points:
        return [0.0, 0.0, 0.0]

    n = len(points)
    centroid = [0.0, 0.0, 0.0]

    for point in points:
        for i in range(3):
            centroid[i] += point[i]

    return [c / n for c in centroid]


def normalize_vector(vector: List[float]) -> List[float]:
    """Normalize a 3D vector to unit length.

    Args:
        vector: 3D vector [x, y, z]

    Returns:
        Normalized vector
    """
    length = math.sqrt(sum(v * v for v in vector))

    if length == 0:
        return [0.0, 0.0, 0.0]

    return [v / length for v in vector]


def cross_product(v1: List[float], v2: List[float]) -> List[float]:
    """Compute cross product of two 3D vectors.

    Args:
        v1: First vector [x, y, z]
        v2: Second vector [x, y, z]

    Returns:
        Cross product vector
    """
    return [
        v1[1] * v2[2] - v1[2] * v2[1],
        v1[2] * v2[0] - v1[0] * v2[2],
        v1[0] * v2[1] - v1[1] * v2[0],
    ]


def dot_product(v1: List[float], v2: List[float]) -> float:
    """Compute dot product of two 3D vectors.

    Args:
        v1: First vector [x, y, z]
        v2: Second vector [x, y, z]

    Returns:
        Dot product (scalar)
    """
    return sum(a * b for a, b in zip(v1, v2))


def vector_length(vector: List[float]) -> float:
    """Compute length (magnitude) of a vector.

    Args:
        vector: 3D vector [x, y, z]

    Returns:
        Length (scalar)
    """
    return math.sqrt(sum(v * v for v in vector))


def distance_between_points(p1: List[float], p2: List[float]) -> float:
    """Compute Euclidean distance between two 3D points.

    Args:
        p1: First point [x, y, z]
        p2: Second point [x, y, z]

    Returns:
        Distance (scalar)
    """
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(p1, p2)))


__all__ = [
    "compute_bounding_box",
    "compute_centroid",
    "normalize_vector",
    "cross_product",
    "dot_product",
    "vector_length",
    "distance_between_points",
]
