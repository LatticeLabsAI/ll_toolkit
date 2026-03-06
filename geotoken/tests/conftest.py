"""Test fixtures for geotoken."""
from __future__ import annotations

import os  # noqa: I001 - must precede torch for OpenMP
import sys

# OpenMP protection (macOS): load torch before numpy to avoid libomp conflicts.
# Use conditional import so pure-numpy tests still run without torch.
if sys.platform == "darwin":
    os.environ.setdefault("OMP_NUM_THREADS", "1")
try:
    import torch  # noqa: F401
except ImportError:
    pass

import pytest  # noqa: E402

import numpy as np


@pytest.fixture
def cube_mesh():
    """Unit cube mesh (8 vertices, 12 triangles)."""
    vertices = np.array([
        [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
        [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1],
    ], dtype=float)
    faces = np.array([
        [0, 1, 2], [0, 2, 3],  # bottom
        [4, 6, 5], [4, 7, 6],  # top
        [0, 4, 5], [0, 5, 1],  # front
        [2, 6, 7], [2, 7, 3],  # back
        [0, 3, 7], [0, 7, 4],  # left
        [1, 5, 6], [1, 6, 2],  # right
    ])
    return vertices, faces


@pytest.fixture
def sphere_mesh():
    """Approximate sphere mesh (UV sphere, 42 vertices)."""
    n_lat = 5
    n_lon = 8
    radius = 1.0

    vertices = [[0.0, 0.0, radius]]  # North pole
    for i in range(1, n_lat):
        lat = np.pi * i / n_lat
        for j in range(n_lon):
            lon = 2 * np.pi * j / n_lon
            x = radius * np.sin(lat) * np.cos(lon)
            y = radius * np.sin(lat) * np.sin(lon)
            z = radius * np.cos(lat)
            vertices.append([x, y, z])
    vertices.append([0.0, 0.0, -radius])  # South pole
    vertices = np.array(vertices)

    faces = []
    # North pole cap
    for j in range(n_lon):
        j_next = (j + 1) % n_lon
        faces.append([0, 1 + j, 1 + j_next])
    # Middle bands
    for i in range(n_lat - 2):
        for j in range(n_lon):
            j_next = (j + 1) % n_lon
            row = 1 + i * n_lon
            next_row = row + n_lon
            faces.append([row + j, next_row + j, next_row + j_next])
            faces.append([row + j, next_row + j_next, row + j_next])
    # South pole cap
    south = len(vertices) - 1
    last_row = 1 + (n_lat - 2) * n_lon
    for j in range(n_lon):
        j_next = (j + 1) % n_lon
        faces.append([south, last_row + j_next, last_row + j])
    faces = np.array(faces)

    return vertices, faces


@pytest.fixture
def cylinder_mesh():
    """Approximate cylinder mesh."""
    n_sides = 16
    height = 2.0
    radius = 0.5

    vertices = []
    for i in range(n_sides):
        angle = 2 * np.pi * i / n_sides
        x = radius * np.cos(angle)
        y = radius * np.sin(angle)
        vertices.append([x, y, 0.0])        # bottom ring
        vertices.append([x, y, height])      # top ring
    # Center points
    vertices.append([0.0, 0.0, 0.0])         # bottom center
    vertices.append([0.0, 0.0, height])       # top center
    vertices = np.array(vertices)

    faces = []
    bc = 2 * n_sides       # bottom center
    tc = 2 * n_sides + 1   # top center
    for i in range(n_sides):
        i_next = (i + 1) % n_sides
        b0 = 2 * i
        t0 = 2 * i + 1
        b1 = 2 * i_next
        t1 = 2 * i_next + 1
        # Side quads (2 triangles)
        faces.append([b0, b1, t1])
        faces.append([b0, t1, t0])
        # Bottom cap
        faces.append([bc, b1, b0])
        # Top cap
        faces.append([tc, t0, t1])
    faces = np.array(faces)

    return vertices, faces
