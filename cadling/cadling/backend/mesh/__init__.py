"""Mesh backend module for generic mesh formats (OBJ, PLY, OFF, GLB, etc.).

Uses trimesh for loading and processing. For STL files, use the dedicated
STLBackend in cadling.backend.stl which has native ASCII/binary parsing.
"""

from cadling.backend.mesh.mesh_backend import MeshBackend, MeshViewBackend

__all__ = ["MeshBackend", "MeshViewBackend"]
