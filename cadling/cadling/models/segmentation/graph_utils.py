"""Graph construction utilities for CAD segmentation.

This module provides utilities for converting CAD data to graph representations:
- mesh_to_pyg_graph: Convert trimesh meshes to PyTorch Geometric graphs
- build_face_adjacency_graph: Build face adjacency graphs from topology
- compute_node_features: Extract geometric features for graph nodes
- compute_edge_features: Extract geometric features for graph edges
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Tuple, Optional

import numpy as np
import torch

if TYPE_CHECKING:
    import trimesh
    from torch_geometric.data import Data

_log = logging.getLogger(__name__)


def mesh_to_pyg_graph(
    mesh: "trimesh.Trimesh",
    use_face_graph: bool = True,
    max_neighbors: int = 10,
) -> "Data":
    """Convert trimesh mesh to PyTorch Geometric graph.

    Args:
        mesh: Input trimesh mesh
        use_face_graph: If True, creates face adjacency graph (nodes=faces).
                       If False, creates vertex graph (nodes=vertices).
        max_neighbors: Maximum number of neighbors per node (for sampling)

    Returns:
        PyTorch Geometric Data object with node features and edge indices
    """
    from torch_geometric.data import Data

    if use_face_graph:
        return _mesh_to_face_graph(mesh, max_neighbors)
    else:
        return _mesh_to_vertex_graph(mesh, max_neighbors)


def _mesh_to_face_graph(mesh: "trimesh.Trimesh", max_neighbors: int) -> "Data":
    """Convert mesh to face adjacency graph.

    Nodes: Faces
    Edges: Face adjacency (shared edges)
    Node features: centroid (3), normal (3), area (1) = 7 dims
    Edge features: dihedral angle (1), edge length (1) = 2 dims
    """
    from torch_geometric.data import Data

    num_faces = len(mesh.faces)

    # Compute face centroids
    face_centroids = np.mean(mesh.vertices[mesh.faces], axis=1)  # [F, 3]

    # Get face normals (or compute if not available)
    # Copy to ensure writable arrays for PyTorch compatibility
    if hasattr(mesh, "face_normals") and mesh.face_normals is not None:
        face_normals = mesh.face_normals.copy()
    else:
        face_normals = mesh.face_normals.copy()  # Will compute automatically

    # Compute face areas
    # Copy to ensure writable arrays for PyTorch compatibility
    face_areas = mesh.area_faces.copy().reshape(-1, 1)  # [F, 1]

    # Node features: [centroid (3) + normal (3) + area (1)] = 7 dims
    node_features = np.concatenate([face_centroids, face_normals, face_areas], axis=1)

    # Build face adjacency using existing utility
    edge_index, edge_features = _build_face_adjacency(mesh, max_neighbors)

    # Convert to torch tensors
    x = torch.from_numpy(node_features).float()
    edge_index = torch.from_numpy(edge_index).long()
    edge_attr = torch.from_numpy(edge_features).float() if edge_features is not None else None

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, num_nodes=num_faces)


def _mesh_to_vertex_graph(mesh: "trimesh.Trimesh", max_neighbors: int) -> "Data":
    """Convert mesh to vertex graph (K-NN or edge-based).

    Nodes: Vertices
    Edges: Mesh edges or K-nearest neighbors
    Node features: position (3), normal (3), curvature (1) = 7 dims
    """
    from torch_geometric.data import Data

    num_vertices = len(mesh.vertices)

    # Node features: vertex positions
    # Copy to ensure writable arrays for PyTorch compatibility
    vertex_positions = mesh.vertices.copy()  # [V, 3]

    # Vertex normals
    # Copy to ensure writable arrays for PyTorch compatibility
    if hasattr(mesh, "vertex_normals") and mesh.vertex_normals is not None:
        vertex_normals = mesh.vertex_normals.copy()
    else:
        vertex_normals = mesh.vertex_normals.copy()  # Will compute automatically

    # Compute vertex curvature (discrete Gaussian curvature)
    vertex_curvature = compute_vertex_curvature(mesh).reshape(-1, 1)  # [V, 1]

    # Node features: [position (3) + normal (3) + curvature (1)] = 7 dims
    node_features = np.concatenate([vertex_positions, vertex_normals, vertex_curvature], axis=1)

    # Build edge index from mesh edges
    edge_index = _build_vertex_adjacency(mesh)

    # Edge features: edge length
    edge_vectors = vertex_positions[edge_index[1]] - vertex_positions[edge_index[0]]
    edge_lengths = np.linalg.norm(edge_vectors, axis=1, keepdims=True)  # [E, 1]

    # Convert to torch tensors
    x = torch.from_numpy(node_features).float()
    edge_index_tensor = torch.from_numpy(edge_index).long()
    edge_attr = torch.from_numpy(edge_lengths).float()

    return Data(x=x, edge_index=edge_index_tensor, edge_attr=edge_attr, num_nodes=num_vertices)


def _build_face_adjacency(
    mesh: "trimesh.Trimesh", max_neighbors: int = 10
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Build face adjacency graph from mesh.

    Two faces are adjacent if they share an edge.

    Returns:
        edge_index: [2, E] array of edge indices
        edge_features: [E, 2] array with dihedral angle and edge length
    """
    # Use trimesh's built-in face adjacency
    face_adjacency = mesh.face_adjacency  # [E, 2] pairs of adjacent faces
    face_adjacency_edges = mesh.face_adjacency_edges  # [E, 2] shared edge vertex indices

    # Create bidirectional edges
    edge_index = np.concatenate([face_adjacency.T, face_adjacency[:, [1, 0]].T], axis=1)

    # Compute edge features
    edge_features = []

    # 1. Dihedral angles
    if hasattr(mesh, "face_adjacency_angles"):
        dihedral_angles = mesh.face_adjacency_angles  # [E]
        # Repeat for bidirectional edges
        dihedral_angles = np.concatenate([dihedral_angles, dihedral_angles])
    else:
        # Compute dihedral angles manually
        dihedral_angles = _compute_dihedral_angles(mesh, face_adjacency, face_adjacency_edges)
        dihedral_angles = np.concatenate([dihedral_angles, dihedral_angles])

    # 2. Edge lengths
    edge_vertices = mesh.vertices[face_adjacency_edges]  # [E, 2, 3]
    edge_lengths = np.linalg.norm(edge_vertices[:, 1] - edge_vertices[:, 0], axis=1)  # [E]
    edge_lengths = np.concatenate([edge_lengths, edge_lengths])  # Repeat for bidirectional

    # Combine edge features: [dihedral_angle, edge_length]
    edge_features = np.stack([dihedral_angles, edge_lengths], axis=1)  # [2E, 2]

    return edge_index, edge_features


def _build_vertex_adjacency(mesh: "trimesh.Trimesh") -> np.ndarray:
    """Build vertex adjacency from mesh edges.

    Returns:
        edge_index: [2, E] array of edge indices
    """
    # Get unique edges (trimesh stores as [E, 2])
    edges = mesh.edges_unique

    # Create bidirectional edges
    edge_index = np.concatenate([edges.T, edges[:, [1, 0]].T], axis=1)

    return edge_index


def _compute_dihedral_angles(
    mesh: "trimesh.Trimesh",
    face_adjacency: np.ndarray,
    face_adjacency_edges: np.ndarray,
) -> np.ndarray:
    """Compute dihedral angles between adjacent faces.

    Args:
        mesh: Input mesh
        face_adjacency: [E, 2] pairs of adjacent face indices
        face_adjacency_edges: [E, 2] shared edge vertex indices

    Returns:
        angles: [E] dihedral angles in radians
    """
    # Get face normals
    normals = mesh.face_normals

    # Get normals of adjacent faces
    n1 = normals[face_adjacency[:, 0]]  # [E, 3]
    n2 = normals[face_adjacency[:, 1]]  # [E, 3]

    # Compute angle between normals
    # angle = arccos(n1 · n2)
    dot_products = np.sum(n1 * n2, axis=1)  # [E]
    dot_products = np.clip(dot_products, -1.0, 1.0)  # Numerical stability
    angles = np.arccos(dot_products)  # [E]

    # Determine concave vs convex
    # If angle > π/2, edge is concave (negative angle)
    # This requires checking edge direction, simplified here
    return angles


def compute_vertex_curvature(mesh: "trimesh.Trimesh") -> np.ndarray:
    """Compute discrete Gaussian curvature at vertices.

    Uses angle defect method: curvature = 2π - sum(angles)

    Args:
        mesh: Input mesh

    Returns:
        curvature: [V] Gaussian curvature at each vertex
    """
    num_vertices = len(mesh.vertices)
    curvature = np.zeros(num_vertices)

    # Compute angle defect at each vertex
    for vertex_idx in range(num_vertices):
        # Find faces containing this vertex
        face_indices = np.where(np.any(mesh.faces == vertex_idx, axis=1))[0]

        if len(face_indices) == 0:
            continue

        # Sum angles at this vertex in all adjacent faces
        angle_sum = 0.0

        for face_idx in face_indices:
            face = mesh.faces[face_idx]

            # Find position of vertex in face
            local_idx = np.where(face == vertex_idx)[0][0]

            # Get the three vertices of the triangle
            v0 = mesh.vertices[face[local_idx]]
            v1 = mesh.vertices[face[(local_idx + 1) % 3]]
            v2 = mesh.vertices[face[(local_idx - 1) % 3]]

            # Compute angle at v0
            edge1 = v1 - v0
            edge2 = v2 - v0

            # Normalize
            edge1_norm = np.linalg.norm(edge1)
            edge2_norm = np.linalg.norm(edge2)

            if edge1_norm > 1e-10 and edge2_norm > 1e-10:
                cos_angle = np.dot(edge1, edge2) / (edge1_norm * edge2_norm)
                cos_angle = np.clip(cos_angle, -1.0, 1.0)
                angle = np.arccos(cos_angle)
                angle_sum += angle

        # Gaussian curvature = angle defect
        curvature[vertex_idx] = 2 * np.pi - angle_sum

    return curvature


def build_face_adjacency_graph(
    faces: list,
    edges: list,
    node_features: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Build face adjacency graph from face and edge lists.

    This is for B-Rep data where faces and edges are provided separately.

    Args:
        faces: List of face objects
        edges: List of edge objects
        node_features: Optional [F, D] node feature array

    Returns:
        edge_index: [2, E] adjacency matrix
        node_features: [F, D] node features (identity if not provided)
    """
    # Build face-to-edge mapping
    face_to_edges = {}
    for face_idx, face in enumerate(faces):
        face_edges = getattr(face, "edges", [])
        face_to_edges[face_idx] = set(face_edges)

    # Find face adjacency (faces sharing edges)
    edge_pairs = []
    for i in range(len(faces)):
        for j in range(i + 1, len(faces)):
            # Check if faces share any edges
            shared_edges = face_to_edges[i].intersection(face_to_edges[j])
            if len(shared_edges) > 0:
                edge_pairs.append([i, j])
                edge_pairs.append([j, i])  # Bidirectional

    if len(edge_pairs) == 0:
        # No adjacency found, create self-loops
        edge_index = np.array([[i, i] for i in range(len(faces))]).T
    else:
        edge_index = np.array(edge_pairs).T

    # Create default node features if not provided
    if node_features is None:
        node_features = np.eye(len(faces))  # Identity features

    return edge_index, node_features


def compute_node_features(
    vertices: np.ndarray,
    normals: np.ndarray,
    faces: np.ndarray,
) -> np.ndarray:
    """Compute geometric node features for vertices.

    Args:
        vertices: [V, 3] vertex positions
        normals: [V, 3] vertex normals
        faces: [F, 3] face indices

    Returns:
        features: [V, 7] node features (position, normal, curvature)
    """
    import trimesh

    # Create temporary mesh for curvature computation
    # Use process=False to prevent mesh modification
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)

    # Use trimesh's built-in vertex_defects property (angle defect method)
    # This computes discrete Gaussian curvature as (2π - sum of angles at vertex)
    curvature = mesh.vertex_defects.reshape(-1, 1)

    # Concatenate features: [position, normal, curvature]
    features = np.concatenate([vertices, normals, curvature], axis=1)

    return features


def compute_edge_features(
    edge_index: np.ndarray,
    vertices: np.ndarray,
    faces: np.ndarray,
    normals: np.ndarray,
) -> np.ndarray:
    """Compute geometric edge features.

    Args:
        edge_index: [2, E] edge indices
        vertices: [V, 3] vertex positions
        faces: [F, 3] face indices
        normals: [F, 3] face normals

    Returns:
        edge_features: [E, 2] edge features (dihedral angle, edge length)
    """
    num_edges = edge_index.shape[1]

    # Compute edge vectors and lengths
    edge_vectors = vertices[edge_index[1]] - vertices[edge_index[0]]
    edge_lengths = np.linalg.norm(edge_vectors, axis=1, keepdims=True)  # [E, 1]

    # Build edge-to-faces mapping
    # Map each edge (as sorted vertex pair) to list of adjacent face indices
    edge_to_faces = {}
    for face_idx, face in enumerate(faces):
        # Each triangular face has 3 edges
        edges = [
            tuple(sorted([face[0], face[1]])),
            tuple(sorted([face[1], face[2]])),
            tuple(sorted([face[2], face[0]])),
        ]
        for edge in edges:
            if edge not in edge_to_faces:
                edge_to_faces[edge] = []
            edge_to_faces[edge].append(face_idx)

    # Compute dihedral angles for each edge
    dihedral_angles = []
    for i in range(num_edges):
        v1, v2 = edge_index[0, i], edge_index[1, i]
        edge_key = tuple(sorted([v1, v2]))

        if edge_key in edge_to_faces and len(edge_to_faces[edge_key]) == 2:
            # Interior edge with exactly 2 adjacent faces - compute dihedral angle
            face_idx1, face_idx2 = edge_to_faces[edge_key]
            n1 = normals[face_idx1]
            n2 = normals[face_idx2]

            # Compute angle between normals using dot product
            # Clip to avoid numerical errors in arccos
            dot_product = np.dot(n1, n2)
            dot_product = np.clip(dot_product, -1.0, 1.0)
            angle = np.arccos(dot_product)
            dihedral_angles.append(angle)
        else:
            # Boundary edge (1 face) or non-manifold edge (>2 faces)
            dihedral_angles.append(0.0)

    dihedral_angles = np.array(dihedral_angles).reshape(-1, 1)

    # Concatenate features: [dihedral_angle, edge_length]
    edge_features = np.concatenate([dihedral_angles, edge_lengths], axis=1)

    return edge_features
