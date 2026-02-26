"""Mesh to PyTorch Geometric graph conversion.

This module converts triangle meshes (from trimesh) to PyTorch Geometric Data objects
for use in graph neural networks. Supports both face-based and vertex-based graphs.
"""

from __future__ import annotations

from typing import Optional, Dict, List, Tuple, Any
import numpy as np

try:
    import torch
    from torch_geometric.data import Data
except ImportError:
    torch = None  # type: ignore
    Data = None  # type: ignore

try:
    import trimesh
except ImportError:
    trimesh = None  # type: ignore

from .features import (
    compute_face_centroid,
    compute_face_normal_normalized,
    compute_face_area,
    compute_dihedral_angle,
    compute_edge_length,
    compute_vertex_curvature,
)


def _build_face_adjacency_graph(mesh: "trimesh.Trimesh") -> Dict[int, List[int]]:  # type: ignore
    """Build face adjacency graph from mesh.

    This is adapted from mesh_chunker._build_face_adjacency() but returns
    an adjacency list suitable for PyG graph construction.

    Args:
        mesh: Trimesh object

    Returns:
        Dictionary mapping face_id -> list of adjacent face_ids
    """
    # Build edge-to-faces mapping
    edge_to_faces: Dict[Tuple[int, int], List[int]] = {}

    for face_idx, face in enumerate(mesh.faces):
        # Get the three edges of this face
        edges = [
            tuple(sorted([face[0], face[1]])),
            tuple(sorted([face[1], face[2]])),
            tuple(sorted([face[2], face[0]])),
        ]

        for edge in edges:
            if edge not in edge_to_faces:
                edge_to_faces[edge] = []
            edge_to_faces[edge].append(face_idx)

    # Build adjacency list
    adjacency: Dict[int, List[int]] = {i: [] for i in range(len(mesh.faces))}

    for edge, face_list in edge_to_faces.items():
        if len(face_list) == 2:
            # Two faces share this edge - they're adjacent
            face0, face1 = face_list
            adjacency[face0].append(face1)
            adjacency[face1].append(face0)

    return adjacency


def compute_mesh_features(
    mesh: "trimesh.Trimesh",  # type: ignore
    include_normals: bool = True,
    include_curvature: bool = False
) -> np.ndarray:
    """Compute feature matrix for all faces in a mesh.

    Args:
        mesh: Trimesh object
        include_normals: Include face normals in features
        include_curvature: Include curvature estimates (slower)

    Returns:
        Feature matrix [N_faces, feature_dim]
        - If include_normals=True, include_curvature=False: [N, 7] (centroid + normal + area)
        - If include_normals=True, include_curvature=True: [N, 8] (centroid + normal + area + curvature)
        - If include_normals=False: [N, 4] (centroid + area)
    """
    n_faces = len(mesh.faces)
    vertices = mesh.vertices
    faces = mesh.faces

    # Vectorized centroid computation: mean of 3 face vertices
    v0 = vertices[faces[:, 0]]  # [N, 3]
    v1 = vertices[faces[:, 1]]  # [N, 3]
    v2 = vertices[faces[:, 2]]  # [N, 3]

    centroids = (v0 + v1 + v2) / 3.0  # [N, 3]

    # Vectorized area computation via cross product
    edge1 = v1 - v0
    edge2 = v2 - v0
    cross = np.cross(edge1, edge2)  # [N, 3]
    areas = 0.5 * np.linalg.norm(cross, axis=1, keepdims=True)  # [N, 1]

    feature_parts = [centroids]

    if include_normals:
        # Vectorized normal computation: normalized cross product
        norms = np.linalg.norm(cross, axis=1, keepdims=True)
        # Avoid division by zero for degenerate triangles
        safe_norms = np.where(norms > 1e-10, norms, 1.0)
        normals = cross / safe_norms  # [N, 3]
        feature_parts.append(normals)

    feature_parts.append(areas)

    if include_curvature:
        # Curvature still requires per-vertex computation (mesh topology dependent)
        curvature_list = []
        for face_idx in range(n_faces):
            face = faces[face_idx]
            curvatures = [
                compute_vertex_curvature(mesh, vid, method="gaussian")
                for vid in face
            ]
            curvature_list.append(np.mean(curvatures))
        curvature_array = np.array(curvature_list, dtype=np.float32).reshape(-1, 1)
        feature_parts.append(curvature_array)

    return np.concatenate(feature_parts, axis=1).astype(np.float32)


def mesh_to_pyg_graph(
    mesh: "trimesh.Trimesh",  # type: ignore
    use_face_graph: bool = True,
    include_normals: bool = True,
    include_curvature: bool = False,
    labels: Optional[np.ndarray] = None
) -> "Data":  # type: ignore
    """Convert trimesh to PyTorch Geometric Data object.

    Args:
        mesh: Trimesh object with vertices and faces
        use_face_graph: If True, nodes=faces; if False, nodes=vertices
        include_normals: Include face/vertex normals in features
        include_curvature: Include curvature estimates (slower)
        labels: Optional labels for faces/vertices [N]

    Returns:
        PyG Data object with:
        - x: Node features [N, feature_dim]
          - If face graph: centroid (3) + normal (3) + area (1) = 7 dims (default)
          - If face graph with curvature: 8 dims
          - If vertex graph: position (3) + normal (3) = 6 dims (default)
          - If vertex graph with curvature: 7 dims
        - edge_index: Adjacency [2, E]
        - edge_attr: Edge features [E, 2]
          - If face graph: (dihedral angle + centroid distance)
          - If vertex graph: (edge length + normal angle)
        - pos: 3D positions [N, 3] (face centroids or vertex positions)
        - y: Labels [N] (if provided)

    Raises:
        ImportError: If torch or trimesh are not installed
    """
    if torch is None:
        raise ImportError("PyTorch is required for mesh_to_pyg_graph. Install with: pip install torch")

    if trimesh is None:
        raise ImportError("trimesh is required for mesh_to_pyg_graph. Install with: pip install trimesh")

    if not use_face_graph:
        # VERTEX-BASED GRAPH IMPLEMENTATION
        # Nodes = vertices, Edges = mesh edges connecting vertices

        # Compute node features for vertices
        node_features = compute_vertex_features(
            mesh,
            include_normals=include_normals,
            include_curvature=include_curvature
        )

        # Convert to torch tensor
        x = torch.from_numpy(node_features).float()

        # Use vertex positions directly
        pos = torch.from_numpy(mesh.vertices).float()

        # Build vertex adjacency graph from mesh edges
        # Each edge in the mesh connects two vertices
        edge_list = []
        edge_features_list = []

        # Get unique edges from mesh
        edges = mesh.edges_unique

        for edge in edges:
            v1, v2 = edge[0], edge[1]

            # Add both directions (undirected graph)
            edge_list.append([v1, v2])
            edge_list.append([v2, v1])

            # Compute edge features
            # 1. Edge length
            vert1 = mesh.vertices[v1]
            vert2 = mesh.vertices[v2]
            edge_length = np.linalg.norm(vert2 - vert1)

            # 2. Edge curvature (angle between vertex normals)
            if include_normals:
                norm1 = mesh.vertex_normals[v1]
                norm2 = mesh.vertex_normals[v2]
                dot_prod = np.clip(np.dot(norm1, norm2), -1.0, 1.0)
                edge_angle = np.arccos(dot_prod)
            else:
                edge_angle = 0.0

            # Add edge features for both directions
            edge_features_list.append([edge_length, edge_angle])
            edge_features_list.append([edge_length, edge_angle])

        if len(edge_list) == 0:
            # Degenerate case: no edges
            edge_index = torch.zeros((2, 0), dtype=torch.long)
            edge_attr = torch.zeros((0, 2), dtype=torch.float32)
        else:
            edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
            edge_attr = torch.tensor(edge_features_list, dtype=torch.float32)

        # Create PyG Data object
        data = Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            pos=pos
        )

        # Add labels if provided (should match vertex count)
        if labels is not None:
            if len(labels) == len(mesh.vertices):
                data.y = torch.from_numpy(labels).long()
            else:
                raise ValueError(
                    f"Labels length ({len(labels)}) does not match vertex count ({len(mesh.vertices)})"
                )

        return data

    # FACE-BASED GRAPH IMPLEMENTATION (original code)
    # Compute node features for faces
    node_features = compute_mesh_features(
        mesh,
        include_normals=include_normals,
        include_curvature=include_curvature
    )

    # Convert to torch tensor
    x = torch.from_numpy(node_features).float()

    # Compute face centroids for positions (vectorized)
    v0 = mesh.vertices[mesh.faces[:, 0]]
    v1 = mesh.vertices[mesh.faces[:, 1]]
    v2 = mesh.vertices[mesh.faces[:, 2]]
    centroids = ((v0 + v1 + v2) / 3.0).astype(np.float32)
    pos = torch.from_numpy(centroids)

    # Build adjacency graph
    adjacency = _build_face_adjacency_graph(mesh)

    # Convert adjacency to edge_index format [2, E]
    edge_list = []
    edge_features_list = []

    for face_i, neighbors in adjacency.items():
        for face_j in neighbors:
            if face_i < face_j:  # Only add each edge once (undirected graph)
                edge_list.append([face_i, face_j])
                edge_list.append([face_j, face_i])  # Add both directions for undirected

                # Compute edge features
                # 1. Dihedral angle
                normal_i = compute_face_normal_normalized(mesh.vertices, mesh.faces[face_i])
                normal_j = compute_face_normal_normalized(mesh.vertices, mesh.faces[face_j])
                dihedral = compute_dihedral_angle(normal_i, normal_j)

                # 2. Edge length (distance between centroids)
                centroid_i = centroids[face_i]
                centroid_j = centroids[face_j]
                dist = np.linalg.norm(centroid_i - centroid_j)

                # Add edge features for both directions
                edge_features_list.append([dihedral, dist])
                edge_features_list.append([dihedral, dist])

    if len(edge_list) == 0:
        # Degenerate case: no edges (single face or disconnected faces)
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_attr = torch.zeros((0, 2), dtype=torch.float32)
    else:
        edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_features_list, dtype=torch.float32)

    # Create PyG Data object
    data = Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        pos=pos
    )

    # Add labels if provided
    if labels is not None:
        data.y = torch.from_numpy(labels).long()

    return data


def compute_vertex_features(
    mesh: "trimesh.Trimesh",  # type: ignore
    include_normals: bool = True,
    include_curvature: bool = False
) -> np.ndarray:
    """Compute feature matrix for all vertices in a mesh.

    This is for future support of vertex-based graphs.

    Args:
        mesh: Trimesh object
        include_normals: Include vertex normals in features
        include_curvature: Include curvature estimates

    Returns:
        Feature matrix [N_vertices, feature_dim]
    """
    n_vertices = len(mesh.vertices)
    features_list = []

    for vertex_idx in range(n_vertices):
        vertex = mesh.vertices[vertex_idx]
        vertex_features = list(vertex)  # Position (3 dims)

        # Compute normal (3 dims)
        if include_normals:
            normal = mesh.vertex_normals[vertex_idx]
            vertex_features.extend(normal)

        # Compute curvature (1 dim)
        if include_curvature:
            curvature = compute_vertex_curvature(mesh, vertex_idx, method="gaussian")
            vertex_features.append(curvature)

        features_list.append(vertex_features)

    return np.array(features_list, dtype=np.float32)
