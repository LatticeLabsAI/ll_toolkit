"""B-Rep to PyTorch Geometric graph conversion.

This module converts STEP B-Rep (boundary representation) data to PyTorch Geometric
Data objects for use in graph neural networks. Uses TopologyBuilder and STEPFeatureExtractor
from the backend.step module to extract face entities, adjacency, and geometric features.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Any
import numpy as np

try:
    import torch
    from torch_geometric.data import Data
except ImportError:
    torch = None  # type: ignore
    Data = None  # type: ignore

from cadling.backend.step.topology_builder import TopologyBuilder
from cadling.backend.step.feature_extractor import STEPFeatureExtractor


# Surface type encoding (10 types)
SURFACE_TYPES = {
    "PLANE": 0,
    "CYLINDRICAL_SURFACE": 1,
    "CONICAL_SURFACE": 2,
    "SPHERICAL_SURFACE": 3,
    "TOROIDAL_SURFACE": 4,
    "B_SPLINE_SURFACE": 5,
    "SURFACE_OF_REVOLUTION": 6,
    "SURFACE_OF_LINEAR_EXTRUSION": 7,
    "SWEPT_SURFACE": 8,
    "UNKNOWN": 9,
}

NUM_SURFACE_TYPES = len(SURFACE_TYPES)


def _extract_face_entities(
    entities: Dict[int, Dict],
    topology_builder: TopologyBuilder
) -> List[int]:
    """Extract face entity IDs from STEP entities.

    Args:
        entities: STEP entities dictionary
        topology_builder: TopologyBuilder instance with topology graph built

    Returns:
        List of face entity IDs
    """
    face_ids = []

    for entity_id, entity_data in entities.items():
        entity_type = entity_data.get("type", "")

        # Check if this is a face entity
        if "FACE" in entity_type.upper() and "SURFACE" in entity_type.upper():
            face_ids.append(entity_id)

    return face_ids


def _build_face_adjacency(
    face_ids: List[int],
    entities: Dict[int, Dict],
    topology_builder: TopologyBuilder
) -> Dict[int, List[int]]:
    """Build face-to-face adjacency graph.

    Two faces are adjacent if they share an edge.

    Args:
        face_ids: List of face entity IDs
        entities: STEP entities dictionary
        topology_builder: TopologyBuilder with topology graph

    Returns:
        Dictionary mapping face_id -> list of adjacent face_ids
    """
    adjacency: Dict[int, List[int]] = {fid: [] for fid in face_ids}

    # Build edge-to-faces mapping
    edge_to_faces: Dict[int, List[int]] = {}

    for face_id in face_ids:
        # Get edges referenced by this face
        neighbors = topology_builder.get_entity_neighborhood(face_id, depth=1)

        for neighbor_id in neighbors:
            neighbor_type = entities.get(neighbor_id, {}).get("type", "")

            if "EDGE" in neighbor_type.upper():
                # This is an edge entity
                if neighbor_id not in edge_to_faces:
                    edge_to_faces[neighbor_id] = []
                edge_to_faces[neighbor_id].append(face_id)

    # Build adjacency from edge-to-faces
    for edge_id, face_list in edge_to_faces.items():
        if len(face_list) >= 2:
            # Multiple faces share this edge - they're adjacent
            for i, face_i in enumerate(face_list):
                for face_j in face_list[i+1:]:
                    if face_i in adjacency and face_j not in adjacency[face_i]:
                        adjacency[face_i].append(face_j)
                    if face_j in adjacency and face_i not in adjacency[face_j]:
                        adjacency[face_j].append(face_i)

    return adjacency


def _encode_surface_type(surface_type: str) -> np.ndarray:
    """Encode surface type as one-hot vector.

    Args:
        surface_type: Surface type string (e.g., "PLANE", "CYLINDRICAL_SURFACE")

    Returns:
        One-hot encoded vector [NUM_SURFACE_TYPES]
    """
    one_hot = np.zeros(NUM_SURFACE_TYPES, dtype=np.float32)

    # Normalize surface type name
    surface_type_upper = surface_type.upper()

    # Find matching surface type
    idx = SURFACE_TYPES.get(surface_type_upper, SURFACE_TYPES["UNKNOWN"])
    one_hot[idx] = 1.0

    return one_hot


def _extract_surface_features(
    face_id: int,
    entities: Dict[int, Dict],
    feature_extractor: STEPFeatureExtractor
) -> np.ndarray:
    """Extract geometric features for a face.

    Args:
        face_id: Face entity ID
        entities: STEP entities dictionary
        feature_extractor: STEPFeatureExtractor instance

    Returns:
        Feature vector [14] containing:
        - area (1)
        - centroid (3)
        - normal (3)
        - curvature (2): mean and gaussian
        - bounding box dimensions (3)
        - surface parameter (2): u_range, v_range for parametric surfaces
    """
    entity_data = entities.get(face_id, {})
    entity_type = entity_data.get("type", "")

    # Extract features using STEPFeatureExtractor
    features_dict = feature_extractor.extract_entity_features(
        face_id, entity_type, entity_data, entities
    )

    # Build feature vector (14 dims)
    feature_vector = np.zeros(14, dtype=np.float32)

    # Area (1)
    feature_vector[0] = features_dict.get("area", 0.0)

    # Centroid (3)
    centroid = features_dict.get("centroid", [0.0, 0.0, 0.0])
    if isinstance(centroid, (list, tuple)):
        feature_vector[1:4] = centroid[:3]

    # Normal (3)
    normal = features_dict.get("normal", [0.0, 0.0, 1.0])
    if isinstance(normal, (list, tuple)):
        feature_vector[4:7] = normal[:3]

    # Curvature (2): mean and gaussian
    feature_vector[7] = features_dict.get("mean_curvature", 0.0)
    feature_vector[8] = features_dict.get("gaussian_curvature", 0.0)

    # Bounding box dimensions (3)
    bbox = features_dict.get("bbox_dimensions", [0.0, 0.0, 0.0])
    if isinstance(bbox, (list, tuple)):
        feature_vector[9:12] = bbox[:3]

    # Surface parameters (2)
    feature_vector[12] = features_dict.get("u_range", 0.0)
    feature_vector[13] = features_dict.get("v_range", 0.0)

    return feature_vector


def compute_brep_features(
    entities: Dict[int, Dict],
    face_ids: List[int],
    topology_builder: TopologyBuilder,
    feature_extractor: STEPFeatureExtractor
) -> Tuple[np.ndarray, List[str]]:
    """Compute feature matrix for B-Rep faces.

    Args:
        entities: STEP entities dictionary
        face_ids: List of face entity IDs
        topology_builder: TopologyBuilder instance
        feature_extractor: STEPFeatureExtractor instance

    Returns:
        Tuple of (features [N, 24], surface_types [N])
        - features: Node feature matrix
          - Surface type one-hot (10)
          - Geometric features (14)
        - surface_types: List of surface type strings
    """
    n_faces = len(face_ids)
    features = np.zeros((n_faces, 24), dtype=np.float32)
    surface_types = []

    for i, face_id in enumerate(face_ids):
        entity_data = entities.get(face_id, {})

        # Get surface type
        surface_type = entity_data.get("surface_type", "UNKNOWN")
        surface_types.append(surface_type)

        # Encode surface type (10 dims)
        surface_one_hot = _encode_surface_type(surface_type)
        features[i, :10] = surface_one_hot

        # Extract geometric features (14 dims)
        geom_features = _extract_surface_features(face_id, entities, feature_extractor)
        features[i, 10:24] = geom_features

    return features, surface_types


def compute_edge_features(
    face_i: int,
    face_j: int,
    entities: Dict[int, Dict],
    feature_extractor: STEPFeatureExtractor
) -> np.ndarray:
    """Compute edge features for a pair of adjacent faces.

    Args:
        face_i: First face entity ID
        face_j: Second face entity ID
        entities: STEP entities dictionary
        feature_extractor: STEPFeatureExtractor instance

    Returns:
        Edge feature vector [8] containing:
        - Edge type (1): 0=concave, 0.5=tangent, 1=convex
        - Dihedral angle (1): angle between face normals
        - Edge length (1): length of shared edge
        - Shared edge centroid (3)
        - Edge curvature (2): mean and gaussian curvature of shared edge
    """
    edge_features = np.zeros(8, dtype=np.float32)

    # Get face data
    face_i_data = entities.get(face_i, {})
    face_j_data = entities.get(face_j, {})

    # Extract features for both faces
    features_i = feature_extractor.extract_entity_features(
        face_i, face_i_data.get("type", ""), face_i_data, entities
    )
    features_j = feature_extractor.extract_entity_features(
        face_j, face_j_data.get("type", ""), face_j_data, entities
    )

    # Compute dihedral angle from normals
    normal_i = features_i.get("normal", [0.0, 0.0, 1.0])
    normal_j = features_j.get("normal", [0.0, 0.0, 1.0])

    if isinstance(normal_i, (list, tuple)) and isinstance(normal_j, (list, tuple)):
        normal_i = np.array(normal_i[:3], dtype=np.float32)
        normal_j = np.array(normal_j[:3], dtype=np.float32)

        # Normalize
        norm_i = np.linalg.norm(normal_i)
        norm_j = np.linalg.norm(normal_j)

        if norm_i > 1e-10 and norm_j > 1e-10:
            normal_i = normal_i / norm_i
            normal_j = normal_j / norm_j

            # Compute angle
            cos_angle = np.clip(np.dot(normal_i, normal_j), -1.0, 1.0)
            dihedral_angle = np.arccos(cos_angle)

            edge_features[1] = dihedral_angle

            # Edge type: concave (0), tangent (0.5), convex (1)
            if dihedral_angle < np.pi / 3:
                edge_features[0] = 1.0  # Convex
            elif dihedral_angle > 2 * np.pi / 3:
                edge_features[0] = 0.0  # Concave
            else:
                edge_features[0] = 0.5  # Tangent/mixed

    # Compute distance between centroids as proxy for edge length
    centroid_i = features_i.get("centroid", [0.0, 0.0, 0.0])
    centroid_j = features_j.get("centroid", [0.0, 0.0, 0.0])

    if isinstance(centroid_i, (list, tuple)) and isinstance(centroid_j, (list, tuple)):
        centroid_i = np.array(centroid_i[:3], dtype=np.float32)
        centroid_j = np.array(centroid_j[:3], dtype=np.float32)

        edge_length = np.linalg.norm(centroid_i - centroid_j)
        edge_features[2] = edge_length

        # Shared edge centroid (midpoint)
        edge_centroid = (centroid_i + centroid_j) / 2.0
        edge_features[3:6] = edge_centroid

    # Edge curvature (average of face curvatures)
    mean_curv_i = features_i.get("mean_curvature", 0.0)
    mean_curv_j = features_j.get("mean_curvature", 0.0)
    edge_features[6] = (mean_curv_i + mean_curv_j) / 2.0

    gauss_curv_i = features_i.get("gaussian_curvature", 0.0)
    gauss_curv_j = features_j.get("gaussian_curvature", 0.0)
    edge_features[7] = (gauss_curv_i + gauss_curv_j) / 2.0

    return edge_features


def brep_to_pyg_graph(
    entities: Dict[int, Dict],
    face_labels: Optional[np.ndarray] = None,
) -> "Data":  # type: ignore
    """Convert STEP B-Rep entities to PyTorch Geometric Data object.

    Args:
        entities: STEP entities dictionary (from STEP parser)
        face_labels: Optional ground-truth labels for faces [N]

    Returns:
        PyG Data object with:
        - x: Node features [N, 24]
          - Surface type one-hot (10 types)
          - Geometric features (14): area, centroid, normal, curvature, bbox
        - edge_index: Face adjacency [2, E]
        - edge_attr: Edge features [E, 8]
          - Edge type, dihedral angle, length, centroid, curvature
        - y: Face labels [N] (if provided)
        - num_nodes: Number of faces

    Raises:
        ImportError: If torch is not installed
        ValueError: If no face entities found in STEP data
    """
    if torch is None:
        raise ImportError("PyTorch is required for brep_to_pyg_graph. Install with: pip install torch")

    # Initialize topology builder and feature extractor
    topology_builder = TopologyBuilder()
    topology_builder.build_topology_graph(entities)

    feature_extractor = STEPFeatureExtractor()

    # Extract face entities
    face_ids = _extract_face_entities(entities, topology_builder)

    if len(face_ids) == 0:
        # No faces found - return empty graph
        return Data(
            x=torch.zeros((0, 24), dtype=torch.float32),
            edge_index=torch.zeros((2, 0), dtype=torch.long),
            edge_attr=torch.zeros((0, 8), dtype=torch.float32),
            num_nodes=0
        )

    # Compute node features
    node_features, surface_types = compute_brep_features(
        entities, face_ids, topology_builder, feature_extractor
    )

    x = torch.from_numpy(node_features).float()

    # Build face adjacency
    adjacency = _build_face_adjacency(face_ids, entities, topology_builder)

    # Create face_id to index mapping
    face_id_to_idx = {face_id: idx for idx, face_id in enumerate(face_ids)}

    # Convert adjacency to edge_index and compute edge features
    edge_list = []
    edge_features_list = []

    for face_i, neighbors in adjacency.items():
        idx_i = face_id_to_idx[face_i]

        for face_j in neighbors:
            idx_j = face_id_to_idx[face_j]

            if idx_i < idx_j:  # Only add each edge once
                # Add both directions for undirected graph
                edge_list.append([idx_i, idx_j])
                edge_list.append([idx_j, idx_i])

                # Compute edge features
                edge_feats = compute_edge_features(
                    face_i, face_j, entities, feature_extractor
                )

                # Add features for both directions
                edge_features_list.append(edge_feats)
                edge_features_list.append(edge_feats)

    if len(edge_list) == 0:
        # No edges (isolated faces)
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_attr = torch.zeros((0, 8), dtype=torch.float32)
    else:
        edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
        edge_attr = torch.from_numpy(np.array(edge_features_list, dtype=np.float32))

    # Create PyG Data object
    data = Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        num_nodes=len(face_ids)
    )

    # Add labels if provided
    if face_labels is not None:
        if len(face_labels) == len(face_ids):
            data.y = torch.from_numpy(face_labels).long()
        else:
            # Labels don't match face count - skip
            pass

    return data
