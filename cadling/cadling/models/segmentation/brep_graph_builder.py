"""B-Rep face adjacency graph builder.

Builds face-level adjacency graphs from STEP B-Rep topology for segmentation models.
Leverages existing TopologyBuilder and STEPFeatureExtractor infrastructure.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

import numpy as np
import torch

if TYPE_CHECKING:
    from torch_geometric.data import Data

    from cadling.datamodel.base_models import CADlingDocument
    from cadling.datamodel.step import STEPEntityItem

_log = logging.getLogger(__name__)


class BRepFaceGraphBuilder:
    """Build face adjacency graphs from B-Rep topology.

    Uses existing TopologyBuilder to extract face entities and build
    face-to-face adjacency relationships. Uses STEPFeatureExtractor
    to compute geometric features for each face.

    Face Graph Structure:
    - Nodes: B-Rep faces (ADVANCED_FACE, FACE_SURFACE entities)
    - Edges: Face adjacency (faces sharing edges/vertices)
    - Node features: surface type (one-hot), area, curvature, normal, centroid
    - Edge features: edge type (concave/convex), dihedral angle, edge length
    """

    def __init__(self):
        """Initialize B-Rep face graph builder."""
        try:
            from stepnet import (
                STEPFeatureExtractor,
                STEPTopologyBuilder,
            )

            self.feature_extractor = STEPFeatureExtractor()
            self.topology_builder = STEPTopologyBuilder()

            _log.info("Initialized B-Rep face graph builder")
        except ImportError:
            _log.error("stepnet (ll_stepnet package) not installed, cannot build B-Rep graphs")
            raise

    def build_face_graph(
        self,
        doc: "CADlingDocument",
        item: Optional["STEPEntityItem"] = None,
    ) -> "Data":
        """Build face adjacency graph from STEP document.

        Args:
            doc: CADlingDocument with STEP entities
            item: Specific STEPEntityItem to build graph for (optional)

        Returns:
            PyTorch Geometric Data object with face graph
        """
        from torch_geometric.data import Data

        # Extract face entities
        face_entities = self._extract_face_entities(doc)

        if len(face_entities) == 0:
            _log.warning("No face entities found in document")
            # Return empty graph
            return Data(
                x=torch.zeros((1, 24)),
                edge_index=torch.zeros((2, 0), dtype=torch.long),
                edge_attr=torch.zeros((0, 8)),
            )

        # Build face-to-face adjacency
        edge_index, edge_features = self._build_face_adjacency(doc, face_entities)

        # Extract face features
        node_features = self._extract_face_features(doc, face_entities)

        # Convert to torch tensors
        x = torch.from_numpy(node_features).float()
        edge_index = torch.from_numpy(edge_index).long()
        edge_attr = (
            torch.from_numpy(edge_features).float() if edge_features is not None else None
        )

        # Store face entity IDs for reference
        face_ids = [f["entity_id"] for f in face_entities]

        graph_data = Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            num_nodes=len(face_entities),
        )

        # Store additional metadata
        graph_data.face_ids = face_ids
        graph_data.faces = face_entities

        return graph_data

    def _extract_face_entities(
        self, doc: "CADlingDocument"
    ) -> List[Dict]:
        """Extract face entities from STEP document.

        Args:
            doc: CADlingDocument with STEP entities

        Returns:
            List of face entity dictionaries
        """
        # Use topology builder to extract hierarchy
        # This assumes doc has an 'entities' attribute or similar
        # Adapt based on actual CADlingDocument structure

        face_entities = []

        # Extract from document items
        for item in doc.items:
            # Check if item represents a face
            if hasattr(item, "entity_type"):
                entity_type = item.entity_type.upper()

                if "FACE" in entity_type:
                    face_entities.append({
                        "entity_id": getattr(item, "entity_id", None),
                        "entity_type": entity_type,
                        "text": item.text if hasattr(item, "text") else "",
                        "item": item,
                    })

        # Alternative: Use topology_builder to extract faces
        if len(face_entities) == 0:
            _log.debug("No face items found, trying topology extraction")
            # This would require access to raw STEP entities
            # For now, return empty list
            pass

        return face_entities

    def _build_face_adjacency(
        self,
        doc: "CADlingDocument",
        face_entities: List[Dict],
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Build face-to-face adjacency graph.

        Two faces are adjacent if they share an edge or vertex.

        Args:
            doc: CADlingDocument
            face_entities: List of face entity dictionaries

        Returns:
            edge_index: [2, E] adjacency matrix
            edge_features: [E, 8] edge features (or None)
        """
        num_faces = len(face_entities)

        # Build face-to-edge mapping
        face_to_edges = {}

        for i, face_entity in enumerate(face_entities):
            # Extract edges associated with this face
            # This requires parsing the face entity geometry
            # For now, use simplified approach

            # Get entity text and parse for edge references
            entity_text = face_entity.get("text", "")

            # Simplified: Extract edge IDs from entity text
            # Format: ADVANCED_FACE('...',(#123,#124,#125),...)
            edge_refs = self._extract_edge_references(entity_text)

            face_to_edges[i] = set(edge_refs)

        # Find face pairs sharing edges
        edge_pairs = []

        for i in range(num_faces):
            for j in range(i + 1, num_faces):
                # Check if faces share edges
                shared_edges = face_to_edges[i].intersection(face_to_edges[j])

                if len(shared_edges) > 0:
                    edge_pairs.append([i, j])
                    edge_pairs.append([j, i])  # Bidirectional

        if len(edge_pairs) == 0:
            # No adjacency found - create self-loops or empty edges
            edge_index = np.array([[i, i] for i in range(num_faces)]).T
        else:
            edge_index = np.array(edge_pairs).T

        # Compute REAL edge features - NO MORE PLACEHOLDERS!
        edge_features = self._compute_edge_features(
            edge_index=edge_index,
            face_entities=face_entities,
            doc=doc,
        )

        return edge_index, edge_features

    def _extract_edge_references(self, entity_text: str) -> List[int]:
        """Extract edge references from STEP entity text.

        Args:
            entity_text: STEP entity text

        Returns:
            List of edge reference IDs
        """
        import re

        # Find all #N references in the entity
        edge_refs = re.findall(r"#(\d+)", entity_text)

        # Convert to integers
        edge_refs = [int(ref) for ref in edge_refs]

        return edge_refs

    def _compute_edge_features(
        self,
        edge_index: np.ndarray,
        face_entities: List[Dict],
        doc: "CADlingDocument",
    ) -> np.ndarray:
        """Compute real edge features for face-to-face adjacency.

        Edge features (8 dims):
        0. Dihedral angle (radians, 0-π)
        1. Edge type (0=concave, 0.5=tangent, 1=convex)
        2. Edge length (normalized)
        3. Normal dot product (cosine of dihedral angle)
        4. Centroid distance between faces
        5. Area ratio (smaller/larger face)
        6. Curvature compatibility (similar curvature = 1)
        7. Surface type compatibility (same type = 1)

        Args:
            edge_index: [2, E] adjacency matrix
            face_entities: List of face entity dictionaries
            doc: CADlingDocument with OCC shape

        Returns:
            [E, 8] edge feature matrix
        """
        num_edges = edge_index.shape[1]
        edge_features = np.zeros((num_edges, 8))

        # Try to load OCC shape for geometric analysis
        shape = self._load_shape_from_document(doc)
        topods_faces = self._extract_faces_from_shape(shape) if shape else []

        # Match entities to faces if OCC shape available
        entity_to_face = {}
        if topods_faces and len(topods_faces) == len(face_entities):
            entity_to_face = self._match_entity_to_face(face_entities, topods_faces)

        has_occ_geometry = len(entity_to_face) > 0

        # Extract face features if not already done
        # We need normals, centroids, areas, curvatures for edge features
        face_normals = np.zeros((len(face_entities), 3))
        face_centroids = np.zeros((len(face_entities), 3))
        face_areas = np.zeros(len(face_entities))
        face_curvatures = np.zeros((len(face_entities), 2))  # Gaussian, mean
        face_types = []

        from cadling.lib.geometry.face_geometry import FaceGeometryExtractor

        geom_extractor = FaceGeometryExtractor() if has_occ_geometry else None

        for i, face_entity in enumerate(face_entities):
            entity_text = face_entity.get("text", "")

            # Try to get geometric features from OCC
            if i in entity_to_face and geom_extractor:
                topods_face = entity_to_face[i]
                geom_features = geom_extractor.extract_features(topods_face)

                if geom_features:
                    face_normals[i] = geom_features["normal"]
                    face_centroids[i] = geom_features["centroid"]
                    face_areas[i] = geom_features["surface_area"]
                    face_curvatures[i, 0] = geom_features["gaussian_curvature"]
                    face_curvatures[i, 1] = geom_features["mean_curvature"]

            # Extract surface type from entity text
            try:
                entity_info = self.feature_extractor.extract_entity_info(entity_text)
                surface_type = entity_info.get("surface_type", "OTHER")
                face_types.append(surface_type)

                # Use entity info for area if OCC not available
                if face_areas[i] == 0:
                    face_areas[i] = entity_info.get("area", 0.0)
            except:
                face_types.append("OTHER")

        # Compute edge features for each pair
        for e in range(num_edges):
            src_face = edge_index[0, e]
            dst_face = edge_index[1, e]

            # Skip self-loops
            if src_face == dst_face:
                continue

            # Feature 0: Dihedral angle
            # Angle between face normals
            n1 = face_normals[src_face]
            n2 = face_normals[dst_face]

            # Normalize if needed
            n1_norm = np.linalg.norm(n1)
            n2_norm = np.linalg.norm(n2)

            if n1_norm > 1e-6 and n2_norm > 1e-6:
                n1 = n1 / n1_norm
                n2 = n2 / n2_norm

                # Compute angle
                dot_product = np.clip(np.dot(n1, n2), -1.0, 1.0)
                dihedral_angle = np.arccos(dot_product)

                edge_features[e, 0] = dihedral_angle
                edge_features[e, 3] = dot_product

                # Feature 1: Edge type (concave/convex/tangent)
                # Simplified: based on angle
                if dihedral_angle < np.pi / 6:  # < 30 degrees
                    edge_features[e, 1] = 0.5  # Tangent
                elif dihedral_angle < np.pi / 2:  # < 90 degrees
                    edge_features[e, 1] = 1.0  # Convex
                else:
                    edge_features[e, 1] = 0.0  # Concave
            else:
                # Invalid normals - use defaults
                edge_features[e, 0] = 0.0
                edge_features[e, 1] = 0.5
                edge_features[e, 3] = 0.0

            # Feature 2: Edge length (use centroid distance as proxy)
            c1 = face_centroids[src_face]
            c2 = face_centroids[dst_face]
            centroid_dist = np.linalg.norm(c2 - c1)

            edge_features[e, 2] = centroid_dist
            edge_features[e, 4] = centroid_dist

            # Feature 5: Area ratio
            a1 = face_areas[src_face]
            a2 = face_areas[dst_face]

            if a1 > 0 and a2 > 0:
                area_ratio = min(a1, a2) / max(a1, a2)
                edge_features[e, 5] = area_ratio
            else:
                edge_features[e, 5] = 1.0  # Default: equal areas

            # Feature 6: Curvature compatibility
            k1 = face_curvatures[src_face]
            k2 = face_curvatures[dst_face]

            # Compare mean curvatures
            if abs(k1[1]) > 1e-6 or abs(k2[1]) > 1e-6:
                curv_diff = abs(k1[1] - k2[1]) / (abs(k1[1]) + abs(k2[1]) + 1e-6)
                curv_compat = np.exp(-curv_diff)  # Similar curvature → 1
                edge_features[e, 6] = curv_compat
            else:
                edge_features[e, 6] = 1.0  # Both flat

            # Feature 7: Surface type compatibility
            t1 = face_types[src_face] if src_face < len(face_types) else "OTHER"
            t2 = face_types[dst_face] if dst_face < len(face_types) else "OTHER"

            edge_features[e, 7] = 1.0 if t1 == t2 else 0.0

        if has_occ_geometry:
            _log.info(
                f"Computed REAL edge features for {num_edges} edges "
                f"(mean dihedral angle: {np.mean(edge_features[:, 0]):.2f} rad)"
            )
        else:
            _log.warning(
                f"Computed edge features for {num_edges} edges WITHOUT OCC geometry. "
                f"Features may be less accurate (default normals/centroids)."
            )

        return edge_features

    def _load_shape_from_document(self, doc: "CADlingDocument"):
        """Load pythonocc shape from document if available.

        Args:
            doc: CADlingDocument (should have _occ_shape if STEP backend loaded it)

        Returns:
            TopoDS_Shape or None if not available
        """
        if hasattr(doc, "_occ_shape") and doc._occ_shape is not None:
            _log.debug("Found cached OCC shape in document")
            return doc._occ_shape

        _log.debug("No OCC shape available in document")
        return None

    def _extract_faces_from_shape(self, shape) -> List:
        """Extract all TopoDS_Face objects from shape.

        Args:
            shape: TopoDS_Shape

        Returns:
            List of TopoDS_Face objects
        """
        if shape is None:
            return []

        try:
            from OCC.Core.TopExp import TopExp_Explorer
            from OCC.Core.TopAbs import TopAbs_FACE

            faces = []
            explorer = TopExp_Explorer(shape, TopAbs_FACE)

            while explorer.More():
                face = explorer.Current()
                faces.append(face)
                explorer.Next()

            _log.debug(f"Extracted {len(faces)} TopoDS_Face objects from shape")
            return faces

        except Exception as e:
            _log.warning(f"Failed to extract faces from shape: {e}")
            return []

    def _match_entity_to_face(
        self, face_entities: List[Dict], topods_faces: List
    ) -> Dict[int, any]:
        """Match STEP entity indices to TopoDS_Face objects.

        Strategy: Match by order (assumes face entities and TopoDS faces are in same order).
        For more robust matching, could use area or centroid comparison.

        Args:
            face_entities: List of face entity dictionaries
            topods_faces: List of TopoDS_Face objects

        Returns:
            Dictionary mapping entity index to TopoDS_Face
        """
        entity_to_face = {}

        # Simple order-based matching
        # NOTE: This assumes entities and faces are in the same order,
        # which is usually the case for STEP files
        min_len = min(len(face_entities), len(topods_faces))

        for i in range(min_len):
            entity_to_face[i] = topods_faces[i]

        if len(face_entities) != len(topods_faces):
            _log.warning(
                f"Entity count ({len(face_entities)}) != Face count ({len(topods_faces)}). "
                f"Matched {min_len} faces."
            )

        return entity_to_face

    def _extract_face_features(
        self,
        doc: "CADlingDocument",
        face_entities: List[Dict],
    ) -> np.ndarray:
        """Extract geometric features for each face.

        Features (24 dims total):
        - Surface type (one-hot, 10 types): PLANE, CYLINDER, CONE, SPHERE, TORUS, NURBS, etc.
        - Surface area (1)
        - Curvature (2): Gaussian, mean
        - Normal vector (3)
        - Centroid (3)
        - Bounding box dimensions (3)
        - Additional geometric properties (2)

        Args:
            doc: CADlingDocument
            face_entities: List of face entity dictionaries

        Returns:
            [F, 24] feature matrix
        """
        from cadling.lib.geometry.face_geometry import FaceGeometryExtractor

        num_faces = len(face_entities)
        feature_dim = 24

        features = np.zeros((num_faces, feature_dim))

        # Surface types for one-hot encoding
        surface_types = [
            "PLANE",
            "CYLINDER",
            "CONE",
            "SPHERE",
            "TORUS",
            "B_SPLINE",
            "BEZIER",
            "NURBS",
            "SURFACE_OF_REVOLUTION",
            "OTHER",
        ]

        # Try to load shape and extract real geometric features
        shape = self._load_shape_from_document(doc)
        topods_faces = self._extract_faces_from_shape(shape) if shape else []
        entity_to_face = (
            self._match_entity_to_face(face_entities, topods_faces)
            if topods_faces
            else {}
        )

        # Create geometry extractor if we have faces
        geom_extractor = None
        has_real_geometry = False

        if len(entity_to_face) > 0:
            geom_extractor = FaceGeometryExtractor()
            has_real_geometry = geom_extractor.has_pythonocc
            _log.info(
                f"Extracting REAL geometric features for {len(entity_to_face)} faces "
                f"(pythonocc available: {has_real_geometry})"
            )
        else:
            _log.warning(
                "No OCC shape available - using placeholder geometric features. "
                "This will result in ZERO-valued curvature and default normals!"
            )

        for i, face_entity in enumerate(face_entities):
            entity_text = face_entity.get("text", "")

            # Use feature extractor to get surface type and basic properties
            try:
                entity_info = self.feature_extractor.extract_entity_info(entity_text)

                # Extract surface type
                surface_type = entity_info.get("surface_type", "OTHER").upper()

                # One-hot encode surface type (first 10 dims)
                if surface_type in surface_types:
                    type_idx = surface_types.index(surface_type)
                    features[i, type_idx] = 1.0
                else:
                    features[i, 9] = 1.0  # OTHER

                # Extract surface area from entity info
                features[i, 10] = entity_info.get("area", 0.0)

                # REAL GEOMETRIC FEATURES - NO MORE PLACEHOLDERS!
                if i in entity_to_face and geom_extractor:
                    topods_face = entity_to_face[i]
                    geom_features = geom_extractor.extract_features(topods_face)

                    if geom_features:
                        # Use REAL geometric features
                        features[i, 11] = geom_features["gaussian_curvature"]
                        features[i, 12] = geom_features["mean_curvature"]
                        features[i, 13:16] = geom_features["normal"]
                        features[i, 16:19] = geom_features["centroid"]
                        features[i, 19:22] = geom_features["bbox_dimensions"]

                        # Override area with real computed area if available
                        if geom_features["surface_area"] > 0:
                            features[i, 10] = geom_features["surface_area"]

                        _log.debug(
                            f"Face {i}: Real features - "
                            f"K={geom_features['gaussian_curvature']:.3f}, "
                            f"H={geom_features['mean_curvature']:.3f}, "
                            f"area={geom_features['surface_area']:.3f}"
                        )
                    else:
                        # Geometry extraction failed - use placeholders with warning
                        _log.warning(f"Failed to extract geometry for face {i}")
                        features[i, 11] = 0.0  # Gaussian curvature
                        features[i, 12] = 0.0  # Mean curvature
                        features[i, 13:16] = [0.0, 0.0, 1.0]  # Default up
                        features[i, 16:19] = [0.0, 0.0, 0.0]  # Centroid
                        features[i, 19:22] = [1.0, 1.0, 1.0]  # Bounding box
                else:
                    # No OCC shape available - MUST use placeholders
                    # Log this so we know the data quality is degraded
                    features[i, 11] = 0.0  # Gaussian curvature
                    features[i, 12] = 0.0  # Mean curvature
                    features[i, 13:16] = [0.0, 0.0, 1.0]  # Default up
                    features[i, 16:19] = [0.0, 0.0, 0.0]  # Centroid
                    features[i, 19:22] = [1.0, 1.0, 1.0]  # Bounding box

                # Additional properties (not used currently)
                features[i, 22] = 0.0
                features[i, 23] = 0.0

            except Exception as e:
                _log.warning(f"Failed to extract features for face {i}: {e}")
                # Use default features
                features[i, 9] = 1.0  # Mark as OTHER

        # Store metadata about feature quality
        if has_real_geometry:
            _log.info(
                f"Successfully extracted REAL geometric features for {len(entity_to_face)}/{num_faces} faces"
            )
        else:
            _log.warning(
                f"Using PLACEHOLDER geometric features for all {num_faces} faces. "
                f"Training on this data will have degraded quality!"
            )

        return features
