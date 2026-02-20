"""B-Rep face adjacency graph builder.

Builds face-level adjacency graphs from STEP B-Rep topology for segmentation models.
Leverages existing TopologyBuilder and STEPFeatureExtractor infrastructure.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

if TYPE_CHECKING:
    from torch_geometric.data import Data

    from cadling.datamodel.base_models import CADlingDocument
    from cadling.datamodel.step import STEPEntityItem
    from cadling.models.segmentation.architectures.brep_net import CoedgeData

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
        doc: CADlingDocument,
        item: STEPEntityItem | None = None,
    ) -> Data:
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

        # If a specific item is provided, filter to faces related to that item
        if item is not None:
            item_id = getattr(item, "entity_id", None)
            if item_id is not None:
                # Filter to faces that reference this item or are referenced by it
                item_text = getattr(item, "text", "") or ""
                filtered = []
                for face_entity in face_entities:
                    face_id = face_entity.get("entity_id")
                    face_text = face_entity.get("text", "")
                    # Include if face references item or item references face
                    if f"#{item_id}" in face_text or (face_id and f"#{face_id}" in item_text):
                        filtered.append(face_entity)
                if filtered:
                    face_entities = filtered
                    _log.debug(f"Filtered to {len(face_entities)} faces related to item #{item_id}")

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
        self, doc: CADlingDocument
    ) -> list[dict]:
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
            # entity_type may be on item directly (STEPEntityItem) or on item.label
            entity_type_raw = getattr(item, "entity_type", None) or getattr(item.label, "entity_type", None)
            if entity_type_raw:
                entity_type = entity_type_raw.upper()

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

            # Try to extract faces from raw STEP text using regex parsing
            try:
                import re

                # Get raw STEP text from document
                step_text = None
                raw_content = getattr(doc, 'raw_content', None)
                if raw_content:
                    step_text = raw_content
                elif hasattr(doc, 'properties') and doc.properties.get("step_text"):
                    step_text = doc.properties.get("step_text")
                else:
                    backend = getattr(doc, '_backend', None)
                    if backend is not None and hasattr(backend, 'content'):
                        step_text = backend.content

                if not step_text:
                    _log.warning("No STEP text available for face extraction")
                    return face_entities

                # Parse face entities directly from STEP text using regex
                # Match patterns like: #123 = ADVANCED_FACE('name', ...);
                face_pattern = re.compile(
                    r"#(\d+)\s*=\s*(ADVANCED_FACE|FACE_SURFACE|FACE_OUTER_BOUND|FACE_BOUND)"
                    r"\s*\(([^;]+)\)\s*;",
                    re.IGNORECASE | re.MULTILINE
                )

                for match in face_pattern.finditer(step_text):
                    entity_id = int(match.group(1))
                    entity_type = match.group(2).upper()
                    entity_text = match.group(0)

                    face_entities.append({
                        "entity_id": entity_id,
                        "entity_type": entity_type,
                        "text": entity_text,
                    })

                _log.debug(f"Extracted {len(face_entities)} faces via regex parsing")

            except Exception as e:
                _log.error(f"Face extraction from STEP text failed: {e}")

        return face_entities

    def _build_face_adjacency(
        self,
        doc: CADlingDocument,
        face_entities: list[dict],
    ) -> tuple[np.ndarray, np.ndarray | None]:
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
            # Extract edges associated with this face by parsing the face entity geometry
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

    def _extract_edge_references(self, entity_text: str) -> list[int]:
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
        face_entities: list[dict],
        doc: CADlingDocument,
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
            except Exception as e:
                _log.debug(f"Failed to extract entity info: {e}")
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

    def _load_shape_from_document(self, doc: CADlingDocument):
        """Load pythonocc shape from document if available.

        Uses multi-pattern loading strategy:
        1. doc._occ_shape (cached shape)
        2. doc._backend._occ_shape (backend cached shape)
        3. doc._backend.get_shape() (lazy loading via method)
        4. doc._backend.load_shape() (alternative lazy loading)

        Args:
            doc: CADlingDocument (should have _occ_shape if STEP backend loaded it)

        Returns:
            TopoDS_Shape or None if not available
        """
        # Strategy 1: Check cached OCC shape on document
        occ_shape = getattr(doc, "_occ_shape", None)
        if occ_shape is not None:
            _log.debug("Found cached OCC shape in document")
            return occ_shape

        # Strategy 2: Try to get from backend
        backend = getattr(doc, "_backend", None)
        if backend is not None:

            # Pattern 2a: backend._occ_shape
            if hasattr(backend, "_occ_shape") and backend._occ_shape is not None:
                _log.debug("Found OCC shape in backend._occ_shape")
                return backend._occ_shape

            # Pattern 2b: backend.get_shape()
            if hasattr(backend, "get_shape") and callable(backend.get_shape):
                try:
                    shape = backend.get_shape()
                    if shape is not None:
                        _log.debug("Loaded OCC shape via backend.get_shape()")
                        return shape
                except Exception as e:
                    _log.debug(f"backend.get_shape() failed: {e}")

            # Pattern 2c: backend.load_shape()
            if hasattr(backend, "load_shape") and callable(backend.load_shape):
                try:
                    shape = backend.load_shape()
                    if shape is not None:
                        _log.debug("Loaded OCC shape via backend.load_shape()")
                        return shape
                except Exception as e:
                    _log.debug(f"backend.load_shape() failed: {e}")

            # Pattern 2d: backend._shape (some backends use this)
            if hasattr(backend, "_shape") and backend._shape is not None:
                _log.debug("Found OCC shape in backend._shape")
                return backend._shape

        _log.debug("No OCC shape available in document or backend")
        return None

    def _extract_faces_from_shape(self, shape) -> list:
        """Extract all TopoDS_Face objects from shape.

        Args:
            shape: TopoDS_Shape

        Returns:
            List of TopoDS_Face objects
        """
        if shape is None:
            return []

        try:
            from OCC.Core.TopAbs import TopAbs_FACE
            from OCC.Core.TopExp import TopExp_Explorer

            faces = []
            explorer = TopExp_Explorer(shape, TopAbs_FACE)  # type: ignore[arg-type]

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
        self, face_entities: list[dict], topods_faces: list
    ) -> dict[int, Any]:
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
        doc: CADlingDocument,
        face_entities: list[dict],
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
                        # OCC extraction failed - try STEP text heuristics
                        _log.warning(f"OCC extraction failed for face {i}, using text heuristics")
                        self._estimate_features_from_step_text(
                            features, i, surface_type, entity_info, entity_text,
                        )
                else:
                    # No OCC shape available - use STEP text heuristics
                    self._estimate_features_from_step_text(
                        features, i, surface_type, entity_info, entity_text,
                    )

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

    @staticmethod
    def _estimate_features_from_step_text(
        features: np.ndarray,
        face_idx: int,
        surface_type: str,
        entity_info: dict,
        entity_text: str,
    ) -> None:
        """Estimate geometric features from surface type and STEP text.

        When OCC shape is not available, we can still derive meaningful
        approximate features from the surface type classification and
        any numeric parameters parsed from the STEP entity text.

        This is better than returning all-zeros because:
        - Planes have known curvature (0, 0) and axis-aligned normals
        - Cylinders/cones/spheres have known curvature from radius
        - STEP text often contains point coordinates for centroids

        Args:
            features: Feature matrix to fill in-place.
            face_idx: Index of the face in the feature matrix.
            surface_type: Classified surface type (PLANE, CYLINDER, etc.).
            entity_info: Parsed entity information dict.
            entity_text: Raw STEP entity text for coordinate extraction.
        """
        import re

        i = face_idx

        # ------ Curvature from surface type ------
        radius = entity_info.get("radius", 0.0)

        if surface_type == "PLANE":
            features[i, 11] = 0.0  # Gaussian curvature
            features[i, 12] = 0.0  # Mean curvature
        elif surface_type == "CYLINDER" and radius > 0:
            features[i, 11] = 0.0           # K = 0 for cylinder
            features[i, 12] = 1.0 / radius  # H = 1/R for cylinder
        elif surface_type == "SPHERE" and radius > 0:
            features[i, 11] = 1.0 / (radius * radius)  # K = 1/R^2
            features[i, 12] = 1.0 / radius              # H = 1/R
        elif surface_type == "CONE" and radius > 0:
            features[i, 11] = 0.0           # K = 0 for cone
            features[i, 12] = 0.5 / radius  # H ~ 1/(2R) approx
        elif surface_type == "TORUS":
            major_r = entity_info.get("major_radius", 0.0)
            minor_r = entity_info.get("minor_radius", radius)
            if major_r > 0 and minor_r > 0:
                features[i, 11] = 1.0 / (major_r * minor_r)  # K approx
                features[i, 12] = 0.5 * (1.0 / major_r + 1.0 / minor_r)
            else:
                features[i, 11] = 0.0
                features[i, 12] = 0.0
        else:
            features[i, 11] = 0.0
            features[i, 12] = 0.0

        # ------ Normal from surface type and axis ------
        axis = entity_info.get("axis", None)
        if axis and len(axis) == 3:
            norm = np.sqrt(sum(a * a for a in axis))
            if norm > 1e-6:
                features[i, 13:16] = [a / norm for a in axis]
            else:
                features[i, 13:16] = [0.0, 0.0, 1.0]
        elif surface_type == "PLANE":
            # Planes often have z-up normal, but try to parse from text
            features[i, 13:16] = [0.0, 0.0, 1.0]
        else:
            features[i, 13:16] = [0.0, 0.0, 1.0]

        # ------ Centroid from STEP text coordinates ------
        location = entity_info.get("location", None)
        if location and len(location) == 3:
            features[i, 16:19] = location
        else:
            # Try to parse CARTESIAN_POINT from entity text
            point_match = re.search(
                r"CARTESIAN_POINT\s*\(\s*'[^']*'\s*,\s*\(\s*"
                r"([-\d.eE+]+)\s*,\s*([-\d.eE+]+)\s*,\s*([-\d.eE+]+)\s*\)",
                entity_text,
            )
            if point_match:
                try:
                    features[i, 16] = float(point_match.group(1))
                    features[i, 17] = float(point_match.group(2))
                    features[i, 18] = float(point_match.group(3))
                except (ValueError, IndexError):
                    features[i, 16:19] = [0.0, 0.0, 0.0]
            else:
                features[i, 16:19] = [0.0, 0.0, 0.0]

        # ------ Bounding box from area or radius ------
        area = entity_info.get("area", 0.0)
        if area > 0:
            # Approximate: assume roughly square face
            side = np.sqrt(area)
            features[i, 19:22] = [side, side, 0.0]
        elif radius > 0:
            features[i, 19:22] = [2 * radius, 2 * radius, 2 * radius]
        else:
            # A real face must have *some* physical extent.
            # Use unit-scale default rather than zero, which would be
            # indistinguishable from "no data" and mislead downstream models.
            features[i, 19:22] = [1.0, 1.0, 0.0]

        _log.debug(
            "Face %d: STEP-text heuristic features - type=%s, "
            "K=%.4f, H=%.4f, normal=%s, centroid=%s",
            i, surface_type,
            features[i, 11], features[i, 12],
            features[i, 13:16].tolist(),
            features[i, 16:19].tolist(),
        )

    # ------------------------------------------------------------------
    # UV-sampled graph construction (Phase 4: GNN Upgrades)
    # ------------------------------------------------------------------

    def build_uv_sampled_graph(
        self,
        doc: CADlingDocument,
        grid_size: int = 10,
    ) -> Data:
        """Build face adjacency graph with UV-grid sampled node features.

        Like build_face_graph but node features are UV-grid samples
        (grid_size x grid_size x 7) per face, flattened to a single vector
        per node for compatibility with PyG Data objects. The raw grids are
        also stored on the Data object as ``face_grids``.

        Args:
            doc: CADlingDocument with STEP entities.
            grid_size: UV grid resolution per face (default 10).

        Returns:
            PyTorch Geometric Data object where:
              - x: [F, grid_size*grid_size*7] flattened UV-grid features
              - face_grids: [F, 7, grid_size, grid_size] for direct CNN use
              - edge_index, edge_attr: face adjacency as in build_face_graph
        """
        from torch_geometric.data import Data

        from cadling.models.segmentation.architectures.uv_net import UVGridSampler

        face_entities = self._extract_face_entities(doc)

        if len(face_entities) == 0:
            _log.warning("No face entities found for UV-sampled graph")
            empty_grid_dim = grid_size * grid_size * 7
            return Data(
                x=torch.zeros((1, empty_grid_dim)),
                edge_index=torch.zeros((2, 0), dtype=torch.long),
                edge_attr=torch.zeros((0, 8)),
            )

        # Build adjacency (reuse existing method)
        edge_index_np, edge_features = self._build_face_adjacency(doc, face_entities)

        # Sample UV grids for each face
        sampler = UVGridSampler(grid_size=grid_size)
        shape = self._load_shape_from_document(doc)
        topods_faces = self._extract_faces_from_shape(shape) if shape else []
        entity_to_face = (
            self._match_entity_to_face(face_entities, topods_faces)
            if topods_faces
            else {}
        )

        grids = []
        for i, _face_entity in enumerate(face_entities):
            if i in entity_to_face:
                grid = sampler.sample_face(entity_to_face[i], grid_size)
            else:
                # Use placeholder with entity index as seed
                grid = sampler.sample_face(i, grid_size)
            grids.append(grid)

        # Stack: [F, grid_size, grid_size, 7]
        face_grids_hwc = torch.stack(grids, dim=0)
        # Rearrange to [F, 7, grid_size, grid_size] for CNN
        face_grids_chw = face_grids_hwc.permute(0, 3, 1, 2).contiguous()
        # Flatten for x: [F, grid_size*grid_size*7]
        x_flat = face_grids_hwc.reshape(len(face_entities), -1)

        edge_index_t = torch.from_numpy(edge_index_np).long()
        edge_attr_t = (
            torch.from_numpy(edge_features).float()
            if edge_features is not None
            else None
        )

        graph_data = Data(
            x=x_flat,
            edge_index=edge_index_t,
            edge_attr=edge_attr_t,
            num_nodes=len(face_entities),
        )

        # Store grids in CNN-ready format and metadata
        graph_data.face_grids = face_grids_chw
        graph_data.grid_size = grid_size
        graph_data.face_ids = [f["entity_id"] for f in face_entities]
        graph_data.faces = face_entities

        _log.info(
            f"Built UV-sampled graph: {len(face_entities)} faces, "
            f"grid_size={grid_size}, "
            f"x shape={x_flat.shape}, "
            f"face_grids shape={face_grids_chw.shape}"
        )

        return graph_data

    # ------------------------------------------------------------------
    # Coedge graph construction (Phase 4: GNN Upgrades)
    # ------------------------------------------------------------------

    def build_coedge_graph(
        self,
        doc: CADlingDocument,
    ) -> CoedgeData:
        """Build coedge-level graph from B-Rep topology.

        Each topological edge in a B-Rep solid has two oriented coedges
        (one per adjacent face). This method constructs the coedge graph
        with next/prev/mate pointers required by BRepNetEncoder.

        Tries OCC-based extraction first (more accurate), falls back to
        STEP text parsing if OCC shape is not available.

        Coedge features (12 dims):
          - edge curve type one-hot (6): LINE, CIRCLE, ELLIPSE, B_SPLINE, PARABOLA, OTHER
          - edge length (1)
          - tangent vector at midpoint (3)
          - curvature at midpoint (1)
          - convexity flag (1): 1=convex, 0=concave, 0.5=tangent

        Args:
            doc: CADlingDocument with STEP entities.

        Returns:
            CoedgeData object for use with BRepNetEncoder.
        """
        from cadling.models.segmentation.architectures.brep_net import CoedgeData

        # Try OCC-based extraction first (more accurate)
        shape = self._load_shape_from_document(doc)
        if shape is not None:
            try:
                coedge_data = self._build_coedge_graph_from_occ(shape)
                if coedge_data is not None and coedge_data.features.shape[0] > 0:
                    _log.info(
                        f"Built coedge graph from OCC shape: "
                        f"{coedge_data.features.shape[0]} coedges"
                    )
                    return coedge_data
            except Exception as e:
                _log.debug(f"OCC coedge extraction failed, falling back to STEP text: {e}")

        # Fall back to STEP text parsing
        _log.debug("Using STEP text parsing for coedge extraction")
        face_entities = self._extract_face_entities(doc)
        num_faces = len(face_entities)

        if num_faces == 0:
            _log.warning("No face entities found for coedge graph")
            return CoedgeData(
                features=torch.zeros((0, 12)),
                next_indices=torch.zeros(0, dtype=torch.long),
                prev_indices=torch.zeros(0, dtype=torch.long),
                mate_indices=torch.zeros(0, dtype=torch.long),
                face_indices=torch.zeros(0, dtype=torch.long),
            )

        # Build face-to-edge mapping: for each face, extract the ordered
        # list of edge references that form its boundary loops.
        face_edge_lists: list[list[int]] = []
        for face_entity in face_entities:
            entity_text = face_entity.get("text", "")
            edge_refs = self._extract_edge_references(entity_text)
            face_edge_lists.append(edge_refs)

        # Build edge -> list of (face_idx, position_in_face_loop)
        # Each topological edge should appear in exactly 2 faces for a manifold solid.
        edge_to_faces: dict[int, list[tuple[int, int]]] = {}
        for face_idx, edge_refs in enumerate(face_edge_lists):
            for pos, edge_ref in enumerate(edge_refs):
                if edge_ref not in edge_to_faces:
                    edge_to_faces[edge_ref] = []
                edge_to_faces[edge_ref].append((face_idx, pos))

        # Create coedges: each (face_idx, edge_ref) pair is a coedge
        coedge_list: list[dict] = []
        coedge_key_to_idx: dict[tuple[int, int], int] = {}

        for face_idx, edge_refs in enumerate(face_edge_lists):
            for pos, edge_ref in enumerate(edge_refs):
                coedge_idx = len(coedge_list)
                coedge_key_to_idx[(face_idx, edge_ref)] = coedge_idx
                coedge_list.append({
                    "face_idx": face_idx,
                    "edge_ref": edge_ref,
                    "pos_in_loop": pos,
                    "loop_size": len(edge_refs),
                })

        num_coedges = len(coedge_list)

        if num_coedges == 0:
            _log.warning("No coedges found in document")
            return CoedgeData(
                features=torch.zeros((0, 12)),
                next_indices=torch.zeros(0, dtype=torch.long),
                prev_indices=torch.zeros(0, dtype=torch.long),
                mate_indices=torch.zeros(0, dtype=torch.long),
                face_indices=torch.zeros(0, dtype=torch.long),
            )

        # Build next/prev/mate indices
        next_indices = torch.zeros(num_coedges, dtype=torch.long)
        prev_indices = torch.zeros(num_coedges, dtype=torch.long)
        mate_indices = torch.zeros(num_coedges, dtype=torch.long)
        face_indices = torch.zeros(num_coedges, dtype=torch.long)

        for ci, coedge in enumerate(coedge_list):
            face_idx = coedge["face_idx"]
            edge_ref = coedge["edge_ref"]
            pos = coedge["pos_in_loop"]
            loop_size = coedge["loop_size"]
            edge_refs_for_face = face_edge_lists[face_idx]

            face_indices[ci] = face_idx

            # Next coedge in the same face loop (cyclic)
            next_pos = (pos + 1) % loop_size
            next_edge_ref = edge_refs_for_face[next_pos]
            next_key = (face_idx, next_edge_ref)
            next_indices[ci] = coedge_key_to_idx.get(next_key, ci)

            # Prev coedge in the same face loop (cyclic)
            prev_pos = (pos - 1) % loop_size
            prev_edge_ref = edge_refs_for_face[prev_pos]
            prev_key = (face_idx, prev_edge_ref)
            prev_indices[ci] = coedge_key_to_idx.get(prev_key, ci)

            # Mate coedge: the coedge for the same edge on the adjacent face
            mate_idx = ci  # Default: self (for boundary edges)
            faces_sharing_edge = edge_to_faces.get(edge_ref, [])
            for other_face_idx, _other_pos in faces_sharing_edge:
                if other_face_idx != face_idx:
                    other_key = (other_face_idx, edge_ref)
                    if other_key in coedge_key_to_idx:
                        mate_idx = coedge_key_to_idx[other_key]
                        break
            mate_indices[ci] = mate_idx

        # Build coedge features (12 dims)
        # Edge curve types for one-hot encoding
        curve_types = ["LINE", "CIRCLE", "ELLIPSE", "B_SPLINE", "PARABOLA", "OTHER"]

        features = torch.zeros(num_coedges, 12, dtype=torch.float32)

        # Try to extract geometric info from document
        shape = self._load_shape_from_document(doc)

        # Extract edges and faces from OCC shape for geometric calculations
        occ_edges = self._extract_edges_from_shape(shape) if shape else []
        topods_faces = self._extract_faces_from_shape(shape) if shape else []

        # Build edge_ref to OCC edge mapping if possible
        edge_ref_to_occ = {}
        if occ_edges and len(face_edge_lists) > 0:
            edge_ref_to_occ = self._match_edge_refs_to_occ_edges(
                face_edge_lists, occ_edges, face_entities
            )

        # Build face_idx to OCC face mapping
        face_idx_to_occ = {}
        if topods_faces and len(topods_faces) == num_faces:
            for idx, occ_face in enumerate(topods_faces):
                face_idx_to_occ[idx] = occ_face

        has_occ_geometry = len(edge_ref_to_occ) > 0 or len(face_idx_to_occ) > 0

        for ci, coedge in enumerate(coedge_list):
            edge_ref = coedge["edge_ref"]
            face_idx = coedge["face_idx"]

            # Try to determine edge curve type from entity text
            curve_type_idx = 5  # Default: OTHER
            try:
                # Use feature extractor to get edge info
                entity_info = self.feature_extractor.extract_entity_info(
                    f"#{edge_ref}"
                )
                edge_type = entity_info.get("curve_type", "OTHER").upper()

                for ct_idx, ct_name in enumerate(curve_types):
                    if ct_name in edge_type:
                        curve_type_idx = ct_idx
                        break
            except Exception:
                pass

            # One-hot curve type (dims 0-5)
            features[ci, curve_type_idx] = 1.0

            # Edge length (dim 6) - compute from OCC or estimate from entity
            edge_length = self._compute_edge_length(edge_ref, edge_ref_to_occ)
            features[ci, 6] = edge_length

            # Tangent vector at midpoint (dims 7-9) - compute from OCC or estimate
            tangent = self._compute_edge_tangent(edge_ref, edge_ref_to_occ, curve_type_idx)
            features[ci, 7] = tangent[0]
            features[ci, 8] = tangent[1]
            features[ci, 9] = tangent[2]

            # Curvature at midpoint (dim 10) - compute from OCC or curve type
            curvature = self._compute_edge_curvature(
                edge_ref, edge_ref_to_occ, curve_type_idx, entity_info if 'entity_info' in dir() else None
            )
            features[ci, 10] = curvature

            # Convexity flag (dim 11): compute from face normals and edge tangent
            mate_ci = int(mate_indices[ci].item())
            convexity = self._compute_edge_convexity(
                ci, mate_ci, face_idx, face_indices, face_idx_to_occ, tangent
            )
            features[ci, 11] = convexity

        if has_occ_geometry:
            _log.info(
                f"Built coedge graph with REAL geometric features: {num_coedges} coedges, "
                f"{num_faces} faces, feature_dim=12"
            )
        else:
            _log.warning(
                f"Built coedge graph with estimated features (no OCC): {num_coedges} coedges, "
                f"{num_faces} faces, feature_dim=12"
            )

        return CoedgeData(
            features=features,
            next_indices=next_indices,
            prev_indices=prev_indices,
            mate_indices=mate_indices,
            face_indices=face_indices,
        )

    def _build_coedge_graph_from_occ(self, shape) -> "CoedgeData | None":
        """Build coedge graph directly from OCC shape using CoedgeExtractor.

        This is more accurate than STEP text parsing because it uses
        BRepTools_WireExplorer for proper edge ordering within face loops.

        Args:
            shape: OCC TopoDS_Shape

        Returns:
            CoedgeData object or None if extraction fails
        """
        from cadling.models.segmentation.architectures.brep_net import CoedgeData

        try:
            from cadling.lib.topology.coedge_extractor import CoedgeExtractor

            extractor = CoedgeExtractor()
            coedge_data = extractor.extract_coedge_data(shape, feature_dim=12)

            if coedge_data["features"].shape[0] == 0:
                return None

            # Convert numpy arrays to torch tensors
            return CoedgeData(
                features=torch.from_numpy(coedge_data["features"]).float(),
                next_indices=torch.from_numpy(coedge_data["next_indices"]).long(),
                prev_indices=torch.from_numpy(coedge_data["prev_indices"]).long(),
                mate_indices=torch.from_numpy(coedge_data["mate_indices"]).long(),
                face_indices=torch.from_numpy(coedge_data["face_indices"]).long(),
            )

        except ImportError:
            _log.debug("CoedgeExtractor not available")
            return None
        except Exception as e:
            _log.debug(f"OCC coedge extraction failed: {e}")
            return None

    def _extract_edges_from_shape(self, shape) -> list:
        """Extract all TopoDS_Edge objects from shape.

        Args:
            shape: TopoDS_Shape

        Returns:
            List of TopoDS_Edge objects
        """
        if shape is None:
            return []

        try:
            from OCC.Core.TopAbs import TopAbs_EDGE
            from OCC.Core.TopExp import TopExp_Explorer

            edges = []
            explorer = TopExp_Explorer(shape, TopAbs_EDGE)  # type: ignore[arg-type]

            while explorer.More():
                edge = explorer.Current()
                edges.append(edge)
                explorer.Next()

            _log.debug(f"Extracted {len(edges)} TopoDS_Edge objects from shape")
            return edges

        except Exception as e:
            _log.warning(f"Failed to extract edges from shape: {e}")
            return []

    def _match_edge_refs_to_occ_edges(
        self,
        face_edge_lists: list[list[int]],
        occ_edges: list,
        face_entities: list[dict],
    ) -> dict[int, Any]:
        """Match STEP edge reference IDs to OCC TopoDS_Edge objects.

        Uses order-based matching within faces as a heuristic.

        Args:
            face_edge_lists: List of edge reference lists per face
            occ_edges: List of TopoDS_Edge objects
            face_entities: Face entity dictionaries

        Returns:
            Dictionary mapping edge_ref to TopoDS_Edge
        """
        edge_ref_to_occ = {}

        # Simple heuristic: collect all unique edge refs and match by order
        all_edge_refs = []
        seen_refs = set()
        for edge_list in face_edge_lists:
            for ref in edge_list:
                if ref not in seen_refs:
                    all_edge_refs.append(ref)
                    seen_refs.add(ref)

        # Match by order (assumes similar ordering)
        min_len = min(len(all_edge_refs), len(occ_edges))
        for i in range(min_len):
            edge_ref_to_occ[all_edge_refs[i]] = occ_edges[i]

        if len(all_edge_refs) != len(occ_edges):
            # Use face_entities for diagnostic context
            face_count = len(face_entities)
            avg_edges_per_face = len(all_edge_refs) / face_count if face_count > 0 else 0
            _log.debug(
                f"Edge ref count ({len(all_edge_refs)}) != OCC edge count ({len(occ_edges)}). "
                f"Matched {min_len} edges across {face_count} faces "
                f"(avg {avg_edges_per_face:.1f} edges/face)."
            )

        return edge_ref_to_occ

    def _compute_edge_length(
        self,
        edge_ref: int,
        edge_ref_to_occ: dict[int, Any],
    ) -> float:
        """Compute edge length from OCC or estimate.

        Args:
            edge_ref: STEP edge reference ID
            edge_ref_to_occ: Mapping from edge refs to OCC edges

        Returns:
            Edge length (always returns a value, never None)
        """
        # Strategy 1: Use OCC if available
        if edge_ref in edge_ref_to_occ:
            try:
                from OCC.Core.BRepGProp import brepgprop
                from OCC.Core.GProp import GProp_GProps

                occ_edge = edge_ref_to_occ[edge_ref]
                props = GProp_GProps()
                brepgprop.LinearProperties(occ_edge, props)
                length = props.Mass()

                if length > 0:
                    return float(length)

            except Exception as e:
                _log.debug(f"OCC edge length computation failed: {e}")

        # Strategy 2: Try to get from feature extractor
        try:
            entity_info = self.feature_extractor.extract_entity_info(f"#{edge_ref}")
            length = entity_info.get("length", 0)
            if length > 0:
                return float(length)
        except Exception:
            pass

        # Strategy 3: Default value
        return 1.0

    def _compute_edge_tangent(
        self,
        edge_ref: int,
        edge_ref_to_occ: dict[int, Any],
        curve_type_idx: int,
    ) -> list[float]:
        """Compute edge tangent vector at midpoint.

        Args:
            edge_ref: STEP edge reference ID
            edge_ref_to_occ: Mapping from edge refs to OCC edges
            curve_type_idx: Curve type index (0=LINE, 1=CIRCLE, etc.)

        Returns:
            Tangent vector [x, y, z] (normalized, always returns a value)
        """
        # Strategy 1: Use OCC BRepAdaptor_Curve
        if edge_ref in edge_ref_to_occ:
            try:
                from OCC.Core.BRepAdaptor import BRepAdaptor_Curve
                from OCC.Core.gp import gp_Pnt, gp_Vec

                occ_edge = edge_ref_to_occ[edge_ref]
                curve_adaptor = BRepAdaptor_Curve(occ_edge)

                # Get parameter at midpoint
                u_start = curve_adaptor.FirstParameter()
                u_end = curve_adaptor.LastParameter()
                u_mid = (u_start + u_end) / 2.0

                # Get tangent via D1 (point and first derivative)
                point = gp_Pnt()
                tangent = gp_Vec()
                curve_adaptor.D1(u_mid, point, tangent)

                # Normalize tangent
                mag = tangent.Magnitude()
                if mag > 1e-10:
                    tangent.Normalize()
                    return [tangent.X(), tangent.Y(), tangent.Z()]

            except Exception as e:
                _log.debug(f"OCC tangent computation failed: {e}")

        # Strategy 2: Estimate from edge endpoints (for lines)
        if edge_ref in edge_ref_to_occ:
            try:
                from OCC.Core.BRepAdaptor import BRepAdaptor_Curve
                from OCC.Core.gp import gp_Vec

                occ_edge = edge_ref_to_occ[edge_ref]
                curve_adaptor = BRepAdaptor_Curve(occ_edge)

                u_start = curve_adaptor.FirstParameter()
                u_end = curve_adaptor.LastParameter()

                p_start = curve_adaptor.Value(u_start)
                p_end = curve_adaptor.Value(u_end)

                direction = gp_Vec(p_start, p_end)
                mag = direction.Magnitude()
                if mag > 1e-10:
                    direction.Normalize()
                    return [direction.X(), direction.Y(), direction.Z()]

            except Exception as e:
                _log.debug(f"Endpoint tangent estimation failed: {e}")

        # Strategy 3: Default based on curve type
        # For circles, default tangent is perpendicular to axis (assume XY plane)
        if curve_type_idx == 1:  # CIRCLE
            return [1.0, 0.0, 0.0]  # Tangent in X direction

        # Default tangent along Z axis
        return [0.0, 0.0, 1.0]

    def _compute_edge_curvature(
        self,
        edge_ref: int,
        edge_ref_to_occ: dict[int, Any],
        curve_type_idx: int,
        entity_info: dict | None = None,
    ) -> float:
        """Compute edge curvature at midpoint.

        Args:
            edge_ref: STEP edge reference ID
            edge_ref_to_occ: Mapping from edge refs to OCC edges
            curve_type_idx: Curve type index (0=LINE, 1=CIRCLE, etc.)
            entity_info: Optional entity info with radius

        Returns:
            Curvature value (always returns a value)
        """
        # Strategy 1: Use OCC BRepLProp_CLProps
        if edge_ref in edge_ref_to_occ:
            try:
                from OCC.Core.BRepAdaptor import BRepAdaptor_Curve
                from OCC.Core.BRepLProp import BRepLProp_CLProps

                occ_edge = edge_ref_to_occ[edge_ref]
                curve_adaptor = BRepAdaptor_Curve(occ_edge)

                u_start = curve_adaptor.FirstParameter()
                u_end = curve_adaptor.LastParameter()
                u_mid = (u_start + u_end) / 2.0

                # Compute curvature using local properties
                props = BRepLProp_CLProps(curve_adaptor, u_mid, 2, 1e-6)

                if props.IsTangentDefined():
                    curvature = props.Curvature()
                    if curvature >= 0 and curvature < 1e10:
                        return float(curvature)

            except Exception as e:
                _log.debug(f"OCC curvature computation failed: {e}")

        # Strategy 2: Compute from curve type
        # LINE: curvature = 0
        if curve_type_idx == 0:
            return 0.0

        # CIRCLE: curvature = 1/radius
        if curve_type_idx == 1:
            radius = None

            # Try to get radius from entity info
            if entity_info and "radius" in entity_info:
                radius = entity_info.get("radius")
            else:
                # Try feature extractor
                try:
                    info = self.feature_extractor.extract_entity_info(f"#{edge_ref}")
                    radius = info.get("radius")
                except Exception:
                    pass

            if radius and radius > 1e-10:
                return 1.0 / radius

        # ELLIPSE: approximate curvature
        if curve_type_idx == 2:
            # Ellipse curvature varies, use average approximation
            return 0.1

        # B_SPLINE, PARABOLA: estimate from control points
        if curve_type_idx in (3, 4):
            return 0.01  # Small curvature for splines

        # Default: zero curvature
        return 0.0

    def _compute_edge_convexity(
        self,
        coedge_idx: int,
        mate_idx: int,
        face_idx: int,
        face_indices: torch.Tensor,
        face_idx_to_occ: dict[int, Any],
        tangent: list[float],
    ) -> float:
        """Compute edge convexity from adjacent face normals.

        Convexity is determined by the dihedral angle between adjacent faces:
        - Convex (1.0): Faces form an outward corner
        - Concave (0.0): Faces form an inward corner
        - Tangent (0.5): Faces are nearly coplanar

        Args:
            coedge_idx: Current coedge index
            mate_idx: Mate coedge index
            face_idx: Face index of current coedge
            face_indices: Tensor mapping coedge to face index
            face_idx_to_occ: Mapping from face index to OCC face
            tangent: Edge tangent vector

        Returns:
            Convexity value in [0, 1] (0=concave, 0.5=tangent, 1=convex)
        """
        # If no mate (boundary edge), return unknown
        if coedge_idx == mate_idx:
            return 0.5

        # Get mate face index
        mate_face_idx = int(face_indices[mate_idx].item())

        # Strategy 1: Compute from OCC face normals
        if face_idx in face_idx_to_occ and mate_face_idx in face_idx_to_occ:
            try:
                from OCC.Core.gp import gp_Vec

                face1 = face_idx_to_occ[face_idx]
                face2 = face_idx_to_occ[mate_face_idx]

                # Get normals at face centers
                n1 = self._get_face_normal_at_center(face1)
                n2 = self._get_face_normal_at_center(face2)

                if n1 is not None and n2 is not None:
                    # Compute cross product of normals
                    cross = gp_Vec(
                        n1[1] * n2[2] - n1[2] * n2[1],
                        n1[2] * n2[0] - n1[0] * n2[2],
                        n1[0] * n2[1] - n1[1] * n2[0],
                    )

                    # Dot with edge tangent to determine convexity
                    dot = cross.X() * tangent[0] + cross.Y() * tangent[1] + cross.Z() * tangent[2]

                    # Map to [0, 1]: positive = convex, negative = concave
                    if abs(dot) < 0.1:
                        return 0.5  # Tangent/coplanar
                    elif dot > 0:
                        return 1.0  # Convex
                    else:
                        return 0.0  # Concave

            except Exception as e:
                _log.debug(f"OCC convexity computation failed: {e}")

        # Strategy 2: Use dihedral angle from face features if available
        # (would require face normals to be pre-computed)

        # Strategy 3: Default to unknown
        return 0.5

    def _get_face_normal_at_center(self, occ_face) -> list[float] | None:
        """Get face normal at center point.

        Args:
            occ_face: OCC TopoDS_Face

        Returns:
            Normal vector [x, y, z] or None
        """
        try:
            from OCC.Core.BRepAdaptor import BRepAdaptor_Surface
            from OCC.Core.gp import gp_Pnt, gp_Vec

            adaptor = BRepAdaptor_Surface(occ_face)

            u_mid = (adaptor.FirstUParameter() + adaptor.LastUParameter()) / 2
            v_mid = (adaptor.FirstVParameter() + adaptor.LastVParameter()) / 2

            # Get normal via D1
            point = gp_Pnt()
            d1u = gp_Vec()
            d1v = gp_Vec()
            adaptor.D1(u_mid, v_mid, point, d1u, d1v)

            normal = d1u.Crossed(d1v)
            mag = normal.Magnitude()

            if mag > 1e-10:
                normal.Normalize()
                return [normal.X(), normal.Y(), normal.Z()]

        except Exception as e:
            _log.debug(f"Face normal computation failed: {e}")

        return None
