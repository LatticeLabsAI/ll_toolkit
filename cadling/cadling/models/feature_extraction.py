"""Feature extraction enrichment model for CAD documents.

This module provides an enrichment model that auto-populates geometric features
on CADItems during the pipeline enrichment stage. Features are extracted via
the OCC wrapper and cached for efficiency.

Example:
    from cadling.models.feature_extraction import FeatureExtractionModel

    model = FeatureExtractionModel(
        extract_face_features=True,
        extract_uv_grids=False,  # expensive, opt-in
        use_cache=True,
    )

    # Add to pipeline
    options = PipelineOptions(enrichment_models=[model])
    pipeline = SimpleCADPipeline(options)

    # Or apply directly
    model(doc, doc.items)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, TYPE_CHECKING

import numpy as np

from cadling.models.base_model import EnrichmentModel

if TYPE_CHECKING:
    from cadling.datamodel.base_models import CADItem, CADlingDocument

_log = logging.getLogger(__name__)


class FeatureExtractionModel(EnrichmentModel):
    """Auto-populate geometric features on CADItems during enrichment.

    This enrichment model extracts geometric features from the OCC shape
    attached to the document and adds them to item properties. Features
    are cached to avoid recomputation on repeated loads.

    Features extracted:
        - Shape-level: volume, surface_area, bbox, num_faces, num_edges
        - Face-level: surface_type, area, normal, centroid, curvature
        - Edge-level: curve_type, length, tangent, curvature
        - Adjacency: face-to-face adjacency graph

    Attributes:
        extract_face_features: Whether to extract per-face features
        extract_edge_features: Whether to extract per-edge features
        extract_uv_grids: Whether to extract UV grids (expensive)
        extract_adjacency: Whether to extract adjacency graph
        use_cache: Whether to use feature cache
        uv_grid_size: Size of UV grid if extracting (default 10x10)
    """

    def __init__(
        self,
        extract_face_features: bool = True,
        extract_edge_features: bool = True,
        extract_uv_grids: bool = False,
        extract_adjacency: bool = True,
        use_cache: bool = True,
        uv_grid_size: int = 10,
    ):
        """Initialize feature extraction model.

        Args:
            extract_face_features: Extract per-face geometric features
            extract_edge_features: Extract per-edge geometric features
            extract_uv_grids: Extract UV grids for faces (expensive)
            extract_adjacency: Extract face adjacency graph
            use_cache: Use persistent feature cache
            uv_grid_size: UV grid size (default 10x10)
        """
        super().__init__()

        self.extract_face_features = extract_face_features
        self.extract_edge_features = extract_edge_features
        self.extract_uv_grids = extract_uv_grids
        self.extract_adjacency = extract_adjacency
        self.use_cache = use_cache
        self.uv_grid_size = uv_grid_size

        self._cache = None
        if use_cache:
            from cadling.lib.cache.feature_cache import FeatureCache
            self._cache = FeatureCache()

        _log.info(
            "Initialized FeatureExtractionModel: "
            f"faces={extract_face_features}, edges={extract_edge_features}, "
            f"uv_grids={extract_uv_grids}, cache={use_cache}"
        )

    def __call__(
        self,
        doc: "CADlingDocument",
        item_batch: List["CADItem"],
    ) -> None:
        """Enrich items with geometric features.

        Extracts features from the OCC shape attached to the document and
        adds them to item properties. Uses cache if available.

        Args:
            doc: The CADlingDocument being enriched
            item_batch: List of CADItem objects to process
        """
        # Get OCC shape from document
        shape = self._get_occ_shape(doc)
        if shape is None:
            _log.debug("No OCC shape available in document, skipping feature extraction")
            return

        # Compute cache key
        cache_key = None
        cached_features = None

        if self._cache is not None:
            cache_key = self._compute_cache_key(doc)
            cached_features = self._cache.get(cache_key)

            if cached_features is not None:
                _log.debug("Using cached features")
                self._apply_features_to_items(cached_features, doc, item_batch)
                return

        # Extract features via OCC wrapper
        features = self._extract_features(shape)

        # Apply to document and items
        self._apply_features_to_items(features, doc, item_batch)

        # Cache for future use
        if self._cache is not None and cache_key is not None:
            self._cache.set(cache_key, features)
            _log.debug("Cached extracted features")

    def _get_occ_shape(self, doc: "CADlingDocument") -> Optional[Any]:
        """Get OCC shape from document.

        Tries multiple strategies:
        1. doc._occ_shape
        2. doc._backend._occ_shape
        3. doc._backend.get_shape()

        Args:
            doc: CADlingDocument

        Returns:
            OCC TopoDS_Shape or None
        """
        # Strategy 1: Direct on document
        shape = getattr(doc, "_occ_shape", None)
        if shape is not None:
            return shape

        # Strategy 2: Via backend
        backend = getattr(doc, "_backend", None)
        if backend is not None:
            # Try _occ_shape attribute
            shape = getattr(backend, "_occ_shape", None)
            if shape is not None:
                return shape

            # Try get_shape method
            if hasattr(backend, "get_shape") and callable(backend.get_shape):
                try:
                    return backend.get_shape()
                except Exception:
                    pass

            # Try load_shape method
            if hasattr(backend, "load_shape") and callable(backend.load_shape):
                try:
                    return backend.load_shape()
                except Exception:
                    pass

        return None

    def _compute_cache_key(self, doc: "CADlingDocument") -> str:
        """Compute cache key for document.

        Args:
            doc: CADlingDocument

        Returns:
            Cache key string
        """
        # Get source file path
        file_path = None
        input_doc = getattr(doc, "input", None)
        if input_doc is not None:
            file_path = getattr(input_doc, "file", None)

        if file_path is None:
            # Use document name as fallback
            file_path = Path(doc.name) if doc.name else Path("unknown")

        # Build params dict
        params = {
            "extract_face_features": self.extract_face_features,
            "extract_edge_features": self.extract_edge_features,
            "extract_uv_grids": self.extract_uv_grids,
            "extract_adjacency": self.extract_adjacency,
            "uv_grid_size": self.uv_grid_size,
        }

        return self._cache.compute_key(file_path, params)

    def _extract_features(self, shape: Any) -> Dict[str, Any]:
        """Extract features from OCC shape.

        Args:
            shape: OCC TopoDS_Shape

        Returns:
            Dictionary with extracted features
        """
        from cadling.lib.occ_wrapper import OCCShape, HAS_OCC

        if not HAS_OCC:
            _log.warning("OCC not available, returning empty features")
            return {}

        wrapper = OCCShape(shape)
        features: Dict[str, Any] = {}

        # Shape-level features
        features["num_faces"] = wrapper.num_faces()
        features["num_edges"] = wrapper.num_edges()
        features["num_vertices"] = wrapper.num_vertices()
        features["volume"] = wrapper.volume()
        features["surface_area"] = wrapper.surface_area()

        bbox_min, bbox_max = wrapper.bbox()
        features["bbox_min"] = bbox_min
        features["bbox_max"] = bbox_max
        features["bbox_dimensions"] = [bbox_max[i] - bbox_min[i] for i in range(3)]

        # Face features
        if self.extract_face_features:
            face_features = []
            for face in wrapper.faces():
                face_features.append(face.extract_features())
            features["faces"] = face_features

            # Also extract as numpy matrix for GNN
            features["face_feature_matrix"] = wrapper.extract_face_feature_matrix()

        # Edge features
        if self.extract_edge_features:
            edge_features = []
            for edge in wrapper.edges():
                edge_features.append(edge.extract_features())
            features["edges"] = edge_features

        # UV grids (expensive)
        if self.extract_uv_grids:
            uv_grids = []
            for face in wrapper.faces():
                grid = face.uv_grid(self.uv_grid_size, self.uv_grid_size)
                uv_grids.append(grid)
            features["uv_grids"] = uv_grids

        # Adjacency graph
        if self.extract_adjacency:
            features["adjacency"] = wrapper.face_adjacency_graph()
            features["edge_index"] = wrapper.extract_edge_index()

        _log.info(
            f"Extracted features: {features['num_faces']} faces, "
            f"{features['num_edges']} edges, volume={features['volume']:.2f}"
        )

        return features

    def _apply_features_to_items(
        self,
        features: Dict[str, Any],
        doc: "CADlingDocument",
        item_batch: List["CADItem"],
    ) -> None:
        """Apply extracted features to document and items.

        Args:
            features: Extracted features dictionary
            doc: CADlingDocument to update
            item_batch: Items to update
        """
        # Handle cached format (features nested under "features" key)
        if "features" in features and isinstance(features.get("features"), dict):
            features = features["features"]

        # Add shape-level features to document properties
        doc_props = getattr(doc, "properties", None)
        if doc_props is None:
            doc.properties = {}
            doc_props = doc.properties

        doc_props["geometry_features"] = {
            "num_faces": features.get("num_faces", 0),
            "num_edges": features.get("num_edges", 0),
            "num_vertices": features.get("num_vertices", 0),
            "volume": features.get("volume", 0.0),
            "surface_area": features.get("surface_area", 0.0),
            "bbox_min": features.get("bbox_min", [0.0, 0.0, 0.0]),
            "bbox_max": features.get("bbox_max", [1.0, 1.0, 1.0]),
            "bbox_dimensions": features.get("bbox_dimensions", [1.0, 1.0, 1.0]),
        }

        if self.extract_adjacency:
            doc_props["adjacency"] = features.get("adjacency", {})

        # Store feature matrices on document for GNN use
        if "face_feature_matrix" in features:
            setattr(doc, "_face_feature_matrix", features["face_feature_matrix"])

        if "edge_index" in features:
            setattr(doc, "_edge_index", features["edge_index"])

        if "uv_grids" in features:
            setattr(doc, "_uv_grids", features["uv_grids"])

        # Add face features to relevant items
        face_features = features.get("faces", [])
        if face_features:
            for item in item_batch:
                # Check if item is a face entity
                entity_type = getattr(item, "entity_type", None)
                if entity_type is None:
                    label = getattr(item, "label", None)
                    if label is not None:
                        entity_type = getattr(label, "entity_type", None)

                if entity_type is not None and "FACE" in str(entity_type).upper():
                    # Try to match by entity_id or index
                    entity_id = getattr(item, "entity_id", None)
                    item_idx = getattr(item, "_index", None)

                    if item_idx is not None and item_idx < len(face_features):
                        item.properties["face_geometry"] = face_features[item_idx]
                    elif entity_id is not None:
                        # Store all face features and let downstream match
                        item.properties["available_face_features"] = face_features

        _log.debug(f"Applied features to {len(item_batch)} items")

    def supports_batch_processing(self) -> bool:
        """This model processes all items together."""
        return True

    def get_batch_size(self) -> int:
        """Process all items at once."""
        return 0  # 0 = all at once

    def requires_gpu(self) -> bool:
        """This model does not require GPU."""
        return False

    def get_model_info(self) -> Dict[str, str]:
        """Get information about this model."""
        info = super().get_model_info()
        info.update({
            "extract_face_features": str(self.extract_face_features),
            "extract_edge_features": str(self.extract_edge_features),
            "extract_uv_grids": str(self.extract_uv_grids),
            "extract_adjacency": str(self.extract_adjacency),
            "use_cache": str(self.use_cache),
        })
        return info
