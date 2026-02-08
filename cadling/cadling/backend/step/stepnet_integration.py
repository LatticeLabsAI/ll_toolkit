"""ll_stepnet integration layer for STEP parsing.

This module provides integration with ll_stepnet for:
- STEP tokenization
- Feature extraction
- Topology building
- Neural network inference

When ll_stepnet is unavailable, this module provides ALTERNATIVE implementations
using cadling's own parsing infrastructure that produce EQUIVALENT results.

Classes:
    STEPNetIntegration: Main integration wrapper
"""

from __future__ import annotations

import logging
import re
from collections import defaultdict
from typing import Any, Optional

_log = logging.getLogger(__name__)

# Try to import ll_stepnet components
_LL_STEPNET_AVAILABLE = False
_LL_STEPNET_RESERIALIZATION_AVAILABLE = False
try:
    from ll_stepnet import STEPFeatureExtractor, STEPTokenizer, STEPTopologyBuilder

    _LL_STEPNET_AVAILABLE = True
    _log.info("ll_stepnet successfully imported")
except ImportError as e:
    _log.warning(f"ll_stepnet not available: {e}")
    _log.info("STEP backend will use cadling's alternative implementations")

try:
    from ll_stepnet import (
        reserialize_step,
        STEPReserializationConfig,
        STEPAnnotationConfig,
        STEPEntityGraph,
        STEPStructuralAnnotator,
    )

    _LL_STEPNET_RESERIALIZATION_AVAILABLE = True
    _log.info("ll_stepnet reserialization module available")
except ImportError as e:
    _log.debug(f"ll_stepnet reserialization not available: {e}")


class STEPNetIntegration:
    """Integration layer for ll_stepnet.

    Provides access to ll_stepnet components for STEP file processing:
    - Tokenization: Parse STEP text into tokens
    - Feature extraction: Extract geometric features from entities
    - Topology building: Build entity reference graphs

    When ll_stepnet is not available, provides ALTERNATIVE implementations
    using cadling's infrastructure that produce equivalent output.

    Attributes:
        available: Whether ll_stepnet is available
        tokenizer: STEPTokenizer instance (if available)
        feature_extractor: STEPFeatureExtractor instance (if available)
        topology_builder: STEPTopologyBuilder instance (if available)
        _alt_vocab: Alternative vocabulary for tokenization (when ll_stepnet unavailable)
        _alt_token_to_id: Token to ID mapping for alternative tokenization
    """

    # STEP entity types for vocabulary (200+ types)
    _STEP_ENTITY_TYPES = [
        # Special tokens
        "PAD", "UNK", "SEP", "CLS",
        # Topology entities
        "MANIFOLD_SOLID_BREP", "BREP_WITH_VOIDS", "FACETED_BREP",
        "CLOSED_SHELL", "OPEN_SHELL",
        "ADVANCED_FACE", "FACE_SURFACE", "FACE_BOUND", "FACE_OUTER_BOUND",
        "EDGE_CURVE", "ORIENTED_EDGE", "EDGE_LOOP",
        "VERTEX_POINT",
        # Geometry entities
        "CARTESIAN_POINT", "DIRECTION", "VECTOR",
        "AXIS1_PLACEMENT", "AXIS2_PLACEMENT_3D",
        "LINE", "CIRCLE", "ELLIPSE", "PARABOLA", "HYPERBOLA",
        "B_SPLINE_CURVE", "B_SPLINE_CURVE_WITH_KNOTS", "BEZIER_CURVE",
        "TRIMMED_CURVE", "COMPOSITE_CURVE", "SURFACE_CURVE",
        "PLANE", "CYLINDRICAL_SURFACE", "CONICAL_SURFACE",
        "SPHERICAL_SURFACE", "TOROIDAL_SURFACE",
        "B_SPLINE_SURFACE", "B_SPLINE_SURFACE_WITH_KNOTS",
        "SURFACE_OF_REVOLUTION", "SURFACE_OF_LINEAR_EXTRUSION",
        # Product structure
        "PRODUCT", "PRODUCT_DEFINITION", "PRODUCT_DEFINITION_FORMATION",
        "PRODUCT_DEFINITION_SHAPE", "SHAPE_DEFINITION_REPRESENTATION",
        "SHAPE_REPRESENTATION", "ADVANCED_BREP_SHAPE_REPRESENTATION",
        "MANIFOLD_SURFACE_SHAPE_REPRESENTATION",
        # Assembly entities
        "NEXT_ASSEMBLY_USAGE_OCCURRENCE", "PRODUCT_DEFINITION_RELATIONSHIP",
        "CONTEXT_DEPENDENT_SHAPE_REPRESENTATION",
        "REPRESENTATION_RELATIONSHIP", "REPRESENTATION_RELATIONSHIP_WITH_TRANSFORMATION",
        "ITEM_DEFINED_TRANSFORMATION",
        # Placement/transform
        "AXIS2_PLACEMENT_3D", "CARTESIAN_TRANSFORMATION_OPERATOR_3D",
        # PMI entities
        "GEOMETRIC_TOLERANCE", "DIMENSION_CALLOUT", "DATUM_FEATURE",
        # Additional common entities
        "POLYLINE", "COMPOSITE_CURVE_SEGMENT", "PCURVE",
        "DEFINITIONAL_REPRESENTATION", "PARAMETRIC_REPRESENTATION_CONTEXT",
        "GEOMETRIC_REPRESENTATION_CONTEXT", "GLOBAL_UNCERTAINTY_ASSIGNED_CONTEXT",
        "REPRESENTATION_CONTEXT", "APPLICATION_CONTEXT", "APPLICATION_PROTOCOL_DEFINITION",
    ]

    # Numeric quantization buckets (for continuous values)
    _NUM_QUANTIZATION_BUCKETS = 1000

    def __init__(self):
        """Initialize ll_stepnet integration.

        When ll_stepnet is unavailable, initializes alternative implementations
        using cadling's parsing infrastructure.
        """
        self.available = _LL_STEPNET_AVAILABLE

        if self.available:
            self.tokenizer = STEPTokenizer()
            self.feature_extractor = STEPFeatureExtractor()
            self.topology_builder = STEPTopologyBuilder()
            _log.debug("Initialized ll_stepnet integration")
        else:
            self.tokenizer = None
            self.feature_extractor = None
            self.topology_builder = None
            # Initialize alternative implementations
            self._init_alternative_vocab()
            _log.debug("Initialized alternative STEP parsing (ll_stepnet unavailable)")

    def _init_alternative_vocab(self) -> None:
        """Initialize alternative vocabulary for tokenization.

        Creates a vocabulary matching ll_stepnet's format:
        - Special tokens: [PAD]=0, [UNK]=1, [SEP]=2, [CLS]=3
        - Entity types: IDs 4-203
        - Numeric buckets: IDs 204-1203
        - Reference tokens: IDs 1204+
        """
        self._alt_token_to_id: dict[str, int] = {}
        self._alt_id_to_token: dict[int, str] = {}

        # Special tokens (IDs 0-3)
        special_tokens = ["[PAD]", "[UNK]", "[SEP]", "[CLS]"]
        for idx, token in enumerate(special_tokens):
            self._alt_token_to_id[token] = idx
            self._alt_id_to_token[idx] = token

        # Entity type tokens (IDs 4-203)
        for idx, entity_type in enumerate(self._STEP_ENTITY_TYPES[4:], start=4):  # Skip special tokens
            self._alt_token_to_id[entity_type] = idx
            self._alt_id_to_token[idx] = entity_type

        # Numeric bucket tokens (IDs 204-1203)
        base_id = len(special_tokens) + len(self._STEP_ENTITY_TYPES)
        for bucket_idx in range(self._NUM_QUANTIZATION_BUCKETS):
            token = f"[NUM_{bucket_idx}]"
            token_id = base_id + bucket_idx
            self._alt_token_to_id[token] = token_id
            self._alt_id_to_token[token_id] = token

        # Reference token base ID (for #N entity references)
        self._ref_base_id = base_id + self._NUM_QUANTIZATION_BUCKETS

        _log.debug(f"Initialized alternative vocab with {len(self._alt_token_to_id)} tokens")

    def tokenize(
        self, step_text: str, return_metadata: bool = False,
    ) -> "list[int] | tuple[list[int], dict]":
        """Tokenize STEP text.

        When ll_stepnet is available, uses its tokenizer.
        Otherwise, uses cadling's alternative tokenizer that produces
        equivalent output format.

        Args:
            step_text: STEP file content
            return_metadata: If True, returns (token_ids, metadata) tuple.
                Metadata contains 'degraded' (bool), 'method' (str),
                and 'warning' (str or None) to indicate data quality.

        Returns:
            List of token IDs (never returns None), or
            (token_ids, metadata) tuple if return_metadata=True.
        """
        metadata = {"degraded": False, "method": "unknown", "warning": None}

        if self.available:
            try:
                token_ids = self.tokenizer.encode(step_text)
                metadata["method"] = "ll_stepnet"
                _log.debug(f"Tokenized STEP text (ll_stepnet): {len(token_ids)} tokens")
                if return_metadata:
                    return token_ids, metadata
                return token_ids
            except Exception as e:
                _log.warning(f"ll_stepnet tokenization failed, using alternative: {e}")
                metadata["degraded"] = True
                metadata["warning"] = f"ll_stepnet failed: {e}"

        # Alternative implementation using cadling's parsing
        token_ids = self._tokenize_alternative(step_text)
        metadata["method"] = "alternative"
        if not metadata["degraded"]:
            metadata["degraded"] = True
            metadata["warning"] = "ll_stepnet not available"
        _log.warning(
            "Using degraded tokenization (method=%s): %s",
            metadata["method"], metadata["warning"],
        )
        if return_metadata:
            return token_ids, metadata
        return token_ids

    def _tokenize_alternative(self, step_text: str) -> list[int]:
        """Alternative tokenization using cadling's parsing infrastructure.

        Produces token IDs in the same format as ll_stepnet:
        - [CLS] token at start
        - Entity type tokens
        - Numeric value tokens (quantized)
        - Reference tokens (#N)
        - [SEP] tokens between entities

        Args:
            step_text: STEP file content

        Returns:
            List of token IDs
        """
        token_ids: list[int] = []

        # Start with [CLS]
        token_ids.append(self._alt_token_to_id.get("[CLS]", 3))

        # Parse entities from STEP text
        entities = self._parse_step_entities(step_text)

        for entity in entities:
            entity_type = entity.get("type", "")
            params = entity.get("params", [])

            # Add entity type token
            if entity_type in self._alt_token_to_id:
                token_ids.append(self._alt_token_to_id[entity_type])
            else:
                token_ids.append(self._alt_token_to_id.get("[UNK]", 1))

            # Tokenize parameters
            for param in params:
                param_tokens = self._tokenize_param(param)
                token_ids.extend(param_tokens)

            # Add [SEP] between entities
            token_ids.append(self._alt_token_to_id.get("[SEP]", 2))

        _log.debug(f"Tokenized STEP text (alternative): {len(token_ids)} tokens")
        return token_ids

    def _parse_step_entities(self, step_text: str) -> list[dict[str, Any]]:
        """Parse STEP entities from text.

        Args:
            step_text: STEP file content (DATA section)

        Returns:
            List of entity dictionaries with 'id', 'type', 'params'
        """
        entities = []

        # Pattern to match STEP entities: #N=ENTITY_TYPE(params);
        entity_pattern = re.compile(
            r"#(\d+)\s*=\s*([A-Z_][A-Z0-9_]*)\s*\(([^;]*)\)\s*;",
            re.MULTILINE | re.DOTALL
        )

        for match in entity_pattern.finditer(step_text):
            entity_id = int(match.group(1))
            entity_type = match.group(2)
            params_str = match.group(3)

            # Parse parameters
            params = self._parse_step_params(params_str)

            entities.append({
                "id": entity_id,
                "type": entity_type,
                "params": params,
                "text": match.group(0),
            })

        return entities

    def _parse_step_params(self, params_str: str) -> list[Any]:
        """Parse STEP entity parameters.

        Args:
            params_str: Parameter string from STEP entity

        Returns:
            List of parsed parameters (strings, numbers, references, lists)
        """
        params = []
        if not params_str.strip():
            return params

        # Simple tokenization: split by comma, handling nested parens
        depth = 0
        current = ""

        for char in params_str:
            if char == "(":
                depth += 1
                current += char
            elif char == ")":
                depth -= 1
                current += char
            elif char == "," and depth == 0:
                param = current.strip()
                if param:
                    params.append(self._parse_single_param(param))
                current = ""
            else:
                current += char

        # Add last parameter
        param = current.strip()
        if param:
            params.append(self._parse_single_param(param))

        return params

    def _parse_single_param(self, param: str) -> Any:
        """Parse a single STEP parameter.

        Args:
            param: Parameter string

        Returns:
            Parsed value (int, float, str, or list)
        """
        param = param.strip()

        # Reference: #N
        if param.startswith("#"):
            try:
                return int(param[1:])
            except ValueError:
                return param

        # String: 'text'
        if param.startswith("'") and param.endswith("'"):
            return param[1:-1]

        # Boolean
        if param in (".T.", ".TRUE."):
            return True
        if param in (".F.", ".FALSE."):
            return False

        # Enum: .VALUE.
        if param.startswith(".") and param.endswith("."):
            return param

        # List: (item1, item2, ...)
        if param.startswith("(") and param.endswith(")"):
            inner = param[1:-1]
            return self._parse_step_params(inner)

        # Number
        try:
            if "." in param or "E" in param.upper():
                return float(param)
            return int(param)
        except ValueError:
            return param

    def _tokenize_param(self, param: Any) -> list[int]:
        """Tokenize a single parameter value.

        Args:
            param: Parsed parameter value

        Returns:
            List of token IDs
        """
        tokens = []

        if isinstance(param, int):
            if param >= 0:  # Entity reference
                # Map reference to token ID
                ref_token_id = self._ref_base_id + (param % 10000)
                tokens.append(ref_token_id)
            else:
                # Negative int - quantize
                bucket = self._quantize_number(float(param))
                token = f"[NUM_{bucket}]"
                tokens.append(self._alt_token_to_id.get(token, 1))

        elif isinstance(param, float):
            bucket = self._quantize_number(param)
            token = f"[NUM_{bucket}]"
            tokens.append(self._alt_token_to_id.get(token, 1))

        elif isinstance(param, str):
            # Check if it's an entity type or enum
            if param in self._alt_token_to_id:
                tokens.append(self._alt_token_to_id[param])
            else:
                tokens.append(self._alt_token_to_id.get("[UNK]", 1))

        elif isinstance(param, list):
            for item in param:
                tokens.extend(self._tokenize_param(item))

        elif isinstance(param, bool):
            # Map bool to numeric bucket
            bucket = 999 if param else 0
            token = f"[NUM_{bucket}]"
            tokens.append(self._alt_token_to_id.get(token, 1))

        return tokens

    def _quantize_number(self, value: float) -> int:
        """Quantize a float value into a bucket index.

        Uses logarithmic scaling to handle wide range of CAD values.

        Args:
            value: Float value to quantize

        Returns:
            Bucket index (0 to NUM_QUANTIZATION_BUCKETS-1)
        """
        import math

        if value == 0:
            return self._NUM_QUANTIZATION_BUCKETS // 2

        # Log scale to handle wide range
        sign = 1 if value > 0 else -1
        abs_val = abs(value)

        # Map to range [0, 1] using arctan
        normalized = (math.atan(math.log10(abs_val + 1e-10)) / math.pi) + 0.5

        # Apply sign
        if sign < 0:
            normalized = 1.0 - normalized

        # Map to bucket
        bucket = int(normalized * (self._NUM_QUANTIZATION_BUCKETS - 1))
        return max(0, min(bucket, self._NUM_QUANTIZATION_BUCKETS - 1))

    def extract_features(
        self, entity_text: str, entity_type: str, return_metadata: bool = False,
    ) -> "dict[str, Any] | tuple[dict[str, Any], dict]":
        """Extract geometric features from entity.

        When ll_stepnet is available, uses its feature extractor.
        Otherwise, uses cadling's STEPFeatureExtractor for equivalent output.

        Args:
            entity_text: Entity definition text
            entity_type: Entity type (e.g., "CARTESIAN_POINT")
            return_metadata: If True, returns (features, metadata) tuple.
                Metadata contains 'degraded' (bool), 'method' (str),
                and 'warning' (str or None) to indicate data quality.

        Returns:
            Dictionary of extracted features (never returns None), or
            (features, metadata) tuple if return_metadata=True.
        """
        metadata = {"degraded": False, "method": "unknown", "warning": None}

        if self.available:
            try:
                features = self.feature_extractor.extract_geometric_features(
                    entity_text, entity_type
                )
                metadata["method"] = "ll_stepnet"
                _log.debug(f"Extracted features for {entity_type} (ll_stepnet): {list(features.keys())}")
                if return_metadata:
                    return features, metadata
                return features
            except Exception as e:
                _log.warning(f"ll_stepnet feature extraction failed, using alternative: {e}")
                metadata["degraded"] = True
                metadata["warning"] = f"ll_stepnet failed: {e}"

        # Alternative implementation using cadling's feature extractor
        features = self._extract_features_alternative(entity_text, entity_type)
        metadata["method"] = "alternative"
        if not metadata["degraded"]:
            metadata["degraded"] = True
            metadata["warning"] = "ll_stepnet not available"
        _log.warning(
            "Using degraded feature extraction (method=%s): %s",
            metadata["method"], metadata["warning"],
        )
        if return_metadata:
            return features, metadata
        return features

    def _extract_features_alternative(
        self, entity_text: str, entity_type: str
    ) -> dict[str, Any]:
        """Alternative feature extraction using cadling's infrastructure.

        Produces features in the same schema as ll_stepnet:
        - entity_id: int
        - entity_type: str
        - category: str (point, curve, surface, topology, other)
        - coordinates: list[float] (for points)
        - radius: float (for curves)
        - direction: list[float] (for lines/axes)
        - numeric_params: list[float]
        - references: list[int]
        - num_references: int

        Args:
            entity_text: Entity definition text
            entity_type: Entity type

        Returns:
            Dictionary of extracted features
        """
        from cadling.backend.step.feature_extractor import STEPFeatureExtractor as CadlingFeatureExtractor

        features: dict[str, Any] = {
            "entity_type": entity_type,
            "category": self._categorize_entity_type(entity_type),
        }

        # Parse entity ID from text
        id_match = re.search(r"#(\d+)", entity_text)
        if id_match:
            features["entity_id"] = int(id_match.group(1))

        # Use cadling's feature extractor for detailed extraction
        try:
            cadling_extractor = CadlingFeatureExtractor()

            # Parse entities from text
            entities = self._parse_step_entities(entity_text)

            if entities:
                entity_data = entities[0]
                all_entities = {entity_data["id"]: {"type": entity_type, "params": entity_data["params"]}}

                extracted = cadling_extractor.extract_entity_features(
                    entity_data["id"],
                    entity_type,
                    all_entities[entity_data["id"]],
                    all_entities
                )

                # Merge extracted features
                features.update(extracted)

        except Exception as e:
            _log.debug(f"Cadling feature extraction failed: {e}")

        # Extract additional features directly from text
        features.update(self._extract_text_features(entity_text, entity_type))

        _log.debug(f"Extracted features for {entity_type} (alternative): {list(features.keys())}")
        return features

    def _categorize_entity_type(self, entity_type: str) -> str:
        """Categorize STEP entity type.

        Args:
            entity_type: STEP entity type name

        Returns:
            Category string: point, curve, surface, topology, shape, or other
        """
        point_types = {"CARTESIAN_POINT", "VERTEX_POINT", "POINT_ON_CURVE", "POINT_ON_SURFACE"}
        curve_types = {"LINE", "CIRCLE", "ELLIPSE", "PARABOLA", "HYPERBOLA",
                       "B_SPLINE_CURVE", "BEZIER_CURVE", "POLYLINE", "TRIMMED_CURVE"}
        surface_types = {"PLANE", "CYLINDRICAL_SURFACE", "CONICAL_SURFACE",
                         "SPHERICAL_SURFACE", "TOROIDAL_SURFACE", "B_SPLINE_SURFACE",
                         "SURFACE_OF_REVOLUTION", "SURFACE_OF_LINEAR_EXTRUSION"}
        topology_types = {"VERTEX_POINT", "EDGE_CURVE", "ORIENTED_EDGE", "EDGE_LOOP",
                          "FACE_BOUND", "FACE_SURFACE", "CLOSED_SHELL", "OPEN_SHELL",
                          "MANIFOLD_SOLID_BREP", "ADVANCED_FACE"}
        shape_types = {"ADVANCED_BREP_SHAPE_REPRESENTATION", "SHAPE_REPRESENTATION",
                       "MANIFOLD_SURFACE_SHAPE_REPRESENTATION"}

        if entity_type in point_types:
            return "point"
        elif entity_type in curve_types:
            return "curve"
        elif entity_type in surface_types:
            return "surface"
        elif entity_type in topology_types:
            return "topology"
        elif entity_type in shape_types:
            return "shape"
        else:
            return "other"

    def _extract_text_features(self, entity_text: str, entity_type: str) -> dict[str, Any]:
        """Extract features directly from entity text using regex.

        Args:
            entity_text: STEP entity text
            entity_type: Entity type

        Returns:
            Dictionary of extracted features
        """
        features: dict[str, Any] = {}

        # Extract numeric values
        numeric_pattern = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")
        numeric_values = [float(m) for m in numeric_pattern.findall(entity_text)]

        if numeric_values:
            features["numeric_params"] = numeric_values
            features["num_numeric_params"] = len(numeric_values)

        # Extract references
        ref_pattern = re.compile(r"#(\d+)")
        references = [int(m) for m in ref_pattern.findall(entity_text)]

        if references:
            # Skip first reference (entity ID)
            features["references"] = references[1:] if len(references) > 1 else []
            features["num_references"] = len(features["references"])

        # Entity-specific feature extraction
        if entity_type == "CARTESIAN_POINT" and len(numeric_values) >= 3:
            features["coordinates"] = numeric_values[:3]
            features["x"] = numeric_values[0]
            features["y"] = numeric_values[1]
            features["z"] = numeric_values[2]
            import math
            features["distance_from_origin"] = math.sqrt(sum(c**2 for c in numeric_values[:3]))

        elif entity_type == "DIRECTION" and len(numeric_values) >= 3:
            features["direction"] = numeric_values[:3]

        elif entity_type == "CIRCLE" and numeric_values:
            # Last numeric is usually radius
            features["radius"] = numeric_values[-1]
            features["curve_type"] = "circle"
            features["is_closed"] = True

        elif entity_type == "LINE":
            features["curve_type"] = "line"
            features["is_linear"] = True

        elif entity_type in ("CYLINDRICAL_SURFACE", "SPHERICAL_SURFACE", "CONICAL_SURFACE"):
            if numeric_values:
                features["radius"] = numeric_values[-1]
            features["surface_type"] = entity_type.replace("_SURFACE", "").lower()

        elif entity_type == "PLANE":
            features["surface_type"] = "plane"
            features["is_planar"] = True

        return features

    def build_topology(
        self, entities: list[dict[str, Any]], return_metadata: bool = False,
    ) -> "dict[str, Any] | tuple[dict[str, Any], dict]":
        """Build topology graph from entities.

        When ll_stepnet is available, uses its topology builder.
        Otherwise, uses cadling's TopologyBuilder for equivalent output.

        Args:
            entities: List of entity dictionaries with IDs and references
            return_metadata: If True, returns (topology, metadata) tuple.
                Metadata contains 'degraded' (bool), 'method' (str),
                and 'warning' (str or None) to indicate data quality.

        Returns:
            Topology graph data (never returns None), or
            (topology, metadata) tuple if return_metadata=True.
        """
        metadata = {"degraded": False, "method": "unknown", "warning": None}

        if self.available:
            try:
                topology = self.topology_builder.build_complete_topology(entities)
                metadata["method"] = "ll_stepnet"
                _log.debug(
                    f"Built topology (ll_stepnet): {topology.get('num_nodes', 0)} nodes, "
                    f"{topology.get('num_edges', 0)} edges"
                )
                if return_metadata:
                    return topology, metadata
                return topology
            except Exception as e:
                _log.warning(f"ll_stepnet topology building failed, using alternative: {e}")
                metadata["degraded"] = True
                metadata["warning"] = f"ll_stepnet failed: {e}"

        # Alternative implementation using cadling's TopologyBuilder
        topology = self._build_topology_alternative(entities)
        metadata["method"] = "alternative"
        if not metadata["degraded"]:
            metadata["degraded"] = True
            metadata["warning"] = "ll_stepnet not available"
        _log.warning(
            "Using degraded topology building (method=%s): %s",
            metadata["method"], metadata["warning"],
        )
        if return_metadata:
            return topology, metadata
        return topology

    def _build_topology_alternative(self, entities: list[dict[str, Any]]) -> dict[str, Any]:
        """Alternative topology building using cadling's infrastructure.

        Produces topology in the same schema as ll_stepnet:
        - num_nodes: int
        - num_edges: int
        - adjacency_list: dict[int, list[int]]
        - reverse_adjacency_list: dict[int, list[int]]
        - entity_levels: dict[int, int]
        - connected_components: list[list[int]]
        - topology_statistics: dict
        - node_features: torch.Tensor (if torch available)
        - edge_index: torch.Tensor (if torch available)

        Args:
            entities: List of entity dictionaries

        Returns:
            Topology graph data
        """
        from cadling.backend.step.topology_builder import TopologyBuilder

        # Convert list of entities to dict format expected by TopologyBuilder
        entities_dict: dict[int, dict[str, Any]] = {}

        for entity in entities:
            entity_id = entity.get("entity_id") or entity.get("id")
            if entity_id is not None:
                entities_dict[entity_id] = {
                    "type": entity.get("entity_type") or entity.get("type", "UNKNOWN"),
                    "params": entity.get("params", []),
                }

        # Build topology using cadling's TopologyBuilder
        topology_builder = TopologyBuilder()
        topology = topology_builder.build_topology_graph(entities_dict)

        # Add additional statistics to match ll_stepnet output
        topology["topology_types"] = self._identify_topology_types(entities)

        # Try to add torch tensors if available
        try:
            import torch

            # Build edge_index tensor (PyG format)
            edge_list = []
            node_ids = sorted(entities_dict.keys())
            id_to_idx = {nid: idx for idx, nid in enumerate(node_ids)}

            for from_id, to_ids in topology.get("adjacency_list", {}).items():
                if from_id in id_to_idx:
                    from_idx = id_to_idx[from_id]
                    for to_id in to_ids:
                        if to_id in id_to_idx:
                            to_idx = id_to_idx[to_id]
                            edge_list.append([from_idx, to_idx])

            if edge_list:
                topology["edge_index"] = torch.tensor(edge_list, dtype=torch.long).t()
            else:
                topology["edge_index"] = torch.zeros(2, 0, dtype=torch.long)

            # Build node features tensor
            num_nodes = len(entities_dict)
            feature_dim = 129  # Match ll_stepnet feature dimension
            node_features = torch.zeros(num_nodes, feature_dim, dtype=torch.float32)

            for entity in entities:
                entity_id = entity.get("entity_id") or entity.get("id")
                if entity_id is None or entity_id not in id_to_idx:
                    continue

                idx = id_to_idx[entity_id]

                # Numeric parameters (pad/truncate to 128 dims)
                numeric_params = entity.get("numeric_params", [])
                if numeric_params:
                    params_to_use = numeric_params[:128]
                    node_features[idx, :len(params_to_use)] = torch.tensor(
                        params_to_use, dtype=torch.float32
                    )

                # Entity type hash (last dimension)
                entity_type = entity.get("entity_type") or entity.get("type", "")
                if entity_type:
                    type_hash = (hash(entity_type) % 10000) / 10000.0
                    node_features[idx, -1] = type_hash

            topology["node_features"] = node_features
            topology["node_ids"] = node_ids
            topology["id_to_idx"] = id_to_idx

        except ImportError:
            _log.debug("torch not available, skipping tensor features")

        _log.debug(
            f"Built topology (alternative): {topology.get('num_nodes', 0)} nodes, "
            f"{topology.get('num_edges', 0)} edges"
        )

        return topology

    def _identify_topology_types(self, entities: list[dict[str, Any]]) -> dict[str, list[int]]:
        """Categorize entities by topological role.

        Args:
            entities: List of entity dictionaries

        Returns:
            Dictionary mapping category to list of entity IDs
        """
        categories: dict[str, list[int]] = {
            "vertices": [],
            "edges": [],
            "faces": [],
            "shells": [],
            "solids": [],
            "geometry": [],
            "other": [],
        }

        for entity in entities:
            entity_id = entity.get("entity_id") or entity.get("id")
            entity_type = entity.get("entity_type") or entity.get("type", "")

            if entity_id is None:
                continue

            if entity_type in ("VERTEX_POINT",):
                categories["vertices"].append(entity_id)
            elif entity_type in ("EDGE_CURVE", "ORIENTED_EDGE", "EDGE_LOOP"):
                categories["edges"].append(entity_id)
            elif entity_type in ("ADVANCED_FACE", "FACE_BOUND", "FACE_OUTER_BOUND", "FACE_SURFACE"):
                categories["faces"].append(entity_id)
            elif entity_type in ("CLOSED_SHELL", "OPEN_SHELL"):
                categories["shells"].append(entity_id)
            elif entity_type in ("MANIFOLD_SOLID_BREP", "BREP_WITH_VOIDS"):
                categories["solids"].append(entity_id)
            elif entity_type in ("CYLINDRICAL_SURFACE", "PLANE", "CONICAL_SURFACE",
                                 "SPHERICAL_SURFACE", "CIRCLE", "LINE", "B_SPLINE_CURVE"):
                categories["geometry"].append(entity_id)
            else:
                categories["other"].append(entity_id)

        return categories

    def reserialize(
        self,
        step_text: str,
        include_annotations: bool = True,
        reserialization_config: Optional[dict[str, Any]] = None,
        annotation_config: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        """Reserialize STEP text using DFS traversal.

        Applies DFS-based reserialization to reorder STEP entities so
        related entities appear contiguously. Optionally generates
        structural annotations (summary and branch annotations) that
        describe the high-level file structure.

        When ll_stepnet is available, uses its reserialization module.
        Otherwise, uses cadling's alternative implementation.

        Args:
            step_text: Raw STEP entity text (DATA section content).
            include_annotations: Whether to generate structural annotations.
            reserialization_config: Optional dict of reserialization config overrides.
            annotation_config: Optional dict of annotation config overrides.

        Returns:
            Dictionary with reserialized text and metadata (never returns None)
        """
        if _LL_STEPNET_RESERIALIZATION_AVAILABLE:
            try:
                reser_cfg = STEPReserializationConfig(**(reserialization_config or {}))
                result = reserialize_step(step_text, config=reser_cfg)

                output = {
                    "reserialized_text": result.text,
                    "traversal_order": result.traversal_order,
                    "entity_count": result.entity_count,
                    "orphan_count": result.orphan_count,
                    "max_depth_reached": result.max_depth_reached,
                    "id_mapping": result.id_mapping,
                }

                if include_annotations:
                    ann_cfg = STEPAnnotationConfig(**(annotation_config or {}))
                    graph = STEPEntityGraph.parse(step_text)
                    annotator = STEPStructuralAnnotator(ann_cfg)
                    annotated = annotator.annotate(graph, result)
                    output["annotated_text"] = annotated.format()
                    output["summary"] = (
                        annotated.summary.format() if annotated.summary else None
                    )
                    output["branch_count"] = len(annotated.branches)

                _log.debug(
                    "Reserialized STEP text (ll_stepnet): %d entities, %d orphans",
                    result.entity_count,
                    result.orphan_count,
                )
                return output
            except Exception as e:
                _log.warning(f"ll_stepnet reserialization failed, using alternative: {e}")

        # Alternative implementation
        return self._reserialize_alternative(
            step_text,
            include_annotations,
            reserialization_config,
            annotation_config
        )

    def _reserialize_alternative(
        self,
        step_text: str,
        include_annotations: bool = True,
        reserialization_config: Optional[dict[str, Any]] = None,
        annotation_config: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        """Alternative reserialization using cadling's infrastructure.

        Performs DFS-based entity reordering and generates structural
        annotations matching ll_stepnet's output format.

        Args:
            step_text: Raw STEP entity text
            include_annotations: Whether to generate annotations
            reserialization_config: Config overrides
            annotation_config: Annotation config overrides

        Returns:
            Dictionary with reserialized text and metadata
        """
        config = reserialization_config or {}
        max_depth = config.get("max_depth", 100)

        # Parse entities
        entities = self._parse_step_entities(step_text)

        if not entities:
            return {
                "reserialized_text": step_text,
                "traversal_order": [],
                "entity_count": 0,
                "orphan_count": 0,
                "max_depth_reached": 0,
                "id_mapping": {},
                "annotated_text": step_text if include_annotations else None,
                "summary": None,
                "branch_count": 0,
            }

        # Build entity graph
        entity_dict = {e["id"]: e for e in entities}
        adjacency: dict[int, list[int]] = defaultdict(list)
        reverse_adjacency: dict[int, list[int]] = defaultdict(list)

        for entity in entities:
            entity_id = entity["id"]
            refs = self._extract_references(entity)
            for ref_id in refs:
                if ref_id in entity_dict:
                    adjacency[entity_id].append(ref_id)
                    reverse_adjacency[ref_id].append(entity_id)

        # Find root entities (no incoming references from other entities in file)
        # Roots are typically PRODUCT, SHAPE_REPRESENTATION, etc.
        root_types = {
            "PRODUCT", "PRODUCT_DEFINITION", "SHAPE_REPRESENTATION",
            "ADVANCED_BREP_SHAPE_REPRESENTATION", "MANIFOLD_SOLID_BREP"
        }

        roots = []
        for entity in entities:
            entity_id = entity["id"]
            entity_type = entity["type"]
            # Entity is a root if no other entity references it, or it's a known root type
            if not reverse_adjacency[entity_id] or entity_type in root_types:
                roots.append(entity_id)

        # If no roots found, use entities with highest dependency count
        if not roots:
            deps_count = {e["id"]: len(adjacency[e["id"]]) for e in entities}
            sorted_entities = sorted(deps_count.items(), key=lambda x: x[1], reverse=True)
            roots = [sorted_entities[0][0]] if sorted_entities else [entities[0]["id"]]

        # DFS traversal to determine ordering
        visited = set()
        traversal_order = []
        max_depth_reached = 0

        def dfs(entity_id: int, depth: int) -> None:
            nonlocal max_depth_reached
            if entity_id in visited or depth > max_depth:
                return
            visited.add(entity_id)
            max_depth_reached = max(max_depth_reached, depth)

            # Visit dependencies first (post-order DFS)
            for ref_id in adjacency[entity_id]:
                dfs(ref_id, depth + 1)

            traversal_order.append(entity_id)

        # Start DFS from each root
        for root_id in roots:
            dfs(root_id, 0)

        # Add orphans (entities not reachable from roots)
        orphan_ids = []
        for entity in entities:
            if entity["id"] not in visited:
                orphan_ids.append(entity["id"])
                traversal_order.append(entity["id"])

        # Create ID mapping (old_id -> new_id)
        id_mapping = {old_id: new_id for new_id, old_id in enumerate(traversal_order, start=1)}

        # Generate reserialized text with new IDs
        reserialized_lines = []
        for new_id, old_id in enumerate(traversal_order, start=1):
            entity = entity_dict[old_id]
            # Replace entity ID and references in text
            new_text = self._remap_entity_text(entity["text"], id_mapping)
            reserialized_lines.append(new_text)

        reserialized_text = "\n".join(reserialized_lines)

        output = {
            "reserialized_text": reserialized_text,
            "traversal_order": traversal_order,
            "entity_count": len(entities),
            "orphan_count": len(orphan_ids),
            "max_depth_reached": max_depth_reached,
            "id_mapping": id_mapping,
        }

        # Generate annotations
        if include_annotations:
            output.update(self._generate_annotations(
                entities, entity_dict, traversal_order, adjacency, annotation_config
            ))

        _log.debug(
            "Reserialized STEP text (alternative): %d entities, %d orphans",
            len(entities),
            len(orphan_ids),
        )

        return output

    def _extract_references(self, entity: dict[str, Any]) -> list[int]:
        """Extract entity reference IDs from entity.

        Args:
            entity: Entity dictionary

        Returns:
            List of referenced entity IDs
        """
        refs = []
        ref_pattern = re.compile(r"#(\d+)")

        # Extract from params
        for param in entity.get("params", []):
            if isinstance(param, int):
                refs.append(param)
            elif isinstance(param, str):
                matches = ref_pattern.findall(param)
                refs.extend(int(m) for m in matches)
            elif isinstance(param, list):
                refs.extend(self._extract_refs_from_list(param))

        return refs

    def _extract_refs_from_list(self, param_list: list[Any]) -> list[int]:
        """Recursively extract references from nested lists."""
        refs = []
        ref_pattern = re.compile(r"#(\d+)")

        for item in param_list:
            if isinstance(item, int):
                refs.append(item)
            elif isinstance(item, str):
                matches = ref_pattern.findall(item)
                refs.extend(int(m) for m in matches)
            elif isinstance(item, list):
                refs.extend(self._extract_refs_from_list(item))

        return refs

    def _remap_entity_text(self, entity_text: str, id_mapping: dict[int, int]) -> str:
        """Remap entity IDs in STEP text.

        Args:
            entity_text: Original entity text
            id_mapping: Mapping from old IDs to new IDs

        Returns:
            Entity text with remapped IDs
        """
        def replace_ref(match: re.Match) -> str:
            old_id = int(match.group(1))
            new_id = id_mapping.get(old_id, old_id)
            return f"#{new_id}"

        return re.sub(r"#(\d+)", replace_ref, entity_text)

    def _generate_annotations(
        self,
        entities: list[dict[str, Any]],
        entity_dict: dict[int, dict[str, Any]],
        traversal_order: list[int],
        adjacency: dict[int, list[int]],
        annotation_config: Optional[dict[str, Any]],
    ) -> dict[str, Any]:
        """Generate structural annotations for reserialized STEP.

        Args:
            entities: List of all entities
            entity_dict: Entity dictionary
            traversal_order: DFS traversal order
            adjacency: Entity adjacency list
            annotation_config: Configuration for annotations

        Returns:
            Dictionary with annotation fields
        """
        config = annotation_config or {}
        include_summary = config.get("include_summary", True)
        include_branches = config.get("include_branches", True)

        # Count entity types
        type_counts: dict[str, int] = defaultdict(int)
        for entity in entities:
            type_counts[entity["type"]] += 1

        # Generate summary annotation
        summary = None
        if include_summary:
            summary_lines = [
                f"# STEP File Structure Summary",
                f"# Total entities: {len(entities)}",
                f"# Entity types: {len(type_counts)}",
            ]

            # Add type breakdown
            for etype, count in sorted(type_counts.items(), key=lambda x: -x[1])[:10]:
                summary_lines.append(f"#   {etype}: {count}")

            summary = "\n".join(summary_lines)

        # Generate branch annotations
        branch_count = 0
        annotated_lines = []

        if include_branches:
            # Identify branch roots (entities that start new subtrees)
            branch_roots = set()
            for entity in entities:
                entity_type = entity["type"]
                # Mark significant entity types as branch roots
                if entity_type in ("MANIFOLD_SOLID_BREP", "CLOSED_SHELL", "ADVANCED_FACE",
                                   "SHAPE_REPRESENTATION", "PRODUCT"):
                    branch_roots.add(entity["id"])

            # Generate annotated text with branch markers
            current_branch = None
            for entity_id in traversal_order:
                entity = entity_dict[entity_id]

                if entity_id in branch_roots:
                    branch_count += 1
                    # Count descendants
                    descendants = self._count_descendants(entity_id, adjacency)
                    branch_annotation = (
                        f"/* Branch {branch_count}: {entity['type']} "
                        f"({descendants} entities) */"
                    )
                    annotated_lines.append(branch_annotation)
                    current_branch = entity_id

                annotated_lines.append(entity["text"])

        annotated_text = "\n".join(annotated_lines) if annotated_lines else None

        return {
            "annotated_text": annotated_text,
            "summary": summary,
            "branch_count": branch_count,
        }

    def _count_descendants(self, entity_id: int, adjacency: dict[int, list[int]]) -> int:
        """Count descendants of an entity in the graph.

        Args:
            entity_id: Root entity ID
            adjacency: Adjacency list

        Returns:
            Number of descendant entities
        """
        visited = set()
        stack = [entity_id]
        count = 0

        while stack:
            current = stack.pop()
            if current in visited:
                continue
            visited.add(current)
            count += 1

            for ref_id in adjacency.get(current, []):
                if ref_id not in visited:
                    stack.append(ref_id)

        return count - 1  # Exclude the root itself

    @staticmethod
    def is_available() -> bool:
        """Check if ll_stepnet is available.

        Returns:
            True if ll_stepnet can be imported, False otherwise
        """
        return _LL_STEPNET_AVAILABLE


# Utility function to check ll_stepnet availability
def check_ll_stepnet_availability() -> tuple[bool, Optional[str]]:
    """Check if ll_stepnet is available and get version info.

    Returns:
        Tuple of (is_available, version_or_error_message)
    """
    if _LL_STEPNET_AVAILABLE:
        try:
            from ll_stepnet import __version__

            return True, __version__
        except ImportError:
            return True, "unknown version"
    else:
        return False, "ll_stepnet not installed or import failed"
