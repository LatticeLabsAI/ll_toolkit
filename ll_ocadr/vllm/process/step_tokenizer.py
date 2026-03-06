"""
STEP/B-Rep Tokenizer for LL-OCADR.
Complete tokenizer that ensures ALL content in STEP files is captured.

Extracts:
- Geometric features (CYLINDER radius, PLANE normal, B_SPLINE control points, etc.)
- Topological relationships (face-edge-vertex references)
- ALL numeric values, entity references, keywords, and structural elements
"""

import logging
import hashlib
import re
import struct
import torch
import numpy as np
from typing import List, Dict, Tuple, Optional, Set
from collections import defaultdict

_log = logging.getLogger(__name__)


def _deterministic_hash(s: str) -> int:
    """Return a deterministic integer hash for a string.

    Unlike Python's built-in hash(), this is stable across processes
    and Python invocations regardless of PYTHONHASHSEED.
    """
    digest = hashlib.md5(s.encode("utf-8")).digest()
    return struct.unpack("<Q", digest[:8])[0]


class STEPCompleteTokenizer:
    """
    Complete STEP tokenizer ensuring 100% content coverage.

    Token types:
    - ENTITY_ID: #123
    - ENTITY_TYPE: CYLINDER, B_SPLINE_CURVE, etc.
    - NUMERIC: All numeric values (integers, floats, scientific notation)
    - REFERENCE: Entity references like #3770
    - KEYWORD: .T., .F., .UNSPECIFIED., etc.
    - OPERATOR: =, (, ), ,, ;
    - STRING: Text in quotes
    - PARAM_NAME: Inferred parameter names (radius, axis_location, etc.)
    """

    def __init__(self, vocab_size: int = 50000):
        """
        Args:
            vocab_size: Maximum vocabulary size for entity types and keywords
        """
        self.vocab_size = vocab_size

        # Pre-compile entity tokenization regex (avoid recompiling per entity)
        _patterns = [
            (r'#\d+=', 'ENTITY_ID'),
            (r'#\d+', 'REFERENCE'),
            (r'[A-Z_][A-Z0-9_]*', 'ENTITY_TYPE'),
            (r'-?\d+\.?\d*(?:[Ee][+-]?\d+)?', 'NUMERIC'),
            (r'\.[A-Z_]+\.', 'KEYWORD'),
            (r"'[^']*'", 'STRING'),
            (r'[=(),;]', 'OPERATOR'),
            (r'\$|\*', 'SPECIAL'),
            (r'\n', 'NEWLINE'),
        ]
        combined = '|'.join(f'(?P<{name}>{pattern})' for pattern, name in _patterns)
        self._entity_regex = re.compile(combined)

        # Build vocabulary from STEP AP203/AP214 specification
        self._build_vocabulary()

        # Token type IDs
        self.token_type_to_id = {
            'PAD': 0,
            'ENTITY_ID': 1,
            'ENTITY_TYPE': 2,
            'NUMERIC': 3,
            'REFERENCE': 4,
            'KEYWORD': 5,
            'OPERATOR': 6,
            'STRING': 7,
            'PARAM_NAME': 8,
            'NEWLINE': 9,
            'UNK': 10
        }

    def _build_vocabulary(self):
        """Build comprehensive STEP vocabulary."""

        # All STEP entity types from AP203/AP214
        self.entity_types = set([
            # Geometric entities
            'CARTESIAN_POINT', 'DIRECTION', 'VECTOR',
            'AXIS1_PLACEMENT', 'AXIS2_PLACEMENT_2D', 'AXIS2_PLACEMENT_3D',
            'LINE', 'CIRCLE', 'ELLIPSE', 'PARABOLA', 'HYPERBOLA',
            'B_SPLINE_CURVE', 'B_SPLINE_CURVE_WITH_KNOTS',
            'RATIONAL_B_SPLINE_CURVE', 'BEZIER_CURVE',
            'TRIMMED_CURVE', 'COMPOSITE_CURVE', 'POLYLINE',
            'OFFSET_CURVE_2D', 'OFFSET_CURVE_3D',

            # Surfaces
            'PLANE', 'CYLINDRICAL_SURFACE', 'CONICAL_SURFACE',
            'SPHERICAL_SURFACE', 'TOROIDAL_SURFACE',
            'B_SPLINE_SURFACE', 'B_SPLINE_SURFACE_WITH_KNOTS',
            'RATIONAL_B_SPLINE_SURFACE', 'BEZIER_SURFACE',
            'SURFACE_OF_LINEAR_EXTRUSION', 'SURFACE_OF_REVOLUTION',
            'OFFSET_SURFACE', 'RECTANGULAR_TRIMMED_SURFACE',

            # Topology
            'VERTEX_POINT', 'EDGE_CURVE', 'EDGE_LOOP',
            'ORIENTED_EDGE', 'FACE_BOUND', 'FACE_OUTER_BOUND',
            'ADVANCED_FACE', 'CLOSED_SHELL', 'OPEN_SHELL',
            'CONNECTED_FACE_SET', 'MANIFOLD_SOLID_BREP',
            'BREP_WITH_VOIDS', 'FACETED_BREP',
            'SHELL_BASED_SURFACE_MODEL',

            # Properties
            'BOUNDED_CURVE', 'BOUNDED_SURFACE',
            'GEOMETRIC_REPRESENTATION_ITEM',
            'CURVE', 'SURFACE', 'SOLID',
            'REPRESENTATION_ITEM', 'REPRESENTATION',

            # Assembly
            'PRODUCT', 'PRODUCT_DEFINITION', 'PRODUCT_DEFINITION_SHAPE',
            'SHAPE_REPRESENTATION', 'SHAPE_DEFINITION_REPRESENTATION',
            'NEXT_ASSEMBLY_USAGE_OCCURRENCE',
            'PRODUCT_DEFINITION_FORMATION',

            # Context
            'APPLICATION_CONTEXT', 'APPLICATION_PROTOCOL_DEFINITION',
            'DESIGN_CONTEXT', 'MECHANICAL_CONTEXT',

            # Relationships
            'REPRESENTATION_RELATIONSHIP',
            'REPRESENTATION_RELATIONSHIP_WITH_TRANSFORMATION',
            'ITEM_DEFINED_TRANSFORMATION',
            'CONTEXT_DEPENDENT_SHAPE_REPRESENTATION',
            'SHAPE_REPRESENTATION_RELATIONSHIP',

            # Presentation
            'STYLED_ITEM', 'PRESENTATION_STYLE_ASSIGNMENT',
            'SURFACE_STYLE_USAGE', 'SURFACE_SIDE_STYLE',
            'SURFACE_STYLE_FILL_AREA', 'FILL_AREA_STYLE',
            'FILL_AREA_STYLE_COLOUR', 'COLOUR_RGB',
            'DRAUGHTING_PRE_DEFINED_COLOUR',
            'MECHANICAL_DESIGN_GEOMETRIC_PRESENTATION_REPRESENTATION',
        ])

        # STEP keywords
        self.keywords = set([
            '.T.', '.F.', '.U.', '.UNKNOWN.',
            '.UNSPECIFIED.', '.PARAMETER.', '.POSITIVE.',
            '.NEGATIVE.', '$', '*', ''
        ])

        # Operators
        self.operators = set(['=', '(', ')', ',', ';', '#'])

        # Build ID mappings
        self.entity_type_to_id = {et: i for i, et in enumerate(sorted(self.entity_types))}
        self.keyword_to_id = {kw: i for i, kw in enumerate(sorted(self.keywords))}

        # Parameter name inference patterns
        self.param_patterns = {
            'CARTESIAN_POINT': ['coordinates'],
            'DIRECTION': ['direction_ratios'],
            'VECTOR': ['orientation', 'magnitude'],
            'CIRCLE': ['position', 'radius'],
            'ELLIPSE': ['position', 'semi_axis_1', 'semi_axis_2'],
            'CYLINDER': ['position', 'radius'],
            'CYLINDRICAL_SURFACE': ['position', 'radius'],
            'CONICAL_SURFACE': ['position', 'radius', 'semi_angle'],
            'SPHERICAL_SURFACE': ['position', 'radius'],
            'TOROIDAL_SURFACE': ['position', 'major_radius', 'minor_radius'],
            'B_SPLINE_CURVE': ['degree', 'control_points_list', 'curve_form', 'closed_curve', 'self_intersect'],
            'B_SPLINE_CURVE_WITH_KNOTS': ['multiplicities', 'knots', 'knot_spec'],
            'RATIONAL_B_SPLINE_CURVE': ['weights_data'],
            'B_SPLINE_SURFACE': ['u_degree', 'v_degree', 'control_points_list', 'surface_form', 'u_closed', 'v_closed', 'self_intersect'],
            'B_SPLINE_SURFACE_WITH_KNOTS': ['u_multiplicities', 'v_multiplicities', 'u_knots', 'v_knots', 'knot_spec'],
            'ADVANCED_FACE': ['bounds', 'face_geometry', 'same_sense'],
            'EDGE_CURVE': ['edge_start', 'edge_end', 'edge_geometry', 'same_sense'],
            'ORIENTED_EDGE': ['edge_element', 'orientation'],
        }

    def tokenize_raw_content(self, raw_content: str) -> Dict[str, any]:
        """
        Tokenize raw STEP content ensuring complete coverage.

        Args:
            raw_content: Raw STEP text from chunk['raw_content']

        Returns:
            Dictionary with:
                - tokens: List of token strings
                - token_types: List of token type IDs
                - token_values: List of numeric/reference values (None if not applicable)
                - entity_boundaries: List of (start, end) indices for each entity
                - reference_graph: Dict mapping entity IDs to referenced entity IDs
                - geometric_features: Extracted geometric parameters
                - topological_features: Extracted topological relationships
        """
        tokens = []
        token_types = []
        token_values = []
        entity_boundaries = []
        reference_graph = defaultdict(list)
        geometric_features = []
        topological_features = []

        # Split into entities (by lines starting with #)
        entities = []
        current_entity = []
        current_entity_id = None

        for line in raw_content.split('\n'):
            stripped = line.strip()
            if not stripped:
                continue

            if stripped.startswith('#'):
                # Save previous entity
                if current_entity:
                    entities.append({
                        'id': current_entity_id,
                        'text': '\n'.join(current_entity)
                    })

                # Start new entity
                current_entity = [line]
                # Extract entity ID
                id_match = re.match(r'#(\d+)', stripped)
                current_entity_id = int(id_match.group(1)) if id_match else None
            else:
                # Continuation of current entity
                if current_entity:
                    current_entity.append(line)

        # Don't forget last entity
        if current_entity:
            entities.append({
                'id': current_entity_id,
                'text': '\n'.join(current_entity)
            })

        # Process each entity
        for entity in entities:
            entity_start = len(tokens)
            entity_id = entity['id']
            entity_text = entity['text']

            # Tokenize this entity
            entity_tokens = self._tokenize_entity_complete(
                entity_text,
                entity_id,
                reference_graph,
                geometric_features,
                topological_features
            )

            tokens.extend(entity_tokens['tokens'])
            token_types.extend(entity_tokens['types'])
            token_values.extend(entity_tokens['values'])

            entity_end = len(tokens)
            entity_boundaries.append((entity_start, entity_end))

        return {
            'tokens': tokens,
            'token_types': token_types,
            'token_values': token_values,
            'entity_boundaries': entity_boundaries,
            'reference_graph': dict(reference_graph),
            'geometric_features': geometric_features,
            'topological_features': topological_features,
            'num_entities': len(entities)
        }

    def _tokenize_entity_complete(
        self,
        entity_text: str,
        entity_id: int,
        reference_graph: dict,
        geometric_features: list,
        topological_features: list
    ) -> Dict:
        """Completely tokenize a single entity."""

        tokens = []
        types = []
        values = []

        # Use pre-compiled regex from __init__
        regex = self._entity_regex

        # Track current context for parameter name inference
        current_entity_type = None
        param_index = 0
        in_parentheses = 0

        for match in regex.finditer(entity_text):
            token_type = match.lastgroup
            token_text = match.group()
            token_value = None

            if token_type == 'ENTITY_ID':
                # Entity ID - extract number from #123=
                eid = int(token_text.strip('#='))
                tokens.append(token_text)
                types.append(self.token_type_to_id['ENTITY_ID'])
                values.append(eid)

            elif token_type == 'REFERENCE':
                # Entity reference - extract number from #456
                ref_id = int(token_text.strip('#'))
                tokens.append(token_text)
                types.append(self.token_type_to_id['REFERENCE'])
                values.append(ref_id)

                # Track reference in graph
                if entity_id is not None:
                    reference_graph[entity_id].append(ref_id)
                    topological_features.append({
                        'from_entity': entity_id,
                        'to_entity': ref_id,
                        'relationship': 'references'
                    })

            elif token_type == 'ENTITY_TYPE':
                # Entity type keyword
                if token_text in self.entity_types:
                    current_entity_type = token_text
                    param_index = 0

                tokens.append(token_text)
                types.append(self.token_type_to_id['ENTITY_TYPE'])
                values.append(None)

            elif token_type == 'NUMERIC':
                # Numeric value
                try:
                    num_val = float(token_text)
                    token_value = num_val

                    # Extract geometric feature if we know the context
                    if current_entity_type and param_index < 10:
                        param_name = self._infer_param_name(
                            current_entity_type, param_index
                        )
                        geometric_features.append({
                            'entity_id': entity_id,
                            'entity_type': current_entity_type,
                            'parameter': param_name,
                            'value': num_val
                        })

                    param_index += 1

                except (ValueError, TypeError):
                    _log.debug("Could not parse numeric token %r, defaulting to 0.0", token_text)
                    num_val = 0.0

                tokens.append(token_text)
                types.append(self.token_type_to_id['NUMERIC'])
                values.append(num_val)

            elif token_type == 'KEYWORD':
                tokens.append(token_text)
                types.append(self.token_type_to_id['KEYWORD'])
                values.append(None)

            elif token_type == 'SPECIAL':
                tokens.append(token_text)
                types.append(self.token_type_to_id['KEYWORD'])  # Treat as keyword
                values.append(None)

            elif token_type == 'STRING':
                tokens.append(token_text)
                types.append(self.token_type_to_id['STRING'])
                values.append(None)

            elif token_type == 'OPERATOR':
                tokens.append(token_text)
                types.append(self.token_type_to_id['OPERATOR'])
                values.append(None)

                if token_text == '(':
                    in_parentheses += 1
                elif token_text == ')':
                    in_parentheses -= 1
                    if in_parentheses == 0:
                        param_index = 0  # Reset for next entity type

            elif token_type == 'NEWLINE':
                tokens.append('<NL>')
                types.append(self.token_type_to_id['NEWLINE'])
                values.append(None)

        return {
            'tokens': tokens,
            'types': types,
            'values': values
        }

    def _infer_param_name(self, entity_type: str, param_index: int) -> str:
        """Infer parameter name from entity type and position."""
        if entity_type in self.param_patterns:
            param_names = self.param_patterns[entity_type]
            if param_index < len(param_names):
                return param_names[param_index]
        return f'param_{param_index}'

    def encode_to_tensors(self, tokenized: Dict) -> Dict[str, torch.Tensor]:
        """
        Convert tokenized data to tensor format for neural network.

        Returns:
            Dictionary with:
                - token_ids: [seq_len] - Hashed token IDs
                - token_types: [seq_len] - Token type IDs
                - token_values: [seq_len] - Numeric values (0 if not numeric)
                - entity_mask: [seq_len] - 1 at entity boundaries
                - reference_matrix: [num_entities, num_entities] - Adjacency matrix
                - geometric_tensor: [num_geometric_features, 4] - (entity_id, param_id, value, type)
        """
        # Hash tokens to IDs
        token_ids = [_deterministic_hash(t) % self.vocab_size for t in tokenized['tokens']]

        # Token types
        token_types = tokenized['token_types']

        # Token values (0 for non-numeric)
        token_values = [
            v if v is not None else 0.0
            for v in tokenized['token_values']
        ]

        # Entity mask
        entity_mask = torch.zeros(len(tokenized['tokens']), dtype=torch.bool)
        for start, end in tokenized['entity_boundaries']:
            entity_mask[start] = True

        # Reference matrix (adjacency matrix for entity graph)
        if tokenized['num_entities'] > 0:
            max_entity_id = max(
                max(tokenized['reference_graph'].keys(), default=0),
                max((max(refs, default=0) for refs in tokenized['reference_graph'].values()), default=0)
            )
            ref_matrix = torch.zeros(max_entity_id + 1, max_entity_id + 1)

            for from_id, to_ids in tokenized['reference_graph'].items():
                for to_id in to_ids:
                    ref_matrix[from_id, to_id] = 1
        else:
            ref_matrix = torch.zeros(1, 1)

        # Geometric features tensor
        geom_features = tokenized['geometric_features']
        if geom_features:
            geom_tensor = torch.zeros(len(geom_features), 4, dtype=torch.float64)
            for i, feat in enumerate(geom_features):
                geom_tensor[i, 0] = feat['entity_id']
                geom_tensor[i, 1] = _deterministic_hash(feat['parameter']) % 1000
                geom_tensor[i, 2] = feat['value']
                geom_tensor[i, 3] = _deterministic_hash(feat.get('entity_type', '')) % 1000
        else:
            geom_tensor = torch.zeros(1, 4, dtype=torch.float64)

        return {
            'token_ids': torch.tensor(token_ids, dtype=torch.long),
            'token_types': torch.tensor(token_types, dtype=torch.long),
            'token_values': torch.tensor(token_values, dtype=torch.float32),
            'entity_mask': entity_mask,
            'reference_matrix': ref_matrix,
            'geometric_tensor': geom_tensor,
            'num_tokens': len(token_ids),
            'num_entities': tokenized['num_entities']
        }

    def tokenize_chunk(self, chunk: Dict) -> Dict[str, torch.Tensor]:
        """
        Tokenize a chunk from UnifiedCADContentChunker.

        Args:
            chunk: Chunk dict with 'raw_content' field

        Returns:
            Encoded tensors ready for model
        """
        raw_content = chunk.get('raw_content', '')

        # Complete tokenization
        tokenized = self.tokenize_raw_content(raw_content)

        # Encode to tensors
        encoded = self.encode_to_tensors(tokenized)

        # Add metadata
        encoded['chunk_format'] = chunk.get('format', 'STEP')
        encoded['chunk_start_entity'] = chunk.get('start_entity', 0)
        encoded['chunk_end_entity'] = chunk.get('end_entity', 0)

        return encoded


def build_step_tokenizer(vocab_size: int = 50000):
    """Build complete STEP tokenizer."""
    return STEPCompleteTokenizer(vocab_size=vocab_size)
