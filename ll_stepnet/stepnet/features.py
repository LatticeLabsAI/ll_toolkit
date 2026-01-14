"""
STEP Feature Extraction Module
Extracts geometric properties from STEP entities.
"""

import re
from typing import List, Dict, Tuple


class STEPFeatureExtractor:
    """
    Extracts geometric features from tokenized STEP content.
    Separate from tokenization - operates on parsed entities.
    """

    def __init__(self):
        """Initialize feature extractor with parameter patterns."""
        # Parameter positions for common entity types
        self.param_patterns = {
            'CARTESIAN_POINT': {
                'params': ['x', 'y', 'z'],
                'num_params': 3
            },
            'CIRCLE': {
                'params': ['position_ref', 'radius'],
                'num_params': 2
            },
            'ELLIPSE': {
                'params': ['position_ref', 'semi_axis_1', 'semi_axis_2'],
                'num_params': 3
            },
            'CYLINDRICAL_SURFACE': {
                'params': ['position_ref', 'radius'],
                'num_params': 2
            },
            'CONICAL_SURFACE': {
                'params': ['position_ref', 'radius', 'semi_angle'],
                'num_params': 3
            },
            'SPHERICAL_SURFACE': {
                'params': ['position_ref', 'radius'],
                'num_params': 2
            },
            'B_SPLINE_CURVE_WITH_KNOTS': {
                'params': ['multiplicities', 'knots', 'knot_spec'],
                'num_params': 3
            },
        }

    def extract_entity_info(self, entity_text: str) -> Dict:
        """
        Parse a single STEP entity to extract basic info.

        Args:
            entity_text: Single STEP entity string (e.g., "#31=CYLINDER(...);")

        Returns:
            Dictionary with entity_id, entity_type, parameters
        """
        # Extract entity ID
        id_match = re.match(r'#(\d+)\s*=', entity_text)
        entity_id = int(id_match.group(1)) if id_match else None

        # Extract entity type
        type_match = re.search(r'([A-Z_][A-Z0-9_]*)\s*\(', entity_text)
        entity_type = type_match.group(1) if type_match else None

        # Extract parameters (everything in parentheses)
        params_match = re.search(r'\((.*?)\);?\s*$', entity_text, re.DOTALL)
        params_text = params_match.group(1) if params_match else ""

        return {
            'entity_id': entity_id,
            'entity_type': entity_type,
            'params_text': params_text,
            'raw_text': entity_text
        }

    def extract_numeric_params(self, params_text: str) -> List[float]:
        """
        Extract all numeric values from parameter string.

        Args:
            params_text: Parameter text from entity

        Returns:
            List of numeric values
        """
        # Find all numbers (int, float, scientific notation)
        pattern = r'-?\d+\.?\d*(?:[Ee][+-]?\d+)?'
        matches = re.findall(pattern, params_text)

        numbers = []
        for match in matches:
            try:
                numbers.append(float(match))
            except ValueError:
                continue

        return numbers

    def extract_references(self, params_text: str) -> List[int]:
        """
        Extract entity reference IDs (#123, #456, etc.).

        Args:
            params_text: Parameter text from entity

        Returns:
            List of referenced entity IDs
        """
        pattern = r'#(\d+)'
        matches = re.findall(pattern, params_text)
        return [int(m) for m in matches]

    def extract_geometric_features(self, entity_text: str) -> Dict:
        """
        Extract complete geometric features from an entity.

        Args:
            entity_text: STEP entity text

        Returns:
            Dictionary with:
                - entity_id: int
                - entity_type: str
                - numeric_params: List[float]
                - references: List[int]
                - named_params: Dict (if known pattern)
        """
        info = self.extract_entity_info(entity_text)

        features = {
            'entity_id': info['entity_id'],
            'entity_type': info['entity_type'],
            'numeric_params': self.extract_numeric_params(info['params_text']),
            'references': self.extract_references(info['params_text']),
        }

        # Add named parameters if we know the pattern
        if info['entity_type'] in self.param_patterns:
            pattern = self.param_patterns[info['entity_type']]
            features['named_params'] = {}

            for i, param_name in enumerate(pattern['params']):
                if i < len(features['numeric_params']):
                    features['named_params'][param_name] = features['numeric_params'][i]

        return features

    def extract_features_from_chunk(self, chunk_text: str) -> List[Dict]:
        """
        Extract features from a chunk of STEP text (multiple entities).

        Args:
            chunk_text: Raw STEP text with multiple entities

        Returns:
            List of feature dictionaries
        """
        features_list = []

        # Split into entities
        entities = []
        current_entity = []

        for line in chunk_text.split('\n'):
            stripped = line.strip()
            if not stripped:
                continue

            if stripped.startswith('#'):
                # Save previous entity
                if current_entity:
                    entity_text = '\n'.join(current_entity)
                    entities.append(entity_text)

                # Start new entity
                current_entity = [line]
            else:
                # Continuation
                if current_entity:
                    current_entity.append(line)

        # Last entity
        if current_entity:
            entity_text = '\n'.join(current_entity)
            entities.append(entity_text)

        # Extract features from each entity
        for entity_text in entities:
            features = self.extract_geometric_features(entity_text)
            features_list.append(features)

        return features_list
