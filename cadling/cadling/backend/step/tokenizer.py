"""STEP tokenizer implementation.

This module provides tokenization for STEP (ISO 10303-21) files,
converting STEP text into tokens and token IDs for processing.

The tokenizer handles:
- Entity references (#123)
- Entity type names (CARTESIAN_POINT, CIRCLE, etc.)
- Numeric parameters (coordinates, radii, etc.)
- Keywords (.T., .F., .UNSPECIFIED., etc.)
- Operators (=, (), ,, ;, etc.)
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional


class STEPTokenizer:
    """Tokenizer for STEP files.

    Converts STEP text to tokens and token IDs following standard
    tokenizer design patterns. Handles entity types, references,
    numeric parameters, and keywords.

    Attributes:
        vocab_size: Maximum vocabulary size
        PAD_ID: Padding token ID
        UNK_ID: Unknown token ID
        SEP_ID: Separator token ID
        CLS_ID: Classification/start token ID
        vocab: Token to ID mapping
        id_to_token: ID to token mapping
    """

    def __init__(self, vocab_size: int = 50000):
        """Initialize STEP tokenizer.

        Args:
            vocab_size: Maximum vocabulary size for hashing unknown tokens
        """
        self.vocab_size = vocab_size

        # Special tokens
        self.PAD_ID = 0
        self.UNK_ID = 1
        self.SEP_ID = 2
        self.CLS_ID = 3

        self.special_tokens = {
            '<PAD>': self.PAD_ID,
            '<UNK>': self.UNK_ID,
            '<SEP>': self.SEP_ID,
            '<CLS>': self.CLS_ID,
        }

        # Build vocabulary
        self.vocab = self._build_vocab()
        self.id_to_token = {v: k for k, v in self.vocab.items()}

        # Parse error tracking
        self.parse_errors = []
        self.parse_warnings = []

    def _build_vocab(self) -> Dict[str, int]:
        """Build STEP vocabulary from common entity types and keywords.

        Returns:
            Dictionary mapping tokens to IDs
        """
        vocab = dict(self.special_tokens)
        idx = len(vocab)

        # Common STEP entity types
        entity_types = [
            # Points and directions
            'CARTESIAN_POINT', 'DIRECTION', 'VECTOR',

            # Placement and axes
            'AXIS2_PLACEMENT_3D', 'AXIS1_PLACEMENT', 'PLACEMENT',

            # Curves
            'LINE', 'CIRCLE', 'ELLIPSE', 'PARABOLA', 'HYPERBOLA',
            'B_SPLINE_CURVE', 'B_SPLINE_CURVE_WITH_KNOTS',
            'RATIONAL_B_SPLINE_CURVE', 'POLYLINE', 'TRIMMED_CURVE',

            # Surfaces
            'PLANE', 'CYLINDRICAL_SURFACE', 'CONICAL_SURFACE',
            'SPHERICAL_SURFACE', 'TOROIDAL_SURFACE',
            'B_SPLINE_SURFACE', 'B_SPLINE_SURFACE_WITH_KNOTS',
            'SURFACE_OF_REVOLUTION', 'SURFACE_OF_LINEAR_EXTRUSION',

            # Topology
            'VERTEX_POINT', 'EDGE_CURVE', 'ORIENTED_EDGE', 'EDGE_LOOP',
            'FACE_BOUND', 'FACE_OUTER_BOUND', 'ADVANCED_FACE',
            'CLOSED_SHELL', 'OPEN_SHELL', 'MANIFOLD_SOLID_BREP',
            'BREP_WITH_VOIDS', 'FACETED_BREP',

            # Representation
            'REPRESENTATION_ITEM', 'GEOMETRIC_REPRESENTATION_ITEM',
            'TOPOLOGICAL_REPRESENTATION_ITEM', 'SHAPE_REPRESENTATION',
            'GEOMETRIC_REPRESENTATION_CONTEXT', 'GLOBAL_UNIT_ASSIGNED_CONTEXT',

            # Properties
            'PRODUCT_DEFINITION', 'PRODUCT', 'PRODUCT_DEFINITION_SHAPE',
            'SHAPE_DEFINITION_REPRESENTATION', 'PROPERTY_DEFINITION',

            # Units and measures
            'LENGTH_MEASURE', 'PLANE_ANGLE_MEASURE', 'SOLID_ANGLE_MEASURE',
            'SI_UNIT', 'CONVERSION_BASED_UNIT', 'NAMED_UNIT',
        ]

        for token in entity_types:
            vocab[token] = idx
            idx += 1

        # STEP keywords
        keywords = [
            '.T.', '.F.',  # Boolean
            '.UNSPECIFIED.', '.PARAMETER.',  # Enumeration
            '$',  # Null reference
            '*',  # Derived value
        ]
        for token in keywords:
            vocab[token] = idx
            idx += 1

        # Operators and delimiters
        operators = ['=', '(', ')', ',', ';', '#', '/', '\'', '"']
        for token in operators:
            vocab[token] = idx
            idx += 1

        return vocab

    def tokenize(self, text: str) -> List[str]:
        """Split STEP text into tokens.

        Tokenizes STEP text using regex to identify:
        - Entity references (#123)
        - Identifiers (CARTESIAN_POINT, MY_ENTITY)
        - Numbers (-123.456, 1.23E-10)
        - Keywords (.T., .UNSPECIFIED.)
        - String literals ('text')
        - Operators (=, (), ,, ;, etc.)

        Args:
            text: Raw STEP text

        Returns:
            List of token strings
        """
        # Regex pattern breakdown:
        # #\d+ - entity references
        # [A-Z_][A-Z0-9_]* - identifiers (entity types, names)
        # -?\d+\.?\d*(?:[Ee][+-]?\d+)? - numbers (int, float, scientific)
        # \.[A-Z_]+\. - keywords (.T., .UNSPECIFIED.)
        # '[^']*' - string literals
        # [=(),;] - operators
        # \$|\* - special symbols
        pattern = r"#\d+|[A-Z_][A-Z0-9_]*|-?\d+\.?\d*(?:[Ee][+-]?\d+)?|\.[A-Z_]+\.|'[^']*'|[=(),;]|\$|\*"
        tokens = re.findall(pattern, text)
        return tokens

    def encode(self, text: str) -> List[int]:
        """Encode STEP text to token IDs.

        Converts text to tokens, then maps each token to an ID.
        Known tokens use vocab mapping. Unknown tokens are hashed
        into the vocab space for consistent representation.

        Args:
            text: Raw STEP text

        Returns:
            List of token IDs
        """
        tokens = self.tokenize(text)
        token_ids = []

        for token in tokens:
            if token in self.vocab:
                token_ids.append(self.vocab[token])
            else:
                # Hash unknown tokens to vocab space for consistency
                token_ids.append(hash(token) % self.vocab_size)

        return token_ids

    def decode(self, token_ids: List[int]) -> str:
        """Decode token IDs back to text (approximate).

        Converts token IDs back to tokens. Unknown IDs are
        represented as <UNK>. Approximate because hashed tokens
        cannot be perfectly recovered.

        Args:
            token_ids: List of token IDs

        Returns:
            Decoded text (space-separated tokens)
        """
        tokens = []
        for tid in token_ids:
            if tid in self.id_to_token:
                tokens.append(self.id_to_token[tid])
            else:
                tokens.append('<UNK>')

        return ' '.join(tokens)

    def batch_encode(
        self,
        texts: List[str],
        add_special_tokens: bool = True,
        max_length: Optional[int] = None,
        padding: bool = False,
    ) -> Dict[str, List[List[int]]]:
        """Batch encode multiple STEP texts.

        Encodes multiple texts with optional special tokens,
        padding, and truncation.

        Args:
            texts: List of STEP text strings
            add_special_tokens: Add CLS and SEP tokens
            max_length: Maximum sequence length (truncate if exceeded)
            padding: Pad sequences to max_length

        Returns:
            Dictionary with 'token_ids' and optional 'attention_mask'
        """
        all_ids = []

        for text in texts:
            ids = self.encode(text)

            if add_special_tokens:
                ids = [self.CLS_ID] + ids + [self.SEP_ID]

            # Truncate if needed
            if max_length and len(ids) > max_length:
                ids = ids[:max_length]

            # Pad if needed
            if padding and max_length:
                pad_length = max_length - len(ids)
                if pad_length > 0:
                    ids = ids + [self.PAD_ID] * pad_length

            all_ids.append(ids)

        result = {'token_ids': all_ids}

        # Add attention mask if padding is used
        if padding and max_length:
            attention_mask = []
            for ids in all_ids:
                mask = [1 if tid != self.PAD_ID else 0 for tid in ids]
                attention_mask.append(mask)
            result['attention_mask'] = attention_mask

        return result

    def get_vocab_size(self) -> int:
        """Get vocabulary size.

        Returns:
            Number of tokens in vocabulary
        """
        return len(self.vocab)

    def get_special_tokens(self) -> Dict[str, int]:
        """Get special token mapping.

        Returns:
            Dictionary of special tokens and their IDs
        """
        return dict(self.special_tokens)

    def _remove_comments(self, text: str) -> str:
        """Remove C-style comments /* ... */ from text.

        Handles single-line and multiline comments.

        Args:
            text: Text potentially containing comments

        Returns:
            Text with comments removed
        """
        result = []
        i = 0
        while i < len(text):
            if i < len(text) - 1 and text[i:i+2] == '/*':
                # Start of comment, skip until */
                i += 2
                while i < len(text) - 1:
                    if text[i:i+2] == '*/':
                        i += 2
                        break
                    i += 1
            else:
                result.append(text[i])
                i += 1

        return ''.join(result)

    def _normalize_whitespace(self, text: str) -> str:
        """Normalize whitespace outside of string literals.

        Converts newlines, tabs, and multiple spaces to single space
        while preserving whitespace inside strings.

        Args:
            text: Text with arbitrary whitespace

        Returns:
            Text with normalized whitespace
        """
        result = []
        in_string = False
        prev_was_space = False

        for char in text:
            if char == "'":
                in_string = not in_string
                result.append(char)
                prev_was_space = False
            elif in_string:
                result.append(char)
                prev_was_space = False
            elif char in ' \n\t\r':
                if not prev_was_space:
                    result.append(' ')
                    prev_was_space = True
            else:
                result.append(char)
                prev_was_space = False

        return ''.join(result).strip()

    def _collect_multiline_entity(self, lines: List[str], start_idx: int) -> tuple:
        """Collect a complete entity that may span multiple lines.

        Args:
            lines: All lines from DATA section
            start_idx: Index of line starting with #

        Returns:
            Tuple of (complete_entity_string, next_line_index)
        """
        if start_idx >= len(lines):
            return "", start_idx

        entity_parts = []
        current_idx = start_idx
        in_string = False
        found_semicolon = False

        while current_idx < len(lines) and not found_semicolon:
            line = lines[current_idx].strip()

            # Skip empty lines
            if not line:
                current_idx += 1
                continue

            # Remove comments
            line = self._remove_comments(line)

            # Process character by character to find semicolon outside strings
            for char in line:
                if char == "'" and (not entity_parts or not entity_parts[-1] or entity_parts[-1][-1] != '\\'):
                    in_string = not in_string
                elif char == ';' and not in_string:
                    found_semicolon = True
                    break

            entity_parts.append(line)
            current_idx += 1

        complete_entity = ' '.join(entity_parts)
        return complete_entity, current_idx

    def parse_step_file(self, content: str) -> Dict[str, Any]:
        """Parse a complete STEP file.

        Parses both HEADER and DATA sections of a STEP file,
        extracting entity definitions and metadata.

        Args:
            content: Complete STEP file content

        Returns:
            Dictionary containing:
                - header: Dict with header metadata (file_description, file_name, file_schema)
                - entities: Dict mapping entity IDs to entity data
        """
        from typing import Any

        result = {
            "header": {},
            "entities": {}
        }

        # Split into sections
        lines = content.split('\n')
        current_section = None
        header_content = []
        data_content = []

        for line in lines:
            line = line.strip()
            if line.startswith('HEADER;'):
                current_section = 'header'
            elif line.startswith('DATA;'):
                current_section = 'data'
            elif line.startswith('ENDSEC;') or line.startswith('END-ISO'):
                current_section = None
            elif current_section == 'header':
                header_content.append(line)
            elif current_section == 'data':
                data_content.append(line)

        # Parse header
        result["header"] = self._parse_header(header_content)

        # Parse entities from data section
        result["entities"] = self._parse_entities(data_content)

        return result

    def _parse_header(self, header_lines: List[str]) -> Dict[str, Any]:
        """Parse STEP header section.

        Args:
            header_lines: Lines from the HEADER section

        Returns:
            Dictionary with header metadata
        """
        from typing import Any

        header = {
            "file_description": None,
            "file_name": None,
            "file_schema": None
        }

        for line in header_lines:
            if line.startswith('FILE_DESCRIPTION'):
                header["file_description"] = line
            elif line.startswith('FILE_NAME'):
                header["file_name"] = line
            elif line.startswith('FILE_SCHEMA'):
                header["file_schema"] = line

        return header

    def _parse_entities(self, data_lines: List[str]) -> Dict[int, Dict[str, Any]]:
        """Parse entities from STEP data section with multiline support.

        Args:
            data_lines: Lines from the DATA section

        Returns:
            Dictionary mapping entity IDs to entity data
        """
        from typing import Any
        import re

        entities = {}

        # Clear previous parse errors
        self.parse_errors = []
        self.parse_warnings = []

        line_idx = 0

        while line_idx < len(data_lines):
            line = data_lines[line_idx].strip()

            # Skip empty lines and non-entity lines
            if not line or not line.startswith('#'):
                line_idx += 1
                continue

            # Collect complete multiline entity
            complete_entity, next_idx = self._collect_multiline_entity(data_lines, line_idx)
            line_idx = next_idx

            # Parse the complete entity
            # Two patterns:
            # 1. Normal: #123 = ENTITY_TYPE(params);
            # 2. Complex: #123 = ( ... );  (multiple entity types in list)

            # Try normal pattern first
            match = re.match(r'#(\d+)\s*=\s*([A-Z_][A-Z0-9_]*)\s*\((.*)\);?', complete_entity, re.DOTALL)

            if match:
                entity_id = int(match.group(1))
                entity_type = match.group(2)
                params_str = match.group(3)

                # Parse parameters
                params = self._parse_params(params_str)

                entities[entity_id] = {
                    "type": entity_type,
                    "params": params,
                    "raw": complete_entity
                }
            else:
                # Try complex entity pattern: #123 = ( ... );
                match_complex = re.match(r'#(\d+)\s*=\s*\((.*)\);?', complete_entity, re.DOTALL)

                if match_complex:
                    entity_id = int(match_complex.group(1))
                    params_str = match_complex.group(2)

                    # For complex entities, the content is the entity type list
                    entities[entity_id] = {
                        "type": "COMPLEX_ENTITY",
                        "params": params_str.strip(),
                        "raw": complete_entity
                    }
                else:
                    # Log parse error
                    self.parse_errors.append({
                        "line": line_idx,
                        "entity": complete_entity[:100] if len(complete_entity) > 100 else complete_entity,
                        "error": "Failed to parse entity format"
                    })

        return entities

    def _parse_params(self, params_str: str) -> List[Any]:
        """Parse parameter string from entity definition with multiline support.

        Handles nested parentheses, lists, and various parameter types.
        Normalizes whitespace from multiline strings.

        Args:
            params_str: Parameter string (content within parentheses, may contain newlines)

        Returns:
            List of parsed parameters
        """
        from typing import Any

        if not params_str.strip():
            return []

        # Normalize whitespace outside of strings
        params_str = self._normalize_whitespace(params_str)

        params = []
        current_param = []
        depth = 0
        in_string = False

        for char in params_str:
            if char == "'" and (not current_param or current_param[-1] != '\\'):
                in_string = not in_string
                current_param.append(char)
            elif in_string:
                current_param.append(char)
            elif char == '(':
                depth += 1
                current_param.append(char)
            elif char == ')':
                depth -= 1
                current_param.append(char)
            elif char == ',' and depth == 0:
                # End of parameter
                param_str = ''.join(current_param).strip()
                if param_str:
                    params.append(self._parse_single_param(param_str))
                current_param = []
            else:
                current_param.append(char)

        # Add last parameter
        if current_param:
            param_str = ''.join(current_param).strip()
            if param_str:
                params.append(self._parse_single_param(param_str))

        return params

    def _parse_single_param(self, param: str) -> Any:
        """Parse a single parameter value.

        Args:
            param: Parameter string

        Returns:
            Parsed parameter (can be string, number, reference, list, etc.)
        """
        from typing import Any

        param = param.strip()

        # Empty/null
        if not param or param == '$':
            return None

        # String literal
        if param.startswith("'") and param.endswith("'"):
            return param[1:-1]

        # Entity reference
        if param.startswith('#'):
            return param

        # Boolean/enumeration
        if param.startswith('.') and param.endswith('.'):
            return param

        # List (nested parentheses)
        if param.startswith('(') and param.endswith(')'):
            inner = param[1:-1]
            return self._parse_params(inner)

        # Try to parse as number
        try:
            if '.' in param or 'E' in param.upper():
                return param  # Keep as string to preserve precision
            else:
                return param
        except ValueError:
            pass

        # Return as-is
        return param
