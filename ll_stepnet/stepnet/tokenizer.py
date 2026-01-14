"""
STEP Tokenizer Module
Converts STEP text to token IDs following standard tokenizer design.
"""

import re
from typing import List, Dict


class STEPTokenizer:
    """
    Standard tokenizer for STEP files.
    Only handles text → token IDs conversion.
    No feature extraction or graph building.
    """

    def __init__(self, vocab_size: int = 50000):
        """
        Args:
            vocab_size: Maximum vocabulary size
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

    def _build_vocab(self) -> Dict[str, int]:
        """Build STEP vocabulary from common entity types and keywords."""
        vocab = dict(self.special_tokens)
        idx = len(vocab)

        # STEP entity types
        entity_types = [
            'CARTESIAN_POINT', 'DIRECTION', 'VECTOR', 'AXIS2_PLACEMENT_3D',
            'LINE', 'CIRCLE', 'ELLIPSE', 'B_SPLINE_CURVE', 'B_SPLINE_CURVE_WITH_KNOTS',
            'PLANE', 'CYLINDRICAL_SURFACE', 'CONICAL_SURFACE', 'SPHERICAL_SURFACE',
            'VERTEX_POINT', 'EDGE_CURVE', 'ORIENTED_EDGE', 'ADVANCED_FACE',
            'CLOSED_SHELL', 'MANIFOLD_SOLID_BREP',
            'REPRESENTATION_ITEM', 'GEOMETRIC_REPRESENTATION_ITEM',
        ]

        for token in entity_types:
            vocab[token] = idx
            idx += 1

        # Keywords
        keywords = ['.T.', '.F.', '.UNSPECIFIED.', '$', '*']
        for token in keywords:
            vocab[token] = idx
            idx += 1

        # Operators
        operators = ['=', '(', ')', ',', ';', '#']
        for token in operators:
            vocab[token] = idx
            idx += 1

        return vocab

    def tokenize(self, text: str) -> List[str]:
        """
        Split STEP text into tokens.

        Args:
            text: Raw STEP text

        Returns:
            List of token strings
        """
        # Regex: entity refs, identifiers, numbers, keywords, strings, operators
        pattern = r"#\d+|[A-Z_][A-Z0-9_]*|-?\d+\.?\d*(?:[Ee][+-]?\d+)?|\.[A-Z_]+\.|'[^']*'|[=(),;]|\$|\*"
        tokens = re.findall(pattern, text)
        return tokens

    def encode(self, text: str) -> List[int]:
        """
        Encode STEP text to token IDs.

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
                # Hash unknown tokens to vocab space
                token_ids.append(hash(token) % self.vocab_size)

        return token_ids

    def decode(self, token_ids: List[int]) -> str:
        """
        Decode token IDs back to text (approximate).

        Args:
            token_ids: List of token IDs

        Returns:
            Decoded text
        """
        tokens = []
        for tid in token_ids:
            if tid in self.id_to_token:
                tokens.append(self.id_to_token[tid])
            else:
                tokens.append('<UNK>')

        return ' '.join(tokens)

    def batch_encode(self, texts: List[str], add_special_tokens: bool = True) -> Dict[str, List[List[int]]]:
        """
        Batch encode multiple texts.

        Args:
            texts: List of STEP text strings
            add_special_tokens: Add CLS and SEP tokens

        Returns:
            Dictionary with token_ids
        """
        all_ids = []

        for text in texts:
            ids = self.encode(text)

            if add_special_tokens:
                ids = [self.CLS_ID] + ids + [self.SEP_ID]

            all_ids.append(ids)

        return {'token_ids': all_ids}
