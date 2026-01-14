"""ll_stepnet integration layer for STEP parsing.

This module provides integration with ll_stepnet for:
- STEP tokenization
- Feature extraction
- Topology building
- Neural network inference

Classes:
    STEPNetIntegration: Main integration wrapper
"""

from __future__ import annotations

import logging
from typing import Any, Optional

_log = logging.getLogger(__name__)

# Try to import ll_stepnet components
_LL_STEPNET_AVAILABLE = False
try:
    from ll_stepnet import STEPFeatureExtractor, STEPTokenizer, STEPTopologyBuilder

    _LL_STEPNET_AVAILABLE = True
    _log.info("ll_stepnet successfully imported")
except ImportError as e:
    _log.warning(f"ll_stepnet not available: {e}")
    _log.warning("STEP backend will use basic parsing without ll_stepnet features")


class STEPNetIntegration:
    """Integration layer for ll_stepnet.

    Provides access to ll_stepnet components for STEP file processing:
    - Tokenization: Parse STEP text into tokens
    - Feature extraction: Extract geometric features from entities
    - Topology building: Build entity reference graphs

    If ll_stepnet is not available, gracefully degrades to basic functionality.

    Attributes:
        available: Whether ll_stepnet is available
        tokenizer: STEPTokenizer instance (if available)
        feature_extractor: STEPFeatureExtractor instance (if available)
        topology_builder: STEPTopologyBuilder instance (if available)
    """

    def __init__(self):
        """Initialize ll_stepnet integration.

        Raises:
            ImportError: If ll_stepnet is required but not available
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
            _log.debug("ll_stepnet not available, using fallback parsing")

    def tokenize(self, step_text: str) -> Optional[list[int]]:
        """Tokenize STEP text.

        Args:
            step_text: STEP file content

        Returns:
            List of token IDs if ll_stepnet available, None otherwise
        """
        if not self.available:
            _log.debug("Tokenization skipped: ll_stepnet not available")
            return None

        try:
            token_ids = self.tokenizer.encode(step_text)
            _log.debug(f"Tokenized STEP text: {len(token_ids)} tokens")
            return token_ids
        except Exception as e:
            _log.error(f"Tokenization failed: {e}")
            return None

    def extract_features(
        self, entity_text: str, entity_type: str
    ) -> Optional[dict[str, Any]]:
        """Extract geometric features from entity.

        Args:
            entity_text: Entity definition text
            entity_type: Entity type (e.g., "CARTESIAN_POINT")

        Returns:
            Dictionary of extracted features if available, None otherwise
        """
        if not self.available:
            return None

        try:
            features = self.feature_extractor.extract_geometric_features(
                entity_text, entity_type
            )
            _log.debug(f"Extracted features for {entity_type}: {list(features.keys())}")
            return features
        except Exception as e:
            _log.error(f"Feature extraction failed for {entity_type}: {e}")
            return None

    def build_topology(
        self, entities: list[dict[str, Any]]
    ) -> Optional[dict[str, Any]]:
        """Build topology graph from entities.

        Args:
            entities: List of entity dictionaries with IDs and references

        Returns:
            Topology graph data if available, None otherwise
        """
        if not self.available:
            return None

        try:
            topology = self.topology_builder.build_complete_topology(entities)
            _log.debug(
                f"Built topology: {topology.get('num_nodes', 0)} nodes, "
                f"{topology.get('num_edges', 0)} edges"
            )
            return topology
        except Exception as e:
            _log.error(f"Topology building failed: {e}")
            return None

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
