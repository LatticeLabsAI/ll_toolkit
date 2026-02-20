"""Topology utilities for B-Rep processing.

This package provides topology analysis tools for CAD models, including
stable shape identity tracking and coedge extraction.

Classes:
    ShapeIdentityRegistry: Stable shape identity using HashCode with fallback
    CoedgeExtractor: Extract ordered coedge sequences from OCC B-Rep
    Coedge: Data class representing a coedge with mate pointers
"""

from cadling.lib.topology.face_identity import ShapeIdentityRegistry
from cadling.lib.topology.coedge_extractor import CoedgeExtractor, Coedge

__all__ = ["ShapeIdentityRegistry", "CoedgeExtractor", "Coedge"]
