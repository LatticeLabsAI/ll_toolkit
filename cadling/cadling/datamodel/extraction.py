"""Data extraction and feature extraction models.

This module provides data models for feature extraction results,
including geometric features, topological features, and semantic features.

Classes:
    GeometricFeatures: Geometric properties extracted from CAD
    TopologicalFeatures: Topological properties
    SemanticFeatures: Semantic/high-level features
    ExtractionResult: Complete extraction result
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field


class GeometricFeatures(BaseModel):
    """Geometric features extracted from CAD geometry.

    Attributes:
        bounding_box: 3D bounding box [x_min, y_min, z_min, x_max, y_max, z_max]
        centroid: Center of mass [x, y, z]
        volume: Solid volume (if applicable)
        surface_area: Total surface area
        perimeter: Perimeter (for 2D profiles)
        principal_axes: Principal axes of inertia
        moments_of_inertia: Moments of inertia
        radii: Radii for cylindrical/spherical features
        angles: Angles for conical/angular features
        coordinates: Point coordinates
        dimensions: Overall dimensions [length, width, height]
    """

    bounding_box: Optional[List[float]] = None
    centroid: Optional[List[float]] = None
    volume: Optional[float] = None
    surface_area: Optional[float] = None
    perimeter: Optional[float] = None
    principal_axes: Optional[List[List[float]]] = None
    moments_of_inertia: Optional[List[float]] = None
    radii: List[float] = Field(default_factory=list)
    angles: List[float] = Field(default_factory=list)
    coordinates: List[List[float]] = Field(default_factory=list)
    dimensions: Optional[List[float]] = None


class TopologicalFeatures(BaseModel):
    """Topological features extracted from CAD topology.

    Attributes:
        num_vertices: Number of vertices
        num_edges: Number of edges
        num_faces: Number of faces
        num_solids: Number of solids
        euler_characteristic: Euler characteristic (V - E + F)
        genus: Topological genus
        num_components: Number of connected components
        num_holes: Number of holes
        is_manifold: Whether geometry is manifold
        is_orientable: Whether geometry is orientable
        adjacency_graph: Adjacency relationships
        entity_references: Entity reference count
    """

    num_vertices: int = 0
    num_edges: int = 0
    num_faces: int = 0
    num_solids: int = 0
    euler_characteristic: Optional[int] = None
    genus: Optional[int] = None
    num_components: int = 0
    num_holes: int = 0
    is_manifold: Optional[bool] = None
    is_orientable: Optional[bool] = None
    adjacency_graph: Optional[Dict[str, List[str]]] = None
    entity_references: int = 0


class SemanticFeatures(BaseModel):
    """Semantic/high-level features extracted from CAD.

    Attributes:
        entity_types: Distribution of entity types
        feature_types: Identified machining features (holes, pockets, etc.)
        symmetries: Detected symmetries (planar, rotational, etc.)
        part_category: Part category (bracket, housing, shaft, etc.)
        complexity_score: Geometric complexity score (0-1)
        manufacturability_score: Manufacturability score (0-1)
        annotations: Extracted annotations (dimensions, tolerances)
        materials: Material information
        finish: Surface finish requirements
    """

    entity_types: Dict[str, int] = Field(default_factory=dict)
    feature_types: List[str] = Field(default_factory=list)
    symmetries: List[str] = Field(default_factory=list)
    part_category: Optional[str] = None
    complexity_score: Optional[float] = None
    manufacturability_score: Optional[float] = None
    annotations: List[Dict[str, Any]] = Field(default_factory=list)
    materials: List[str] = Field(default_factory=list)
    finish: List[str] = Field(default_factory=list)


class ExtractionResult(BaseModel):
    """Complete feature extraction result.

    Combines geometric, topological, and semantic features into a single
    comprehensive result object.

    Attributes:
        geometric: Geometric features
        topological: Topological features
        semantic: Semantic features
        metadata: Additional metadata
        extraction_time_ms: Time taken for extraction (milliseconds)
        extractor_version: Version of extraction algorithms
    """

    geometric: GeometricFeatures = Field(default_factory=GeometricFeatures)
    topological: TopologicalFeatures = Field(default_factory=TopologicalFeatures)
    semantic: SemanticFeatures = Field(default_factory=SemanticFeatures)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    extraction_time_ms: Optional[float] = None
    extractor_version: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation.

        Returns:
            Dictionary representation of extraction result
        """
        return {
            "geometric": self.geometric.model_dump(),
            "topological": self.topological.model_dump(),
            "semantic": self.semantic.model_dump(),
            "metadata": self.metadata,
            "extraction_time_ms": self.extraction_time_ms,
            "extractor_version": self.extractor_version,
        }

    def get_feature_vector(self) -> List[float]:
        """Get a fixed-size feature vector for ML models.

        Returns:
            Feature vector combining key numeric features
        """
        features = []

        # Geometric features
        if self.geometric.volume is not None:
            features.append(self.geometric.volume)
        else:
            features.append(0.0)

        if self.geometric.surface_area is not None:
            features.append(self.geometric.surface_area)
        else:
            features.append(0.0)

        # Topological features
        features.extend([
            float(self.topological.num_vertices),
            float(self.topological.num_edges),
            float(self.topological.num_faces),
            float(self.topological.num_solids),
        ])

        # Semantic features
        if self.semantic.complexity_score is not None:
            features.append(self.semantic.complexity_score)
        else:
            features.append(0.0)

        if self.semantic.manufacturability_score is not None:
            features.append(self.semantic.manufacturability_score)
        else:
            features.append(0.0)

        return features
