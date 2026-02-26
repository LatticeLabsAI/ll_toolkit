"""STL format data models.

This module provides data models specific to STL (STereoLithography) files,
including mesh representations and validation.

Classes:
    MeshItem: Mesh data (vertices, facets, normals)
    STLDocument: STL-specific document structure
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from pydantic import Field, field_validator

from cadling.datamodel.base_models import (
    BoundingBox3D,
    CADItem,
    CADItemLabel,
    CADlingDocument,
    InputFormat,
)


class STLFacet(CADItem):
    """Individual STL facet (triangle) with vertices and normal.

    Represents a single triangular facet from an STL file with
    its normal vector and three vertex coordinates.

    Attributes:
        item_type: Always "stl_facet"
        normal: Normal vector (nx, ny, nz)
        v1: First vertex coordinates (x, y, z)
        v2: Second vertex coordinates (x, y, z)
        v3: Third vertex coordinates (x, y, z)
        facet_index: Index of this facet in the original mesh
    """

    item_type: str = "stl_facet"

    normal: tuple[float, float, float] = (0.0, 0.0, 1.0)
    v1: tuple[float, float, float] = (0.0, 0.0, 0.0)
    v2: tuple[float, float, float] = (0.0, 0.0, 0.0)
    v3: tuple[float, float, float] = (0.0, 0.0, 0.0)
    facet_index: int = 0

    def compute_area(self) -> float:
        """Compute area of this triangle facet.

        Returns:
            Area of the triangular facet
        """
        import numpy as np

        a = np.array(self.v1)
        b = np.array(self.v2)
        c = np.array(self.v3)

        return 0.5 * np.linalg.norm(np.cross(b - a, c - a))

    def compute_centroid(self) -> tuple[float, float, float]:
        """Compute centroid of this triangle facet.

        Returns:
            Centroid coordinates (x, y, z)
        """
        cx = (self.v1[0] + self.v2[0] + self.v3[0]) / 3.0
        cy = (self.v1[1] + self.v2[1] + self.v3[1]) / 3.0
        cz = (self.v1[2] + self.v2[2] + self.v3[2]) / 3.0
        return (cx, cy, cz)


class MeshItem(CADItem):
    """Mesh data item.

    Represents mesh data from STL, OBJ, or other mesh formats.
    Contains vertices, normals, and facets (triangles).

    Attributes:
        item_type: Always "mesh"
        vertices: List of vertex coordinates [[x,y,z], ...]
        normals: List of normal vectors [[nx,ny,nz], ...]
        facets: List of facet indices [[v1,v2,v3], ...]
        num_vertices: Number of vertices
        num_facets: Number of facets
        is_manifold: Whether mesh is manifold
        is_watertight: Whether mesh is watertight (closed)
        volume: Mesh volume (if watertight)
        surface_area: Total surface area
    """

    item_type: str = "mesh"

    # Mesh data
    vertices: List[List[float]] = Field(default_factory=list)
    normals: List[List[float]] = Field(default_factory=list)
    facets: List[List[int]] = Field(default_factory=list)

    @property
    def num_vertices(self) -> int:
        """Number of vertices derived from vertices list."""
        return len(self.vertices)

    @property
    def num_facets(self) -> int:
        """Number of facets derived from facets list."""
        return len(self.facets)
    is_manifold: Optional[bool] = None
    is_watertight: Optional[bool] = None
    volume: Optional[float] = None
    surface_area: Optional[float] = None

    @field_validator("vertices")
    @classmethod
    def validate_vertices(cls, v):
        """Validate vertices are 3D points."""
        for vertex in v:
            if len(vertex) != 3:
                raise ValueError(f"Vertex must have 3 coordinates, got {len(vertex)}")
        return v

    @field_validator("normals")
    @classmethod
    def validate_normals(cls, v):
        """Validate normals are 3D vectors."""
        for normal in v:
            if len(normal) != 3:
                raise ValueError(f"Normal must have 3 components, got {len(normal)}")
        return v

    @field_validator("facets")
    @classmethod
    def validate_facets(cls, v):
        """Validate facets are triangles."""
        for facet in v:
            if len(facet) != 3:
                raise ValueError(f"Facet must have 3 vertices, got {len(facet)}")
        return v

    def compute_bounding_box(self) -> BoundingBox3D:
        """Compute 3D bounding box from vertices.

        Returns:
            BoundingBox3D enclosing all vertices
        """
        if not self.vertices:
            return BoundingBox3D(
                x_min=0, y_min=0, z_min=0,
                x_max=0, y_max=0, z_max=0
            )

        xs = [v[0] for v in self.vertices]
        ys = [v[1] for v in self.vertices]
        zs = [v[2] for v in self.vertices]

        return BoundingBox3D(
            x_min=min(xs), y_min=min(ys), z_min=min(zs),
            x_max=max(xs), y_max=max(ys), z_max=max(zs)
        )

    def to_numpy(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return (vertices, faces) as numpy arrays for GeoTokenizer.

        Converts the internal list-of-lists storage to numpy arrays
        matching the format expected by geotoken.GeoTokenizer.tokenize().

        Returns:
            Tuple of:
                - vertices: np.ndarray of shape (N, 3), dtype float32
                - faces: np.ndarray of shape (F, 3), dtype int64
        """
        if not self.vertices:
            return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.int64)
        vertices = np.array(self.vertices, dtype=np.float32)
        faces = np.array(self.facets, dtype=np.int64) if self.facets else np.zeros((0, 3), dtype=np.int64)
        return vertices, faces

    def to_numpy_vertices(self) -> np.ndarray:
        """Return vertices only as (N, 3) float32 array.

        Returns:
            np.ndarray of shape (N, 3), dtype float32
        """
        if not self.vertices:
            return np.zeros((0, 3), dtype=np.float32)
        return np.array(self.vertices, dtype=np.float32)

    def to_numpy_normals(self) -> np.ndarray:
        """Return face normals as (F, 3) float32 array.

        Returns:
            np.ndarray of shape (F, 3), dtype float32
        """
        if not self.normals:
            return np.zeros((0, 3), dtype=np.float32)
        return np.array(self.normals, dtype=np.float32)


class STLDocument(CADlingDocument):
    """STL-specific document structure.

    Extends CADlingDocument with STL-specific fields like ASCII/binary
    detection and mesh data.

    Attributes:
        format: Always InputFormat.STL
        is_ascii: Whether file is ASCII STL (vs binary)
        mesh: Primary mesh item
        metadata: Additional metadata dictionary
    """

    format: InputFormat = InputFormat.STL

    # STL-specific fields
    is_ascii: bool = False
    mesh: Optional[MeshItem] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)

    def set_mesh(self, mesh: MeshItem):
        """Set the primary mesh.

        Args:
            mesh: MeshItem to set
        """
        # Remove old mesh from items if present
        if self.mesh is not None:
            self.items = [item for item in self.items if item is not self.mesh]
        self.mesh = mesh
        self.add_item(mesh)

    def export_to_json(self) -> Dict[str, Any]:
        """Export STL document to JSON format.

        Returns:
            Dictionary representation with STL-specific fields.
        """
        # Get base JSON from parent class
        json_data = super().export_to_json()

        # Add STL-specific fields
        json_data["is_ascii"] = self.is_ascii

        # Add mesh data if available
        if self.mesh:
            json_data["mesh"] = {
                "num_vertices": self.mesh.num_vertices,
                "num_facets": self.mesh.num_facets,
                "is_manifold": self.mesh.is_manifold,
                "is_watertight": self.mesh.is_watertight,
                "volume": self.mesh.volume,
                "surface_area": self.mesh.surface_area,
                "bounding_box": self.mesh.bbox.model_dump(mode='json') if self.mesh.bbox else None,
            }

        # Add bounding box
        if self.bounding_box:
            json_data["bounding_box"] = self.bounding_box.model_dump(mode='json')

        # Add metadata
        json_data["metadata"] = self.metadata

        return json_data


class AssemblyItem(CADItem):
    """Assembly node for multi-part CAD.

    Represents a hierarchical assembly with multiple components,
    each potentially having transforms applied.

    Attributes:
        item_type: Always "assembly"
        components: List of child item IDs
        transform_matrix: 4x4 transformation matrix
        component_names: Names of components
    """

    item_type: str = "assembly"

    components: List[str] = Field(default_factory=list)
    transform_matrix: Optional[List[List[float]]] = None
    component_names: List[str] = Field(default_factory=list)

    def add_component(self, component_id: str, name: Optional[str] = None):
        """Add a component to assembly.

        Args:
            component_id: ID of component item
            name: Optional name for component
        """
        self.components.append(component_id)
        if name:
            self.component_names.append(name)


class AnnotationItem(CADItem):
    """Annotation from rendered CAD (dimensions, tolerances, notes).

    Extracted using vision models from rendered CAD images.
    Contains the annotation value and position in the source image.

    Attributes:
        item_type: Always "annotation"
        annotation_type: Type ("dimension", "tolerance", "note", "label")
        value: Annotation text/value
        image_bbox: 2D bounding box in source image
        source_view: View name where annotation was found
        confidence: Confidence score from vision model
        unit: Unit for dimensions (mm, inch, etc.)
    """

    item_type: str = "annotation"

    annotation_type: str  # "dimension", "tolerance", "note", "label"
    value: Optional[str] = None

    # Image provenance (2D bbox in rendered image)
    image_bbox: Optional[List[float]] = None  # [x_min, y_min, x_max, y_max]
    source_view: Optional[str] = None  # "front", "top", etc.
    confidence: Optional[float] = None

    # Dimension-specific
    unit: Optional[str] = None  # "mm", "inch", etc.
