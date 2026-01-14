"""STL format data models.

This module provides data models specific to STL (STereoLithography) files,
including mesh representations and validation.

Classes:
    MeshItem: Mesh data (vertices, facets, normals)
    STLDocument: STL-specific document structure
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import Field, field_validator

from cadling.datamodel.base_models import (
    BoundingBox3D,
    CADItem,
    CADItemLabel,
    CADlingDocument,
    InputFormat,
)


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

    # Computed properties
    num_vertices: int = 0
    num_facets: int = 0
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
