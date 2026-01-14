"""BRep (Boundary Representation) format data models.

This module provides data models specific to BRep files, which represent
solid geometry through boundaries (faces, edges, vertices).

Classes:
    BRepFaceItem: Individual face in BRep model
    BRepEdgeItem: Individual edge in BRep model
    BRepVertexItem: Individual vertex in BRep model
    BRepDocument: BRep-specific document structure
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import Field

from cadling.datamodel.base_models import (
    BoundingBox3D,
    CADItem,
    CADItemLabel,
    CADlingDocument,
    InputFormat,
)


class BRepVertexItem(CADItem):
    """BRep vertex item.

    Represents a topological vertex in the boundary representation,
    which is a 0D point in space.

    Attributes:
        item_type: Always "brep_vertex"
        vertex_id: Unique vertex identifier
        coordinates: 3D coordinates [x, y, z]
        tolerance: Geometric tolerance
    """

    item_type: str = "brep_vertex"

    vertex_id: int
    coordinates: List[float] = Field(default_factory=list)
    tolerance: Optional[float] = None


class BRepEdgeItem(CADItem):
    """BRep edge item.

    Represents a topological edge in the boundary representation,
    which is a 1D curve connecting two vertices.

    Attributes:
        item_type: Always "brep_edge"
        edge_id: Unique edge identifier
        curve_type: Type of curve ("line", "circle", "ellipse", "bspline", etc.)
        start_vertex_id: ID of start vertex
        end_vertex_id: ID of end vertex
        length: Edge length
        is_degenerate: Whether edge is degenerate (zero length)
    """

    item_type: str = "brep_edge"

    edge_id: int
    curve_type: str
    start_vertex_id: Optional[int] = None
    end_vertex_id: Optional[int] = None
    length: Optional[float] = None
    is_degenerate: bool = False


class BRepFaceItem(CADItem):
    """BRep face item.

    Represents a topological face in the boundary representation,
    which is a 2D surface bounded by edges.

    Attributes:
        item_type: Always "brep_face"
        face_id: Unique face identifier
        surface_type: Type of surface ("plane", "cylinder", "sphere", "bspline", etc.)
        edge_ids: List of bounding edge IDs
        area: Surface area
        orientation: Face orientation ("forward" or "reversed")
        is_planar: Whether face is planar
    """

    item_type: str = "brep_face"

    face_id: int
    surface_type: str
    edge_ids: List[int] = Field(default_factory=list)
    area: Optional[float] = None
    orientation: Optional[str] = None
    is_planar: bool = False


class BRepSolidItem(CADItem):
    """BRep solid item.

    Represents a complete solid bounded by faces.

    Attributes:
        item_type: Always "brep_solid"
        solid_id: Unique solid identifier
        face_ids: List of bounding face IDs
        volume: Solid volume
        surface_area: Total surface area
        num_shells: Number of shells
        is_closed: Whether solid is closed (watertight)
    """

    item_type: str = "brep_solid"

    solid_id: int
    face_ids: List[int] = Field(default_factory=list)
    volume: Optional[float] = None
    surface_area: Optional[float] = None
    num_shells: int = 1
    is_closed: bool = False


class BRepDocument(CADlingDocument):
    """BRep-specific document structure.

    Extends CADlingDocument with BRep-specific fields for managing
    the topological hierarchy (solids -> faces -> edges -> vertices).

    Attributes:
        format: Always InputFormat.BREP
        vertex_index: Dict mapping vertex ID to BRepVertexItem
        edge_index: Dict mapping edge ID to BRepEdgeItem
        face_index: Dict mapping face ID to BRepFaceItem
        solid_index: Dict mapping solid ID to BRepSolidItem
        num_solids: Number of solids in model
        num_faces: Number of faces in model
        num_edges: Number of edges in model
        num_vertices: Number of vertices in model
    """

    format: InputFormat = InputFormat.BREP

    # BRep-specific indices for fast lookups
    vertex_index: Dict[int, BRepVertexItem] = Field(default_factory=dict)
    edge_index: Dict[int, BRepEdgeItem] = Field(default_factory=dict)
    face_index: Dict[int, BRepFaceItem] = Field(default_factory=dict)
    solid_index: Dict[int, BRepSolidItem] = Field(default_factory=dict)

    # Counts
    num_solids: int = 0
    num_faces: int = 0
    num_edges: int = 0
    num_vertices: int = 0

    def add_vertex(self, vertex: BRepVertexItem):
        """Add a vertex to the document.

        Args:
            vertex: BRepVertexItem to add
        """
        self.add_item(vertex)
        self.vertex_index[vertex.vertex_id] = vertex
        self.num_vertices += 1

    def add_edge(self, edge: BRepEdgeItem):
        """Add an edge to the document.

        Args:
            edge: BRepEdgeItem to add
        """
        self.add_item(edge)
        self.edge_index[edge.edge_id] = edge
        self.num_edges += 1

    def add_face(self, face: BRepFaceItem):
        """Add a face to the document.

        Args:
            face: BRepFaceItem to add
        """
        self.add_item(face)
        self.face_index[face.face_id] = face
        self.num_faces += 1

    def add_solid(self, solid: BRepSolidItem):
        """Add a solid to the document.

        Args:
            solid: BRepSolidItem to add
        """
        self.add_item(solid)
        self.solid_index[solid.solid_id] = solid
        self.num_solids += 1

    def get_vertex(self, vertex_id: int) -> Optional[BRepVertexItem]:
        """Get vertex by ID.

        Args:
            vertex_id: Vertex ID

        Returns:
            BRepVertexItem if found, None otherwise
        """
        return self.vertex_index.get(vertex_id)

    def get_edge(self, edge_id: int) -> Optional[BRepEdgeItem]:
        """Get edge by ID.

        Args:
            edge_id: Edge ID

        Returns:
            BRepEdgeItem if found, None otherwise
        """
        return self.edge_index.get(edge_id)

    def get_face(self, face_id: int) -> Optional[BRepFaceItem]:
        """Get face by ID.

        Args:
            face_id: Face ID

        Returns:
            BRepFaceItem if found, None otherwise
        """
        return self.face_index.get(face_id)

    def get_solid(self, solid_id: int) -> Optional[BRepSolidItem]:
        """Get solid by ID.

        Args:
            solid_id: Solid ID

        Returns:
            BRepSolidItem if found, None otherwise
        """
        return self.solid_index.get(solid_id)
