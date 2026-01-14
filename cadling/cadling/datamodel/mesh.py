"""Generic mesh data models.

This module provides generic mesh data models that can be used across
different formats (STL, OBJ, PLY, etc.). For format-specific models,
see the individual format modules (stl.py, etc.).

Note: MeshItem is already defined in stl.py and used there. This module
provides additional mesh-related utilities and advanced mesh types.

Classes:
    TriangleMesh: Triangle mesh with additional geometric properties
    QuadMesh: Quadrilateral mesh
    PointCloud: Point cloud data
    MeshMetadata: Metadata about mesh quality and properties
"""

from __future__ import annotations

from typing import List, Optional, Tuple

from pydantic import BaseModel, Field, field_validator
import numpy as np


class MeshMetadata(BaseModel):
    """Metadata about mesh quality and properties.

    Attributes:
        num_vertices: Number of vertices
        num_faces: Number of faces
        num_edges: Number of edges
        is_manifold: Whether mesh is manifold
        is_watertight: Whether mesh is watertight (closed)
        has_duplicates: Whether mesh has duplicate vertices
        has_degenerate_faces: Whether mesh has degenerate (zero-area) faces
        euler_characteristic: Euler characteristic (V - E + F)
        genus: Topological genus
        volume: Mesh volume (if watertight)
        surface_area: Total surface area
        edge_length_min: Minimum edge length
        edge_length_max: Maximum edge length
        edge_length_mean: Mean edge length
    """

    num_vertices: int
    num_faces: int
    num_edges: int
    is_manifold: Optional[bool] = None
    is_watertight: Optional[bool] = None
    has_duplicates: bool = False
    has_degenerate_faces: bool = False
    euler_characteristic: Optional[int] = None
    genus: Optional[int] = None
    volume: Optional[float] = None
    surface_area: Optional[float] = None
    edge_length_min: Optional[float] = None
    edge_length_max: Optional[float] = None
    edge_length_mean: Optional[float] = None

    model_config = {"arbitrary_types_allowed": True}


class TriangleMesh(BaseModel):
    """Triangle mesh with additional geometric properties.

    This extends the basic MeshItem from stl.py with additional
    computational geometry features.

    Attributes:
        vertices: List of vertex coordinates [[x,y,z], ...]
        faces: List of face indices [[v1,v2,v3], ...]
        normals: List of normal vectors (per-face or per-vertex)
        colors: Optional vertex colors [[r,g,b], ...]
        texture_coords: Optional texture coordinates [[u,v], ...]
        metadata: Mesh quality metadata
    """

    vertices: List[List[float]] = Field(default_factory=list)
    faces: List[List[int]] = Field(default_factory=list)
    normals: List[List[float]] = Field(default_factory=list)
    colors: Optional[List[List[float]]] = None
    texture_coords: Optional[List[List[float]]] = None
    metadata: Optional[MeshMetadata] = None

    model_config = {"arbitrary_types_allowed": True}

    @field_validator("vertices")
    @classmethod
    def validate_vertices(cls, v):
        """Validate vertices are 3D points."""
        for vertex in v:
            if len(vertex) != 3:
                raise ValueError(f"Vertex must have 3 coordinates, got {len(vertex)}")
        return v

    @field_validator("faces")
    @classmethod
    def validate_faces(cls, v):
        """Validate faces are triangles."""
        for face in v:
            if len(face) != 3:
                raise ValueError(f"Face must have 3 vertices, got {len(face)}")
        return v

    def compute_face_normals(self) -> List[List[float]]:
        """Compute per-face normals.

        Returns:
            List of normal vectors, one per face
        """
        normals = []
        for face in self.faces:
            v0 = self.vertices[face[0]]
            v1 = self.vertices[face[1]]
            v2 = self.vertices[face[2]]

            # Compute edge vectors
            e1 = [v1[i] - v0[i] for i in range(3)]
            e2 = [v2[i] - v0[i] for i in range(3)]

            # Cross product
            normal = [
                e1[1] * e2[2] - e1[2] * e2[1],
                e1[2] * e2[0] - e1[0] * e2[2],
                e1[0] * e2[1] - e1[1] * e2[0],
            ]

            # Normalize
            length = sum(n * n for n in normal) ** 0.5
            if length > 0:
                normal = [n / length for n in normal]

            normals.append(normal)

        return normals

    def compute_bounding_box(self) -> Tuple[List[float], List[float]]:
        """Compute axis-aligned bounding box.

        Returns:
            Tuple of (min_coords, max_coords)
        """
        if not self.vertices:
            return ([0, 0, 0], [0, 0, 0])

        min_coords = [float('inf')] * 3
        max_coords = [float('-inf')] * 3

        for vertex in self.vertices:
            for i in range(3):
                min_coords[i] = min(min_coords[i], vertex[i])
                max_coords[i] = max(max_coords[i], vertex[i])

        return (min_coords, max_coords)


class QuadMesh(BaseModel):
    """Quadrilateral mesh.

    Similar to TriangleMesh but with quad faces instead of triangles.
    Common in CAD modeling and subdivision surfaces.

    Attributes:
        vertices: List of vertex coordinates [[x,y,z], ...]
        quads: List of quad indices [[v1,v2,v3,v4], ...]
        normals: List of normal vectors (per-face or per-vertex)
        metadata: Mesh quality metadata
    """

    vertices: List[List[float]] = Field(default_factory=list)
    quads: List[List[int]] = Field(default_factory=list)
    normals: List[List[float]] = Field(default_factory=list)
    metadata: Optional[MeshMetadata] = None

    model_config = {"arbitrary_types_allowed": True}

    @field_validator("quads")
    @classmethod
    def validate_quads(cls, v):
        """Validate quads have 4 vertices."""
        for quad in v:
            if len(quad) != 4:
                raise ValueError(f"Quad must have 4 vertices, got {len(quad)}")
        return v

    def triangulate(self) -> TriangleMesh:
        """Convert quad mesh to triangle mesh.

        Returns:
            TriangleMesh with quads split into triangles
        """
        triangle_faces = []
        for quad in self.quads:
            # Split quad into two triangles
            triangle_faces.append([quad[0], quad[1], quad[2]])
            triangle_faces.append([quad[0], quad[2], quad[3]])

        return TriangleMesh(
            vertices=self.vertices,
            faces=triangle_faces,
            normals=self.normals,
        )


class PointCloud(BaseModel):
    """Point cloud data.

    Represents unstructured 3D points, optionally with colors and normals.

    Attributes:
        points: List of point coordinates [[x,y,z], ...]
        colors: Optional point colors [[r,g,b], ...]
        normals: Optional point normals [[nx,ny,nz], ...]
        intensities: Optional intensity values
        num_points: Number of points
    """

    points: List[List[float]] = Field(default_factory=list)
    colors: Optional[List[List[float]]] = None
    normals: Optional[List[List[float]]] = None
    intensities: Optional[List[float]] = None
    num_points: int = 0

    model_config = {"arbitrary_types_allowed": True}

    @field_validator("points")
    @classmethod
    def validate_points(cls, v):
        """Validate points are 3D."""
        for point in v:
            if len(point) != 3:
                raise ValueError(f"Point must have 3 coordinates, got {len(point)}")
        return v

    def compute_bounding_box(self) -> Tuple[List[float], List[float]]:
        """Compute axis-aligned bounding box.

        Returns:
            Tuple of (min_coords, max_coords)
        """
        if not self.points:
            return ([0, 0, 0], [0, 0, 0])

        min_coords = [float('inf')] * 3
        max_coords = [float('-inf')] * 3

        for point in self.points:
            for i in range(3):
                min_coords[i] = min(min_coords[i], point[i])
                max_coords[i] = max(max_coords[i], point[i])

        return (min_coords, max_coords)

    def downsample(self, voxel_size: float) -> PointCloud:
        """Downsample point cloud using voxel grid.

        Args:
            voxel_size: Size of voxel grid

        Returns:
            Downsampled PointCloud
        """
        if not self.points:
            return self

        # Simple voxel grid downsampling
        voxel_dict = {}
        for i, point in enumerate(self.points):
            # Compute voxel coordinates
            voxel_coords = tuple(int(p / voxel_size) for p in point)

            if voxel_coords not in voxel_dict:
                voxel_dict[voxel_coords] = {
                    'points': [],
                    'colors': [] if self.colors else None,
                    'normals': [] if self.normals else None,
                }

            voxel_dict[voxel_coords]['points'].append(point)
            if self.colors:
                voxel_dict[voxel_coords]['colors'].append(self.colors[i])
            if self.normals:
                voxel_dict[voxel_coords]['normals'].append(self.normals[i])

        # Average points in each voxel
        downsampled_points = []
        downsampled_colors = [] if self.colors else None
        downsampled_normals = [] if self.normals else None

        for voxel_data in voxel_dict.values():
            # Average position
            avg_point = [
                sum(p[i] for p in voxel_data['points']) / len(voxel_data['points'])
                for i in range(3)
            ]
            downsampled_points.append(avg_point)

            # Average color
            if voxel_data['colors']:
                avg_color = [
                    sum(c[i] for c in voxel_data['colors']) / len(voxel_data['colors'])
                    for i in range(3)
                ]
                downsampled_colors.append(avg_color)

            # Average normal (and renormalize)
            if voxel_data['normals']:
                avg_normal = [
                    sum(n[i] for n in voxel_data['normals']) / len(voxel_data['normals'])
                    for i in range(3)
                ]
                length = sum(n * n for n in avg_normal) ** 0.5
                if length > 0:
                    avg_normal = [n / length for n in avg_normal]
                downsampled_normals.append(avg_normal)

        return PointCloud(
            points=downsampled_points,
            colors=downsampled_colors,
            normals=downsampled_normals,
            num_points=len(downsampled_points),
        )
