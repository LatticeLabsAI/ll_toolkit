"""
STL Backend - Complete backend for STL file processing.

Implements parsing for both ASCII and binary STL files from scratch,
along with mesh analysis and rendering capabilities.
"""

from __future__ import annotations

import logging
import struct
from io import BytesIO
from pathlib import Path
from typing import List, Optional, Set, Union, Dict, Any, Tuple
import math

from PIL import Image

from cadling.backend.abstract_backend import (
    AbstractCADBackend,
    DeclarativeCADBackend,
    RenderableCADBackend,
    CADViewBackend,
)
from cadling.datamodel.base_models import (
    InputFormat,
    BoundingBox3D,
)
from cadling.datamodel.stl import MeshItem, STLDocument

_log = logging.getLogger(__name__)


class STLBackend(DeclarativeCADBackend, RenderableCADBackend):
    """
    Complete STL backend with parsing and rendering capabilities.

    This backend:
    1. Parses both ASCII and binary STL files from scratch
    2. Extracts mesh data (vertices, normals, facets)
    3. Computes mesh properties (manifold, watertight, volume, surface area)
    4. Optionally renders views using trimesh or matplotlib

    Attributes:
        is_ascii: Whether file is ASCII format
        parsed_mesh: Cached parsed mesh data
        has_trimesh: Whether trimesh is available for rendering
    """

    def __init__(
        self,
        in_doc: "CADInputDocument",
        path_or_stream: Union[Path, str, BytesIO],
        options: Optional["BackendOptions"] = None,
    ):
        """Initialize STL backend."""
        super().__init__(in_doc, path_or_stream, options)

        # Cache for parsed data
        self.is_ascii: Optional[bool] = None
        self.parsed_mesh: Optional[Dict[str, Any]] = None
        self._file_content: Optional[bytes] = None

        # Check for trimesh availability
        self.has_trimesh = False
        try:
            import trimesh

            self.has_trimesh = True
            _log.debug("trimesh available for rendering")
        except ImportError:
            _log.warning(
                "trimesh not available. Rendering capabilities limited. "
                "Install with: pip install trimesh"
            )

    @classmethod
    def supported_formats(cls) -> Set[InputFormat]:
        """STL backend supports STL format."""
        return {InputFormat.STL}

    @classmethod
    def supports_text_parsing(cls) -> bool:
        """STL backend always supports text parsing (ASCII STL)."""
        return True

    @classmethod
    def supports_rendering(cls) -> bool:
        """STL backend supports rendering if trimesh is available."""
        try:
            import trimesh

            return True
        except ImportError:
            return False

    def is_valid(self) -> bool:
        """Validate that file is a valid STL file."""
        try:
            content = self._read_file_content()

            # Check for ASCII STL
            if content.startswith(b"solid"):
                return True

            # Check for binary STL (80-byte header + 4-byte triangle count)
            if len(content) >= 84:
                # Extract triangle count from bytes 80-84
                triangle_count = struct.unpack("<I", content[80:84])[0]
                # Binary STL: 80 header + 4 count + 50 bytes per triangle
                expected_size = 80 + 4 + (triangle_count * 50)
                return len(content) == expected_size

            return False

        except Exception as e:
            _log.error(f"Failed to validate STL file: {e}")
            return False

    def _read_file_content(self) -> bytes:
        """Read file content as bytes, using converter cache when available."""
        if self._file_content is not None:
            return self._file_content

        try:
            # Use content cache from document converter to avoid redundant disk read
            if self.in_doc._content_cache is not None:
                self._file_content = self.in_doc._content_cache
            elif isinstance(self.path_or_stream, BytesIO):
                self._file_content = self.path_or_stream.read()
                self.path_or_stream.seek(0)  # Reset stream
            else:
                path = Path(self.path_or_stream)
                with open(path, "rb") as f:
                    self._file_content = f.read()

            return self._file_content

        except Exception as e:
            _log.error(f"Failed to read STL file: {e}")
            raise

    def _detect_format(self, content: bytes) -> bool:
        """
        Detect if STL is ASCII or binary.

        Returns:
            True if ASCII, False if binary
        """
        # ASCII STL starts with "solid"
        if content.startswith(b"solid"):
            # Sample a prefix to check if content is ASCII text.
            # Full decode would double peak memory for large files.
            sample_size = min(len(content), 1024)
            try:
                content[:sample_size].decode("ascii")
                return True
            except UnicodeDecodeError:
                # Binary file that happens to start with "solid" in header
                return False

        # Binary STL
        return False

    def _parse_ascii_stl(self, content: bytes) -> Dict[str, Any]:
        """Parse ASCII STL format from scratch."""
        _log.debug("Parsing ASCII STL")

        vertices = []
        normals = []
        facets = []

        try:
            text = content.decode("ascii", errors="ignore")
            lines = text.split("\n")

            current_normal = None
            current_vertices = []

            for line in lines:
                line = line.strip().lower()

                if line.startswith("facet normal"):
                    # Extract normal vector
                    parts = line.split()
                    if len(parts) >= 5:
                        normal = [float(parts[2]), float(parts[3]), float(parts[4])]
                        current_normal = normal
                        current_vertices = []

                elif line.startswith("vertex"):
                    # Extract vertex coordinates
                    parts = line.split()
                    if len(parts) >= 4:
                        vertex = [float(parts[1]), float(parts[2]), float(parts[3])]
                        vertices.append(vertex)
                        current_vertices.append(len(vertices) - 1)

                elif line.startswith("endfacet"):
                    # Complete facet
                    if len(current_vertices) == 3 and current_normal:
                        facets.append(current_vertices[:])
                        normals.append(current_normal)
                    current_vertices = []
                    current_normal = None

            _log.info(
                f"Parsed ASCII STL: {len(vertices)} vertices, {len(facets)} facets"
            )

            return {
                "vertices": vertices,
                "normals": normals,
                "facets": facets,
                "is_ascii": True,
            }

        except Exception as e:
            _log.error(f"Failed to parse ASCII STL: {e}")
            raise

    def _parse_binary_stl(self, content: bytes) -> Dict[str, Any]:
        """Parse binary STL format from scratch."""
        _log.debug("Parsing binary STL")

        try:
            # Binary STL format:
            # - 80 bytes: header
            # - 4 bytes: number of triangles (uint32)
            # - For each triangle (50 bytes):
            #   - 12 bytes: normal vector (3 x float32)
            #   - 12 bytes: vertex 1 (3 x float32)
            #   - 12 bytes: vertex 2 (3 x float32)
            #   - 12 bytes: vertex 3 (3 x float32)
            #   - 2 bytes: attribute byte count (uint16)

            if len(content) < 84:
                raise ValueError("Binary STL too short")

            # Read triangle count
            triangle_count = struct.unpack("<I", content[80:84])[0]

            # Verify size
            expected_size = 84 + (triangle_count * 50)
            if len(content) != expected_size:
                _log.warning(
                    f"Binary STL size mismatch: expected {expected_size}, "
                    f"got {len(content)}"
                )

            vertices = []
            normals = []
            facets = []
            vertex_map = {}  # Map (x,y,z) tuple to vertex index

            offset = 84

            for i in range(triangle_count):
                if offset + 50 > len(content):
                    break

                # Read normal (3 floats)
                normal = struct.unpack("<fff", content[offset : offset + 12])
                normals.append(list(normal))
                offset += 12

                # Read 3 vertices
                facet_indices = []
                for j in range(3):
                    vertex = struct.unpack("<fff", content[offset : offset + 12])
                    vertex_tuple = vertex
                    offset += 12

                    # Check if vertex already exists (for deduplication)
                    if vertex_tuple in vertex_map:
                        vertex_idx = vertex_map[vertex_tuple]
                    else:
                        vertex_idx = len(vertices)
                        vertices.append(list(vertex))
                        vertex_map[vertex_tuple] = vertex_idx

                    facet_indices.append(vertex_idx)

                facets.append(facet_indices)

                # Skip attribute bytes
                offset += 2

            _log.info(
                f"Parsed binary STL: {len(vertices)} vertices, {len(facets)} facets"
            )

            return {
                "vertices": vertices,
                "normals": normals,
                "facets": facets,
                "is_ascii": False,
            }

        except Exception as e:
            _log.error(f"Failed to parse binary STL: {e}")
            raise

    def _compute_mesh_properties(self, mesh_data: Dict[str, Any]) -> None:
        """Compute mesh properties like manifold, watertight, volume, surface area.

        Uses numpy vectorized operations for performance on large meshes.
        """
        import numpy as np

        verts = np.asarray(mesh_data["vertices"], dtype=np.float64)
        faces = np.asarray(mesh_data["facets"], dtype=np.int64)

        # Gather triangle vertices: (N, 3) each
        v1 = verts[faces[:, 0]]
        v2 = verts[faces[:, 1]]
        v3 = verts[faces[:, 2]]

        # --- Surface area (vectorized cross product) ---
        edge1 = v2 - v1
        edge2 = v3 - v1
        cross_vecs = np.cross(edge1, edge2)
        tri_areas = 0.5 * np.linalg.norm(cross_vecs, axis=1)
        surface_area = float(tri_areas.sum())
        mesh_data["surface_area"] = surface_area

        # --- Manifold check (each edge shared by exactly 2 faces) ---
        # Build sorted edge pairs, then count occurrences
        e0 = np.sort(faces[:, [0, 1]], axis=1)
        e1 = np.sort(faces[:, [1, 2]], axis=1)
        e2 = np.sort(faces[:, [2, 0]], axis=1)
        all_edges = np.concatenate([e0, e1, e2], axis=0)

        # Use structured array for unique counting
        edge_view = np.ascontiguousarray(all_edges).view(
            np.dtype((np.void, all_edges.dtype.itemsize * 2))
        )
        _, counts = np.unique(edge_view, return_counts=True)

        is_manifold = bool(np.all(counts == 2))
        mesh_data["is_manifold"] = is_manifold

        # Watertight means manifold and no boundary edges
        is_watertight = is_manifold
        mesh_data["is_watertight"] = is_watertight

        # --- Volume (signed tetrahedron method, vectorized) ---
        if is_watertight:
            # V = (1/6) * sum( v1 . (v2 x v3) )
            cross_v2v3 = np.cross(v2, v3)
            signed_volumes = np.einsum("ij,ij->i", v1, cross_v2v3) / 6.0
            mesh_data["volume"] = float(abs(signed_volumes.sum()))
        else:
            mesh_data["volume"] = None

        _log.debug(
            f"Mesh properties: surface_area={surface_area:.2f}, "
            f"manifold={is_manifold}, watertight={is_watertight}, "
            f"volume={mesh_data.get('volume')}"
        )

    def _parse_file(self) -> Dict[str, Any]:
        """Parse STL file."""
        if self.parsed_mesh is not None:
            return self.parsed_mesh

        _log.info(f"Parsing STL file: {self.file.name}")

        # Read content
        content = self._read_file_content()

        # Detect format
        self.is_ascii = self._detect_format(content)

        # Parse based on format
        if self.is_ascii:
            mesh_data = self._parse_ascii_stl(content)
        else:
            mesh_data = self._parse_binary_stl(content)

        # Compute mesh properties
        self._compute_mesh_properties(mesh_data)

        self.parsed_mesh = mesh_data
        return mesh_data

    def convert(self) -> STLDocument:
        """
        Convert STL file to STLDocument.

        Returns:
            Fully populated STLDocument with mesh data and properties.
        """
        # Parse the file
        mesh_data = self._parse_file()

        # Create document
        doc = STLDocument(name=self.file.name, is_ascii=mesh_data["is_ascii"])

        # Create mesh item
        mesh_item = MeshItem(
            label={"text": "STL Mesh"},
            text=f"Mesh with {len(mesh_data['vertices'])} vertices and {len(mesh_data['facets'])} facets",
            vertices=mesh_data["vertices"],
            normals=mesh_data["normals"],
            facets=mesh_data["facets"],
            num_vertices=len(mesh_data["vertices"]),
            num_facets=len(mesh_data["facets"]),
            is_manifold=mesh_data.get("is_manifold"),
            is_watertight=mesh_data.get("is_watertight"),
            volume=mesh_data.get("volume"),
            surface_area=mesh_data.get("surface_area"),
        )

        # Compute bounding box
        bbox = mesh_item.compute_bounding_box()
        mesh_item.bbox = bbox
        doc.bounding_box = bbox

        # Set mesh
        doc.set_mesh(mesh_item)

        # Add metadata
        doc.metadata = {
            "is_ascii": mesh_data["is_ascii"],
            "num_vertices": len(mesh_data["vertices"]),
            "num_facets": len(mesh_data["facets"]),
            "is_manifold": mesh_data.get("is_manifold"),
            "is_watertight": mesh_data.get("is_watertight"),
            "volume": mesh_data.get("volume"),
            "surface_area": mesh_data.get("surface_area"),
        }

        _log.info(f"Converted STL file to document")

        return doc

    def available_views(self) -> List[str]:
        """List available rendering views."""
        if not self.has_trimesh:
            _log.warning("trimesh not available, no views available")
            return []

        return [
            "front",
            "back",
            "top",
            "bottom",
            "right",
            "left",
            "isometric",
            "isometric2",
        ]

    def load_view(self, view_name: str) -> CADViewBackend:
        """Load a specific view for rendering."""
        if not self.has_trimesh:
            raise RuntimeError(
                "trimesh not available. Cannot load views. "
                "Install with: pip install trimesh"
            )

        return STLViewBackend(view_name, self)

    def render_view(self, view_name: str, resolution: int = 1024) -> Image.Image:
        """Render a specific view to image."""
        view_backend = self.load_view(view_name)
        return view_backend.render(resolution=resolution)


class STLViewBackend(CADViewBackend):
    """View backend for rendering specific views of STL meshes."""

    def __init__(self, view_name: str, parent_backend: STLBackend):
        """Initialize STL view backend."""
        super().__init__(view_name, parent_backend)
        self.stl_backend = parent_backend
        self._mesh = None

    def _load_mesh(self):
        """Load mesh using trimesh."""
        if self._mesh is not None:
            return self._mesh

        try:
            import trimesh

            # Parse mesh data
            mesh_data = self.stl_backend._parse_file()

            # Create trimesh object
            self._mesh = trimesh.Trimesh(
                vertices=mesh_data["vertices"], faces=mesh_data["facets"]
            )

            _log.debug(f"Loaded mesh with {len(self._mesh.vertices)} vertices")
            return self._mesh

        except ImportError as e:
            raise RuntimeError(
                f"trimesh not available: {e}. Install with: pip install trimesh"
            )
        except Exception as e:
            _log.error(f"Failed to load mesh: {e}")
            raise

    def render(self, resolution: int = 1024) -> Image.Image:
        """Render this view to an image."""
        try:
            import trimesh
            import numpy as np

            # Load mesh
            mesh = self._load_mesh()

            # Get camera parameters
            camera_params = self.get_camera_parameters()

            # Create scene
            scene = mesh.scene()

            # Set camera transform based on view
            camera_pos = np.array(camera_params["position"])
            camera_dir = np.array(camera_params["direction"])
            camera_up = np.array(camera_params["up"])

            # Apply camera transform using computed parameters
            # Build rotation angles from camera direction
            dx, dy, dz = camera_dir
            elevation = math.degrees(math.atan2(dz, math.sqrt(dx * dx + dy * dy)))
            azimuth = math.degrees(math.atan2(dy, dx))
            scene.set_camera(
                angles=[elevation, azimuth, 0],
                distance=np.linalg.norm(camera_pos),
                center=mesh.centroid,
            )

            # Render to PNG
            png = scene.save_image(resolution=(resolution, resolution))

            # Convert to PIL Image
            from PIL import Image as PILImage

            image = PILImage.open(BytesIO(png))

            return image

        except ImportError as e:
            # NO MORE LIES! Renderer's job is to RENDER, not return gray boxes!
            _log.error("trimesh is required for STL rendering")
            raise ImportError(
                "trimesh is required for STL rendering. "
                "Install with: pip install trimesh. "
                "We will NOT return fake gray placeholder images!"
            ) from e

        except Exception as e:
            # NO MORE LIES! If rendering fails, FAIL LOUDLY!
            _log.error(f"STL rendering failed for view {self.view_name}: {e}")
            raise RuntimeError(
                f"STL rendering failed for view {self.view_name}: {e}. "
                f"We will NOT return fake gray placeholder images!"
            ) from e

    def get_camera_parameters(self) -> dict:
        """Get camera parameters for this view."""
        from cadling.backend.abstract_backend import DEFAULT_CAMERA_PARAMETERS

        return DEFAULT_CAMERA_PARAMETERS.get(
            self.view_name, DEFAULT_CAMERA_PARAMETERS["front"]
        )
