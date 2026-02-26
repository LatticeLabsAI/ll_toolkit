"""Generic mesh backend for OBJ, PLY, OFF, and other mesh formats.

Handles loading, validation, conversion to CADlingDocument, and rendering
of generic mesh files via trimesh. For STL files, use the dedicated
STLBackend which has native ASCII/binary parsing.

Classes:
    MeshBackend: Backend for generic mesh formats
    MeshViewBackend: View backend for rendering mesh views
"""

from __future__ import annotations

import logging
import math
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import numpy as np
from PIL import Image

from cadling.backend.abstract_backend import (
    CADViewBackend,
    DeclarativeCADBackend,
    RenderableCADBackend,
)
from cadling.datamodel.base_models import (
    BoundingBox3D,
    CADDocumentOrigin,
    CADlingDocument,
    InputFormat,
)
from cadling.datamodel.stl import MeshItem, STLDocument

_log = logging.getLogger(__name__)

# Mesh file extensions we handle (not STL — that has its own backend)
_MESH_EXTENSIONS = {".obj", ".ply", ".off", ".glb", ".gltf", ".3mf", ".dae"}


def _load_trimesh():
    """Load trimesh, raising ImportError with install instructions if missing."""
    try:
        import trimesh
        return trimesh
    except ImportError:
        raise ImportError(
            "trimesh is required for MeshBackend. Install with: pip install trimesh"
        )


class MeshBackend(DeclarativeCADBackend, RenderableCADBackend):
    """Backend for generic mesh formats (OBJ, PLY, OFF, GLB, etc.).

    Uses trimesh for loading and processing mesh files. Provides:
    1. Mesh loading and validation
    2. Conversion to CADlingDocument with MeshItem data
    3. Mesh property computation (watertight, volume, surface area)
    4. Multi-view rendering via trimesh's scene renderer

    Attributes:
        _trimesh_mesh: Loaded trimesh.Trimesh object
        _file_path: Path to the mesh file
        _mesh_properties: Cached mesh analysis properties
    """

    def __init__(
        self,
        in_doc: "CADInputDocument",
        path_or_stream: Union[Path, str, BytesIO],
        options: Optional["BackendOptions"] = None,
    ):
        """Initialize mesh backend and load the mesh.

        Args:
            in_doc: Input document descriptor.
            path_or_stream: Path to mesh file or byte stream.
            options: Backend-specific options.
        """
        super().__init__(in_doc, path_or_stream, options)

        self._trimesh_mesh = None
        self._mesh_properties: Optional[Dict[str, Any]] = None
        self._file_path: Optional[Path] = None

        # Resolve file path
        if isinstance(path_or_stream, (str, Path)):
            self._file_path = Path(path_or_stream)
        elif isinstance(path_or_stream, BytesIO):
            self._file_path = None  # Will load from stream
        else:
            self._file_path = Path(str(path_or_stream))

        # Load the mesh immediately so we fail fast on bad files
        self._load_mesh()

    def _load_mesh(self):
        """Load mesh using trimesh."""
        trimesh = _load_trimesh()

        try:
            if self._file_path is not None:
                scene_or_mesh = trimesh.load(str(self._file_path))
            else:
                # Load from BytesIO — need file type hint from extension
                ext = self.file.suffix if hasattr(self.file, "suffix") else ".obj"
                scene_or_mesh = trimesh.load(self.path_or_stream, file_type=ext)

            # trimesh.load can return a Scene (multi-mesh) or Trimesh (single)
            if isinstance(scene_or_mesh, trimesh.Scene):
                # Combine all meshes in the scene into one
                meshes = list(scene_or_mesh.geometry.values())
                if not meshes:
                    raise ValueError("Scene contains no geometry")
                self._trimesh_mesh = trimesh.util.concatenate(meshes)
                _log.info(
                    "Loaded scene with %d meshes, combined into %d vertices / %d faces",
                    len(meshes),
                    len(self._trimesh_mesh.vertices),
                    len(self._trimesh_mesh.faces),
                )
            elif isinstance(scene_or_mesh, trimesh.Trimesh):
                self._trimesh_mesh = scene_or_mesh
                _log.info(
                    "Loaded mesh: %d vertices, %d faces",
                    len(self._trimesh_mesh.vertices),
                    len(self._trimesh_mesh.faces),
                )
            else:
                raise ValueError(
                    f"Unsupported trimesh load result: {type(scene_or_mesh).__name__}"
                )

        except ImportError:
            raise
        except Exception as e:
            _log.error("Failed to load mesh from %s: %s", self._file_path or "stream", e)
            raise

    # ----- Abstract method implementations -----

    @classmethod
    def supports_text_parsing(cls) -> bool:
        """Mesh formats are parsed via trimesh, not as text."""
        return True  # We produce text representation of mesh data

    @classmethod
    def supports_rendering(cls) -> bool:
        """Rendering supported when trimesh is installed."""
        try:
            _load_trimesh()
            return True
        except ImportError:
            return False

    @classmethod
    def supported_formats(cls) -> Set[InputFormat]:
        """Formats supported by this backend.

        Note: We don't handle STL (has its own backend) or STEP/IGES/BREP
        (those are B-Rep, not mesh). We handle OBJ/PLY/OFF etc. but since
        InputFormat doesn't yet have those enums, we return an empty set.
        Callers should use MeshBackend directly for those formats.
        """
        # InputFormat doesn't yet include OBJ/PLY/OFF — return empty
        # This backend is used directly, not via format-based dispatch
        return set()

    def is_valid(self) -> bool:
        """Validate that the mesh loaded successfully and has geometry."""
        if self._trimesh_mesh is None:
            return False
        if len(self._trimesh_mesh.vertices) == 0:
            return False
        if len(self._trimesh_mesh.faces) == 0:
            return False
        return True

    def convert(self) -> CADlingDocument:
        """Convert loaded mesh to CADlingDocument.

        Extracts vertices, normals, facets, and computes mesh properties.

        Returns:
            CADlingDocument populated with mesh data and analysis.
        """
        if self._trimesh_mesh is None:
            raise RuntimeError("No mesh loaded — call _load_mesh() first or check is_valid()")

        mesh = self._trimesh_mesh

        # Build document
        # Detect actual format from file extension
        ext = self.file.suffix.lower() if hasattr(self.file, 'suffix') else ''
        detected_format = InputFormat.STL  # Default fallback
        # Map mesh extensions to closest InputFormat
        _ext_map = {'.stl': InputFormat.STL, '.obj': InputFormat.STL,
                    '.ply': InputFormat.STL, '.off': InputFormat.STL,
                    '.glb': InputFormat.STL, '.gltf': InputFormat.STL}
        detected_format = _ext_map.get(ext, InputFormat.STL)

        doc = STLDocument(
            name=self.file.name,
            format=detected_format,
            origin=CADDocumentOrigin(
                filename=self.file.name,
                format=detected_format,
                binary_hash=self.document_hash,
            ),
            hash=self.document_hash,
        )

        # Convert to MeshItem
        vertices = mesh.vertices.tolist()
        faces = mesh.faces.tolist()
        normals = mesh.face_normals.tolist() if mesh.face_normals is not None else []

        mesh_item = MeshItem(
            label=self._make_label("mesh"),
            text=self._build_mesh_text_representation(mesh),
            vertices=vertices,
            normals=normals,
            facets=faces,
            properties=self._compute_mesh_properties(mesh),
        )

        # Compute bounding box
        bounds_min = mesh.vertices.min(axis=0)
        bounds_max = mesh.vertices.max(axis=0)
        mesh_item.bbox = BoundingBox3D(
            x_min=float(bounds_min[0]),
            y_min=float(bounds_min[1]),
            z_min=float(bounds_min[2]),
            x_max=float(bounds_max[0]),
            y_max=float(bounds_max[1]),
            z_max=float(bounds_max[2]),
        )

        doc.items.append(mesh_item)
        doc.mesh = mesh_item

        _log.info(
            "Converted mesh to document: %d vertices, %d faces, watertight=%s",
            len(vertices),
            len(faces),
            mesh.is_watertight,
        )

        return doc

    def _make_label(self, text: str):
        """Create a label for an item."""
        from cadling.datamodel.base_models import CADLabel
        return CADLabel(text=text)

    def _build_mesh_text_representation(self, mesh) -> str:
        """Build human-readable text representation of mesh.

        Args:
            mesh: trimesh.Trimesh object

        Returns:
            Multi-line text description of the mesh
        """
        lines = [
            f"Mesh: {self.file.name}",
            f"Vertices: {len(mesh.vertices)}",
            f"Faces: {len(mesh.faces)}",
        ]

        # Bounds
        bounds_min = mesh.vertices.min(axis=0)
        bounds_max = mesh.vertices.max(axis=0)
        extent = bounds_max - bounds_min
        lines.append(
            f"Bounds: ({bounds_min[0]:.4f}, {bounds_min[1]:.4f}, {bounds_min[2]:.4f}) "
            f"to ({bounds_max[0]:.4f}, {bounds_max[1]:.4f}, {bounds_max[2]:.4f})"
        )
        lines.append(f"Extent: {extent[0]:.4f} x {extent[1]:.4f} x {extent[2]:.4f}")

        # Properties
        lines.append(f"Watertight: {mesh.is_watertight}")
        if mesh.is_watertight:
            lines.append(f"Volume: {mesh.volume:.6f}")

        lines.append(f"Surface Area: {mesh.area:.6f}")

        # Euler characteristic
        euler = len(mesh.vertices) - len(mesh.edges_unique) + len(mesh.faces)
        lines.append(f"Euler Characteristic: {euler}")

        return "\n".join(lines)

    def _compute_mesh_properties(self, mesh) -> Dict[str, Any]:
        """Compute mesh analysis properties.

        Args:
            mesh: trimesh.Trimesh object

        Returns:
            Dictionary of computed mesh properties
        """
        props: Dict[str, Any] = {
            "num_vertices": len(mesh.vertices),
            "num_faces": len(mesh.faces),
            "num_edges": len(mesh.edges_unique),
            "is_watertight": bool(mesh.is_watertight),
            "surface_area": float(mesh.area),
            "euler_characteristic": int(
                len(mesh.vertices) - len(mesh.edges_unique) + len(mesh.faces)
            ),
        }

        if mesh.is_watertight:
            props["volume"] = float(mesh.volume)
            props["center_of_mass"] = mesh.center_mass.tolist()

        # Bounding box info
        bounds_min = mesh.vertices.min(axis=0)
        bounds_max = mesh.vertices.max(axis=0)
        extent = bounds_max - bounds_min
        props["bounds_min"] = bounds_min.tolist()
        props["bounds_max"] = bounds_max.tolist()
        props["extent"] = extent.tolist()
        props["bounding_box_volume"] = float(np.prod(extent))

        # Face area statistics
        face_areas = mesh.area_faces
        if len(face_areas) > 0:
            props["face_area_mean"] = float(np.mean(face_areas))
            props["face_area_std"] = float(np.std(face_areas))
            props["face_area_min"] = float(np.min(face_areas))
            props["face_area_max"] = float(np.max(face_areas))

        return props

    # ----- Rendering -----

    _VIEW_ANGLES = {
        "front":     (0.0, 0.0),
        "back":      (0.0, 180.0),
        "top":       (90.0, 0.0),
        "bottom":    (-90.0, 0.0),
        "right":     (0.0, 90.0),
        "left":      (0.0, -90.0),
        "isometric": (35.264, 45.0),
    }

    def available_views(self) -> List[str]:
        """List available rendering views."""
        return list(self._VIEW_ANGLES.keys())

    def load_view(self, view_name: str) -> MeshViewBackend:
        """Load a specific view for rendering.

        Args:
            view_name: One of the available view names.

        Returns:
            MeshViewBackend for the requested view.
        """
        if view_name not in self._VIEW_ANGLES:
            raise ValueError(
                f"Unknown view '{view_name}'. Available: {list(self._VIEW_ANGLES.keys())}"
            )
        return MeshViewBackend(view_name, self)

    def render_view(
        self,
        view_name: str,
        resolution: int = 1024,
    ) -> Image.Image:
        """Render a specific view to a PIL Image.

        Args:
            view_name: View name (front, top, isometric, etc.)
            resolution: Image resolution in pixels.

        Returns:
            Rendered PIL Image.
        """
        view = self.load_view(view_name)
        return view.render(resolution)


class MeshViewBackend(CADViewBackend):
    """View backend for rendering a specific camera angle of a mesh.

    Uses trimesh's built-in scene rendering to produce images from
    predetermined camera angles.
    """

    def __init__(self, view_name: str, parent: MeshBackend):
        super().__init__(view_name, parent)
        self._mesh_backend = parent

    def get_camera_parameters(self) -> dict:
        """Get camera parameters for this view.

        Returns:
            Dictionary with elevation, azimuth, and computed distance.
        """
        elevation, azimuth = self._mesh_backend._VIEW_ANGLES[self.view_name]
        mesh = self._mesh_backend._trimesh_mesh
        distance = float(mesh.bounding_sphere.primitive.radius * 3.0)

        el_rad = math.radians(elevation)
        az_rad = math.radians(azimuth)

        return {
            "position": [
                distance * math.cos(el_rad) * math.sin(az_rad),
                distance * math.sin(el_rad),
                distance * math.cos(el_rad) * math.cos(az_rad),
            ],
            "direction": [
                -math.cos(el_rad) * math.sin(az_rad),
                -math.sin(el_rad),
                -math.cos(el_rad) * math.cos(az_rad),
            ],
            "up": [0.0, 1.0, 0.0],
            "fov": 60.0,
            "distance": distance,
            "elevation": elevation,
            "azimuth": azimuth,
        }

    def render(self, resolution: int = 1024) -> Image.Image:
        """Render this view to a PIL Image.

        Uses trimesh's scene rendering. Falls back to a wireframe
        matplotlib render if trimesh's offscreen rendering isn't available.

        Args:
            resolution: Image width and height in pixels.

        Returns:
            PIL Image of the rendered view.
        """
        trimesh = _load_trimesh()
        mesh = self._mesh_backend._trimesh_mesh

        # Try trimesh scene rendering first
        try:
            scene = trimesh.Scene(mesh)

            # Build camera transform from view angles
            params = self.get_camera_parameters()
            camera_transform = trimesh.transformations.compose_matrix(
                translate=params["position"]
            )
            scene.camera_transform = camera_transform

            png_data = scene.save_image(resolution=(resolution, resolution))

            if png_data is not None:
                from io import BytesIO as _BytesIO
                return Image.open(_BytesIO(png_data))

        except Exception as e:
            _log.debug("trimesh scene rendering failed: %s, trying matplotlib", e)

        # Fallback: matplotlib wireframe
        return self._render_matplotlib(mesh, resolution)

    def _render_matplotlib(self, mesh, resolution: int) -> Image.Image:
        """Render mesh using matplotlib as fallback.

        Args:
            mesh: trimesh.Trimesh object
            resolution: Image resolution

        Returns:
            PIL Image
        """
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            from mpl_toolkits.mplot3d.art3d import Poly3DCollection

            fig = plt.figure(figsize=(resolution / 100, resolution / 100), dpi=100)
            ax = fig.add_subplot(111, projection="3d")

            # Sample faces if too many (for performance)
            max_render_faces = 5000
            faces = mesh.faces
            if len(faces) > max_render_faces:
                rng = np.random.default_rng(seed=42)
                indices = rng.choice(len(faces), max_render_faces, replace=False)
                faces = faces[indices]

            # Build polygon collection
            verts = mesh.vertices[faces]
            poly = Poly3DCollection(verts, alpha=0.6, edgecolor="k", linewidth=0.1)
            poly.set_facecolor([0.5, 0.7, 0.9])
            ax.add_collection3d(poly)

            # Set axis limits from mesh bounds
            bounds_min = mesh.vertices.min(axis=0)
            bounds_max = mesh.vertices.max(axis=0)
            center = (bounds_min + bounds_max) / 2
            extent = (bounds_max - bounds_min).max() / 2

            ax.set_xlim(center[0] - extent, center[0] + extent)
            ax.set_ylim(center[1] - extent, center[1] + extent)
            ax.set_zlim(center[2] - extent, center[2] + extent)

            # Set view angle
            elevation, azimuth = self._mesh_backend._VIEW_ANGLES[self.view_name]
            ax.view_init(elev=elevation, azim=azimuth)
            ax.set_axis_off()

            # Render to PIL Image
            fig.tight_layout(pad=0)
            fig.canvas.draw()
            width, height = fig.canvas.get_width_height()
            image_data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            image_data = image_data.reshape(height, width, 3)
            plt.close(fig)

            return Image.fromarray(image_data)

        except ImportError:
            _log.warning("matplotlib not available for fallback rendering")
            # Return a blank image as absolute last resort
            return Image.new("RGB", (resolution, resolution), (200, 200, 200))
