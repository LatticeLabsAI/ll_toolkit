"""
BRep Backend - Complete backend for OpenCASCADE BRep file processing.

BRep (Boundary Representation) is OpenCASCADE's native binary format
for storing 3D geometry. This backend provides parsing and rendering
capabilities using pythonocc-core.
"""

from __future__ import annotations

import logging
from io import BytesIO
from pathlib import Path
from typing import List, Optional, Set, Union, Dict, Any

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
    TopologyGraph,
)
from cadling.datamodel.base_models import CADlingDocument, CADItem

_log = logging.getLogger(__name__)


class BRepBackend(DeclarativeCADBackend, RenderableCADBackend):
    """
    Complete BRep backend with parsing and rendering capabilities.

    BRep files are OpenCASCADE's native format and require pythonocc-core
    for processing. This backend:
    1. Loads BRep files using pythonocc-core
    2. Extracts topology information (solids, shells, faces, edges, vertices)
    3. Computes geometric properties (volume, surface area, bounding box)
    4. Supports rendering from multiple views

    Attributes:
        has_pythonocc: Whether pythonocc-core is available
        shape: Cached loaded shape
        topology_data: Cached topology analysis
    """

    def __init__(
        self,
        in_doc: "CADInputDocument",
        path_or_stream: Union[Path, str, BytesIO],
        options: Optional["BackendOptions"] = None,
    ):
        """Initialize BRep backend."""
        super().__init__(in_doc, path_or_stream, options)

        # Cache for parsed data
        self.shape = None
        self.topology_data: Optional[Dict[str, Any]] = None
        self._file_content: Optional[bytes] = None

        # Check for pythonocc-core availability
        self.has_pythonocc = False
        try:
            from OCC.Core.BRepTools import breptools

            self.has_pythonocc = True
            _log.debug("pythonocc-core available for BRep processing")
        except ImportError:
            _log.warning(
                "pythonocc-core not available. BRep backend requires pythonocc-core. "
                "Install with: conda install pythonocc-core"
            )

    @classmethod
    def supported_formats(cls) -> Set[InputFormat]:
        """BRep backend supports BREP format."""
        return {InputFormat.BREP}

    @classmethod
    def supports_text_parsing(cls) -> bool:
        """BRep is a binary format, but we can extract topology as text."""
        return True

    @classmethod
    def supports_rendering(cls) -> bool:
        """BRep backend supports rendering via pythonocc-core."""
        try:
            from OCC.Core.BRepTools import breptools

            return True
        except ImportError:
            return False

    def is_valid(self) -> bool:
        """Validate that file is a valid BRep file."""
        try:
            # Try to load the shape
            self._load_shape()
            return self.shape is not None
        except Exception as e:
            _log.error(f"Failed to validate BRep file: {e}")
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
                self.path_or_stream.seek(0)
            else:
                path = Path(self.path_or_stream)
                with open(path, "rb") as f:
                    self._file_content = f.read()

            return self._file_content

        except Exception as e:
            _log.error(f"Failed to read BRep file: {e}")
            raise

    def _load_shape(self):
        """Load BRep shape using pythonocc-core."""
        if self.shape is not None:
            return self.shape

        try:
            from OCC.Core.BRepTools import breptools
            from OCC.Core.TopoDS import TopoDS_Shape
            from OCC.Core.BRep import BRep_Builder

            builder = BRep_Builder()
            shape = TopoDS_Shape()

            # Load from file or stream
            if isinstance(self.path_or_stream, BytesIO):
                # Write to temp file for pythonocc
                import tempfile

                with tempfile.NamedTemporaryFile(suffix=".brep", delete=False) as tmp:
                    content = self._read_file_content()
                    tmp.write(content)
                    tmp_path = tmp.name

                success = breptools.Read(shape, tmp_path, builder)

                import os

                os.unlink(tmp_path)
            else:
                success = breptools.Read(
                    shape, str(self.path_or_stream), builder
                )

            if not success:
                raise RuntimeError("Failed to read BRep file")

            self.shape = shape
            _log.debug(f"Loaded BRep shape from {self.file.name}")

            return self.shape

        except Exception as e:
            _log.error(f"Failed to load BRep shape: {e}")
            raise

    def _analyze_topology(self) -> Dict[str, Any]:
        """Analyze topology of the loaded shape."""
        if self.topology_data is not None:
            return self.topology_data

        try:
            from OCC.Core.TopAbs import (
                TopAbs_SOLID,
                TopAbs_SHELL,
                TopAbs_FACE,
                TopAbs_EDGE,
                TopAbs_VERTEX,
            )
            from OCC.Core.TopExp import TopExp_Explorer
            from OCC.Core.GProp import GProp_GProps
            from OCC.Core.BRepGProp import brepgprop
            from OCC.Core.Bnd import Bnd_Box
            from OCC.Core.BRepBndLib import brepbndlib

            shape = self._load_shape()

            # Count topology elements
            topology_counts = {
                "solids": 0,
                "shells": 0,
                "faces": 0,
                "edges": 0,
                "vertices": 0,
            }

            # Count solids
            explorer = TopExp_Explorer(shape, TopAbs_SOLID)
            while explorer.More():
                topology_counts["solids"] += 1
                explorer.Next()

            # Count shells
            explorer = TopExp_Explorer(shape, TopAbs_SHELL)
            while explorer.More():
                topology_counts["shells"] += 1
                explorer.Next()

            # Count faces
            explorer = TopExp_Explorer(shape, TopAbs_FACE)
            while explorer.More():
                topology_counts["faces"] += 1
                explorer.Next()

            # Count edges
            explorer = TopExp_Explorer(shape, TopAbs_EDGE)
            while explorer.More():
                topology_counts["edges"] += 1
                explorer.Next()

            # Count vertices
            explorer = TopExp_Explorer(shape, TopAbs_VERTEX)
            while explorer.More():
                topology_counts["vertices"] += 1
                explorer.Next()

            # Compute geometric properties
            props = GProp_GProps()
            brepgprop.VolumeProperties(shape, props)
            volume = props.Mass()
            center_of_mass = props.CentreOfMass()

            # Compute surface area
            brepgprop.SurfaceProperties(shape, props)
            surface_area = props.Mass()

            # Compute bounding box
            bbox = Bnd_Box()
            brepbndlib.Add(shape, bbox)
            xmin, ymin, zmin, xmax, ymax, zmax = bbox.Get()

            self.topology_data = {
                "topology_counts": topology_counts,
                "volume": volume,
                "surface_area": surface_area,
                "center_of_mass": (
                    center_of_mass.X(),
                    center_of_mass.Y(),
                    center_of_mass.Z(),
                ),
                "bounding_box": {
                    "x_min": xmin,
                    "y_min": ymin,
                    "z_min": zmin,
                    "x_max": xmax,
                    "y_max": ymax,
                    "z_max": zmax,
                },
            }

            _log.info(
                f"BRep topology: {topology_counts['solids']} solids, "
                f"{topology_counts['faces']} faces, {topology_counts['edges']} edges, "
                f"volume={volume:.2f}, surface_area={surface_area:.2f}"
            )

            return self.topology_data

        except Exception as e:
            _log.error(f"Failed to analyze topology: {e}")
            raise

    def _build_topology_graph(self) -> TopologyGraph:
        """Build complete topology graph from BRep shape.

        Constructs a graph representation of the BRep topology showing
        relationships between solids, shells, faces, edges, and vertices.

        Returns:
            TopologyGraph with nodes and adjacency relationships
        """
        from OCC.Core.TopExp import TopExp_Explorer
        from OCC.Core.TopAbs import (
            TopAbs_SOLID,
            TopAbs_SHELL,
            TopAbs_FACE,
            TopAbs_EDGE,
            TopAbs_VERTEX,
        )
        from OCC.Core.TopoDS import topods

        adjacency_list: Dict[int, List[int]] = {}
        node_id_counter = 0
        shape_to_id: Dict[int, int] = {}

        def get_or_create_id(shape) -> int:
            """Get or create unique ID for a shape."""
            nonlocal node_id_counter
            try:
                shape_hash = shape.HashCode(2147483647)
            except Exception:
                shape_hash = hash(shape)
            if shape_hash not in shape_to_id:
                shape_to_id[shape_hash] = node_id_counter
                adjacency_list[node_id_counter] = []
                node_id_counter += 1
            return shape_to_id[shape_hash]

        # Build adjacency relationships following topology hierarchy
        # Solids -> Shells
        solid_explorer = TopExp_Explorer(self.shape, TopAbs_SOLID)
        while solid_explorer.More():
            solid = solid_explorer.Current()
            solid_id = get_or_create_id(solid)

            # Explore shells in this solid
            shell_explorer = TopExp_Explorer(solid, TopAbs_SHELL)
            while shell_explorer.More():
                shell = shell_explorer.Current()
                shell_id = get_or_create_id(shell)
                adjacency_list[solid_id].append(shell_id)
                shell_explorer.Next()

            solid_explorer.Next()

        # Shells -> Faces
        shell_explorer = TopExp_Explorer(self.shape, TopAbs_SHELL)
        while shell_explorer.More():
            shell = shell_explorer.Current()
            shell_id = get_or_create_id(shell)

            # Explore faces in this shell
            face_explorer = TopExp_Explorer(shell, TopAbs_FACE)
            while face_explorer.More():
                face = face_explorer.Current()
                face_id = get_or_create_id(face)
                if face_id not in adjacency_list[shell_id]:
                    adjacency_list[shell_id].append(face_id)
                face_explorer.Next()

            shell_explorer.Next()

        # Faces -> Edges
        face_explorer = TopExp_Explorer(self.shape, TopAbs_FACE)
        while face_explorer.More():
            face = face_explorer.Current()
            face_id = get_or_create_id(face)

            # Explore edges in this face
            edge_explorer = TopExp_Explorer(face, TopAbs_EDGE)
            while edge_explorer.More():
                edge = edge_explorer.Current()
                edge_id = get_or_create_id(edge)
                if edge_id not in adjacency_list[face_id]:
                    adjacency_list[face_id].append(edge_id)
                edge_explorer.Next()

            face_explorer.Next()

        # Edges -> Vertices
        edge_explorer = TopExp_Explorer(self.shape, TopAbs_EDGE)
        while edge_explorer.More():
            edge = edge_explorer.Current()
            edge_id = get_or_create_id(edge)

            # Explore vertices in this edge
            vertex_explorer = TopExp_Explorer(edge, TopAbs_VERTEX)
            while vertex_explorer.More():
                vertex = vertex_explorer.Current()
                vertex_id = get_or_create_id(vertex)
                if vertex_id not in adjacency_list[edge_id]:
                    adjacency_list[edge_id].append(vertex_id)
                vertex_explorer.Next()

            edge_explorer.Next()

        # Count total edges in graph
        num_edges = sum(len(neighbors) for neighbors in adjacency_list.values())

        _log.info(
            f"Built topology graph: {node_id_counter} nodes, {num_edges} edges"
        )

        return TopologyGraph(
            num_nodes=node_id_counter,
            num_edges=num_edges,
            adjacency_list=adjacency_list,
        )

    def convert(self) -> CADlingDocument:
        """
        Convert BRep file to CADlingDocument.

        Returns:
            Fully populated CADlingDocument with topology and properties.
        """
        # Load shape and analyze topology
        topology_data = self._analyze_topology()

        # Create document
        doc = CADlingDocument(name=self.file.name)

        # Create items for each topology element
        topology_counts = topology_data["topology_counts"]

        # Create summary item
        summary_item = CADItem(
            label={"text": "BRep Model"},
            text=f"BRep model with {topology_counts['solids']} solids, "
            f"{topology_counts['faces']} faces, {topology_counts['edges']} edges, "
            f"{topology_counts['vertices']} vertices",
        )
        doc.add_item(summary_item)

        # Add bounding box
        bb = topology_data["bounding_box"]
        doc.bounding_box = BoundingBox3D(
            x_min=bb["x_min"],
            x_max=bb["x_max"],
            y_min=bb["y_min"],
            y_max=bb["y_max"],
            z_min=bb["z_min"],
            z_max=bb["z_max"],
        )

        # Add metadata
        doc.metadata = {
            "topology_counts": topology_counts,
            "volume": topology_data["volume"],
            "surface_area": topology_data["surface_area"],
            "center_of_mass": topology_data["center_of_mass"],
            "representation_type": "brep",
        }

        # Build full topology graph
        doc.topology = self._build_topology_graph()

        _log.info(f"Converted BRep file to document")

        return doc

    def available_views(self) -> List[str]:
        """List available rendering views."""
        if not self.has_pythonocc:
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
        if not self.has_pythonocc:
            raise RuntimeError(
                "pythonocc-core not available. Cannot load views. "
                "Install with: conda install pythonocc-core"
            )

        return BRepViewBackend(view_name, self)

    def render_view(self, view_name: str, resolution: int = 1024) -> Image.Image:
        """Render a specific view to image."""
        view_backend = self.load_view(view_name)
        return view_backend.render(resolution=resolution)


class BRepViewBackend(CADViewBackend):
    """View backend for rendering specific views of BRep models."""

    def __init__(self, view_name: str, parent_backend: BRepBackend):
        """Initialize BRep view backend."""
        super().__init__(view_name, parent_backend)

    def render(self, resolution: int = 1024) -> Image.Image:
        """Render this view to an image using pythonocc-core offscreen rendering."""
        from cadling.backend.pythonocc_core_backend import render_shape_to_image

        try:
            shape = self.parent_backend._load_shape()
            _log.info(f"Rendering BRep {self.view_name} view at {resolution}x{resolution}")
            img = render_shape_to_image(shape, self.view_name, resolution)
            _log.info(f"Successfully rendered {self.view_name} view")
            return img
        except ImportError as e:
            raise RuntimeError(
                f"pythonocc-core not available: {e}. "
                "Install with: conda install -c conda-forge pythonocc-core"
            ) from e
        except Exception as e:
            _log.error(f"BRep rendering failed for view {self.view_name}: {e}")
            raise RuntimeError(
                f"BRep rendering failed for view {self.view_name}: {e}"
            ) from e

    def _set_view_orientation(self, view, view_name: str):
        """Set camera orientation for the specified view.

        Args:
            view: V3d_View instance
            view_name: Name of the view orientation
        """
        if view_name == "front":
            view.SetProj(0, -1, 0)
        elif view_name == "back":
            view.SetProj(0, 1, 0)
        elif view_name == "top":
            view.SetProj(0, 0, -1)
            view.SetUp(0, 1, 0)
        elif view_name == "bottom":
            view.SetProj(0, 0, 1)
            view.SetUp(0, 1, 0)
        elif view_name == "left":
            view.SetProj(-1, 0, 0)
        elif view_name == "right":
            view.SetProj(1, 0, 0)
        elif view_name == "isometric":
            view.SetProj(1, -1, 1)
        elif view_name in ("isometric_back", "isometric2"):
            view.SetProj(-1, 1, 1)
        else:
            # Default to front
            view.SetProj(0, -1, 0)

    def _add_lighting(self, display):
        """Add default directional lighting.

        Args:
            display: Viewer3d instance
        """
        try:
            from OCC.Core.V3d import V3d_DirectionalLight
            from OCC.Core.gp import gp_Dir
            from OCC.Core.Quantity import Quantity_Color, Quantity_TOC_RGB

            # Main light
            main_light_dir = gp_Dir(0.5, -0.5, -0.7)
            main_light_color = Quantity_Color(1.0, 1.0, 1.0, Quantity_TOC_RGB)
            main_light = V3d_DirectionalLight(main_light_dir, main_light_color)
            main_light.SetIntensity(1.0)
            main_light.SetEnabled(True)
            display.Viewer.AddLight(main_light)

            # Fill light
            fill_light_dir = gp_Dir(-0.3, 0.3, -0.5)
            fill_light_color = Quantity_Color(0.6, 0.6, 0.6, Quantity_TOC_RGB)
            fill_light = V3d_DirectionalLight(fill_light_dir, fill_light_color)
            fill_light.SetIntensity(0.5)
            fill_light.SetEnabled(True)
            display.Viewer.AddLight(fill_light)

            display.Viewer.SetLightOn()

        except Exception as e:
            _log.warning(f"Could not add lighting: {e}")

    def get_camera_parameters(self) -> dict:
        """Get camera parameters for this view."""
        from cadling.backend.abstract_backend import DEFAULT_CAMERA_PARAMETERS

        return DEFAULT_CAMERA_PARAMETERS.get(
            self.view_name, DEFAULT_CAMERA_PARAMETERS["front"]
        )
