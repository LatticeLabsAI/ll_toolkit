"""PythonOCC-Core backend for advanced CAD processing.

This backend provides advanced CAD processing capabilities using pythonocc-core,
including topology analysis, rendering, and geometric operations. It's used as
a base for STEP, IGES, and BREP backends.

Classes:
    PythonOCCBackend: Base backend using pythonocc-core
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional, Set, Tuple
from io import BytesIO

from PIL import Image

from cadling.backend.abstract_backend import AbstractCADBackend, RenderableCADBackend
from cadling.datamodel.base_models import CADlingDocument, InputFormat

_log = logging.getLogger(__name__)


def render_shape_to_image(
    shape,
    view_name: str,
    resolution: int = 1024,
    background_color: Tuple[int, int, int] = (255, 255, 255),
) -> Image:
    """Render an OCC shape to a PIL Image using offscreen Viewer3d.

    Shared utility used by PythonOCC, BRep, STEP, and IGES view backends
    to avoid duplicating rendering, lighting, and temp-file logic.

    Args:
        shape: OCC TopoDS_Shape to render.
        view_name: Camera orientation name (front, back, top, bottom, left, right, isometric, isometric_back, isometric2).
        resolution: Image resolution (square).
        background_color: Background RGB tuple (0-255).

    Returns:
        PIL Image.
    """
    import os
    import tempfile

    from OCC.Display.OCCViewer import Viewer3d
    from OCC.Core.Quantity import Quantity_Color, Quantity_TOC_RGB
    from OCC.Core.Graphic3d import Graphic3d_TOSM_FRAGMENT

    display = Viewer3d()
    display.Create()
    display.SetSize(resolution, resolution)

    bg_color = Quantity_Color(
        background_color[0] / 255.0,
        background_color[1] / 255.0,
        background_color[2] / 255.0,
        Quantity_TOC_RGB,
    )
    display.View.SetBackgroundColor(bg_color)
    display.DisplayShape(shape, update=True)

    view = display.View
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
        view.SetProj(0, -1, 0)

    display.FitAll()

    # Lighting
    try:
        from OCC.Core.V3d import V3d_DirectionalLight
        from OCC.Core.gp import gp_Dir

        main_light_dir = gp_Dir(0.5, -0.5, -0.7)
        main_light_color = Quantity_Color(1.0, 1.0, 1.0, Quantity_TOC_RGB)
        main_light = V3d_DirectionalLight(main_light_dir, main_light_color)
        main_light.SetIntensity(1.0)
        main_light.SetEnabled(True)
        display.Viewer.AddLight(main_light)

        fill_light_dir = gp_Dir(-0.3, 0.3, -0.5)
        fill_light_color = Quantity_Color(0.6, 0.6, 0.6, Quantity_TOC_RGB)
        fill_light = V3d_DirectionalLight(fill_light_dir, fill_light_color)
        fill_light.SetIntensity(0.5)
        fill_light.SetEnabled(True)
        display.Viewer.AddLight(fill_light)

        display.Viewer.SetLightOn()
    except Exception as e:
        _log.warning(f"Could not add lighting: {e}")

    display.View.SetShadingModel(Graphic3d_TOSM_FRAGMENT)

    try:
        display.EnableAntiAliasing()
    except Exception as e:
        _log.debug(f"Antialiasing not available: {e}")

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_file:
        tmp_path = tmp_file.name

    try:
        display.View.Dump(tmp_path)
        img = Image.open(tmp_path)
        if img.mode != "RGB":
            img = img.convert("RGB")
        return img
    finally:
        try:
            os.unlink(tmp_path)
        except OSError as e:
            _log.debug(f"Failed to clean up temporary file {tmp_path}: {e}")


class PythonOCCBackend(RenderableCADBackend):
    """Backend using pythonocc-core for CAD processing.

    This backend leverages pythonocc-core (Python bindings for OpenCASCADE)
    to provide advanced CAD capabilities including:
    - STEP/IGES/BREP file loading
    - Topology analysis (faces, edges, vertices)
    - Geometric property calculation (volume, area, etc.)
    - 3D rendering to images
    - Boolean operations
    - Geometric transformations

    Note: Requires pythonocc-core to be installed (usually via conda).
    """

    def __init__(self, in_doc, path_or_stream, options):
        """Initialize PythonOCC backend.

        Args:
            in_doc: Input document descriptor
            path_or_stream: Path to file or byte stream
            options: Backend options
        """
        super().__init__(in_doc, path_or_stream, options)

        self.shape = None  # Will hold OCC TopoDS_Shape
        self._load_shape()

    def _load_shape(self):
        """Load CAD shape using appropriate reader.

        Raises:
            ImportError: If pythonocc-core is not installed
            RuntimeError: If shape loading fails
        """
        try:
            from OCC.Core.STEPControl import STEPControl_Reader
            from OCC.Core.IGESControl import IGESControl_Reader
            from OCC.Core.BRep import BRep_Builder
            from OCC.Core.BRepTools import breptools
            from OCC.Core.TopoDS import TopoDS_Shape
        except ImportError:
            raise ImportError(
                "pythonocc-core is required for PythonOCCBackend. "
                "Install with: conda install -c conda-forge pythonocc-core"
            )

        # Convert path to string
        if isinstance(self.path_or_stream, BytesIO):
            raise ValueError("PythonOCCBackend requires file path, not BytesIO")

        file_path = str(self.path_or_stream)

        # Select reader based on format
        if self.input_format == InputFormat.STEP:
            reader = STEPControl_Reader()
            status = reader.ReadFile(file_path)
            if status != 1:  # IFSelect_RetDone
                raise RuntimeError(f"Failed to read STEP file: {file_path}")
            reader.TransferRoots()
            self.shape = reader.OneShape()

        elif self.input_format == InputFormat.IGES:
            reader = IGESControl_Reader()
            status = reader.ReadFile(file_path)
            if status != 1:  # IFSelect_RetDone
                raise RuntimeError(f"Failed to read IGES file: {file_path}")
            reader.TransferRoots()
            self.shape = reader.OneShape()

        elif self.input_format == InputFormat.BREP:
            builder = BRep_Builder()
            self.shape = TopoDS_Shape()
            breptools.Read(self.shape, file_path, builder)

        else:
            raise ValueError(f"Unsupported format for PythonOCCBackend: {self.input_format}")

        _log.info(f"Loaded shape from {file_path}")

    @classmethod
    def supports_text_parsing(cls) -> bool:
        """Whether backend can parse text representation.

        Returns:
            False - This backend works with 3D geometry, not text
        """
        return False

    @classmethod
    def supports_rendering(cls) -> bool:
        """Whether backend can render to images.

        Returns:
            True - This backend supports rendering via pythonocc
        """
        return True

    @classmethod
    def supported_formats(cls) -> Set[InputFormat]:
        """Get supported formats.

        Returns:
            Set of InputFormat that pythonocc-core can handle
        """
        return {InputFormat.STEP, InputFormat.IGES, InputFormat.BREP}

    def is_valid(self) -> bool:
        """Check if file is valid.

        Returns:
            True if shape was loaded successfully
        """
        return self.shape is not None

    def available_views(self) -> List[str]:
        """List available view orientations.

        Returns:
            List of view names
        """
        return [
            "front",
            "back",
            "top",
            "bottom",
            "left",
            "right",
            "isometric",
            "isometric2",
        ]

    def load_view(self, view_name: str) -> "PythonOCCViewBackend":
        """Load a specific view for rendering.

        Args:
            view_name: Name of the view orientation (e.g. 'front', 'isometric').

        Returns:
            PythonOCCViewBackend instance for the requested view.

        Raises:
            ValueError: If view_name is not in available_views().
        """
        if view_name not in self.available_views():
            raise ValueError(
                f"Invalid view name: {view_name}. "
                f"Available: {self.available_views()}"
            )
        return PythonOCCViewBackend(view_name, self)

    def render_view(
        self,
        view_name: str,
        resolution: int = 1024,
        background_color: Tuple[int, int, int] = (255, 255, 255),
    ) -> Image:
        """Render view to image using pythonocc-core offscreen rendering.

        Args:
            view_name: View orientation name
            resolution: Image resolution (square, width=height)
            background_color: Background RGB color

        Returns:
            PIL Image

        Raises:
            ImportError: If pythonocc-core is not installed
            ValueError: If view_name is invalid
        """
        if view_name not in self.available_views():
            raise ValueError(
                f"Invalid view name: {view_name}. "
                f"Available: {self.available_views()}"
            )

        _log.info(f"Rendering {view_name} view at {resolution}x{resolution}")
        img = render_shape_to_image(self.shape, view_name, resolution, background_color)
        _log.info(f"Successfully rendered {view_name} view")
        return img

    def _set_view_orientation(self, display, view_name: str):
        """Set camera orientation for the specified view.

        Args:
            display: Viewer3d instance
            view_name: Name of the view orientation
        """
        view = display.View

        if view_name == "front":
            view.SetProj(0, -1, 0)  # Look from +Y towards -Y
        elif view_name == "back":
            view.SetProj(0, 1, 0)   # Look from -Y towards +Y
        elif view_name == "top":
            view.SetProj(0, 0, -1)  # Look from +Z towards -Z
            view.SetUp(0, 1, 0)     # Y is up
        elif view_name == "bottom":
            view.SetProj(0, 0, 1)   # Look from -Z towards +Z
            view.SetUp(0, 1, 0)     # Y is up
        elif view_name == "left":
            view.SetProj(-1, 0, 0)  # Look from +X towards -X
        elif view_name == "right":
            view.SetProj(1, 0, 0)   # Look from -X towards +X
        elif view_name == "isometric":
            view.SetProj(1, -1, 1)  # Isometric view
        elif view_name in ("isometric_back", "isometric2"):
            view.SetProj(-1, 1, 1)  # Back isometric view
        else:
            # Default to front view
            view.SetProj(0, -1, 0)

    def _add_default_lighting(self, display):
        """Add default directional lighting to the scene.

        Args:
            display: Viewer3d instance
        """
        try:
            from OCC.Core.V3d import V3d_DirectionalLight
            from OCC.Core.gp import gp_Dir
            from OCC.Core.Quantity import Quantity_Color, Quantity_TOC_RGB

            # Add main directional light from top-right-front
            main_light_dir = gp_Dir(0.5, -0.5, -0.7)
            main_light_color = Quantity_Color(1.0, 1.0, 1.0, Quantity_TOC_RGB)
            main_light = V3d_DirectionalLight(main_light_dir, main_light_color)
            main_light.SetIntensity(1.0)
            main_light.SetEnabled(True)
            display.Viewer.AddLight(main_light)

            # Add fill light from opposite side (softer)
            fill_light_dir = gp_Dir(-0.3, 0.3, -0.5)
            fill_light_color = Quantity_Color(0.6, 0.6, 0.6, Quantity_TOC_RGB)
            fill_light = V3d_DirectionalLight(fill_light_dir, fill_light_color)
            fill_light.SetIntensity(0.5)
            fill_light.SetEnabled(True)
            display.Viewer.AddLight(fill_light)

            # Turn on lighting
            display.Viewer.SetLightOn()

        except Exception as e:
            _log.warning(f"Could not add lighting: {e}")

    def get_bounding_box(self) -> Tuple[float, float, float, float, float, float]:
        """Get 3D bounding box of the shape.

        Returns:
            Tuple of (x_min, y_min, z_min, x_max, y_max, z_max)
        """
        try:
            from OCC.Core.Bnd import Bnd_Box
            from OCC.Core.BRepBndLib import brepbndlib
        except ImportError:
            raise ImportError("pythonocc-core required")

        bbox = Bnd_Box()
        brepbndlib.Add(self.shape, bbox)

        x_min, y_min, z_min, x_max, y_max, z_max = bbox.Get()

        return (x_min, y_min, z_min, x_max, y_max, z_max)

    def get_volume(self) -> float:
        """Get volume of the shape.

        Returns:
            Volume in cubic units
        """
        try:
            from OCC.Core.GProp import GProp_GProps
            from OCC.Core.BRepGProp import brepgprop
        except ImportError:
            raise ImportError("pythonocc-core required")

        props = GProp_GProps()
        brepgprop.VolumeProperties(self.shape, props)

        return props.Mass()

    def get_surface_area(self) -> float:
        """Get surface area of the shape.

        Returns:
            Surface area in square units
        """
        try:
            from OCC.Core.GProp import GProp_GProps
            from OCC.Core.BRepGProp import brepgprop
        except ImportError:
            raise ImportError("pythonocc-core required")

        props = GProp_GProps()
        brepgprop.SurfaceProperties(self.shape, props)

        return props.Mass()

    def _get_default_options(self):
        """Get default backend options.

        Returns:
            Default BackendOptions
        """
        from cadling.datamodel.backend_options import BackendOptions

        return BackendOptions()


class PythonOCCViewBackend:
    """View backend for rendering a specific view of a PythonOCC model.

    Wraps PythonOCCBackend to provide single-view rendering.

    Attributes:
        view_name: Name of the view (e.g. 'front', 'isometric').
        parent_backend: Parent PythonOCCBackend instance.
    """

    def __init__(self, view_name: str, parent_backend: PythonOCCBackend):
        """Initialize view backend.

        Args:
            view_name: Name of the view.
            parent_backend: Parent backend that created this view.
        """
        self.view_name = view_name
        self.parent_backend = parent_backend

    def render(self, resolution: int = 1024) -> Image:
        """Render this view to an image.

        Args:
            resolution: Image resolution (width and height).

        Returns:
            PIL Image of the rendered view.
        """
        return self.parent_backend.render_view(
            self.view_name, resolution=resolution
        )

    def get_camera_parameters(self) -> dict:
        """Get camera parameters for this view.

        Camera distance is computed from the bounding box diagonal when
        a shape is available, falling back to 100 otherwise.

        Returns:
            Dictionary with camera position, direction, up vector, and FOV.
        """
        import math

        # Compute camera distance from bounding box if shape is available
        distance = 100.0  # default fallback
        try:
            bbox = self.parent_backend.get_bounding_box()
            dx = bbox[3] - bbox[0]
            dy = bbox[4] - bbox[1]
            dz = bbox[5] - bbox[2]
            diagonal = math.sqrt(dx * dx + dy * dy + dz * dz)
            if diagonal > 0:
                # Place camera at ~2x the bounding diagonal for good framing
                distance = diagonal * 2.0
        except Exception:
            pass

        d = distance
        iso_d = d / math.sqrt(3)  # component along each axis for isometric

        view_params = {
            "front": {"position": [0, d, 0], "direction": [0, -1, 0], "up": [0, 0, 1], "fov": 45.0},
            "back": {"position": [0, -d, 0], "direction": [0, 1, 0], "up": [0, 0, 1], "fov": 45.0},
            "top": {"position": [0, 0, d], "direction": [0, 0, -1], "up": [0, 1, 0], "fov": 45.0},
            "bottom": {"position": [0, 0, -d], "direction": [0, 0, 1], "up": [0, 1, 0], "fov": 45.0},
            "left": {"position": [-d, 0, 0], "direction": [1, 0, 0], "up": [0, 0, 1], "fov": 45.0},
            "right": {"position": [d, 0, 0], "direction": [-1, 0, 0], "up": [0, 0, 1], "fov": 45.0},
            "isometric": {"position": [iso_d, -iso_d, iso_d], "direction": [-1, 1, -1], "up": [0, 0, 1], "fov": 45.0},
            "isometric_back": {"position": [-iso_d, iso_d, iso_d], "direction": [1, -1, -1], "up": [0, 0, 1], "fov": 45.0},
            "isometric2": {"position": [-iso_d, iso_d, iso_d], "direction": [1, -1, -1], "up": [0, 0, 1], "fov": 45.0},
        }
        return view_params.get(self.view_name, view_params["front"])
