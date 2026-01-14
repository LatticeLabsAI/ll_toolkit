"""BRep backend for parsing BRep files.

This backend handles Boundary Representation (BRep) file processing
with full rendering support via pythonocc-core.

Classes:
    BRepBackend: Main backend for BRep files with rendering support
    BRepViewBackend: View backend for rendering specific views
"""

from __future__ import annotations

import logging
from io import BytesIO
from pathlib import Path
from typing import List, Optional, Union

from PIL import Image

from cadling.backend.abstract_backend import (
    DeclarativeCADBackend,
    RenderableCADBackend,
    CADViewBackend,
)
from cadling.datamodel.backend_options import BackendOptions
from cadling.datamodel.base_models import (
    CADDocumentOrigin,
    CADInputDocument,
    CADlingDocument,
    InputFormat,
)

_log = logging.getLogger(__name__)


class BRepBackend(DeclarativeCADBackend, RenderableCADBackend):
    """BRep file backend with full rendering support.

    This backend provides:
    1. Text parsing of BRep files (basic implementation)
    2. Full rendering support via pythonocc-core

    Attributes:
        has_pythonocc: Whether pythonocc-core is available for rendering
        _shape: Cached OCC shape for rendering
    """

    def __init__(
        self,
        in_doc: CADInputDocument,
        path_or_stream: Union[Path, str, BytesIO],
        options: Optional[BackendOptions] = None,
    ):
        """Initialize BRep backend with rendering support."""
        super().__init__(in_doc, path_or_stream, options)

        # Check for pythonocc-core availability
        self.has_pythonocc = False
        try:
            from OCC.Core.BRep import BRep_Builder

            self.has_pythonocc = True
            _log.debug("pythonocc-core available for BRep rendering")
        except ImportError:
            _log.warning(
                "pythonocc-core not available. Rendering will be disabled. "
                "Install with: conda install -c conda-forge pythonocc-core"
            )

        # Cache for OCC shape
        self._shape = None

    @classmethod
    def supports_text_parsing(cls) -> bool:
        """BRep supports text parsing."""
        return True

    @classmethod
    def supports_rendering(cls) -> bool:
        """BRep backend supports rendering if pythonocc-core is available."""
        try:
            from OCC.Core.BRep import BRep_Builder

            return True
        except ImportError:
            return False

    @classmethod
    def supported_formats(cls) -> set[InputFormat]:
        """BRep backend supports BREP format."""
        return {InputFormat.BREP}

    def is_valid(self) -> bool:
        """Validate BRep file format."""
        # Basic validation - will be enhanced in Phase 9
        return True

    def convert(self) -> CADlingDocument:
        """Parse BRep file (basic implementation)."""
        _log.info(f"Parsing BRep file: {self.file.name}")

        doc = CADlingDocument(
            name=self.file.name,
            format=InputFormat.BREP,
            origin=CADDocumentOrigin(
                filename=self.file.name,
                format=InputFormat.BREP,
                binary_hash=self.document_hash,
            ),
            hash=self.document_hash,
        )

        # Load shape if pythonocc available (for future geometric analysis)
        if self.has_pythonocc:
            try:
                self._load_shape()
                _log.debug("BRep shape loaded successfully")
            except Exception as e:
                _log.warning(f"Failed to load BRep shape: {e}")

        return doc

    def _load_shape(self):
        """Load BRep shape using pythonocc-core."""
        if self._shape is not None:
            return self._shape

        try:
            from OCC.Core.BRep import BRep_Builder
            from OCC.Core.BRepTools import breptools
            from OCC.Core.TopoDS import TopoDS_Shape

            builder = BRep_Builder()
            shape = TopoDS_Shape()

            # Read BRep file
            if isinstance(self.path_or_stream, BytesIO):
                # Write to temp file for pythonocc
                import tempfile

                with tempfile.NamedTemporaryFile(
                    suffix=".brep", delete=False
                ) as tmp:
                    self.path_or_stream.seek(0)
                    tmp.write(self.path_or_stream.read())
                    self.path_or_stream.seek(0)
                    tmp_path = tmp.name

                success = breptools.Read(shape, tmp_path, builder)
                import os

                os.unlink(tmp_path)
            else:
                success = breptools.Read(shape, str(self.path_or_stream), builder)

            if not success:
                raise RuntimeError("Failed to read BRep file with pythonocc")

            self._shape = shape
            _log.debug("Loaded BRep shape successfully")
            return self._shape

        except ImportError as e:
            raise RuntimeError(
                f"pythonocc-core not available: {e}. "
                "Install with: conda install -c conda-forge pythonocc-core"
            ) from e
        except Exception as e:
            _log.error(f"Failed to load BRep shape: {e}")
            raise

    def available_views(self) -> List[str]:
        """List available rendering views."""
        if not self.has_pythonocc:
            _log.warning("pythonocc-core not available, no views available")
            return []

        # Standard orthographic and isometric views
        return [
            "front",  # XY plane, looking along +Z
            "back",  # XY plane, looking along -Z
            "top",  # XZ plane, looking along -Y
            "bottom",  # XZ plane, looking along +Y
            "right",  # YZ plane, looking along -X
            "left",  # YZ plane, looking along +X
            "isometric",  # Isometric view (1,1,1)
            "isometric2",  # Alternate isometric (-1,1,1)
        ]

    def load_view(self, view_name: str) -> CADViewBackend:
        """Load a specific view for rendering."""
        if not self.has_pythonocc:
            raise RuntimeError(
                "pythonocc-core not available. Cannot load views. "
                "Install with: conda install -c conda-forge pythonocc-core"
            )

        return BRepViewBackend(view_name, self)

    def render_view(
        self, view_name: str, resolution: int = 1024
    ) -> Image.Image:
        """Render a specific view to image."""
        view_backend = self.load_view(view_name)
        return view_backend.render(resolution=resolution)


class BRepViewBackend(CADViewBackend):
    """View backend for rendering specific views of BRep models."""

    def __init__(self, view_name: str, parent_backend: BRepBackend):
        """Initialize BRep view backend."""
        super().__init__(view_name, parent_backend)
        self.brep_backend = parent_backend

    def render(self, resolution: int = 1024) -> Image.Image:
        """Render this view to an image.

        Args:
            resolution: Image resolution (width and height in pixels)

        Returns:
            PIL Image object with the rendered view

        Raises:
            RuntimeError: If rendering fails (NO MORE PLACEHOLDER IMAGES!)
        """
        try:
            from OCC.Display.OCCViewer import Viewer3d
            from OCC.Core.BRepBndLib import brepbndlib
            from OCC.Core.Bnd import Bnd_Box
            from OCC.Core.V3d import V3d_XposYnegZpos, V3d_Zneg, V3d_Yneg, V3d_Ypos, V3d_Xneg, V3d_Xpos
            from PIL import Image as PILImage

            # Load shape
            shape = self.brep_backend._load_shape()

            if shape is None:
                raise RuntimeError("BRep shape is None, cannot render")

            # Calculate bounding box for camera positioning
            bbox = Bnd_Box()
            brepbndlib.Add(shape, bbox)

            # Create offscreen viewer
            viewer = Viewer3d()

            # Display shape
            viewer.DisplayShape(shape, update=True)

            # Set view direction based on view_name
            view_obj = viewer.View

            # Map view names to OCC view directions
            if self.view_name == "front":
                view_obj.SetProj(V3d_Zneg)  # Looking along -Z
            elif self.view_name == "back":
                view_obj.SetProj(V3d_Xneg)  # Opposite of front
            elif self.view_name == "top":
                view_obj.SetProj(V3d_Yneg)  # Looking along -Y
            elif self.view_name == "bottom":
                view_obj.SetProj(V3d_Ypos)  # Looking along +Y
            elif self.view_name == "right":
                view_obj.SetProj(V3d_Xneg)  # Looking along -X
            elif self.view_name == "left":
                view_obj.SetProj(V3d_Xpos)  # Looking along +X
            elif self.view_name == "isometric":
                view_obj.SetProj(V3d_XposYnegZpos)  # Isometric (1,1,1)
            elif self.view_name == "isometric2":
                # Custom isometric view (-1,1,1)
                view_obj.SetProj(-1, 1, 1)
            else:
                # Default to isometric
                view_obj.SetProj(V3d_XposYnegZpos)

            # Fit all to view
            viewer.FitAll()

            # Render to image buffer
            viewer.View.Dump(f"/tmp/brep_render_{self.view_name}.png")

            # Read the dumped image
            image = PILImage.open(f"/tmp/brep_render_{self.view_name}.png")

            # Resize if needed
            if image.size != (resolution, resolution):
                # Use LANCZOS for high-quality downsampling
                try:
                    from PIL.Image import Resampling
                    image = image.resize((resolution, resolution), Resampling.LANCZOS)
                except ImportError:
                    # Fallback for older Pillow versions
                    image = image.resize((resolution, resolution), PILImage.LANCZOS)

            _log.debug(f"Successfully rendered BRep view '{self.view_name}' at {resolution}x{resolution}")
            return image

        except ImportError as e:
            raise RuntimeError(
                f"pythonocc-core not available: {e}. "
                "Install with: conda install -c conda-forge pythonocc-core"
            ) from e
        except Exception as e:
            # NO MORE LIES! If rendering fails, FAIL LOUDLY!
            _log.error(f"BRep rendering failed for view {self.view_name}: {e}")
            raise RuntimeError(
                f"BRep rendering failed for view {self.view_name}: {e}. "
                f"We will NOT return fake gray placeholder images!"
            ) from e

    def get_camera_parameters(self) -> dict:
        """Get camera parameters for this view."""
        # Define camera parameters for each standard view
        view_params = {
            "front": {
                "position": [0, 0, 100],
                "direction": [0, 0, -1],
                "up": [0, 1, 0],
            },
            "back": {
                "position": [0, 0, -100],
                "direction": [0, 0, 1],
                "up": [0, 1, 0],
            },
            "top": {
                "position": [0, 100, 0],
                "direction": [0, -1, 0],
                "up": [0, 0, -1],
            },
            "bottom": {
                "position": [0, -100, 0],
                "direction": [0, 1, 0],
                "up": [0, 0, 1],
            },
            "right": {
                "position": [100, 0, 0],
                "direction": [-1, 0, 0],
                "up": [0, 1, 0],
            },
            "left": {
                "position": [-100, 0, 0],
                "direction": [1, 0, 0],
                "up": [0, 1, 0],
            },
            "isometric": {
                "position": [100, 100, 100],
                "direction": [-1, -1, -1],
                "up": [0, 1, 0],
            },
            "isometric2": {
                "position": [-100, 100, 100],
                "direction": [1, -1, -1],
                "up": [0, 1, 0],
            },
        }

        return view_params.get(
            self.view_name,
            view_params["isometric"],  # Default to isometric
        )
