"""IGES backend for parsing IGES files.

This backend handles Initial Graphics Exchange Specification (IGES) file processing.
IGES is a neutral CAD file format for exchanging geometric data between CAD systems.

File structure:
- Start Section: ASCII text (80 chars/line, marked with 'S')
- Global Section: Parameter delimiters, file info (marked with 'G')
- Directory Entry Section: Entity metadata (marked with 'D')
- Parameter Data Section: Entity data (marked with 'P')
- Terminate Section: Counts (marked with 'T')

Classes:
    IGESBackend: Main backend for IGES files
"""

from __future__ import annotations

import logging
import re
from io import BytesIO
from pathlib import Path
from typing import List, Optional, Union

from PIL import Image

from cadling.backend.abstract_backend import (
    CADViewBackend,
    DeclarativeCADBackend,
    RenderableCADBackend,
)
from cadling.datamodel.backend_options import BackendOptions
from cadling.datamodel.base_models import (
    CADDocumentOrigin,
    CADInputDocument,
    CADItemLabel,
    CADlingDocument,
    InputFormat,
)

_log = logging.getLogger(__name__)


class IGESBackend(DeclarativeCADBackend, RenderableCADBackend):
    """IGES file backend with entity parsing and rendering support.

    Parses IGES files and extracts entities. IGES is a neutral CAD format
    used for data exchange between different CAD systems. Also supports
    rendering via pythonocc-core.

    Attributes:
        iges_text: Raw IGES file content
        start_section: Start section lines
        global_section: Global section lines
        directory_entries: Directory entry lines
        parameter_data: Parameter data lines
        has_pythonocc: Whether pythonocc-core is available for rendering
        shape: Loaded OpenCASCADE shape (if rendering enabled)
    """

    def __init__(
        self,
        in_doc: CADInputDocument,
        path_or_stream: Union[Path, str, BytesIO],
        options: Optional[BackendOptions] = None,
    ):
        """Initialize IGES backend.

        Args:
            in_doc: Input document descriptor
            path_or_stream: Path to IGES file or byte stream
            options: Backend options
        """
        super().__init__(in_doc, path_or_stream, options)

        # Load IGES file content
        self.iges_text = self._load_iges_text()
        self.start_section = []
        self.global_section = []
        self.directory_entries = []
        self.parameter_data = []

        # Cache for loaded shape
        self.shape = None

        # Check for pythonocc-core availability for rendering
        self.has_pythonocc = False
        try:
            from OCC.Core.IGESControl import IGESControl_Reader

            self.has_pythonocc = True
            _log.debug("pythonocc-core available for IGES rendering")
        except ImportError:
            _log.warning(
                "pythonocc-core not available. Rendering will be disabled. "
                "Install with: conda install pythonocc-core"
            )

        _log.debug(f"Loaded IGES file: {len(self.iges_text)} characters")

    @classmethod
    def supports_text_parsing(cls) -> bool:
        """IGES supports text parsing."""
        return True

    @classmethod
    def supports_rendering(cls) -> bool:
        """IGES rendering supported if pythonocc-core is available."""
        try:
            from OCC.Core.IGESControl import IGESControl_Reader

            return True
        except ImportError:
            return False

    @classmethod
    def supported_formats(cls) -> set[InputFormat]:
        """IGES backend supports IGES format."""
        return {InputFormat.IGES}

    def is_valid(self) -> bool:
        """Validate IGES file format by checking section structure.

        IGES files are organized into sections identified by a letter
        in column 73 (1-indexed) of each 80-character record:
        S (Start), G (Global), D (Directory Entry), P (Parameter Data),
        T (Terminate). A valid file must have at least S, G, D, and T.

        Returns:
            True if file has proper IGES section structure
        """
        if not self.iges_text or len(self.iges_text.strip()) < 80:
            _log.warning("IGES file too short for valid structure")
            return False

        lines = self.iges_text.split('\n')
        found_sections: set[str] = set()
        required_sections = {'S', 'G', 'D', 'T'}

        for line in lines:
            # IGES records are 80 chars; section marker is column 73 (index 72)
            if len(line) >= 73:
                marker = line[72].strip()
                if marker in ('S', 'G', 'D', 'P', 'T'):
                    found_sections.add(marker)

        missing = required_sections - found_sections
        if missing:
            _log.warning(
                "IGES validation failed: missing required sections %s "
                "(found: %s)", missing, found_sections,
            )
            return False

        _log.debug(
            "IGES validation passed: found sections %s", found_sections,
        )
        return True

    def convert(self) -> CADlingDocument:
        """Parse IGES file and convert to CADlingDocument.

        Returns:
            CADlingDocument with parsed entities
        """
        _log.info(f"Converting IGES file: {self.file.name}")

        # Create document
        doc = CADlingDocument(
            name=self.file.name,
            format=InputFormat.IGES,
            origin=CADDocumentOrigin(
                filename=self.file.name,
                format=InputFormat.IGES,
                binary_hash=self.document_hash,
            ),
            hash=self.document_hash,
        )

        # Parse sections
        self._parse_sections()

        # Extract entities from directory entries and parameter data
        entities = self._parse_entities()

        _log.info(f"Parsed {len(entities)} entities from IGES file")

        # Add entities to document
        from cadling.datamodel.base_models import CADItem

        for entity_type, entity_data in entities:
            item = CADItem(
                label=CADItemLabel(text=f"{entity_type}"),
                text=entity_data,
                item_type="iges_entity",
            )
            item.properties["entity_type"] = entity_type
            doc.add_item(item)

        _log.info(f"Conversion complete: {len(doc.items)} items")

        return doc

    def _load_iges_text(self) -> str:
        """Load IGES file content as text.

        Returns:
            IGES file content as string
        """
        if isinstance(self.path_or_stream, (str, Path)):
            with open(self.path_or_stream, "r", encoding="utf-8", errors="ignore") as f:
                return f.read()
        else:
            # BytesIO
            content = self.path_or_stream.read()
            self.path_or_stream.seek(0)
            return content.decode("utf-8", errors="ignore")

    def _parse_sections(self) -> None:
        """Parse IGES file into sections.

        IGES files have 5 sections marked by letters in column 73:
        - S: Start (comments)
        - G: Global (file metadata)
        - D: Directory Entry (entity index)
        - P: Parameter Data (entity data)
        - T: Terminate (section counts)
        """
        lines = self.iges_text.split("\n")

        for line in lines:
            if len(line) < 73:
                continue

            # Section marker is at column 72 (0-indexed)
            section_marker = line[72] if len(line) > 72 else ""

            if section_marker == "S":
                self.start_section.append(line[:72].strip())
            elif section_marker == "G":
                self.global_section.append(line[:72].strip())
            elif section_marker == "D":
                self.directory_entries.append(line[:72].strip())
            elif section_marker == "P":
                self.parameter_data.append(line[:72].strip())

        _log.debug(
            f"Parsed sections: S={len(self.start_section)}, "
            f"G={len(self.global_section)}, D={len(self.directory_entries)}, "
            f"P={len(self.parameter_data)}"
        )

    def _parse_entities(self) -> List[tuple[str, str]]:
        """Parse entities from directory entries and parameter data.

        Returns:
            List of (entity_type, entity_data) tuples
        """
        entities = []

        # IGES directory entries come in pairs (2 lines per entity)
        for i in range(0, len(self.directory_entries), 2):
            if i + 1 >= len(self.directory_entries):
                break

            # First line contains entity type and other metadata
            line1 = self.directory_entries[i]

            # Entity type number is in columns 1-8
            try:
                entity_type_num = int(line1[:8].strip())

                # Map common entity type numbers to names
                entity_types = {
                    100: "CIRCULAR_ARC",
                    102: "COMPOSITE_CURVE",
                    104: "CONIC_ARC",
                    106: "COPIOUS_DATA",
                    108: "PLANE",
                    110: "LINE",
                    112: "PARAMETRIC_SPLINE_CURVE",
                    114: "PARAMETRIC_SPLINE_SURFACE",
                    116: "POINT",
                    118: "RULED_SURFACE",
                    120: "SURFACE_OF_REVOLUTION",
                    122: "TABULATED_CYLINDER",
                    124: "TRANSFORMATION_MATRIX",
                    126: "RATIONAL_B_SPLINE_CURVE",
                    128: "RATIONAL_B_SPLINE_SURFACE",
                    130: "OFFSET_CURVE",
                    140: "OFFSET_SURFACE",
                    142: "CURVE_ON_PARAMETRIC_SURFACE",
                    144: "TRIMMED_SURFACE",
                }

                entity_type = entity_types.get(
                    entity_type_num, f"ENTITY_{entity_type_num}"
                )

                # Extract parameter data pointer (columns 9-16)
                param_pointer = int(line1[8:16].strip()) if line1[8:16].strip() else 0

                # Combine directory entry data
                entity_data = f"{line1}\n{self.directory_entries[i+1] if i+1 < len(self.directory_entries) else ''}"

                entities.append((entity_type, entity_data))

            except (ValueError, IndexError) as e:
                _log.debug(f"Failed to parse directory entry {i}: {e}")

        return entities

    def _load_shape(self):
        """Load IGES file as OpenCASCADE shape."""
        if self.shape is not None:
            return self.shape

        if not self.has_pythonocc:
            _log.warning("Cannot load shape: pythonocc-core not available")
            return None

        try:
            from OCC.Core.IGESControl import IGESControl_Reader
            from OCC.Core.IFSelect import IFSelect_RetDone

            # Create reader
            reader = IGESControl_Reader()

            # Read file
            if isinstance(self.path_or_stream, (str, Path)):
                status = reader.ReadFile(str(self.path_or_stream))
            else:
                # For BytesIO, write to temp file
                import tempfile

                with tempfile.NamedTemporaryFile(suffix=".iges", delete=False) as tmp:
                    tmp.write(self.path_or_stream.read())
                    tmp_path = tmp.name
                self.path_or_stream.seek(0)
                status = reader.ReadFile(tmp_path)
                Path(tmp_path).unlink()  # Clean up

            if status != IFSelect_RetDone:
                _log.error("Failed to read IGES file with pythonocc")
                return None

            # Transfer shapes
            reader.TransferRoots()
            self.shape = reader.OneShape()

            _log.info("Successfully loaded IGES shape")
            return self.shape

        except Exception as e:
            _log.error(f"Failed to load IGES shape: {e}")
            return None

    def available_views(self) -> List[str]:
        """Return available rendering views."""
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
            raise RuntimeError("pythonocc-core not available for rendering")

        return IGESViewBackend(view_name, self)

    def render_view(self, view_name: str, resolution: int = 512) -> Image.Image:
        """Render a specific view."""
        view = self.load_view(view_name)
        return view.render(resolution)


class IGESViewBackend(CADViewBackend):
    """View backend for rendering IGES files."""

    def __init__(self, view_name: str, parent_backend: IGESBackend):
        """Initialize IGES view backend."""
        super().__init__(view_name, parent_backend)
        self.parent = parent_backend

    def render(self, resolution: int = 512) -> Image.Image:
        """Render the view to an image."""
        if not self.parent.has_pythonocc:
            # NO MORE LIES! Renderer's job is to RENDER, not return gray boxes!
            _log.error("pythonocc-core is required for IGES rendering")
            raise ImportError(
                "pythonocc-core is required for IGES rendering. "
                "Install with: conda install -c conda-forge pythonocc-core. "
                "We will NOT return fake gray placeholder images!"
            )

        # Load shape
        shape = self.parent._load_shape()
        if shape is None:
            _log.error("Failed to load IGES shape for rendering")
            raise RuntimeError(
                "Failed to load IGES shape for rendering. "
                "IGES file may be invalid. "
                "We will NOT return fake gray placeholder images!"
            )

        try:
            from OCC.Display.WebGl.jupyter_renderer import JupyterRenderer

            # Create offscreen renderer
            renderer = JupyterRenderer(size=(resolution, resolution))
            renderer.DisplayShape(shape, transparency=False, color=(0.7, 0.7, 0.7))

            # Set camera view
            view = renderer._display.View

            # Set camera position based on view
            if self.view_name == "front":
                view.SetProj(0, -1, 0)
            elif self.view_name == "back":
                view.SetProj(0, 1, 0)
            elif self.view_name == "top":
                view.SetProj(0, 0, -1)
            elif self.view_name == "bottom":
                view.SetProj(0, 0, 1)
            elif self.view_name == "right":
                view.SetProj(1, 0, 0)
            elif self.view_name == "left":
                view.SetProj(-1, 0, 0)
            elif self.view_name == "isometric":
                view.SetProj(1, -1, 1)
            elif self.view_name == "isometric2":
                view.SetProj(-1, -1, 1)

            view.FitAll()

            # Render to image
            renderer.Render()
            img_data = renderer.GetImageData()

            # Convert to PIL Image
            img = Image.frombytes("RGB", (resolution, resolution), img_data)
            return img

        except Exception as e:
            # NO MORE LIES! If rendering fails, FAIL LOUDLY!
            _log.error(f"IGES rendering failed for view {self.view_name}: {e}")
            raise RuntimeError(
                f"IGES rendering failed for view {self.view_name}: {e}. "
                f"We will NOT return fake gray placeholder images!"
            ) from e

    def get_camera_parameters(self) -> dict:
        """Get camera parameters for the view."""
        views = {
            "front": {"position": (0, -10, 0), "up": (0, 0, 1)},
            "back": {"position": (0, 10, 0), "up": (0, 0, 1)},
            "top": {"position": (0, 0, 10), "up": (0, 1, 0)},
            "bottom": {"position": (0, 0, -10), "up": (0, 1, 0)},
            "right": {"position": (10, 0, 0), "up": (0, 0, 1)},
            "left": {"position": (-10, 0, 0), "up": (0, 0, 1)},
            "isometric": {"position": (7, -7, 7), "up": (0, 0, 1)},
            "isometric2": {"position": (-7, -7, 7), "up": (0, 0, 1)},
        }
        return views.get(self.view_name, views["front"])
