"""Abstract backend interfaces for CAD file processing.

This module provides the foundational backend hierarchy for cadling, adapted from
docling's proven architecture. Backends handle format-specific parsing, feature
extraction, and optionally rendering CAD files to images.

Classes:
    AbstractCADBackend: Base backend for all CAD formats.
    DeclarativeCADBackend: For formats with direct text-to-document conversion.
    RenderableCADBackend: For backends that support rendering to images.

Example:
    class STEPBackend(DeclarativeCADBackend, RenderableCADBackend):
        '''Hybrid backend supporting both text parsing and rendering'''

        def convert(self) -> CADlingDocument:
            # Parse STEP entities
            pass

        def render_view(self, view_name: str) -> Image:
            # Render using pythonocc-core
            pass
"""

from __future__ import annotations

import hashlib
import logging
from abc import ABC, abstractmethod
from io import BytesIO
from pathlib import Path
from typing import List, Optional, Set, Union

try:
    from PIL import Image
except ImportError:
    Image = None

_log = logging.getLogger(__name__)

# Shared camera parameters for all view backends.
# "isometric_back" is an alias for "isometric2" — both map to the same camera position.
DEFAULT_CAMERA_PARAMETERS = {
    "front": {"position": [0, 100, 0], "direction": [0, -1, 0], "up": [0, 0, 1], "fov": 45.0},
    "back": {"position": [0, -100, 0], "direction": [0, 1, 0], "up": [0, 0, 1], "fov": 45.0},
    "top": {"position": [0, 0, 100], "direction": [0, 0, -1], "up": [0, 1, 0], "fov": 45.0},
    "bottom": {"position": [0, 0, -100], "direction": [0, 0, 1], "up": [0, 1, 0], "fov": 45.0},
    "left": {"position": [-100, 0, 0], "direction": [1, 0, 0], "up": [0, 0, 1], "fov": 45.0},
    "right": {"position": [100, 0, 0], "direction": [-1, 0, 0], "up": [0, 0, 1], "fov": 45.0},
    "isometric": {"position": [100, -100, 100], "direction": [-1, 1, -1], "up": [0, 0, 1], "fov": 45.0},
    "isometric2": {"position": [-100, 100, 100], "direction": [1, -1, -1], "up": [0, 0, 1], "fov": 45.0},
    "isometric_back": {"position": [-100, 100, 100], "direction": [1, -1, -1], "up": [0, 0, 1], "fov": 45.0},
}


class AbstractCADBackend(ABC):
    """Base backend for all CAD formats.

    This is the foundational abstract class that all CAD backends must inherit from.
    It defines the contract for CAD file processing, similar to docling's
    AbstractDocumentBackend but adapted for 3D geometry.

    Attributes:
        file: Path to the input file.
        path_or_stream: File path or byte stream.
        document_hash: Hash of the input file for caching.
        input_format: Format of the input file (STEP, STL, etc.).
        options: Backend-specific options.
    """

    def __init__(
        self,
        in_doc: "CADInputDocument",
        path_or_stream: Union[Path, str, BytesIO],
        options: Optional["BackendOptions"] = None,
    ):
        """Initialize backend.

        Args:
            in_doc: Input document descriptor.
            path_or_stream: Path to file or byte stream.
            options: Backend-specific options.
        """
        self.file = in_doc.file
        self.path_or_stream = path_or_stream
        self.input_format = in_doc.format
        self.options = options or self._get_default_options()

        # Use _compute_hash to set document_hash from file content,
        # falling back to the hash provided by the input document
        self.document_hash = self._compute_hash_from_source(path_or_stream) or in_doc.document_hash

        _log.debug(
            f"Initialized {self.__class__.__name__} for {self.file.name} "
            f"(format={self.input_format}, hash={self.document_hash[:8]}...)"
        )

    @classmethod
    @abstractmethod
    def supports_text_parsing(cls) -> bool:
        """Whether this backend can parse text representation.

        Returns:
            True if backend can parse CAD file as text (e.g., STEP entities).
            False if backend only works with binary or rendered representations.

        Example:
            STEPBackend.supports_text_parsing() -> True (can parse STEP text)
            BinarySTLBackend.supports_text_parsing() -> False (binary only)
        """
        pass

    @classmethod
    @abstractmethod
    def supports_rendering(cls) -> bool:
        """Whether this backend can render CAD to images.

        Returns:
            True if backend can render 3D geometry to images.
            False if backend cannot render (e.g., text-only parsing).

        Example:
            STEPBackend.supports_rendering() -> True (via pythonocc-core)
            AsciiSTLBackend.supports_rendering() -> True (via trimesh)
        """
        pass

    @classmethod
    @abstractmethod
    def supported_formats(cls) -> Set["InputFormat"]:
        """Formats supported by this backend.

        Returns:
            Set of InputFormat enum values.

        Example:
            STEPBackend.supported_formats() -> {InputFormat.STEP}
            STLBackend.supported_formats() -> {InputFormat.STL}
        """
        pass

    @abstractmethod
    def is_valid(self) -> bool:
        """Validate the input file format.

        Returns:
            True if file is valid for this backend, False otherwise.

        Example:
            def is_valid(self) -> bool:
                return self.content.startswith("ISO-10303-21")
        """
        pass

    @classmethod
    def _get_default_options(cls) -> Optional["BackendOptions"]:
        """Get default backend options.

        Returns:
            Default options for this backend, or None.
        """
        return None

    def _compute_hash_from_source(self, path_or_stream: Union[Path, str, BytesIO]) -> Optional[str]:
        """Compute hash from the source file or stream.

        Reads file content and computes SHA256 hash using _compute_hash.

        Args:
            path_or_stream: Path to file or byte stream.

        Returns:
            SHA256 hash as hex string, or None if content cannot be read.
        """
        try:
            if isinstance(path_or_stream, (Path, str)):
                with open(path_or_stream, "rb") as f:
                    content = f.read()
            else:
                content = path_or_stream.read()
                path_or_stream.seek(0)
            return self._compute_hash(content)
        except Exception as e:
            _log.debug(f"Could not compute hash from source: {e}")
            return None

    def _compute_hash(self, content: bytes) -> str:
        """Compute hash of file content.

        Args:
            content: File content as bytes.

        Returns:
            SHA256 hash as hex string.
        """
        return hashlib.sha256(content).hexdigest()


class DeclarativeCADBackend(AbstractCADBackend):
    """Backend for formats with direct text-to-document conversion.

    DeclarativeCADBackend is used for CAD formats that have structured text
    representations that can be directly parsed into CADlingDocument without
    requiring page-level or view-level processing.

    Examples:
        - STEP files (ISO-10303-21): Entities in text format
        - ASCII STL files: Vertex and facet data in text
        - IGES files: Entity-based text format

    This mirrors docling's DeclarativeDocumentBackend pattern.
    """

    @classmethod
    def supports_text_parsing(cls) -> bool:
        """DeclarativeCADBackend always supports text parsing."""
        return True

    @abstractmethod
    def convert(self) -> "CADlingDocument":
        """Parse and convert CAD file to CADlingDocument.

        This is the core method that backends must implement. It should:
        1. Parse the CAD file format
        2. Extract entities, geometry, or mesh data
        3. Build the CADlingDocument structure

        Returns:
            Fully populated CADlingDocument.

        Example:
            def convert(self) -> CADlingDocument:
                doc = CADlingDocument(name=self.file.name)

                # Parse entities
                for entity in self._parse_entities():
                    item = self._entity_to_item(entity)
                    doc.add_item(item)

                # Build topology
                doc.topology = self._build_topology()

                return doc
        """
        pass


class RenderableCADBackend(AbstractCADBackend):
    """Backend for CAD formats that support rendering to images.

    RenderableCADBackend is used for backends that can render 3D geometry to
    images. This is analogous to docling's PaginatedDocumentBackend, but instead
    of "pages" we have "views" (front, top, isometric, etc.).

    The key difference from documents:
    - Documents have sequential pages (page 1, 2, 3...)
    - CAD has named views (front, top, isometric...)

    Examples:
        - STEP files (with pythonocc-core)
        - BRep files (with pythonocc-core)
        - STL files (with trimesh or Open3D)
    """

    @classmethod
    def supports_rendering(cls) -> bool:
        """RenderableCADBackend always supports rendering."""
        return True

    @abstractmethod
    def available_views(self) -> List[str]:
        """List available rendering views.

        Returns:
            List of view names (e.g., ["front", "top", "isometric"]).

        Example:
            def available_views(self) -> List[str]:
                return ["front", "top", "right", "isometric",
                        "bottom", "left", "back"]
        """
        pass

    @abstractmethod
    def load_view(self, view_name: str) -> "CADViewBackend":
        """Load a specific view for rendering.

        Args:
            view_name: Name of the view (e.g., "front", "top").

        Returns:
            CADViewBackend instance for this view.

        Example:
            view = backend.load_view("front")
            image = view.render(resolution=1024)
        """
        pass

    @abstractmethod
    def render_view(
        self,
        view_name: str,
        resolution: int = 1024,
    ) -> Image.Image:
        """Render a specific view to image.

        Args:
            view_name: Name of the view to render.
            resolution: Image resolution (width and height in pixels).

        Returns:
            PIL Image of the rendered view.

        Example:
            image = backend.render_view("isometric", resolution=2048)
            image.save("part_isometric.png")
        """
        pass


class CADViewBackend(ABC):
    """Backend for a specific view of a CAD model.

    This is analogous to docling's PageBackend but for CAD views instead of
    document pages. Each view represents a different camera angle.

    Attributes:
        view_name: Name of this view (e.g., "front", "isometric").
        parent_backend: Parent RenderableCADBackend.
    """

    def __init__(
        self,
        view_name: str,
        parent_backend: RenderableCADBackend,
    ):
        """Initialize view backend.

        Args:
            view_name: Name of this view.
            parent_backend: Parent backend that created this view.
        """
        self.view_name = view_name
        self.parent_backend = parent_backend

        _log.debug(
            f"Initialized {self.__class__.__name__} for view '{view_name}'"
        )

    @abstractmethod
    def render(self, resolution: int = 1024) -> Image.Image:
        """Render this view to an image.

        Args:
            resolution: Image resolution.

        Returns:
            PIL Image of the rendered view.
        """
        pass

    @abstractmethod
    def get_camera_parameters(self) -> dict:
        """Get camera parameters for this view.

        Returns:
            Dictionary with camera parameters (position, direction, up, etc.).

        Example:
            {
                "position": [0, 0, 100],
                "direction": [0, 0, -1],
                "up": [0, 1, 0],
                "fov": 45.0
            }
        """
        pass
