"""
STEP Backend - Complete backend for STEP file processing.

Integrates tokenizer, feature extractor, and topology builder for comprehensive
STEP file parsing and analysis.
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
from cadling.backend.step.tokenizer import STEPTokenizer
from cadling.backend.step.feature_extractor import STEPFeatureExtractor
from cadling.backend.step.topology_builder import TopologyBuilder
from cadling.datamodel.base_models import (
    CADDocumentOrigin,
    InputFormat,
    BoundingBox3D,
    TopologyGraph,
)
from cadling.datamodel.step import STEPEntityItem, STEPDocument, STEPHeader

_log = logging.getLogger(__name__)


class STEPBackend(DeclarativeCADBackend, RenderableCADBackend):
    """
    Complete STEP backend with text parsing and rendering capabilities.

    This backend:
    1. Parses STEP files using custom tokenizer (from scratch)
    2. Extracts geometric features using feature extractor (from scratch)
    3. Builds topology graphs using topology builder (from scratch)
    4. Optionally renders views using pythonocc-core

    Attributes:
        tokenizer: STEP tokenizer for parsing
        feature_extractor: Feature extractor for geometric properties
        topology_builder: Topology builder for entity graphs
        parsed_data: Cached parsed data
        has_pythonocc: Whether pythonocc-core is available for rendering
    """

    def __init__(
        self,
        in_doc: "CADInputDocument",
        path_or_stream: Union[Path, str, BytesIO],
        options: Optional["BackendOptions"] = None,
    ):
        """Initialize STEP backend."""
        super().__init__(in_doc, path_or_stream, options)

        # Get STEP-specific options
        from cadling.datamodel.backend_options import STEPBackendOptions
        if isinstance(options, STEPBackendOptions):
            self.step_options = options
        else:
            self.step_options = STEPBackendOptions()

        # Initialize ll_stepnet integration if enabled
        self.ll_stepnet = None
        self.use_ll_stepnet = False

        if self.step_options.enable_ll_stepnet:
            try:
                from cadling.backend.step.stepnet_integration import STEPNetIntegration
                self.ll_stepnet = STEPNetIntegration()
                self.use_ll_stepnet = self.ll_stepnet.available
                if self.use_ll_stepnet:
                    _log.info("ll_stepnet integration enabled for enhanced parsing")
                else:
                    _log.warning("ll_stepnet requested but not available, using basic parsing")
            except Exception as e:
                _log.warning(f"Failed to initialize ll_stepnet: {e}. Using basic parsing")
                self.use_ll_stepnet = False

        # Initialize basic components (always available as fallback)
        self.tokenizer = STEPTokenizer()
        self.feature_extractor = STEPFeatureExtractor()
        self.topology_builder = TopologyBuilder()

        # Cache for parsed data
        self.parsed_data: Optional[Dict[str, Any]] = None
        self._file_content: Optional[str] = None

        # Cache for pythonocc shape (for geometric feature extraction)
        self._occ_shape = None
        self._occ_reader = None

        # Check for pythonocc-core availability
        self.has_pythonocc = False
        try:
            from OCC.Core.STEPControl import STEPControl_Reader

            self.has_pythonocc = True
            _log.debug("pythonocc-core available for rendering")
        except ImportError:
            _log.warning(
                "pythonocc-core not available. Rendering will be disabled. "
                "Install with: conda install pythonocc-core"
            )

    @classmethod
    def supported_formats(cls) -> Set[InputFormat]:
        """STEP backend supports STEP format."""
        return {InputFormat.STEP}

    @classmethod
    def supports_text_parsing(cls) -> bool:
        """STEP backend always supports text parsing."""
        return True

    @classmethod
    def supports_rendering(cls) -> bool:
        """STEP backend supports rendering if pythonocc-core is available."""
        try:
            from OCC.Core.STEPControl import STEPControl_Reader

            return True
        except ImportError:
            return False

    def is_valid(self) -> bool:
        """Validate that file is a valid STEP file."""
        try:
            content = self._read_file_content()
            # Check for ISO-10303-21 header
            return (
                "ISO-10303-21" in content[:200]
                and "HEADER;" in content
                and "DATA;" in content
            )
        except Exception as e:
            _log.error(f"Failed to validate STEP file: {e}")
            return False

    def _read_file_content(self) -> str:
        """Read file content as string, using converter cache when available."""
        if self._file_content is not None:
            return self._file_content

        try:
            # Use content cache from document converter to avoid redundant disk read
            if self.in_doc._content_cache is not None:
                self._file_content = self.in_doc._content_cache.decode("utf-8", errors="ignore")
            elif isinstance(self.path_or_stream, BytesIO):
                content = self.path_or_stream.read()
                if isinstance(content, bytes):
                    self._file_content = content.decode("utf-8", errors="ignore")
                else:
                    self._file_content = content
                self.path_or_stream.seek(0)  # Reset stream
            else:
                path = Path(self.path_or_stream)
                with open(path, "r", encoding="utf-8", errors="ignore") as f:
                    self._file_content = f.read()

            return self._file_content

        except Exception as e:
            _log.error(f"Failed to read STEP file: {e}")
            raise

    def _parse_file(self) -> Dict[str, Any]:
        """Parse STEP file and extract all data."""
        if self.parsed_data is not None:
            return self.parsed_data

        _log.info(f"Parsing STEP file: {self.file.name}")

        # Read content
        content = self._read_file_content()

        # Choose parsing strategy based on ll_stepnet availability
        if self.use_ll_stepnet:
            _log.debug("Using ll_stepnet for enhanced parsing")
            return self._parse_with_ll_stepnet(content)
        else:
            _log.debug("Using basic parsing (ll_stepnet not available)")
            return self._parse_with_basic(content)

    def _parse_with_basic(self, content: str) -> Dict[str, Any]:
        """Parse STEP file using basic parsing (original implementation)."""
        # Parse using tokenizer
        parse_result = self.tokenizer.parse_step_file(content)

        # Extract features
        _log.debug("Extracting features from entities")
        features = self.feature_extractor.extract_features(parse_result["entities"])

        # Build topology
        _log.debug("Building topology graph")
        topology_data = self.topology_builder.build_topology_graph(
            parse_result["entities"]
        )

        # Compute global features
        global_features = self.feature_extractor.compute_global_features(features)

        # Analyze topology type
        topology_type = self.topology_builder.analyze_topology_type(
            parse_result["entities"]
        )

        # Extract hierarchy
        hierarchy = self.topology_builder.extract_topology_hierarchy(
            parse_result["entities"]
        )

        # Cache results
        self.parsed_data = {
            "header": parse_result["header"],
            "entities": parse_result["entities"],
            "features": features,
            "topology": topology_data,
            "global_features": global_features,
            "topology_type": topology_type,
            "hierarchy": hierarchy,
            "used_ll_stepnet": False,
        }

        _log.info(
            f"Parsed {len(parse_result['entities'])} entities from {self.file.name}"
        )
        _log.debug(f"Topology type: {topology_type['representation_type']}")

        return self.parsed_data

    def _parse_with_ll_stepnet(self, content: str) -> Dict[str, Any]:
        """Parse STEP file using ll_stepnet for enhanced features."""
        # First parse structure using basic tokenizer
        parse_result = self.tokenizer.parse_step_file(content)

        # Extract DATA section for ll_stepnet processing
        data_section = self._extract_data_section(content)

        # Use ll_stepnet to extract semantic features
        _log.debug("Extracting semantic features with ll_stepnet")
        ll_features_list = []
        try:
            # Convert entities to format expected by ll_stepnet
            for entity_id, entity_data in parse_result["entities"].items():
                entity_text = entity_data.get("raw_text", "")
                entity_type = entity_data.get("type", "")

                features = self.ll_stepnet.extract_features(entity_text, entity_type)
                if features:
                    features["entity_id"] = entity_id
                    ll_features_list.append(features)

            _log.debug(f"Extracted ll_stepnet features for {len(ll_features_list)} entities")
        except Exception as e:
            _log.error(f"ll_stepnet feature extraction failed: {e}. Falling back to basic features")
            return self._parse_with_basic(content)

        # Build topology using ll_stepnet
        _log.debug("Building topology graph with ll_stepnet")
        try:
            topology_data = self.ll_stepnet.build_topology(ll_features_list)
            if topology_data is None:
                _log.warning("ll_stepnet topology building failed, using basic topology")
                topology_data = self.topology_builder.build_topology_graph(parse_result["entities"])
            else:
                _log.debug(f"Built ll_stepnet topology: {topology_data.get('num_nodes', 0)} nodes, "
                          f"{topology_data.get('num_edges', 0)} edges")
        except Exception as e:
            _log.error(f"ll_stepnet topology building failed: {e}. Using basic topology")
            topology_data = self.topology_builder.build_topology_graph(parse_result["entities"])

        # Merge ll_stepnet features with basic features
        merged_features = {}
        basic_features = self.feature_extractor.extract_features(parse_result["entities"])

        # Build O(1) lookup for ll_stepnet features
        ll_features_by_id = {
            f.get("entity_id"): f for f in ll_features_list if f.get("entity_id") is not None
        }

        for entity_id in parse_result["entities"]:
            merged_features[entity_id] = basic_features.get(entity_id, {})
            # Add ll_stepnet features if available
            ll_entity_features = ll_features_by_id.get(entity_id)
            if ll_entity_features:
                merged_features[entity_id].update(ll_entity_features)

        # Compute global features
        global_features = self.feature_extractor.compute_global_features(merged_features)

        # Analyze topology type
        topology_type = self.topology_builder.analyze_topology_type(parse_result["entities"])

        # Extract hierarchy
        hierarchy = self.topology_builder.extract_topology_hierarchy(parse_result["entities"])

        # Cache results
        self.parsed_data = {
            "header": parse_result["header"],
            "entities": parse_result["entities"],
            "features": merged_features,
            "topology": topology_data,
            "global_features": global_features,
            "topology_type": topology_type,
            "hierarchy": hierarchy,
            "used_ll_stepnet": True,
        }

        _log.info(
            f"Parsed {len(parse_result['entities'])} entities from {self.file.name} using ll_stepnet"
        )
        _log.debug(f"Topology type: {topology_type['representation_type']}")

        return self.parsed_data

    def _extract_data_section(self, content: str) -> str:
        """Extract DATA section from STEP file."""
        if "DATA;" in content:
            data_start = content.find("DATA;") + 5
            data_end = content.find("ENDSEC;", data_start)
            if data_end == -1:
                return content[data_start:]
            else:
                return content[data_start:data_end]
        return content

    def _load_occ_shape(self):
        """Load STEP file as OpenCASCADE shape for geometric feature extraction.

        Returns:
            TopoDS_Shape or None if loading fails or pythonocc unavailable
        """
        if self._occ_shape is not None:
            return self._occ_shape

        if not self.has_pythonocc:
            _log.debug("pythonocc-core not available, cannot load OCC shape")
            return None

        try:
            from OCC.Core.STEPControl import STEPControl_Reader
            from OCC.Core.IFSelect import IFSelect_RetDone

            reader = STEPControl_Reader()

            # Read STEP file
            if isinstance(self.path_or_stream, BytesIO):
                # Write to temp file for pythonocc
                import tempfile

                with tempfile.NamedTemporaryFile(suffix=".step", delete=False) as tmp:
                    content = self._read_file_content()
                    tmp.write(content.encode("utf-8"))
                    tmp_path = tmp.name

                try:
                    status = reader.ReadFile(tmp_path)
                finally:
                    import os
                    os.unlink(tmp_path)
            else:
                status = reader.ReadFile(str(self.path_or_stream))

            if status != IFSelect_RetDone:
                _log.warning("Failed to read STEP file with pythonocc for geometric features")
                return None

            # Transfer shapes
            reader.TransferRoots()
            self._occ_shape = reader.OneShape()
            self._occ_reader = reader

            _log.debug("Loaded STEP shape for geometric feature extraction")
            return self._occ_shape

        except Exception as e:
            _log.warning(f"Failed to load OCC shape for geometric features: {e}")
            return None

    def convert(self) -> STEPDocument:
        """
        Convert STEP file to STEPDocument.

        Returns:
            Fully populated STEPDocument with entities, features, and topology.
        """
        # Parse the file
        parsed_data = self._parse_file()

        # Create document
        doc = STEPDocument(
            name=self.file.name,
            format=InputFormat.STEP,
            origin=CADDocumentOrigin(
                filename=self.file.name,
                format=InputFormat.STEP,
                binary_hash=self.document_hash,
            ),
            hash=self.document_hash,
        )

        # Set header
        doc.header = parsed_data["header"]

        # Create entity items
        _log.debug("Creating entity items")
        for entity_id, entity_data in parsed_data["entities"].items():
            # Get features for this entity
            entity_features = parsed_data["features"].get(entity_id, {})

            # Extract numeric and reference params
            numeric_params = []
            reference_params = []

            for param in entity_data.get("params", []):
                if isinstance(param, (int, float)):
                    numeric_params.append(float(param))
                elif isinstance(param, str) and param.startswith("#"):
                    # Extract reference ID
                    import re

                    match = re.search(r"#(\d+)", param)
                    if match:
                        reference_params.append(int(match.group(1)))

            # Create entity item
            entity_item = STEPEntityItem(
                entity_id=entity_id,
                entity_type=entity_data["type"],
                label={"text": f"#{entity_id} {entity_data['type']}"},
                text=entity_data.get("raw", ""),
                raw_text=entity_data.get("raw", ""),
                numeric_params=numeric_params,
                reference_params=reference_params,
                features=entity_features,
            )

            doc.add_entity(entity_item)

        # Build topology graph
        _log.debug("Building topology graph for document")
        topology = TopologyGraph(
            num_nodes=parsed_data["topology"]["num_nodes"],
            adjacency_list=parsed_data["topology"]["adjacency_list"],
        )
        doc.topology = topology

        # Add global metadata
        doc.metadata = {
            "global_features": parsed_data["global_features"],
            "topology_type": parsed_data["topology_type"],
            "hierarchy": parsed_data["hierarchy"],
            "topology_statistics": parsed_data["topology"]["topology_statistics"],
            "entity_levels": parsed_data["topology"]["entity_levels"],
            "connected_components": parsed_data["topology"]["connected_components"],
            "used_ll_stepnet": parsed_data.get("used_ll_stepnet", False),
            "parsing_method": "ll_stepnet" if parsed_data.get("used_ll_stepnet") else "basic",
        }

        # Compute bounding box if available
        if "bounding_box" in parsed_data["global_features"]:
            bb = parsed_data["global_features"]["bounding_box"]
            doc.bounding_box = BoundingBox3D(
                x_min=bb["x_min"],
                x_max=bb["x_max"],
                y_min=bb["y_min"],
                y_max=bb["y_max"],
                z_min=bb["z_min"],
                z_max=bb["z_max"],
            )

        # Load pythonocc shape for geometric feature extraction
        # This enables BRepFaceGraphBuilder to extract REAL geometric features
        occ_shape = self._load_occ_shape()
        if occ_shape is not None:
            # Store shape in document for downstream use
            doc._occ_shape = occ_shape
            doc.metadata["has_occ_shape"] = True
            _log.debug("Stored OCC shape in document for geometric feature extraction")
        else:
            doc._occ_shape = None
            doc.metadata["has_occ_shape"] = False

        _log.info(f"Converted STEP file to document with {len(doc.items)} entities")

        return doc

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
                "Install with: conda install pythonocc-core"
            )

        return STEPViewBackend(view_name, self)

    def render_view(
        self, view_name: str, resolution: int = 1024
    ) -> Image.Image:
        """Render a specific view to image."""
        view_backend = self.load_view(view_name)
        return view_backend.render(resolution=resolution)


class STEPViewBackend(CADViewBackend):
    """View backend for rendering specific views of STEP models."""

    def __init__(self, view_name: str, parent_backend: STEPBackend):
        """Initialize STEP view backend."""
        super().__init__(view_name, parent_backend)
        self.step_backend = parent_backend
        self._shape = None

    def _load_shape(self):
        """Load STEP shape, reusing parent backend's cached shape when available.

        This avoids redundantly re-parsing the STEP file. The parent
        STEPBackend already loads the OCC shape via ``_load_occ_shape()``
        during ``convert()``, so we reuse that cached result.  If the
        parent has not yet loaded the shape we delegate to its loader
        rather than creating a second STEPControl_Reader.
        """
        if self._shape is not None:
            return self._shape

        # Try to reuse the parent backend's already-parsed shape
        parent_shape = self.step_backend._occ_shape
        if parent_shape is not None:
            self._shape = parent_shape
            _log.debug(
                "Reused parent STEPBackend cached shape for view %s",
                self.view_name,
            )
            return self._shape

        # Parent hasn't loaded yet -- ask it to load (single parse)
        parent_shape = self.step_backend._load_occ_shape()
        if parent_shape is not None:
            self._shape = parent_shape
            _log.debug(
                "Loaded STEP shape via parent backend for view %s",
                self.view_name,
            )
            return self._shape

        raise RuntimeError(
            "Failed to load STEP shape for rendering. "
            "Ensure pythonocc-core is installed and the STEP file is valid."
        )

    def render(self, resolution: int = 1024) -> Image.Image:
        """Render this view to an image with proper offscreen rendering.

        Args:
            resolution: Image resolution (width and height in pixels)

        Returns:
            PIL Image object with the rendered view

        Raises:
            RuntimeError: If rendering fails (NO MORE PLACEHOLDER IMAGES!)
        """
        try:
            from cadling.backend.pythonocc_core_backend import render_shape_to_image

            shape = self._load_shape()

            if shape is None:
                raise RuntimeError("STEP shape is None, cannot render")

            image = render_shape_to_image(shape, self.view_name, resolution)

            _log.debug(f"Successfully rendered STEP view '{self.view_name}' at {resolution}x{resolution}")
            return image

        except ImportError as e:
            raise RuntimeError(
                f"pythonocc-core not available: {e}. "
                "Install with: conda install -c conda-forge pythonocc-core"
            ) from e
        except Exception as e:
            _log.error(f"STEP rendering failed for view {self.view_name}: {e}")
            raise RuntimeError(
                f"STEP rendering failed for view {self.view_name}: {e}. "
                f"We will NOT return fake gray placeholder images!"
            ) from e

    def get_camera_parameters(self) -> dict:
        """Get camera parameters for this view."""
        from cadling.backend.abstract_backend import DEFAULT_CAMERA_PARAMETERS

        return DEFAULT_CAMERA_PARAMETERS.get(
            self.view_name, DEFAULT_CAMERA_PARAMETERS["front"]
        )
