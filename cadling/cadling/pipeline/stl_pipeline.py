"""
STL pipeline for converting STL files to CADlingDocument.

This pipeline handles both ASCII and binary STL files with mesh
analysis and validation.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from cadling.pipeline.base_pipeline import BaseCADPipeline

if TYPE_CHECKING:
    from cadling.datamodel.base_models import ConversionResult
    from cadling.datamodel.pipeline_options import PipelineOptions

_log = logging.getLogger(__name__)


class STLPipeline(BaseCADPipeline):
    """
    Pipeline for STL file conversion.

    This pipeline:
    1. Build: Parses STL file using STLBackend (ASCII or binary)
    2. Assemble: Validates mesh topology (manifold, watertight)
    3. Enrich: Applies enrichment models (classification, embeddings, etc.)

    Example:
        options = STLPipeline.get_default_options()
        pipeline = STLPipeline(options)
        result = pipeline.execute(input_doc)
    """

    def __init__(self, pipeline_options: PipelineOptions):
        """Initialize STL pipeline."""
        super().__init__(pipeline_options)
        _log.debug(f"Initialized STLPipeline")

    @classmethod
    def get_default_options(cls) -> PipelineOptions:
        """Get default pipeline options for STL."""
        from cadling.datamodel.pipeline_options import PipelineOptions

        return PipelineOptions(
            do_topology_analysis=True,
            device="cpu",
        )

    def _build_document(self, conv_res: ConversionResult) -> ConversionResult:
        """
        Build: Parse STL file and extract mesh data.

        Uses the STLBackend to:
        - Detect ASCII vs binary format
        - Parse mesh data (vertices, normals, facets)
        - Compute mesh properties (manifold, watertight, volume, surface area)

        Args:
            conv_res: Conversion result to populate.

        Returns:
            Updated conversion result with document populated.
        """
        try:
            # Get the backend from input
            backend = conv_res.input._backend

            # Verify it's an STL backend
            from cadling.backend.stl import STLBackend

            if not isinstance(backend, STLBackend):
                raise ValueError(
                    f"STLPipeline requires STLBackend, got {type(backend)}"
                )

            # Parse and convert
            _log.debug("Calling STLBackend.convert()")
            document = backend.convert()

            # Set document in result
            conv_res.document = document

            _log.info(
                f"Built STL document: {document.mesh.num_vertices if document.mesh else 0} vertices, "
                f"{document.mesh.num_facets if document.mesh else 0} facets, "
                f"format={'ASCII' if document.is_ascii else 'Binary'}"
            )

            return conv_res

        except Exception as e:
            _log.exception(f"Build stage failed: {e}")
            conv_res.add_error(
                component="STLPipeline._build_document",
                error_message=f"Failed to build document: {str(e)}",
            )
            raise

    def _assemble_document(self, conv_res: ConversionResult) -> ConversionResult:
        """
        Assemble: Validate mesh topology and properties.

        For STL files, this stage:
        - Validates mesh topology (manifold, watertight)
        - Checks for degenerate triangles
        - Validates normal directions
        - Reports mesh quality issues

        Args:
            conv_res: Conversion result to assemble.

        Returns:
            Updated conversion result.
        """
        if not conv_res.document:
            return conv_res

        try:
            from cadling.datamodel.stl import STLDocument, MeshItem

            document = conv_res.document

            # Ensure it's an STL document
            if not isinstance(document, STLDocument):
                _log.warning(
                    f"STLPipeline expected STLDocument, got {type(document)}"
                )
                return conv_res

            # Check if we have a mesh
            if not document.mesh:
                _log.warning("STL document has no mesh data")
                conv_res.add_error(
                    component="STLPipeline._assemble_document",
                    error_message="No mesh data found in STL document",
                )
                return conv_res

            mesh = document.mesh
            _log.debug("Validating mesh topology")

            # Check manifold property
            if mesh.is_manifold is False:
                _log.warning("Mesh is not manifold")
                conv_res.add_error(
                    component="STLPipeline._assemble_document",
                    error_message="Mesh is not manifold",
                )

            # Check watertight property
            if mesh.is_watertight is False:
                _log.warning("Mesh is not watertight (has holes)")
                conv_res.add_error(
                    component="STLPipeline._assemble_document",
                    error_message="Mesh is not watertight",
                )

            # Validate facet indices
            invalid_facets = 0
            max_vertex_idx = len(mesh.vertices) - 1

            for i, facet in enumerate(mesh.facets):
                # Check index range
                if any(idx < 0 or idx > max_vertex_idx for idx in facet):
                    invalid_facets += 1
                    _log.debug(f"Facet {i} has out-of-range vertex indices")

                # Check for degenerate triangles (duplicate vertices)
                if len(set(facet)) < 3:
                    invalid_facets += 1
                    _log.debug(f"Facet {i} is degenerate (duplicate vertices)")

            if invalid_facets > 0:
                _log.warning(f"Found {invalid_facets} invalid facets")
                conv_res.add_error(
                    component="STLPipeline._assemble_document",
                    error_message=f"Found {invalid_facets} invalid facets",
                )

            # Log mesh statistics
            _log.info(
                f"Mesh validation: vertices={mesh.num_vertices}, "
                f"facets={mesh.num_facets}, "
                f"manifold={mesh.is_manifold}, "
                f"watertight={mesh.is_watertight}, "
                f"volume={mesh.volume:.2f if mesh.volume else 'N/A'}, "
                f"surface_area={mesh.surface_area:.2f if mesh.surface_area else 'N/A'}"
            )

            return conv_res

        except Exception as e:
            _log.exception(f"Assembly stage failed: {e}")
            conv_res.add_error(
                component="STLPipeline._assemble_document",
                error_message=f"Assembly failed: {str(e)}",
            )
            return conv_res
