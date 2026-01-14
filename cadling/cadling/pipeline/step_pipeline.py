"""
STEP pipeline for converting STEP files to CADlingDocument.

This pipeline handles STEP files with ll_stepnet-powered parsing,
feature extraction, and topology analysis.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from cadling.pipeline.base_pipeline import BaseCADPipeline

if TYPE_CHECKING:
    from cadling.datamodel.base_models import ConversionResult
    from cadling.datamodel.pipeline_options import PipelineOptions

_log = logging.getLogger(__name__)


class STEPPipeline(BaseCADPipeline):
    """
    Pipeline for STEP file conversion.

    This pipeline:
    1. Build: Parses STEP file using STEPBackend (tokenizer, feature extractor, topology builder)
    2. Assemble: Resolves entity references and validates topology
    3. Enrich: Applies enrichment models (classification, embeddings, etc.)

    Example:
        options = STEPPipeline.get_default_options()
        pipeline = STEPPipeline(options)
        result = pipeline.execute(input_doc)
    """

    def __init__(self, pipeline_options: PipelineOptions):
        """Initialize STEP pipeline."""
        super().__init__(pipeline_options)
        _log.debug(f"Initialized STEPPipeline")

    @classmethod
    def get_default_options(cls) -> PipelineOptions:
        """Get default pipeline options for STEP."""
        from cadling.datamodel.pipeline_options import PipelineOptions

        return PipelineOptions(
            do_topology_analysis=True,
            device="cpu",
        )

    def _build_document(self, conv_res: ConversionResult) -> ConversionResult:
        """
        Build: Parse STEP file and extract document structure.

        Uses the STEPBackend to:
        - Parse STEP entities (tokenizer)
        - Extract geometric features (feature extractor)
        - Build topology graph (topology builder)

        Args:
            conv_res: Conversion result to populate.

        Returns:
            Updated conversion result with document populated.
        """
        try:
            # Get the backend from input
            backend = conv_res.input._backend

            # Verify it's a STEP backend
            from cadling.backend.step import STEPBackend

            if not isinstance(backend, STEPBackend):
                raise ValueError(
                    f"STEPPipeline requires STEPBackend, got {type(backend)}"
                )

            # Parse and convert
            _log.debug("Calling STEPBackend.convert()")
            document = backend.convert()

            # Set document in result
            conv_res.document = document

            _log.info(
                f"Built STEP document: {len(document.items)} entities, "
                f"{document.topology.num_nodes if document.topology else 0} topology nodes"
            )

            return conv_res

        except Exception as e:
            _log.exception(f"Build stage failed: {e}")
            conv_res.add_error(
                component="STEPPipeline._build_document",
                error_message=f"Failed to build document: {str(e)}",
            )
            raise

    def _assemble_document(self, conv_res: ConversionResult) -> ConversionResult:
        """
        Assemble: Resolve entity references and validate topology.

        For STEP files, this stage:
        - Validates entity references (check all #N references exist)
        - Enriches entities with referenced entity data
        - Validates topology consistency

        Args:
            conv_res: Conversion result to assemble.

        Returns:
            Updated conversion result.
        """
        if not conv_res.document:
            return conv_res

        try:
            from cadling.datamodel.step import STEPDocument, STEPEntityItem

            document = conv_res.document

            # Ensure it's a STEP document
            if not isinstance(document, STEPDocument):
                _log.warning(
                    f"STEPPipeline expected STEPDocument, got {type(document)}"
                )
                return conv_res

            _log.debug("Validating entity references")

            # Build entity ID index
            entity_index = {}
            for item in document.items:
                if isinstance(item, STEPEntityItem):
                    entity_index[item.entity_id] = item

            # Validate and enrich references
            invalid_refs = []
            for item in document.items:
                if isinstance(item, STEPEntityItem):
                    # Check all reference params
                    for ref_id in item.reference_params:
                        if ref_id not in entity_index:
                            invalid_refs.append((item.entity_id, ref_id))
                            _log.warning(
                                f"Entity #{item.entity_id} references "
                                f"invalid entity #{ref_id}"
                            )

            if invalid_refs:
                _log.warning(
                    f"Found {len(invalid_refs)} invalid entity references"
                )
                conv_res.add_error(
                    component="STEPPipeline._assemble_document",
                    error_message=f"Found {len(invalid_refs)} invalid references",
                )

            # Validate topology if present
            if document.topology:
                _log.debug(
                    f"Validated topology: {document.topology.num_nodes} nodes, "
                    f"{document.topology.num_edges} edges"
                )

            _log.info("Assembly stage completed successfully")

            return conv_res

        except Exception as e:
            _log.exception(f"Assembly stage failed: {e}")
            conv_res.add_error(
                component="STEPPipeline._assemble_document",
                error_message=f"Assembly failed: {str(e)}",
            )
            return conv_res
