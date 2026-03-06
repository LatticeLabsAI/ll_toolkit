"""Base pipeline for CAD conversion.

This module provides the foundational pipeline architecture for cadling, adapted
from docling's pipeline pattern. Pipelines orchestrate the conversion workflow:
Build → Assemble → Enrich.

Classes:
    BaseCADPipeline: Abstract base class for all pipelines.
    EnrichmentModel: Abstract base class for enrichment models (imported from models.base_model).

Example:
    class MyPipeline(BaseCADPipeline):
        def _build_document(self, conv_res):
            # Parse CAD file
            conv_res.document = backend.convert()
            return conv_res
"""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, List, Optional

# Import EnrichmentModel from models.base_model (re-exported for convenience)
from cadling.models.base_model import EnrichmentModel

if TYPE_CHECKING:
    from cadling.datamodel.base_models import (
        CADItem,
        CADlingDocument,
        ConversionResult,
        ConversionStatus,
    )
    from cadling.datamodel.pipeline_options import PipelineOptions

_log = logging.getLogger(__name__)


class BaseCADPipeline(ABC):
    """Base pipeline for CAD conversion.

    This is the foundational abstract class that all CAD pipelines inherit from.
    It defines the three-stage conversion workflow:

    1. Build: Parse and extract structure from CAD file
    2. Assemble: Combine components and resolve references
    3. Enrich: Apply enrichment models (classification, embeddings, etc.)

    Similar to docling's BasePipeline but adapted for CAD-specific processing.

    Attributes:
        pipeline_options: Configuration options for this pipeline.
        enrichment_pipe: List of enrichment models to apply.
    """

    def __init__(self, pipeline_options: PipelineOptions):
        """Initialize pipeline.

        Args:
            pipeline_options: Pipeline configuration options.
        """
        self.pipeline_options = pipeline_options
        self.enrichment_pipe: List[EnrichmentModel] = (
            pipeline_options.enrichment_models or []
        )

        _log.info(
            f"Initialized {self.__class__.__name__} with "
            f"{len(self.enrichment_pipe)} enrichment models"
        )

    @classmethod
    def get_default_options(cls) -> PipelineOptions:
        """Get default pipeline options.

        Returns:
            Default PipelineOptions for this pipeline.

        Example:
            options = SimpleCADPipeline.get_default_options()
            pipeline = SimpleCADPipeline(options)
        """
        from cadling.datamodel.pipeline_options import PipelineOptions

        return PipelineOptions()

    def execute(self, in_doc: "CADInputDocument") -> ConversionResult:
        """Execute the full conversion pipeline.

        This is the main entry point that orchestrates the three-stage workflow:
        Build → Assemble → Enrich.

        Args:
            in_doc: Input document descriptor with backend attached.

        Returns:
            ConversionResult with status and converted document (or errors).

        Example:
            result = pipeline.execute(input_doc)
            if result.status == ConversionStatus.SUCCESS:
                print(f"Converted {result.document.name}")
        """
        from cadling.datamodel.base_models import ConversionResult, ConversionStatus

        # Initialize result
        conv_res = ConversionResult(input=in_doc)

        start_time = time.time()

        try:
            # Stage 1: Build - Parse and extract structure
            _log.info(f"[Build] Starting for {in_doc.file.name}")
            build_start = time.time()
            conv_res = self._build_document(conv_res)
            build_duration = (time.time() - build_start) * 1000

            if conv_res.document:
                conv_res.document.add_processing_step(
                    step_name="build",
                    component=self.__class__.__name__,
                    duration_ms=build_duration,
                )
                _log.info(
                    f"[Build] Completed in {build_duration:.2f}ms, "
                    f"extracted {len(conv_res.document.items)} items"
                )

            # Stage 2: Assemble - Combine components
            _log.info("[Assemble] Starting")
            assemble_start = time.time()
            conv_res = self._assemble_document(conv_res)
            assemble_duration = (time.time() - assemble_start) * 1000

            if conv_res.document:
                conv_res.document.add_processing_step(
                    step_name="assemble",
                    component=self.__class__.__name__,
                    duration_ms=assemble_duration,
                )
                _log.info(f"[Assemble] Completed in {assemble_duration:.2f}ms")

            # Stage 3: Enrich - Apply models
            if self.enrichment_pipe and conv_res.document:
                _log.info(
                    f"[Enrich] Starting with {len(self.enrichment_pipe)} models"
                )
                enrich_start = time.time()
                conv_res = self._enrich_document(conv_res)
                enrich_duration = (time.time() - enrich_start) * 1000

                conv_res.document.add_processing_step(
                    step_name="enrich",
                    component=self.__class__.__name__,
                    duration_ms=enrich_duration,
                )
                _log.info(f"[Enrich] Completed in {enrich_duration:.2f}ms")

            # Determine final status
            conv_res.status = self._determine_status(conv_res)

            total_duration = (time.time() - start_time) * 1000
            _log.info(
                f"Pipeline completed in {total_duration:.2f}ms "
                f"with status {conv_res.status}"
            )

        except Exception as e:
            _log.exception(f"Pipeline failed with error: {e}")
            conv_res.status = ConversionStatus.FAILURE
            conv_res.add_error(
                component=self.__class__.__name__,
                error_message=f"Pipeline execution failed: {str(e)}",
            )

        return conv_res

    @abstractmethod
    def _build_document(self, conv_res: ConversionResult) -> ConversionResult:
        """Build: Parse and extract document structure.

        This is the core method that pipelines must implement. It should use
        the backend to parse the CAD file and populate conv_res.document.

        Args:
            conv_res: Conversion result to populate.

        Returns:
            Updated conversion result with document populated.

        Example:
            def _build_document(self, conv_res):
                backend = conv_res.input._backend
                conv_res.document = backend.convert()
                return conv_res
        """
        pass

    def _assemble_document(self, conv_res: ConversionResult) -> ConversionResult:
        """Assemble: Combine components and resolve references.

        Default implementation is a no-op. Override for pipelines that need
        to combine multiple components (e.g., multi-part assemblies) or
        resolve entity references.

        Args:
            conv_res: Conversion result to assemble.

        Returns:
            Updated conversion result.

        Example:
            def _assemble_document(self, conv_res):
                # Resolve STEP entity references
                for item in conv_res.document.items:
                    if isinstance(item, STEPEntityItem):
                        self._resolve_references(item, conv_res.document)
                return conv_res
        """
        return conv_res

    def _enrich_document(self, conv_res: ConversionResult) -> ConversionResult:
        """Enrich: Apply enrichment models.

        Default implementation applies all enrichment models in the pipeline
        to the document items. Can be overridden for custom enrichment logic.

        Args:
            conv_res: Conversion result to enrich.

        Returns:
            Updated conversion result with enriched items.
        """
        if not conv_res.document or not self.enrichment_pipe:
            return conv_res

        doc = conv_res.document
        items = doc.items

        # Apply each enrichment model
        for model in self.enrichment_pipe:
            model_name = model.__class__.__name__
            _log.debug(f"Applying enrichment model: {model_name}")

            try:
                # Apply model to all items (models can filter internally).
                # Provenance is the model's responsibility — each model stamps
                # only the items it actually processed (per EnrichmentModel contract).
                model(doc, items)

            except Exception as e:
                _log.error(
                    f"Enrichment model {model_name} failed: {e}",
                    exc_info=True,
                )
                conv_res.add_error(
                    component=model_name,
                    error_message=f"Enrichment failed: {str(e)}",
                )

        return conv_res

    def _determine_status(self, conv_res: ConversionResult) -> ConversionStatus:
        """Determine final conversion status.

        Args:
            conv_res: Conversion result.

        Returns:
            Appropriate ConversionStatus based on result state.
        """
        from cadling.datamodel.base_models import ConversionStatus

        # No document means failure
        if not conv_res.document:
            return ConversionStatus.FAILURE

        # Errors present means partial success
        if conv_res.errors:
            return ConversionStatus.PARTIAL

        # Otherwise success
        return ConversionStatus.SUCCESS
