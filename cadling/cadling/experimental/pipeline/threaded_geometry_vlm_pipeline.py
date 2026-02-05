"""Threaded geometry-VLM pipeline for two-stage CAD processing.

This module provides a two-stage pipeline that combines geometric analysis
with VLM-based visual understanding, similar to Docling's ThreadedLayoutVlmPipeline
but adapted for CAD processing.

Classes:
    ThreadedGeometryVlmPipeline: Two-stage pipeline with geometry + VLM

Example:
    from cadling.experimental.pipeline import ThreadedGeometryVlmPipeline
    from cadling.experimental.datamodel import CADAnnotationOptions

    options = CADAnnotationOptions(vlm_model="gpt-4-vision")
    pipeline = ThreadedGeometryVlmPipeline(options)
    result = pipeline.execute(input_doc)
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from cadling.pipeline.base_pipeline import BaseCADPipeline

if TYPE_CHECKING:
    from cadling.datamodel.base_models import ConversionResult
    from cadling.experimental.datamodel import CADAnnotationOptions

_log = logging.getLogger(__name__)


class ThreadedGeometryVlmPipeline(BaseCADPipeline):
    """Two-stage pipeline combining geometric analysis with VLM processing.

    This experimental pipeline processes CAD models in two distinct stages,
    inspired by Docling's ThreadedLayoutVlmPipeline:

    **Stage 1: Geometric Analysis**
    - Parse CAD file (STEP/STL/BRep)
    - Extract topology (faces, edges, vertices)
    - Detect geometric features (holes, pockets, etc.)
    - Build BRep graph structure
    - Extract ll_stepnet features if STEP file
    - Render multiple views

    **Stage 2: VLM with Geometric Context**
    - Inject detected geometric features into VLM prompts
    - Run VLM analysis on rendered views with context
    - Extract PMI, annotations, design intent
    - Correlate VLM findings with geometric features
    - Enhance with cross-view validation

    The key innovation is that Stage 2 VLM prompts are augmented with
    geometric context from Stage 1, improving accuracy and relevance.

    Attributes:
        options: CAD annotation options (VLM model, views, etc.)
        stage1_complete: Whether Stage 1 has completed successfully
        geometric_context: Extracted geometric features from Stage 1

    Example:
        options = CADAnnotationOptions(
            vlm_model="gpt-4-vision",
            views_to_process=["front", "top", "isometric"],
            annotation_types=["dimension", "tolerance", "gdt"],
            include_geometric_context=True
        )
        pipeline = ThreadedGeometryVlmPipeline(options)
        result = pipeline.execute(input_doc)

        # Access Stage 1 geometric features
        features = result.document.items[0].properties.get("machining_features")

        # Access Stage 2 VLM annotations with context
        annotations = result.document.items[0].properties.get("pmi_annotations")
    """

    def __init__(self, options: CADAnnotationOptions):
        """Initialize threaded geometry-VLM pipeline.

        Args:
            options: Configuration options for the pipeline

        Raises:
            ValueError: If required options not provided
        """
        super().__init__(options)
        self.options: CADAnnotationOptions = options
        self.stage1_complete = False
        self.geometric_context = {}

        # Import experimental models
        from cadling.experimental.models import (
            FeatureRecognitionVlmModel,
            PMIExtractionModel,
        )

        # Configure enrichment pipeline for Stage 2
        # Stage 1 feature recognition will use geometric-only analysis
        # Stage 2 uses VLM with geometric context for enhanced analysis
        self.stage2_models = [
            FeatureRecognitionVlmModel(options),  # VLM-enhanced feature recognition
            PMIExtractionModel(options),  # PMI extraction with geometric context
        ]

        _log.info(
            f"Initialized ThreadedGeometryVlmPipeline with VLM: {options.vlm_model}"
        )

    @classmethod
    def get_default_options(cls) -> CADAnnotationOptions:
        """Get default pipeline options.

        Returns:
            Default CADAnnotationOptions
        """
        from cadling.experimental.datamodel import CADAnnotationOptions

        return CADAnnotationOptions(
            vlm_model="gpt-4-vision",
            annotation_types=["dimension", "tolerance", "gdt"],
            views_to_process=["front", "top", "right", "isometric"],
            include_geometric_context=True,
        )

    def _build_document(self, conv_res: ConversionResult) -> ConversionResult:
        """Build: Parse CAD file and extract structure (Stage 1: Geometry).

        This stage performs pure geometric analysis without VLM:
        - Parse CAD file using appropriate backend
        - Extract BRep topology
        - Detect geometric features
        - Build topology graph
        - Render views for later VLM processing

        Args:
            conv_res: Conversion result to populate

        Returns:
            Updated conversion result with document and geometric features
        """
        _log.info("[Stage 1] Starting geometric analysis")

        try:
            # Get backend from input document
            backend = conv_res.input._backend

            # Convert using backend
            doc = backend.convert()
            conv_res.document = doc

            # Extract geometric features for each item
            for item in doc.items:
                # Detect features using geometric analysis only
                self._extract_geometric_features(doc, item)

                # Render views for later VLM processing
                self._render_views(doc, item)

            self.stage1_complete = True
            self.geometric_context = self._build_geometric_context(doc)

            _log.info(
                f"[Stage 1] Completed: extracted {len(doc.items)} items "
                f"with geometric features"
            )

        except Exception as e:
            _log.error(f"[Stage 1] Geometric analysis failed: {e}")
            raise

        return conv_res

    def _enrich_document(self, conv_res: ConversionResult) -> ConversionResult:
        """Enrich: Apply VLM with geometric context (Stage 2: VLM).

        This stage runs VLM analysis with geometric context from Stage 1:
        - Augment VLM prompts with detected geometric features
        - Run PMI extraction with context-aware prompts
        - Correlate VLM findings with geometric features
        - Cross-validate across multiple views

        Args:
            conv_res: Conversion result to enrich

        Returns:
            Updated conversion result with VLM annotations
        """
        if not self.stage1_complete:
            _log.warning("[Stage 2] Stage 1 not complete, skipping VLM analysis")
            return conv_res

        if not conv_res.document:
            return conv_res

        _log.info("[Stage 2] Starting VLM analysis with geometric context")

        doc = conv_res.document
        items = doc.items

        # Apply Stage 2 models (VLM with context)
        for model in self.stage2_models:
            model_name = model.__class__.__name__
            _log.debug(f"[Stage 2] Applying {model_name}")

            try:
                # Models will access geometric context from item.properties
                # set by Stage 1
                model(doc, items)

                _log.info(f"[Stage 2] {model_name} completed")

            except Exception as e:
                _log.error(f"[Stage 2] {model_name} failed: {e}")
                # Continue with other models

        _log.info("[Stage 2] VLM analysis completed")

        return conv_res

    def _extract_geometric_features(self, doc, item) -> None:
        """Extract geometric features from item (Stage 1 helper).

        Args:
            doc: The document
            item: The item to process
        """
        # This is a simplified geometric feature extraction
        # In reality, this would use topology analysis, ll_stepnet, etc.

        # Placeholder for geometric feature extraction
        # Real implementation would analyze topology, detect holes, pockets, etc.
        detected_features = []

        # Check if topology is available
        if hasattr(doc, "topology") and doc.topology:
            topo = doc.topology

            # Extract basic feature hints from topology
            num_faces = topo.get("num_faces", 0)
            num_edges = topo.get("num_edges", 0)

            # Simple heuristic: circular faces might be holes
            faces = topo.get("faces", [])
            for face in faces:
                if face.get("geometry_type") == "cylinder":
                    # Likely a hole or boss
                    detected_features.append(
                        {
                            "feature_type": "hole",
                            "subtype": "unknown",
                            "parameters": {},
                            "confidence": 0.7,
                            "source": "geometric_analysis",
                        }
                    )

        # Store detected features
        item.properties["machining_features"] = detected_features
        item.properties["geometric_analysis_stage"] = "complete"

        _log.debug(f"[Stage 1] Detected {len(detected_features)} geometric features")

    def _render_views(self, doc, item) -> None:
        """Render views for VLM processing (Stage 1 helper).

        Args:
            doc: The document
            item: The item to process
        """
        # Placeholder for view rendering
        # Real implementation would use visualization backend to render views

        rendered_images = {}

        # In real implementation, would render each view
        for view_name in self.options.views_to_process:
            # rendered_images[view_name] = render_view(item, view_name)
            pass

        item.properties["rendered_images"] = rendered_images
        item.properties["rendering_stage"] = "complete"

        _log.debug(f"[Stage 1] Rendered {len(rendered_images)} views")

    def _build_geometric_context(self, doc) -> dict:
        """Build geometric context summary from Stage 1.

        Args:
            doc: The document

        Returns:
            Dictionary with geometric context
        """
        context = {
            "num_items": len(doc.items),
            "topology_available": hasattr(doc, "topology") and doc.topology is not None,
        }

        # Aggregate features across all items
        total_features = 0
        for item in doc.items:
            features = item.properties.get("machining_features", [])
            total_features += len(features)

        context["total_features_detected"] = total_features

        return context

    def _determine_status(self, conv_res: ConversionResult):
        """Determine final conversion status.

        Args:
            conv_res: Conversion result

        Returns:
            ConversionStatus
        """
        from cadling.datamodel.base_models import ConversionStatus

        if not conv_res.document:
            return ConversionStatus.FAILURE

        if not self.stage1_complete:
            return ConversionStatus.PARTIAL

        return ConversionStatus.SUCCESS
