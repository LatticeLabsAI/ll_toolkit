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
from cadling.experimental.models import (
    FeatureRecognitionVlmModel,
    PMIExtractionModel,
)

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

        Uses BRepGraphBuilder and geometry extractors (HoleGeometryExtractor,
        PocketGeometryExtractor) for robust feature detection.

        Args:
            doc: The document
            item: The item to process
        """
        detected_features = []

        # Try to use BRepGraphBuilder for topology-based feature extraction
        try:
            from cadling.models.segmentation.brep_graph_builder import BRepFaceGraphBuilder
            from cadling.models.segmentation.geometry_extractors import (
                HoleGeometryExtractor,
                PocketGeometryExtractor,
            )

            graph_builder = BRepFaceGraphBuilder()
            hole_extractor = HoleGeometryExtractor()
            pocket_extractor = PocketGeometryExtractor()

            # Build face graph from item
            graph = graph_builder.build_face_graph(doc, item)

            if graph is not None and graph.num_nodes > 0:
                # Analyze face surface types from graph
                for face_idx in range(graph.num_nodes):
                    face_features = graph.x[face_idx] if hasattr(graph, "x") else None

                    if face_features is not None:
                        # Decode surface type from features (simplified)
                        # Real implementation would use trained classifier
                        surface_type = self._classify_surface_type(face_features)

                        if surface_type == "cylindrical":
                            # Extract hole parameters
                            hole_params = hole_extractor.extract_from_face(
                                graph, face_idx
                            ) if hasattr(hole_extractor, "extract_from_face") else {}

                            detected_features.append({
                                "feature_type": "hole",
                                "subtype": hole_params.get("hole_type", "unknown"),
                                "parameters": {
                                    "diameter": hole_params.get("diameter"),
                                    "depth": hole_params.get("depth"),
                                },
                                "confidence": 0.75,
                                "source": "brep_graph_analysis",
                                "face_ids": [face_idx],
                            })

                        elif surface_type == "planar_recessed":
                            # Extract pocket parameters
                            pocket_params = pocket_extractor.extract_from_face(
                                graph, face_idx
                            ) if hasattr(pocket_extractor, "extract_from_face") else {}

                            detected_features.append({
                                "feature_type": "pocket",
                                "subtype": pocket_params.get("pocket_type", "rectangular"),
                                "parameters": {
                                    "width": pocket_params.get("width"),
                                    "length": pocket_params.get("length"),
                                    "depth": pocket_params.get("depth"),
                                },
                                "confidence": 0.7,
                                "source": "brep_graph_analysis",
                                "face_ids": [face_idx],
                            })

                _log.debug(
                    f"[Stage 1] BRepGraphBuilder: {graph.num_nodes} faces analyzed"
                )

        except ImportError as e:
            _log.debug(f"BRepGraphBuilder not available: {e}")
        except Exception as e:
            _log.warning(f"BRepGraphBuilder feature extraction failed: {e}")

        # Fallback: check if topology is available in doc
        if not detected_features and hasattr(doc, "topology") and doc.topology:
            topo = doc.topology

            # Simple heuristic: circular faces might be holes
            faces = topo.get("faces", [])
            for face in faces:
                if face.get("geometry_type") == "cylinder":
                    detected_features.append({
                        "feature_type": "hole",
                        "subtype": "unknown",
                        "parameters": {},
                        "confidence": 0.5,
                        "source": "topology_heuristic",
                    })

        # Store detected features
        item.properties["machining_features"] = detected_features
        item.properties["geometric_analysis_stage"] = "complete"

        _log.debug(f"[Stage 1] Detected {len(detected_features)} geometric features")

    def _classify_surface_type(self, face_features) -> str:
        """Classify surface type from face feature vector.

        Args:
            face_features: Feature tensor for the face

        Returns:
            Surface type string
        """
        # Simplified classification based on feature patterns
        # Real implementation would use trained classifier
        try:
            import torch

            if isinstance(face_features, torch.Tensor):
                features = face_features.detach().cpu().numpy()
            else:
                features = face_features

            # Check for cylindrical surface indicators
            # (curvature patterns, normal distribution, etc.)
            if len(features) > 3:
                # Example heuristic: high curvature in one direction
                curvature_idx = min(3, len(features) - 1)
                if abs(features[curvature_idx]) > 0.1:
                    return "cylindrical"

            return "planar"

        except Exception:
            return "unknown"

    def _render_views(self, doc, item) -> None:
        """Render views for VLM processing (Stage 1 helper).

        Args:
            doc: The document
            item: The item to process
        """
        rendered_images = {}

        # Try to get shape from item or document for rendering
        shape = getattr(item, "_shape", None)
        if shape is None:
            shape = item.properties.get("_shape")
        if shape is None and hasattr(doc, "_backend") and doc._backend is not None:
            # Try to get shape from document backend
            backend = doc._backend
            if hasattr(backend, "shape"):
                shape = backend.shape
            elif hasattr(backend, "_shape"):
                shape = backend._shape

        if shape is None:
            _log.debug(f"[Stage 1] No shape available for rendering")
            item.properties["rendered_images"] = rendered_images
            item.properties["rendering_stage"] = "skipped"
            return

        # Try to get rendering backend
        backend = None
        if hasattr(doc, "_backend") and doc._backend is not None:
            backend = doc._backend
        elif hasattr(shape, "render_view") and callable(shape.render_view):
            backend = shape

        # Render each configured view
        for view_name in self.options.views_to_process:
            try:
                if backend is not None and hasattr(backend, "render_view"):
                    rendered_image = backend.render_view(
                        view_name,
                        resolution=getattr(self.options, "resolution", 512),
                    )
                    if rendered_image is not None:
                        rendered_images[view_name] = rendered_image
                        _log.debug(f"[Stage 1] Rendered view: {view_name}")
                else:
                    _log.debug(f"[Stage 1] Skipping view {view_name}: no rendering backend")

            except Exception as e:
                _log.warning(f"[Stage 1] Failed to render view {view_name}: {e}")

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
