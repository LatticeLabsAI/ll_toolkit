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
import math
from typing import TYPE_CHECKING

import numpy as np

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
        """Extract machining features from the item's B-Rep face graph (Stage 1).

        Builds the face graph with ``BRepFaceGraphBuilder`` and reads each face's
        REAL geometry from its 24-dim node features: the parsed surface type
        (one-hot), area, curvatures, normal, centroid and bbox. Cylindrical faces
        become holes (diameter from mean curvature, depth from lateral area);
        planar faces become pockets only when geometrically recessed.

        Args:
            doc: The document
            item: The item to process
        """
        detected_features = []

        # Use BRepFaceGraphBuilder for topology-based feature detection. The
        # graph's 24-dim node features carry REAL per-face geometry, so we read
        # the parsed surface type and measured quantities directly rather than
        # guessing from a curvature threshold.
        try:
            from cadling.models.segmentation.brep_graph_builder import BRepFaceGraphBuilder

            graph_builder = BRepFaceGraphBuilder()
            graph = graph_builder.build_face_graph(doc, item)

            node_feats = (
                self._features_to_numpy(graph.x)
                if graph is not None and hasattr(graph, "x")
                else None
            )
            if (
                node_feats is not None
                and node_feats.ndim == 2
                and node_feats.shape[1] >= 22
            ):
                for face_idx in range(node_feats.shape[0]):
                    feat = node_feats[face_idx]
                    surface_type, type_conf = self._classify_surface_type(feat)

                    if surface_type == "cylindrical":
                        # Cylindrical face -> hole, with REAL parameters measured
                        # from the node features (diameter from mean curvature,
                        # depth from lateral area, location from centroid).
                        params = self._hole_parameters(feat)
                        detected_features.append({
                            "feature_type": "hole",
                            "subtype": "cylindrical",
                            "parameters": params,
                            "confidence": round(min(0.5 + 0.35 * type_conf, 0.85), 3),
                            "source": "brep_graph_geometry",
                            "detection_method": "surface_type_and_curvature",
                            "face_ids": [face_idx],
                        })

                    elif surface_type == "planar":
                        # Planar face -> pocket ONLY if it is geometrically
                        # recessed (real recession test over face centroids).
                        pocket = self._planar_recession(face_idx, node_feats)
                        if pocket is not None:
                            detected_features.append({
                                "feature_type": "pocket",
                                "subtype": "recessed_planar",
                                "parameters": pocket,
                                "confidence": round(min(0.45 + 0.3 * type_conf, 0.8), 3),
                                "source": "brep_graph_geometry",
                                "detection_method": "planar_recession",
                                "face_ids": [face_idx],
                            })

                _log.debug(
                    f"[Stage 1] BRepFaceGraphBuilder: {node_feats.shape[0]} faces "
                    f"analyzed, {len(detected_features)} features detected"
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

    # Node-feature layout produced by
    # cadling.models.segmentation.brep_graph_builder._extract_face_features
    # (feature_dim=24): [0:10] surface-type one-hot, [10] area,
    # [11] gaussian curvature, [12] mean curvature, [13:16] normal,
    # [16:19] centroid, [19:22] bbox dimensions.
    _SURFACE_TYPES = (
        "PLANE", "CYLINDER", "CONE", "SPHERE", "TORUS",
        "B_SPLINE", "BEZIER", "NURBS", "SURFACE_OF_REVOLUTION", "OTHER",
    )

    @staticmethod
    def _features_to_numpy(face_features):
        """Coerce a node-feature tensor/array to a numpy array, or None."""
        if face_features is None:
            return None
        try:
            import torch

            if isinstance(face_features, torch.Tensor):
                return face_features.detach().cpu().numpy()
        except Exception:
            pass
        try:
            return np.asarray(face_features, dtype=float)
        except Exception:
            return None

    def _classify_surface_type(self, face_features) -> tuple[str, float]:
        """Classify a face's surface type from its REAL parsed surface-type
        one-hot (NOT a curvature threshold).

        Args:
            face_features: A single face's node-feature vector.

        Returns:
            ``(surface_type, confidence)`` with ``surface_type`` one of
            "cylindrical", "planar", "other" or "unknown", and confidence equal
            to the winning one-hot magnitude (1.0 when the STEP surface type was
            parsed confidently).
        """
        features = self._features_to_numpy(face_features)
        if features is None or len(features) < 10:
            return "unknown", 0.0
        onehot = features[:10]
        idx = int(np.argmax(onehot))
        conf = float(onehot[idx])
        if conf <= 0.0:
            return "unknown", 0.0
        name = self._SURFACE_TYPES[idx]
        if name == "CYLINDER":
            return "cylindrical", conf
        if name == "PLANE":
            return "planar", conf
        return "other", conf

    @staticmethod
    def _hole_parameters(feat) -> dict:
        """Measure hole parameters from a cylindrical face's node features.

        Diameter from the cylinder's mean curvature (``|H| = 1/(2R)`` so
        ``d = 1/|H|``), falling back to the smallest bbox cross-section; depth
        from the lateral area (``A = pi*d*h`` so ``h = A/(pi*d)``), falling back
        to the largest bbox dimension; location from the centroid.
        """
        area = float(feat[10])
        mean_curv = abs(float(feat[12]))
        centroid = [float(x) for x in feat[16:19]]
        bbox = [float(x) for x in feat[19:22]]

        if mean_curv > 1e-6:
            diameter = 1.0 / mean_curv
        else:
            cross = sorted(d for d in bbox if d > 1e-9)
            diameter = cross[0] if cross else 0.0

        if diameter > 1e-9 and area > 0:
            depth = area / (math.pi * diameter)
        else:
            depth = max(bbox) if bbox else 0.0

        return {
            "diameter": round(diameter, 4),
            "depth": round(depth, 4),
            "location": centroid,
        }

    @staticmethod
    def _planar_recession(face_idx: int, node_feats) -> dict | None:
        """Detect a recessed planar face (pocket floor) from real geometry.

        A planar face is recessed when other faces lie farther out along its
        outward normal (there is material "above" it). Every face centroid is
        projected onto this face's normal; if the face sits below the outermost
        projection by more than 10% of the model's extent along that normal, it
        is a pocket floor whose depth is that recession distance. Width/length
        come from the face's bbox dimensions, location from its centroid.

        Returns the pocket parameters dict, or None when the face is not
        recessed (so no pocket is fabricated for flat outer faces).
        """
        if node_feats.shape[0] < 3:
            return None
        normal = np.asarray(node_feats[face_idx, 13:16], dtype=float)
        norm = float(np.linalg.norm(normal))
        if norm < 1e-9:
            return None
        normal = normal / norm

        centroids = np.asarray(node_feats[:, 16:19], dtype=float)
        projections = centroids @ normal
        self_proj = float(projections[face_idx])
        max_proj = float(projections.max())
        recession = max_proj - self_proj
        extent = float(projections.max() - projections.min())
        if extent < 1e-9 or recession < 0.1 * extent:
            return None

        dims = sorted((float(d) for d in node_feats[face_idx, 19:22]), reverse=True)
        width = dims[0] if dims else 0.0
        length = dims[1] if len(dims) > 1 else width
        return {
            "width": round(width, 4),
            "length": round(length, 4),
            "depth": round(recession, 4),
            "location": [float(x) for x in node_feats[face_idx, 16:19]],
        }

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
