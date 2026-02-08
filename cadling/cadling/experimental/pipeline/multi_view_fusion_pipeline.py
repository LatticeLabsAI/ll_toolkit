"""Multi-view fusion pipeline for comprehensive CAD analysis.

This module provides a pipeline that renders multiple views of a CAD model,
runs VLM analysis on each view independently, and fuses the results to
achieve more robust and complete feature extraction.

Classes:
    MultiViewFusionPipeline: Pipeline with multi-view rendering and fusion

Example:
    from cadling.experimental.pipeline import MultiViewFusionPipeline
    from cadling.experimental.datamodel import MultiViewOptions

    options = MultiViewOptions(fusion_strategy="weighted_consensus")
    pipeline = MultiViewFusionPipeline(options)
    result = pipeline.execute(input_doc)
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import TYPE_CHECKING, Any, Dict, List

from cadling.pipeline.base_pipeline import BaseCADPipeline
from cadling.experimental.models import (
    FeatureRecognitionVlmModel,
    PMIExtractionModel,
)

if TYPE_CHECKING:
    from cadling.datamodel.base_models import ConversionResult
    from cadling.experimental.datamodel import MultiViewOptions

_log = logging.getLogger(__name__)


class MultiViewFusionPipeline(BaseCADPipeline):
    """Multi-view rendering and fusion pipeline for CAD analysis.

    This experimental pipeline processes CAD models by:

    **Multi-View Rendering:**
    - Render 6+ standard engineering views (front, top, right, back, bottom, left)
    - Include isometric and custom views if specified
    - Configurable resolution and rendering settings
    - Parallel rendering for performance

    **Per-View VLM Analysis:**
    - Run VLM analysis independently on each view
    - Extract features, annotations, dimensions from each perspective
    - Maintain per-view confidence scores

    **Information Fusion:**
    - Fuse overlapping/conflicting information across views
    - Multiple fusion strategies:
      * `weighted_consensus`: Weight by confidence, require majority agreement
      * `majority_vote`: Simple majority voting across views
      * `hierarchical`: Priority-based (front/top > others)
    - Identify consensus features with high confidence
    - Flag conflicts for human review

    **Advantages:**
    - More complete feature detection (features visible in some views but not others)
    - Improved accuracy through cross-validation
    - Reduced false positives (features must appear in multiple views)
    - Better handling of complex geometries

    Attributes:
        options: Multi-view pipeline options
        per_view_results: Results from each view (for debugging/analysis)
        fusion_strategy: Strategy for fusing multi-view results

    Example:
        from cadling.experimental.datamodel import MultiViewOptions, ViewConfig

        options = MultiViewOptions(
            views=[
                ViewConfig(name="front", azimuth=0, elevation=0),
                ViewConfig(name="top", azimuth=0, elevation=90),
                ViewConfig(name="isometric", azimuth=45, elevation=35.264),
            ],
            fusion_strategy="weighted_consensus",
            resolution=2048,
            parallel_rendering=True
        )
        pipeline = MultiViewFusionPipeline(options)
        result = pipeline.execute(input_doc)

        # Access fused features
        features = result.document.items[0].properties.get("fused_features")

        # Access per-view results
        per_view = result.document.items[0].properties.get("per_view_results")
    """

    def __init__(self, options: MultiViewOptions):
        """Initialize multi-view fusion pipeline.

        Args:
            options: Configuration options for the pipeline

        Raises:
            ValueError: If fusion strategy not recognized
        """
        super().__init__(options)
        self.options: MultiViewOptions = options
        self.per_view_results: Dict[str, Dict] = {}

        # Validate fusion strategy
        valid_strategies = ["weighted_consensus", "majority_vote", "hierarchical"]
        if options.fusion_strategy not in valid_strategies:
            raise ValueError(
                f"Invalid fusion_strategy: {options.fusion_strategy}. "
                f"Must be one of {valid_strategies}"
            )

        self.fusion_strategy = options.fusion_strategy

        # Configure enrichment models
        from cadling.experimental.datamodel import CADAnnotationOptions

        ann_options = CADAnnotationOptions(
            vlm_model=getattr(options, "vlm_model", "gpt-4-vision"),
            views_to_process=[v.name for v in options.views],
        )

        self.enrichment_pipe = [
            PMIExtractionModel(ann_options),
            FeatureRecognitionVlmModel(ann_options),
        ]

        _log.info(
            f"Initialized MultiViewFusionPipeline with {len(options.views)} views "
            f"using {options.fusion_strategy} fusion"
        )

    @classmethod
    def get_default_options(cls) -> MultiViewOptions:
        """Get default pipeline options.

        Returns:
            Default MultiViewOptions
        """
        from cadling.experimental.datamodel import MultiViewOptions

        return MultiViewOptions()

    def _build_document(self, conv_res: ConversionResult) -> ConversionResult:
        """Build: Parse CAD file and render multiple views.

        This stage:
        - Parses CAD file using appropriate backend
        - Renders all configured views
        - Stores rendered images for VLM processing

        Args:
            conv_res: Conversion result to populate

        Returns:
            Updated conversion result with document and rendered views
        """
        _log.info("[Build] Starting CAD parsing and multi-view rendering")

        try:
            # Get backend from input document
            backend = conv_res.input._backend

            # Convert using backend
            doc = backend.convert()
            conv_res.document = doc

            # Render all views for each item
            for item in doc.items:
                rendered_views = self._render_all_views(item)
                item.properties["rendered_images"] = rendered_views
                item.properties["num_views"] = len(rendered_views)

            _log.info(
                f"[Build] Completed: extracted {len(doc.items)} items, "
                f"rendered {len(self.options.views)} views per item"
            )

        except Exception as e:
            _log.error(f"[Build] Build failed: {e}")
            raise

        return conv_res

    def _enrich_document(self, conv_res: ConversionResult) -> ConversionResult:
        """Enrich: Run VLM on each view and fuse results.

        This stage:
        - Runs VLM analysis on each view independently
        - Collects per-view results
        - Fuses results using configured strategy
        - Adds fused results to document

        Args:
            conv_res: Conversion result to enrich

        Returns:
            Updated conversion result with fused annotations
        """
        if not conv_res.document:
            return conv_res

        _log.info("[Enrich] Starting multi-view VLM analysis and fusion")

        doc = conv_res.document
        items = doc.items

        # Apply enrichment models (they process all views)
        for model in self.enrichment_pipe:
            model_name = model.__class__.__name__
            _log.debug(f"[Enrich] Applying {model_name} across all views")

            try:
                model(doc, items)
            except Exception as e:
                _log.error(f"[Enrich] {model_name} failed: {e}")

        # Fuse results across views for each item
        for item in items:
            self._fuse_multi_view_results(item)

        _log.info("[Enrich] Multi-view fusion completed")

        return conv_res

    def _render_all_views(self, item) -> Dict[str, Any]:
        """Render all configured views for an item.

        Attempts rendering in order of preference:
        1. Item's associated backend (if it has ``render_view()``)
        2. Shape's own ``render_view()`` method
        3. Software fallback via Open3D / trimesh (if available)
        4. Placeholder metadata (always succeeds — stores view params)

        Args:
            item: The CAD item to render

        Returns:
            Dictionary mapping view name to rendered image or metadata dict
        """
        rendered_views = {}

        # Try to get shape from item for rendering
        shape = getattr(item, "_shape", None)
        if shape is None and hasattr(item, "properties") and isinstance(item.properties, dict):
            shape = item.properties.get("_shape")

        # Try to get rendering backend
        backend = None
        try:
            if hasattr(item, "_backend") and item._backend is not None:
                backend = item._backend
            elif shape is not None and hasattr(shape, "render_view") and callable(shape.render_view):
                backend = shape
        except Exception as e:
            _log.debug(f"Could not get rendering backend: {e}")

        # Build render kwargs from options (uses resolution, lighting, etc.)
        render_opts = {
            "resolution": getattr(self.options, "resolution", 1024),
            "enable_lighting": getattr(self.options, "enable_lighting", True),
            "background_color": getattr(self.options, "background_color", "white"),
            "anti_aliasing": getattr(self.options, "anti_aliasing", True),
            "render_edges": getattr(self.options, "render_edges", False),
        }

        # Render each configured view
        for view_config in self.options.views:
            view_name = view_config.name

            try:
                rendered_image = None

                # Strategy 1: backend with render_view
                if backend is not None and hasattr(backend, "render_view"):
                    rendered_image = backend.render_view(
                        view_name,
                        azimuth=view_config.azimuth,
                        elevation=view_config.elevation,
                        distance=getattr(view_config, "distance", None),
                        **render_opts,
                    )

                # Strategy 2: software fallback via Open3D
                if rendered_image is None and shape is not None:
                    rendered_image = self._software_render(
                        shape, view_config, render_opts
                    )

                # Strategy 3: placeholder metadata (always succeeds)
                if rendered_image is None:
                    rendered_image = {
                        "type": "placeholder",
                        "view_name": view_name,
                        "azimuth": view_config.azimuth,
                        "elevation": view_config.elevation,
                        "distance": getattr(view_config, "distance", None),
                        "render_options": render_opts,
                    }
                    _log.debug(
                        f"[Render] Using placeholder for view {view_name}: "
                        f"no rendering backend available"
                    )

                rendered_views[view_name] = rendered_image
                _log.debug(
                    f"[Render] View {view_name}: "
                    f"az={view_config.azimuth}, el={view_config.elevation}"
                )

            except Exception as e:
                _log.warning(f"Failed to render view {view_name}: {e}")
                # Still store placeholder on error
                rendered_views[view_name] = {
                    "type": "error",
                    "view_name": view_name,
                    "error": str(e),
                }

        return rendered_views

    @staticmethod
    def _software_render(shape, view_config, render_opts) -> Any:
        """Attempt software rendering via Open3D or trimesh.

        Args:
            shape: OCC shape object
            view_config: View configuration
            render_opts: Rendering options dict

        Returns:
            Rendered image data, or None if libraries unavailable
        """
        try:
            import numpy as np

            # Try Open3D first
            try:
                import open3d as o3d
                from OCC.Extend.DataExchange import write_stl_file
                import tempfile
                import os

                # Export shape to temporary STL for Open3D
                with tempfile.NamedTemporaryFile(suffix=".stl", delete=False) as tmp:
                    tmp_path = tmp.name

                write_stl_file(shape, tmp_path)
                mesh = o3d.io.read_triangle_mesh(tmp_path)
                os.unlink(tmp_path)

                if mesh.is_empty():
                    return None

                mesh.compute_vertex_normals()

                # Set up off-screen renderer
                resolution = render_opts.get("resolution", 1024)
                renderer = o3d.visualization.rendering.OffscreenRenderer(
                    resolution, resolution
                )

                # Configure material
                material = o3d.visualization.rendering.MaterialRecord()
                material.shader = "defaultLit" if render_opts.get("enable_lighting", True) else "defaultUnlit"

                renderer.scene.add_geometry("mesh", mesh, material)

                # Set camera from view config
                az = np.radians(view_config.azimuth)
                el = np.radians(view_config.elevation)
                dist = getattr(view_config, "distance", None)
                if dist is None:
                    bounds = mesh.get_axis_aligned_bounding_box()
                    dist = np.linalg.norm(bounds.get_extent()) * 2.0

                eye = np.array([
                    dist * np.cos(el) * np.sin(az),
                    dist * np.sin(el),
                    dist * np.cos(el) * np.cos(az),
                ])
                center = np.array([0.0, 0.0, 0.0])
                up = np.array([0.0, 1.0, 0.0])

                renderer.setup_camera(60.0, center, eye, up)
                image = renderer.render_to_image()

                return np.asarray(image)

            except ImportError:
                _log.debug("Open3D not available, trying trimesh fallback")

            # Try trimesh as fallback
            try:
                import trimesh
                from OCC.Extend.DataExchange import write_stl_file
                import tempfile
                import os

                with tempfile.NamedTemporaryFile(suffix=".stl", delete=False) as tmp:
                    tmp_path = tmp.name

                write_stl_file(shape, tmp_path)
                mesh = trimesh.load(tmp_path)
                os.unlink(tmp_path)

                if mesh.is_empty:
                    return None

                # Use trimesh scene rendering
                resolution = render_opts.get("resolution", 1024)
                scene = trimesh.Scene(mesh)

                # Set camera
                az = np.radians(view_config.azimuth)
                el = np.radians(view_config.elevation)
                dist = getattr(view_config, "distance", None) or mesh.bounding_sphere.primitive.radius * 3.0

                camera_transform = trimesh.transformations.compose_matrix(
                    translate=[
                        dist * np.cos(el) * np.sin(az),
                        dist * np.sin(el),
                        dist * np.cos(el) * np.cos(az),
                    ]
                )
                scene.camera_transform = camera_transform

                # Render to PNG bytes
                png_data = scene.save_image(resolution=(resolution, resolution))
                return png_data

            except ImportError:
                _log.debug("trimesh not available, software rendering unavailable")

        except Exception as e:
            _log.debug(f"Software rendering failed: {e}")

        return None

    def _fuse_multi_view_results(self, item) -> None:
        """Fuse results from multiple views using configured strategy.

        Args:
            item: The CAD item with per-view results
        """
        # Get per-view results
        pmi_annotations = item.properties.get("pmi_annotations", [])
        machining_features = item.properties.get("machining_features", [])

        # Fuse PMI annotations
        if pmi_annotations:
            fused_pmi = self._fuse_annotations(pmi_annotations)
            item.properties["fused_pmi_annotations"] = fused_pmi

        # Fuse machining features
        if machining_features:
            fused_features = self._fuse_features(machining_features)
            item.properties["fused_machining_features"] = fused_features

        # Store fusion metadata
        item.properties["fusion_strategy"] = self.fusion_strategy
        item.properties["num_views_processed"] = len(self.options.views)

        _log.debug(
            f"[Fusion] Fused {len(pmi_annotations)} PMI annotations "
            f"and {len(machining_features)} features"
        )

    def _fuse_annotations(self, annotations: List[Dict]) -> List[Dict]:
        """Fuse PMI annotations across views.

        Args:
            annotations: List of annotations from all views

        Returns:
            List of fused annotations
        """
        if self.fusion_strategy == "weighted_consensus":
            return self._fuse_weighted_consensus(annotations)
        elif self.fusion_strategy == "majority_vote":
            return self._fuse_majority_vote(annotations)
        elif self.fusion_strategy == "hierarchical":
            return self._fuse_hierarchical(annotations)
        else:
            return annotations  # No fusion

    def _fuse_features(self, features: List[Dict]) -> List[Dict]:
        """Fuse detected features across views.

        Args:
            features: List of features from all views

        Returns:
            List of fused features
        """
        # Use same strategy as annotations
        return self._fuse_annotations(features)

    def _fuse_weighted_consensus(self, items: List[Dict]) -> List[Dict]:
        """Fuse using weighted consensus strategy.

        Items that appear in multiple views with consistent values
        are weighted by confidence and view count.

        Args:
            items: Items to fuse

        Returns:
            Fused items
        """
        # Group by text/type
        grouped = defaultdict(list)
        for item in items:
            # Create key from text or type
            key = item.get("text", item.get("feature_type", "unknown"))
            grouped[key].append(item)

        fused = []
        for key, group in grouped.items():
            if len(group) == 1:
                # Single occurrence - keep as is but mark
                item_copy = group[0].copy()
                item_copy["view_count"] = 1
                item_copy["fusion_method"] = "single_view"
                fused.append(item_copy)

            else:
                # Multiple occurrences - check consistency
                views = [item.get("view", "unknown") for item in group]
                confidences = [item.get("confidence", 0.5) for item in group]
                avg_confidence = sum(confidences) / len(confidences)

                # Check if values are consistent (for numeric parameters)
                consistent = self._check_consistency(group)

                if consistent:
                    # Merge with weighted confidence
                    merged = group[0].copy()
                    # Boost confidence based on view agreement
                    merged["confidence"] = min(1.0, avg_confidence * (1 + 0.2 * (len(group) - 1)))
                    merged["view_count"] = len(group)
                    merged["views"] = views
                    merged["fusion_method"] = "weighted_consensus"
                    fused.append(merged)

                else:
                    # Keep all but flag conflict
                    for item in group:
                        item_copy = item.copy()
                        item_copy["conflict_detected"] = True
                        item_copy["conflicting_views"] = views
                        fused.append(item_copy)

        return fused

    def _fuse_majority_vote(self, items: List[Dict]) -> List[Dict]:
        """Fuse using simple majority voting.

        Only keep items that appear in majority of views.

        Args:
            items: Items to fuse

        Returns:
            Fused items
        """
        grouped = defaultdict(list)
        for item in items:
            key = item.get("text", item.get("feature_type", "unknown"))
            grouped[key].append(item)

        num_views = len(self.options.views)
        majority_threshold = num_views // 2 + 1

        fused = []
        for key, group in grouped.items():
            if len(group) >= majority_threshold:
                # Appears in majority - include
                merged = group[0].copy()
                merged["view_count"] = len(group)
                merged["views"] = [item.get("view") for item in group]
                merged["fusion_method"] = "majority_vote"
                fused.append(merged)

        return fused

    def _fuse_hierarchical(self, items: List[Dict]) -> List[Dict]:
        """Fuse using hierarchical priority strategy.

        Front and top views have priority over other views.

        Args:
            items: Items to fuse

        Returns:
            Fused items
        """
        view_priority = {"front": 3, "top": 3, "right": 2, "isometric": 2}

        # Group by key
        grouped = defaultdict(list)
        for item in items:
            key = item.get("text", item.get("feature_type", "unknown"))
            grouped[key].append(item)

        fused = []
        for key, group in grouped.items():
            # Weight by view priority
            best_item = None
            best_score = 0

            for item in group:
                view = item.get("view", "unknown")
                priority = view_priority.get(view, 1)
                confidence = item.get("confidence", 0.5)
                score = priority * confidence

                if score > best_score:
                    best_score = score
                    best_item = item

            if best_item:
                merged = best_item.copy()
                merged["view_count"] = len(group)
                merged["fusion_method"] = "hierarchical"
                merged["priority_score"] = best_score
                fused.append(merged)

        return fused

    def _check_consistency(self, items: List[Dict]) -> bool:
        """Check if items are consistent across views.

        Args:
            items: Items to check

        Returns:
            True if consistent
        """
        # Check numeric parameters for consistency
        for key in ["value", "diameter", "length", "width", "depth"]:
            values = [
                item.get("parameters", {}).get(key) or item.get(key)
                for item in items
                if item.get("parameters", {}).get(key) or item.get(key)
            ]

            if len(values) > 1:
                avg_val = sum(values) / len(values)
                # Allow 10% variation
                for val in values:
                    if abs(val - avg_val) / max(avg_val, 1e-6) > 0.1:
                        return False

        return True

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

        return ConversionStatus.SUCCESS
