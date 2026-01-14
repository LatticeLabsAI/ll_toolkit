"""Mesh quality assessment model for CAD parts.

This module provides mesh quality metrics including:
- Triangle aspect ratio (shape quality)
- Skewness (deviation from ideal shape)
- Edge length distribution
- Degenerate triangle detection
- Non-manifold edge detection
- Normal consistency checking

Classes:
    MeshQualityModel: Main model for mesh quality assessment
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Optional

import numpy as np

from cadling.models.base_model import EnrichmentModel

if TYPE_CHECKING:
    from cadling.datamodel.base_models import CADItem, CADlingDocument

_log = logging.getLogger(__name__)


class MeshQualityModel(EnrichmentModel):
    """Mesh quality assessment enrichment model.

    Assesses mesh quality using various metrics:
    - Aspect ratio: Ratio of longest to shortest edge in triangles
    - Skewness: How much triangles deviate from equilateral
    - Edge length distribution: Min/max/mean edge lengths
    - Degenerate triangles: Zero or near-zero area triangles
    - Non-manifold edges: Edges shared by != 2 faces

    This model primarily works with trimesh for STL meshes and can also
    analyze tessellated surfaces from STEP/IGES/BRep.

    Attributes:
        has_trimesh: Whether trimesh is available
        has_pythonocc: Whether pythonocc-core is available
        aspect_ratio_threshold: Threshold for flagging bad aspect ratios
        skewness_threshold: Threshold for flagging high skewness

    Example:
        model = MeshQualityModel(
            aspect_ratio_threshold=10.0,
            skewness_threshold=0.8
        )
        result = converter.convert(
            "part.stl",
            pipeline_options=PipelineOptions(
                enrichment_models=[model]
            )
        )
        for item in result.document.items:
            quality = item.properties.get("mesh_quality", {})
            print(f"Mean aspect ratio: {quality['aspect_ratio']['mean']}")
    """

    def __init__(
        self,
        aspect_ratio_threshold: float = 10.0,
        skewness_threshold: float = 0.8,
        min_area_threshold: float = 1e-10,
    ):
        """Initialize mesh quality assessment model.

        Args:
            aspect_ratio_threshold: Flag triangles with aspect ratio > this
            skewness_threshold: Flag triangles with skewness > this
            min_area_threshold: Minimum area to consider triangle valid
        """
        super().__init__()

        self.aspect_ratio_threshold = aspect_ratio_threshold
        self.skewness_threshold = skewness_threshold
        self.min_area_threshold = min_area_threshold

        # Check for trimesh availability
        self.has_trimesh = False
        try:
            import trimesh

            self.has_trimesh = True
            _log.debug("trimesh available for mesh quality assessment")
        except ImportError:
            _log.warning("trimesh not available. Mesh quality assessment disabled.")

        # Check for pythonocc-core (for tessellation of CAD)
        self.has_pythonocc = False
        try:
            from OCC.Core.BRepMesh import BRepMesh_IncrementalMesh

            self.has_pythonocc = True
            _log.debug("pythonocc-core available for CAD tessellation")
        except ImportError:
            _log.warning("pythonocc-core not available for CAD tessellation.")

        if not self.has_trimesh:
            _log.error("trimesh not available. Mesh quality assessment disabled.")

    def __call__(
        self,
        doc: CADlingDocument,
        item_batch: list[CADItem],
    ) -> None:
        """Assess mesh quality for CAD items.

        Args:
            doc: CADlingDocument being enriched
            item_batch: List of CADItem objects to assess
        """
        if not self.has_trimesh:
            _log.debug("Mesh quality assessment skipped: trimesh not available")
            return

        for item in item_batch:
            try:
                # Assess mesh quality
                quality_result = self._assess_item(doc, item)

                if quality_result:
                    item.properties["mesh_quality"] = quality_result

                    # Add provenance
                    item.add_provenance(
                        component_type="enrichment_model",
                        component_name="MeshQualityModel",
                    )

                    # Log warnings for poor quality
                    if quality_result.get("quality_score", 1.0) < 0.5:
                        _log.warning(
                            f"Poor mesh quality for item '{item.label.text}': "
                            f"score={quality_result['quality_score']:.2f}"
                        )
                    else:
                        _log.debug(
                            f"Mesh quality assessed for item '{item.label.text}': "
                            f"score={quality_result['quality_score']:.2f}"
                        )

            except Exception as e:
                _log.error(
                    f"Mesh quality assessment failed for item {item.label.text}: {e}"
                )

    def _assess_item(
        self, doc: CADlingDocument, item: CADItem
    ) -> Optional[dict]:
        """Assess mesh quality of a single CAD item.

        Args:
            doc: Document containing the item
            item: Item to assess

        Returns:
            Dictionary with quality metrics, or None if assessment failed
        """
        # Try to get mesh
        mesh = self._get_mesh_for_item(doc, item)

        if mesh is None:
            _log.debug(f"Could not get mesh for item {item.label.text}")
            return None

        # Assess mesh quality
        return self._assess_mesh(mesh)

    def _get_mesh_for_item(
        self, doc: CADlingDocument, item: CADItem
    ) -> Optional[any]:
        """Get mesh object for assessment.

        Args:
            doc: Document containing the item
            item: Item to get mesh for

        Returns:
            Trimesh object, or None
        """
        # Check if item has mesh stored
        if hasattr(item, "_mesh") and item._mesh is not None:
            return item._mesh

        # Try to load from backend
        format_str = str(doc.format).lower()

        if format_str == "stl":
            return self._load_trimesh(doc)
        elif format_str in ["step", "iges", "brep"] and self.has_pythonocc:
            # Would need to tessellate CAD surface
            return self._tessellate_cad(doc)

        return None

    def _load_trimesh(self, doc: CADlingDocument):
        """Load mesh via trimesh."""
        _log.debug("Trimesh loading not yet implemented in enrichment stage")
        return None

    def _tessellate_cad(self, doc: CADlingDocument):
        """Tessellate CAD surface to mesh."""
        _log.debug("CAD tessellation not yet implemented in enrichment stage")
        return None

    def _assess_mesh(self, mesh) -> dict:
        """Assess mesh quality.

        Args:
            mesh: Trimesh object to assess

        Returns:
            Dictionary with quality metrics
        """
        import trimesh

        results = {}

        # Basic mesh properties
        num_vertices = len(mesh.vertices)
        num_faces = len(mesh.faces)

        results["num_vertices"] = num_vertices
        results["num_faces"] = num_faces

        # Compute edge lengths for all edges
        edges = mesh.edges_unique
        edge_vectors = mesh.vertices[edges[:, 1]] - mesh.vertices[edges[:, 0]]
        edge_lengths = np.linalg.norm(edge_vectors, axis=1)

        results["edge_lengths"] = {
            "min": float(np.min(edge_lengths)),
            "max": float(np.max(edge_lengths)),
            "mean": float(np.mean(edge_lengths)),
            "std": float(np.std(edge_lengths)),
        }

        # Compute aspect ratios for triangles
        aspect_ratios = self._compute_aspect_ratios(mesh)
        results["aspect_ratio"] = {
            "min": float(np.min(aspect_ratios)),
            "max": float(np.max(aspect_ratios)),
            "mean": float(np.mean(aspect_ratios)),
            "std": float(np.std(aspect_ratios)),
        }

        # Count triangles with poor aspect ratio
        poor_aspect_ratio = np.sum(aspect_ratios > self.aspect_ratio_threshold)
        results["num_poor_aspect_ratio"] = int(poor_aspect_ratio)
        results["percent_poor_aspect_ratio"] = (
            float(poor_aspect_ratio) / num_faces * 100 if num_faces > 0 else 0.0
        )

        # Compute skewness for triangles
        skewness = self._compute_skewness(mesh)
        results["skewness"] = {
            "min": float(np.min(skewness)),
            "max": float(np.max(skewness)),
            "mean": float(np.mean(skewness)),
            "std": float(np.std(skewness)),
        }

        # Count triangles with high skewness
        high_skewness = np.sum(skewness > self.skewness_threshold)
        results["num_high_skewness"] = int(high_skewness)
        results["percent_high_skewness"] = (
            float(high_skewness) / num_faces * 100 if num_faces > 0 else 0.0
        )

        # Detect degenerate triangles (zero or very small area)
        face_areas = mesh.area_faces
        degenerate = face_areas < self.min_area_threshold
        results["num_degenerate_faces"] = int(np.sum(degenerate))
        results["percent_degenerate_faces"] = (
            float(np.sum(degenerate)) / num_faces * 100 if num_faces > 0 else 0.0
        )

        # Check for non-manifold edges
        # Trimesh provides this via mesh.edges_face count
        results["is_manifold"] = mesh.is_watertight and mesh.is_winding_consistent

        # Overall quality score (0-1, higher is better)
        # Based on percentage of good triangles
        good_triangles = num_faces - poor_aspect_ratio - high_skewness - np.sum(degenerate)
        quality_score = max(0.0, float(good_triangles) / num_faces) if num_faces > 0 else 0.0
        results["quality_score"] = quality_score

        # Quality classification
        if quality_score >= 0.9:
            results["quality_class"] = "excellent"
        elif quality_score >= 0.75:
            results["quality_class"] = "good"
        elif quality_score >= 0.5:
            results["quality_class"] = "fair"
        elif quality_score >= 0.25:
            results["quality_class"] = "poor"
        else:
            results["quality_class"] = "very_poor"

        _log.debug(
            f"Mesh quality assessment: {num_faces} faces, "
            f"score={quality_score:.2f}, class={results['quality_class']}"
        )

        return results

    def _compute_aspect_ratios(self, mesh) -> np.ndarray:
        """Compute aspect ratio for each triangle.

        Aspect ratio = longest_edge / shortest_edge
        Ideal equilateral triangle has aspect ratio = 1.0

        Args:
            mesh: Trimesh object

        Returns:
            Array of aspect ratios for each face
        """
        # Get triangle edges
        faces = mesh.faces
        vertices = mesh.vertices

        # For each triangle, compute 3 edge lengths
        v0 = vertices[faces[:, 0]]
        v1 = vertices[faces[:, 1]]
        v2 = vertices[faces[:, 2]]

        edge1 = np.linalg.norm(v1 - v0, axis=1)
        edge2 = np.linalg.norm(v2 - v1, axis=1)
        edge3 = np.linalg.norm(v0 - v2, axis=1)

        # Stack edge lengths
        edge_lengths = np.stack([edge1, edge2, edge3], axis=1)

        # Aspect ratio = max / min
        max_edges = np.max(edge_lengths, axis=1)
        min_edges = np.min(edge_lengths, axis=1)

        # Avoid division by zero
        aspect_ratios = np.divide(
            max_edges,
            min_edges,
            out=np.full_like(max_edges, np.inf),
            where=min_edges > 1e-10,
        )

        return aspect_ratios

    def _compute_skewness(self, mesh) -> np.ndarray:
        """Compute skewness for each triangle.

        Skewness measures how much a triangle deviates from equilateral.
        Skewness = 1 - (area_actual / area_ideal)

        where area_ideal is the area of an equilateral triangle with the same perimeter.

        Args:
            mesh: Trimesh object

        Returns:
            Array of skewness values for each face (0 = equilateral, 1 = degenerate)
        """
        # Get triangle edges
        faces = mesh.faces
        vertices = mesh.vertices

        v0 = vertices[faces[:, 0]]
        v1 = vertices[faces[:, 1]]
        v2 = vertices[faces[:, 2]]

        edge1 = np.linalg.norm(v1 - v0, axis=1)
        edge2 = np.linalg.norm(v2 - v1, axis=1)
        edge3 = np.linalg.norm(v0 - v2, axis=1)

        # Perimeter
        perimeter = edge1 + edge2 + edge3

        # Actual area (use Heron's formula or cross product)
        # Using cross product: area = 0.5 * ||cross(v1-v0, v2-v0)||
        cross = np.cross(v1 - v0, v2 - v0)
        actual_area = 0.5 * np.linalg.norm(cross, axis=1)

        # Ideal area for equilateral triangle with same perimeter
        # For equilateral: side = perimeter / 3
        # Area = (sqrt(3) / 4) * side^2
        side_ideal = perimeter / 3.0
        ideal_area = (np.sqrt(3) / 4.0) * side_ideal**2

        # Skewness = 1 - (actual / ideal)
        # Avoid division by zero
        skewness = 1.0 - np.divide(
            actual_area,
            ideal_area,
            out=np.ones_like(actual_area),
            where=ideal_area > 1e-10,
        )

        # Clamp to [0, 1]
        skewness = np.clip(skewness, 0.0, 1.0)

        return skewness

    def supports_batch_processing(self) -> bool:
        """Mesh quality assessment can process items independently."""
        return True

    def get_batch_size(self) -> int:
        """Process items one at a time (can be expensive for large meshes)."""
        return 1

    def requires_gpu(self) -> bool:
        """Mesh quality assessment does not require GPU."""
        return False
