"""STL mesh segmentation chunker.

This module performs geometric mesh segmentation on STL files,
chunking the actual 3D geometry into meaningful parts using
mesh processing algorithms.

Strategies:
    - Region growing: Segments mesh by growing regions based on geometric properties
    - Watershed: Segments mesh using watershed algorithm on geometric fields
    - Curvature-based: Segments based on principal curvature analysis
    - Connected components: Segments disconnected mesh parts

Classes:
    STLMeshChunker: Main mesh segmentation chunker for STL files
"""

from __future__ import annotations

import logging
import numpy as np
from collections import defaultdict
from typing import TYPE_CHECKING, Iterator, Optional, List, Set, Tuple

from cadling.chunker.base_chunker import BaseCADChunker, CADChunk, CADChunkMeta
from cadling.datamodel.stl import STLDocument, MeshItem

if TYPE_CHECKING:
    from cadling.datamodel.base_models import CADlingDocument

_log = logging.getLogger(__name__)


class MeshSegment:
    """Represents a segmented portion of a mesh.

    Attributes:
        facet_indices: Indices of facets in this segment
        vertices: Unique vertices in this segment
        boundary_edges: Edges on segment boundary
        properties: Geometric properties (area, volume, curvature, etc.)
    """

    def __init__(self, facet_indices: List[int]):
        self.facet_indices = facet_indices
        self.vertices: Set[Tuple[float, float, float]] = set()
        self.boundary_edges: Set[Tuple[int, int]] = set()
        self.properties: dict = {}


class STLMeshChunker(BaseCADChunker):
    """STL mesh segmentation chunker.

    Performs geometric segmentation of STL mesh data using
    various mesh processing algorithms.

    Attributes:
        strategy: Segmentation strategy ("region_growing", "watershed", "curvature", "connected_components")
        max_facets_per_chunk: Maximum facets per chunk (for size control)
        curvature_threshold: Threshold for curvature-based segmentation
        normal_threshold: Angle threshold for region growing (degrees)
    """

    def __init__(
        self,
        strategy: str = "region_growing",
        max_facets_per_chunk: int = 5000,
        max_tokens: int = 512,
        overlap_tokens: int = 50,
        curvature_threshold: float = 0.1,
        normal_threshold: float = 30.0,
    ):
        """Initialize STL mesh chunker.

        Args:
            strategy: Segmentation strategy
            max_facets_per_chunk: Maximum facets per segment
            max_tokens: Maximum tokens per chunk (for text representation)
            overlap_tokens: Overlap tokens
            curvature_threshold: Curvature threshold for segmentation
            normal_threshold: Normal angle threshold in degrees
        """
        super().__init__(max_tokens, overlap_tokens)
        self.strategy = strategy
        self.max_facets_per_chunk = max_facets_per_chunk
        self.curvature_threshold = curvature_threshold
        self.normal_threshold = normal_threshold

    def chunk(self, doc: CADlingDocument) -> Iterator[CADChunk]:
        """Segment STL mesh into geometric chunks.

        Args:
            doc: CADlingDocument (should be STLDocument)

        Yields:
            CADChunk objects representing mesh segments
        """
        if not isinstance(doc, STLDocument):
            _log.warning(f"STLMeshChunker used on non-STL document: {type(doc)}")
            return

        # Extract facets
        facets = [item for item in doc.items if isinstance(item, STLFacet)]

        if not facets:
            _log.warning("No facets found in STL document")
            return

        _log.info(f"Segmenting mesh with {len(facets)} facets using strategy: {self.strategy}")

        # Build mesh data structures
        mesh_data = self._build_mesh_data(facets)

        # Perform segmentation based on strategy
        if self.strategy == "region_growing":
            segments = self._segment_by_region_growing(facets, mesh_data)
        elif self.strategy == "watershed":
            segments = self._segment_by_watershed(facets, mesh_data)
        elif self.strategy == "curvature":
            segments = self._segment_by_curvature(facets, mesh_data)
        elif self.strategy == "connected_components":
            segments = self._segment_by_connected_components(facets, mesh_data)
        else:
            _log.warning(f"Unknown strategy: {self.strategy}, using region_growing")
            segments = self._segment_by_region_growing(facets, mesh_data)

        _log.info(f"Segmented mesh into {len(segments)} segments")

        # Convert segments to chunks
        for i, segment in enumerate(segments):
            yield self._create_chunk_from_segment(segment, facets, doc, i)

    def _build_mesh_data(self, facets: List[STLFacet]) -> dict:
        """Build mesh connectivity and geometric data structures.

        Args:
            facets: List of STL facets

        Returns:
            Dictionary containing mesh data structures:
                - adjacency: facet adjacency graph
                - vertex_to_facets: vertex to facet mapping
                - edge_to_facets: edge to facet mapping
        """
        adjacency = defaultdict(set)  # facet_id -> set of adjacent facet_ids
        vertex_to_facets = defaultdict(set)
        edge_to_facets = defaultdict(set)

        # Build vertex and edge mappings
        for i, facet in enumerate(facets):
            vertices = [
                self._vertex_to_key(facet.v1),
                self._vertex_to_key(facet.v2),
                self._vertex_to_key(facet.v3)
            ]

            # Map vertices to facets
            for v in vertices:
                vertex_to_facets[v].add(i)

            # Map edges to facets
            edges = [
                self._edge_to_key(vertices[0], vertices[1]),
                self._edge_to_key(vertices[1], vertices[2]),
                self._edge_to_key(vertices[2], vertices[0])
            ]

            for edge in edges:
                edge_to_facets[edge].add(i)

        # Build adjacency graph from shared edges
        for edge, facet_set in edge_to_facets.items():
            facet_list = list(facet_set)
            for i, f1 in enumerate(facet_list):
                for f2 in facet_list[i+1:]:
                    adjacency[f1].add(f2)
                    adjacency[f2].add(f1)

        return {
            "adjacency": adjacency,
            "vertex_to_facets": vertex_to_facets,
            "edge_to_facets": edge_to_facets
        }

    def _segment_by_region_growing(
        self,
        facets: List[STLFacet],
        mesh_data: dict
    ) -> List[MeshSegment]:
        """Segment mesh by region growing based on normal similarity.

        Args:
            facets: List of facets
            mesh_data: Mesh data structures

        Returns:
            List of mesh segments
        """
        adjacency = mesh_data["adjacency"]
        visited = set()
        segments = []

        for seed_idx in range(len(facets)):
            if seed_idx in visited:
                continue

            # Grow region from seed
            region = [seed_idx]
            queue = [seed_idx]
            visited.add(seed_idx)
            seed_normal = np.array(facets[seed_idx].normal)

            while queue and len(region) < self.max_facets_per_chunk:
                current_idx = queue.pop(0)
                current_normal = np.array(facets[current_idx].normal)

                # Check adjacent facets
                for neighbor_idx in adjacency.get(current_idx, []):
                    if neighbor_idx in visited:
                        continue

                    neighbor_normal = np.array(facets[neighbor_idx].normal)

                    # Check normal similarity
                    angle = self._angle_between_normals(current_normal, neighbor_normal)

                    if angle < self.normal_threshold:
                        region.append(neighbor_idx)
                        queue.append(neighbor_idx)
                        visited.add(neighbor_idx)

            if region:
                segment = MeshSegment(region)
                self._compute_segment_properties(segment, facets)
                segments.append(segment)

        return segments

    def _segment_by_connected_components(
        self,
        facets: List[STLFacet],
        mesh_data: dict
    ) -> List[MeshSegment]:
        """Segment mesh by finding disconnected components.

        Args:
            facets: List of facets
            mesh_data: Mesh data structures

        Returns:
            List of mesh segments (one per component)
        """
        adjacency = mesh_data["adjacency"]
        visited = set()
        segments = []

        for start_idx in range(len(facets)):
            if start_idx in visited:
                continue

            # BFS to find connected component
            component = []
            queue = [start_idx]
            visited.add(start_idx)

            while queue:
                current_idx = queue.pop(0)
                component.append(current_idx)

                for neighbor_idx in adjacency.get(current_idx, []):
                    if neighbor_idx not in visited:
                        visited.add(neighbor_idx)
                        queue.append(neighbor_idx)

            # Split large components
            if len(component) > self.max_facets_per_chunk:
                # Split into smaller chunks
                for i in range(0, len(component), self.max_facets_per_chunk):
                    chunk = component[i:i + self.max_facets_per_chunk]
                    segment = MeshSegment(chunk)
                    self._compute_segment_properties(segment, facets)
                    segments.append(segment)
            else:
                segment = MeshSegment(component)
                self._compute_segment_properties(segment, facets)
                segments.append(segment)

        return segments

    def _segment_by_curvature(
        self,
        facets: List[STLFacet],
        mesh_data: dict
    ) -> List[MeshSegment]:
        """Segment mesh based on curvature analysis.

        Args:
            facets: List of facets
            mesh_data: Mesh data structures

        Returns:
            List of mesh segments
        """
        # Compute discrete curvature for each facet
        curvatures = self._compute_discrete_curvatures(facets, mesh_data)

        # Classify facets by curvature magnitude
        low_curv = []  # Planar regions
        high_curv = []  # High curvature regions (edges, corners)

        for idx, curv in enumerate(curvatures):
            if abs(curv) < self.curvature_threshold:
                low_curv.append(idx)
            else:
                high_curv.append(idx)

        segments = []

        # Create segments for each curvature class
        for facet_group in [low_curv, high_curv]:
            if not facet_group:
                continue

            # Further segment by connectivity within curvature class
            visited = set()
            adjacency = mesh_data["adjacency"]

            for start_idx in facet_group:
                if start_idx in visited:
                    continue

                # Grow connected region within same curvature class
                region = []
                queue = [start_idx]
                visited.add(start_idx)

                while queue and len(region) < self.max_facets_per_chunk:
                    current_idx = queue.pop(0)
                    region.append(current_idx)

                    for neighbor_idx in adjacency.get(current_idx, []):
                        if neighbor_idx in visited or neighbor_idx not in facet_group:
                            continue
                        visited.add(neighbor_idx)
                        queue.append(neighbor_idx)

                if region:
                    segment = MeshSegment(region)
                    self._compute_segment_properties(segment, facets)
                    segments.append(segment)

        return segments

    def _segment_by_watershed(
        self,
        facets: List[STLFacet],
        mesh_data: dict
    ) -> List[MeshSegment]:
        """Segment mesh using watershed algorithm on geometric field.

        Args:
            facets: List of facets
            mesh_data: Mesh data structures

        Returns:
            List of mesh segments
        """
        # Simplified watershed: use height field (z-coordinate)
        # Find local minima and grow regions

        # Compute centroid heights
        heights = []
        for facet in facets:
            centroid_z = (facet.v1[2] + facet.v2[2] + facet.v3[2]) / 3.0
            heights.append(centroid_z)

        heights = np.array(heights)
        adjacency = mesh_data["adjacency"]

        # Find local minima
        local_minima = []
        for idx in range(len(facets)):
            neighbors = adjacency.get(idx, [])
            if not neighbors:
                local_minima.append(idx)
                continue

            is_minimum = all(heights[idx] <= heights[n] for n in neighbors)
            if is_minimum:
                local_minima.append(idx)

        _log.debug(f"Found {len(local_minima)} watershed seeds")

        # Grow basins from minima
        visited = set()
        segments = []

        # Sort minima by height
        local_minima.sort(key=lambda idx: heights[idx])

        for seed_idx in local_minima:
            if seed_idx in visited:
                continue

            # Grow basin
            basin = [seed_idx]
            queue = [(heights[seed_idx], seed_idx)]
            visited.add(seed_idx)

            while queue and len(basin) < self.max_facets_per_chunk:
                _, current_idx = queue.pop(0)

                for neighbor_idx in adjacency.get(current_idx, []):
                    if neighbor_idx in visited:
                        continue

                    # Add if height is ascending
                    if heights[neighbor_idx] >= heights[current_idx]:
                        basin.append(neighbor_idx)
                        visited.add(neighbor_idx)
                        queue.append((heights[neighbor_idx], neighbor_idx))
                        queue.sort()  # Keep queue sorted by height

            if basin:
                segment = MeshSegment(basin)
                self._compute_segment_properties(segment, facets)
                segments.append(segment)

        # Handle unvisited facets
        unvisited = set(range(len(facets))) - visited
        if unvisited:
            segment = MeshSegment(list(unvisited))
            self._compute_segment_properties(segment, facets)
            segments.append(segment)

        return segments

    def _compute_discrete_curvatures(
        self,
        facets: List[STLFacet],
        mesh_data: dict
    ) -> np.ndarray:
        """Compute discrete curvature estimates for facets.

        Args:
            facets: List of facets
            mesh_data: Mesh data structures

        Returns:
            Array of curvature values
        """
        adjacency = mesh_data["adjacency"]
        curvatures = np.zeros(len(facets))

        for idx, facet in enumerate(facets):
            normal = np.array(facet.normal)

            # Estimate curvature from normal variation with neighbors
            neighbors = list(adjacency.get(idx, []))
            if not neighbors:
                curvatures[idx] = 0.0
                continue

            # Average normal difference with neighbors
            normal_diffs = []
            for neighbor_idx in neighbors:
                neighbor_normal = np.array(facets[neighbor_idx].normal)
                diff = np.linalg.norm(normal - neighbor_normal)
                normal_diffs.append(diff)

            curvatures[idx] = np.mean(normal_diffs) if normal_diffs else 0.0

        return curvatures

    def _compute_segment_properties(self, segment: MeshSegment, facets: List[STLFacet]):
        """Compute geometric properties of a segment.

        Args:
            segment: Mesh segment
            facets: Full list of facets
        """
        segment_facets = [facets[i] for i in segment.facet_indices]

        # Collect vertices
        for facet in segment_facets:
            segment.vertices.add(self._vertex_to_key(facet.v1))
            segment.vertices.add(self._vertex_to_key(facet.v2))
            segment.vertices.add(self._vertex_to_key(facet.v3))

        # Compute bounds
        vertices_array = np.array([list(v) for v in segment.vertices])
        min_bounds = vertices_array.min(axis=0)
        max_bounds = vertices_array.max(axis=0)

        # Compute surface area
        total_area = 0.0
        for facet in segment_facets:
            area = self._compute_triangle_area(facet.v1, facet.v2, facet.v3)
            total_area += area

        segment.properties = {
            "num_facets": len(segment.facet_indices),
            "num_vertices": len(segment.vertices),
            "surface_area": total_area,
            "bounds_min": min_bounds.tolist(),
            "bounds_max": max_bounds.tolist(),
            "volume_estimate": self._estimate_volume(segment_facets)
        }

    def _compute_triangle_area(
        self,
        v1: Tuple[float, float, float],
        v2: Tuple[float, float, float],
        v3: Tuple[float, float, float]
    ) -> float:
        """Compute area of triangle.

        Args:
            v1, v2, v3: Triangle vertices

        Returns:
            Triangle area
        """
        a = np.array(v1)
        b = np.array(v2)
        c = np.array(v3)

        return 0.5 * np.linalg.norm(np.cross(b - a, c - a))

    def _estimate_volume(self, facets: List[STLFacet]) -> float:
        """Estimate volume of mesh segment using signed volume of tetrahedra.

        Args:
            facets: Facets in segment

        Returns:
            Estimated volume
        """
        volume = 0.0
        origin = np.array([0.0, 0.0, 0.0])

        for facet in facets:
            v1 = np.array(facet.v1)
            v2 = np.array(facet.v2)
            v3 = np.array(facet.v3)

            # Signed volume of tetrahedron (origin, v1, v2, v3)
            tet_vol = np.dot(v1, np.cross(v2, v3)) / 6.0
            volume += tet_vol

        return abs(volume)

    def _angle_between_normals(self, n1: np.ndarray, n2: np.ndarray) -> float:
        """Compute angle between two normal vectors in degrees.

        Args:
            n1, n2: Normal vectors

        Returns:
            Angle in degrees
        """
        dot = np.clip(np.dot(n1, n2), -1.0, 1.0)
        return np.degrees(np.arccos(dot))

    def _vertex_to_key(self, vertex: Tuple[float, float, float]) -> Tuple[float, float, float]:
        """Convert vertex to hashable key with precision rounding.

        Args:
            vertex: Vertex coordinates

        Returns:
            Rounded vertex tuple
        """
        precision = 6
        return tuple(round(v, precision) for v in vertex)

    def _edge_to_key(self, v1: Tuple, v2: Tuple) -> Tuple:
        """Convert edge to canonical hashable key.

        Args:
            v1, v2: Edge vertices

        Returns:
            Canonical edge tuple (sorted)
        """
        return tuple(sorted([v1, v2]))

    def _create_chunk_from_segment(
        self,
        segment: MeshSegment,
        facets: List[STLFacet],
        doc: STLDocument,
        segment_id: int
    ) -> CADChunk:
        """Create CADChunk from mesh segment.

        Args:
            segment: Mesh segment
            facets: Full list of facets
            doc: Parent document
            segment_id: Segment identifier

        Returns:
            CADChunk with segment data
        """
        # Generate text representation of segment
        segment_facets = [facets[i] for i in segment.facet_indices]

        text_lines = [f"Mesh Segment {segment_id} ({len(segment_facets)} facets)"]
        text_lines.append(f"Bounds: {segment.properties['bounds_min']} to {segment.properties['bounds_max']}")
        text_lines.append(f"Surface Area: {segment.properties['surface_area']:.6f}")
        text_lines.append(f"Volume: {segment.properties['volume_estimate']:.6f}")
        text_lines.append("")

        # Include sample of facets (not all, to respect token limits)
        max_sample = min(50, len(segment_facets))
        for facet in segment_facets[:max_sample]:
            normal_str = f"normal {facet.normal[0]:.6f} {facet.normal[1]:.6f} {facet.normal[2]:.6f}"
            text_lines.append(f"facet {normal_str}")
            text_lines.append(f"  vertex {facet.v1[0]:.6f} {facet.v1[1]:.6f} {facet.v1[2]:.6f}")
            text_lines.append(f"  vertex {facet.v2[0]:.6f} {facet.v2[1]:.6f} {facet.v2[2]:.6f}")
            text_lines.append(f"  vertex {facet.v3[0]:.6f} {facet.v3[1]:.6f} {facet.v3[2]:.6f}")
            text_lines.append("endfacet")

        if len(segment_facets) > max_sample:
            text_lines.append(f"... ({len(segment_facets) - max_sample} more facets)")

        text = "\n".join(text_lines)

        meta = CADChunkMeta(
            properties=segment.properties
        )

        return CADChunk(
            text=text,
            meta=meta,
            chunk_id=f"{doc.name}_segment_{segment_id}",
            doc_name=doc.name,
        )


# Convenience aliases
RegionGrowingChunker = lambda **kwargs: STLMeshChunker(strategy="region_growing", **kwargs)
WatershedChunker = lambda **kwargs: STLMeshChunker(strategy="watershed", **kwargs)
CurvatureChunker = lambda **kwargs: STLMeshChunker(strategy="curvature", **kwargs)
ConnectedComponentsChunker = lambda **kwargs: STLMeshChunker(strategy="connected_components", **kwargs)
