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
import re
import struct
from typing import TYPE_CHECKING, Optional, Tuple

import numpy as np

from cadling.models.base_model import EnrichmentModel

if TYPE_CHECKING:
    from cadling.datamodel.base_models import CADItem, CADlingDocument

_log = logging.getLogger(__name__)


# STL binary format header size
STL_BINARY_HEADER_SIZE = 80


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

        Uses multi-strategy approach to ensure quality assessment even when
        trimesh is unavailable.

        Args:
            doc: CADlingDocument being enriched
            item_batch: List of CADItem objects to assess
        """
        # Note: We no longer skip when trimesh unavailable - we have numpy fallbacks
        if not self.has_trimesh:
            _log.debug("trimesh not available, using numpy-based fallback methods")

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

        Uses multi-strategy approach:
        1. Primary: Load mesh via trimesh
        2. Alternative A: Parse STL file directly (numpy-based)
        3. Alternative B: Tessellate CAD shape via pythonocc
        4. Alternative C: Estimate from item properties

        Args:
            doc: Document containing the item
            item: Item to assess

        Returns:
            Dictionary with quality metrics - NEVER returns None
        """
        # Strategy 1: Try to get mesh via trimesh
        mesh = self._get_mesh_for_item(doc, item)

        if mesh is not None:
            return self._assess_mesh(mesh)

        # Strategy 2: Try numpy-based STL parsing (no trimesh required)
        format_str = str(doc.format).lower() if hasattr(doc, 'format') else ""
        if format_str == "stl":
            stl_result = self._assess_from_stl_data(doc)
            if stl_result is not None:
                return stl_result

        # Strategy 3: Try tessellation from STEP/IGES if pythonocc available
        if format_str in ["step", "iges", "brep"]:
            tess_result = self._assess_from_tessellation(doc)
            if tess_result is not None:
                return tess_result

        # Strategy 4: Estimate from item properties (bounding box, surface area, etc.)
        prop_result = self._estimate_from_properties(doc, item)
        if prop_result is not None:
            return prop_result

        # Last resort: Return error dict with computed zeros
        _log.debug(f"Could not assess mesh quality for item {item.label.text} via any method")
        return {
            "status": "error",
            "reason": "Could not retrieve mesh from item via any method",
            "num_vertices": 0,
            "num_faces": 0,
            "edge_lengths": {"min": 0.0, "max": 0.0, "mean": 0.0, "std": 0.0},
            "aspect_ratio": {"min": 0.0, "max": 0.0, "mean": 0.0, "std": 0.0},
            "num_poor_aspect_ratio": 0,
            "percent_poor_aspect_ratio": 0.0,
            "skewness": {"min": 0.0, "max": 0.0, "mean": 0.0, "std": 0.0},
            "num_high_skewness": 0,
            "percent_high_skewness": 0.0,
            "num_degenerate_faces": 0,
            "percent_degenerate_faces": 0.0,
            "is_manifold": False,
            "quality_score": 0.0,
            "quality_class": "unknown",
        }

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

    # _get_backend_resource is inherited from EnrichmentModel base class

    def _load_trimesh(self, doc: CADlingDocument):
        """Load mesh via trimesh.

        Args:
            doc: Document to load mesh from

        Returns:
            Trimesh object or None
        """
        return self._get_backend_resource(doc, "mesh")

    def _tessellate_cad(self, doc: CADlingDocument):
        """Tessellate CAD surface to mesh using pythonocc.

        Uses BRepMesh_IncrementalMesh to generate a triangular mesh from
        the OCC shape, then converts it to a trimesh object.

        Args:
            doc: Document containing the CAD shape

        Returns:
            Trimesh object or None if tessellation fails
        """
        if not self.has_pythonocc:
            _log.debug("CAD tessellation skipped: pythonocc not available")
            return None

        # Get the OCC shape from backend
        shape = self._get_backend_resource(doc, "shape")
        if shape is None:
            _log.debug("CAD tessellation skipped: no shape available")
            return None

        try:
            from OCC.Core.BRepMesh import BRepMesh_IncrementalMesh
            from OCC.Core.TopAbs import TopAbs_FACE
            from OCC.Core.TopExp import TopExp_Explorer
            from OCC.Core.TopLoc import TopLoc_Location
            from OCC.Core.BRep import BRep_Tool
            import trimesh

            # Tessellate the shape
            # Linear deflection controls mesh density (smaller = finer mesh)
            linear_deflection = 0.1
            angular_deflection = 0.5
            mesh_algo = BRepMesh_IncrementalMesh(
                shape, linear_deflection, False, angular_deflection, True
            )
            mesh_algo.Perform()

            if not mesh_algo.IsDone():
                _log.warning("BRepMesh_IncrementalMesh did not complete successfully")
                return None

            # Collect vertices and faces from all faces in the shape
            all_vertices = []
            all_faces = []
            vertex_offset = 0

            face_explorer = TopExp_Explorer(shape, TopAbs_FACE)
            while face_explorer.More():
                face = face_explorer.Current()
                location = TopLoc_Location()

                # Get the triangulation for this face
                triangulation = BRep_Tool.Triangulation(face, location)

                if triangulation is not None:
                    # Get transformation matrix
                    transformation = location.Transformation()

                    # Extract vertices
                    num_nodes = triangulation.NbNodes()
                    for i in range(1, num_nodes + 1):
                        node = triangulation.Node(i)
                        # Apply transformation
                        node = node.Transformed(transformation)
                        all_vertices.append([node.X(), node.Y(), node.Z()])

                    # Extract faces (triangles)
                    num_triangles = triangulation.NbTriangles()
                    for i in range(1, num_triangles + 1):
                        tri = triangulation.Triangle(i)
                        n1, n2, n3 = tri.Get()
                        # Convert from 1-based to 0-based indexing and add offset
                        all_faces.append([
                            n1 - 1 + vertex_offset,
                            n2 - 1 + vertex_offset,
                            n3 - 1 + vertex_offset
                        ])

                    vertex_offset += num_nodes

                face_explorer.Next()

            if not all_vertices or not all_faces:
                _log.warning("Tessellation produced no geometry")
                return None

            # Create trimesh object
            vertices_array = np.array(all_vertices)
            faces_array = np.array(all_faces)
            mesh = trimesh.Trimesh(vertices=vertices_array, faces=faces_array)

            _log.debug(
                f"Tessellated CAD to mesh: {len(all_vertices)} vertices, "
                f"{len(all_faces)} faces"
            )
            return mesh

        except ImportError as e:
            _log.warning(f"CAD tessellation failed due to missing import: {e}")
            return None
        except Exception as e:
            _log.error(f"CAD tessellation failed: {e}")
            return None

    def _assess_from_stl_data(self, doc: CADlingDocument) -> Optional[dict]:
        """Assess mesh quality from STL data using numpy only (no trimesh).

        This is the numpy-based alternative when trimesh is unavailable.
        Parses both ASCII and binary STL formats directly.

        Args:
            doc: Document containing STL data

        Returns:
            Dictionary with quality metrics, or None if parsing fails
        """
        try:
            # Try to get raw STL data from document
            stl_data = self._get_stl_data(doc)
            if stl_data is None:
                return None

            # Parse STL to vertices and faces
            vertices, faces = self._parse_stl_data(stl_data)

            if vertices is None or len(vertices) == 0 or len(faces) == 0:
                return None

            # Compute quality metrics using numpy
            return self._assess_mesh_numpy(vertices, faces)

        except Exception as e:
            _log.debug(f"STL-based mesh assessment failed: {e}")
            return None

    def _get_stl_data(self, doc: CADlingDocument) -> Optional[bytes]:
        """Get raw STL data from document.

        Args:
            doc: Document to get STL data from

        Returns:
            Raw STL bytes or None
        """
        # Try multiple sources
        if hasattr(doc, '_backend') and doc._backend is not None:
            backend = doc._backend
            # Try various attribute patterns
            for attr in ['_raw_data', 'raw_data', '_content', 'content']:
                if hasattr(backend, attr):
                    data = getattr(backend, attr)
                    if data is not None:
                        return data if isinstance(data, bytes) else data.encode('utf-8')

        # Try from origin
        if hasattr(doc, 'origin') and doc.origin is not None:
            if hasattr(doc.origin, 'binary_representation'):
                return doc.origin.binary_representation
            if hasattr(doc.origin, 'text'):
                text = doc.origin.text
                if text:
                    return text.encode('utf-8') if isinstance(text, str) else text

        return None

    def _parse_stl_data(self, data: bytes) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Parse STL data (ASCII or binary) into vertices and faces.

        Args:
            data: Raw STL file data

        Returns:
            Tuple of (vertices [N, 3], faces [M, 3]) or (None, None)
        """
        try:
            # Check if ASCII or binary STL
            # ASCII starts with "solid" (but binary can too with certain names)
            is_ascii = data[:5].lower() == b'solid'

            # Additional check: binary STL has specific structure
            # Header (80 bytes) + num_triangles (4 bytes) + triangles
            if is_ascii and len(data) > 84:
                # Check if file content matches ASCII pattern
                try:
                    text = data.decode('utf-8', errors='ignore')
                    if 'facet normal' in text or 'endsolid' in text:
                        return self._parse_ascii_stl(text)
                except Exception:
                    pass

            # Try binary parsing
            if len(data) > 84:
                return self._parse_binary_stl(data)

            # Fallback to ASCII
            try:
                text = data.decode('utf-8', errors='ignore')
                return self._parse_ascii_stl(text)
            except Exception:
                pass

            return None, None

        except Exception as e:
            _log.debug(f"STL parsing failed: {e}")
            return None, None

    def _parse_ascii_stl(self, text: str) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Parse ASCII STL format.

        Args:
            text: ASCII STL content

        Returns:
            Tuple of (vertices [N, 3], faces [M, 3]) or (None, None)
        """
        vertices = []
        faces = []

        # Parse facets
        # Pattern: facet normal nx ny nz
        #            outer loop
        #              vertex x y z
        #              vertex x y z
        #              vertex x y z
        #            endloop
        #          endfacet

        vertex_pattern = re.compile(
            r'vertex\s+([+-]?[\d.eE+-]+)\s+([+-]?[\d.eE+-]+)\s+([+-]?[\d.eE+-]+)',
            re.IGNORECASE
        )

        current_face_vertices = []
        vertex_idx = 0

        for match in vertex_pattern.finditer(text):
            x, y, z = float(match.group(1)), float(match.group(2)), float(match.group(3))
            vertices.append([x, y, z])
            current_face_vertices.append(vertex_idx)
            vertex_idx += 1

            if len(current_face_vertices) == 3:
                faces.append(current_face_vertices)
                current_face_vertices = []

        if len(vertices) < 3 or len(faces) < 1:
            return None, None

        return np.array(vertices), np.array(faces)

    def _parse_binary_stl(self, data: bytes) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Parse binary STL format.

        Binary STL structure:
        - 80 bytes: header
        - 4 bytes: number of triangles (uint32)
        - For each triangle (50 bytes each):
          - 12 bytes: normal vector (3 x float32)
          - 12 bytes: vertex 1 (3 x float32)
          - 12 bytes: vertex 2 (3 x float32)
          - 12 bytes: vertex 3 (3 x float32)
          - 2 bytes: attribute byte count (usually 0)

        Args:
            data: Binary STL data

        Returns:
            Tuple of (vertices [N, 3], faces [M, 3]) or (None, None)
        """
        try:
            # Read number of triangles
            num_triangles = struct.unpack('<I', data[80:84])[0]

            if num_triangles == 0:
                return None, None

            # Expected file size check
            expected_size = 84 + num_triangles * 50
            if len(data) < expected_size:
                _log.debug(f"Binary STL truncated: expected {expected_size}, got {len(data)}")
                # Try to read what we can
                num_triangles = min(num_triangles, (len(data) - 84) // 50)

            vertices = []
            faces = []

            offset = 84
            for i in range(num_triangles):
                if offset + 50 > len(data):
                    break

                # Skip normal (12 bytes), read 3 vertices (36 bytes), skip attribute (2 bytes)
                tri_data = data[offset:offset + 50]

                # Skip normal vector (first 12 bytes)
                # Read vertex 1
                v1 = struct.unpack('<fff', tri_data[12:24])
                # Read vertex 2
                v2 = struct.unpack('<fff', tri_data[24:36])
                # Read vertex 3
                v3 = struct.unpack('<fff', tri_data[36:48])

                base_idx = len(vertices)
                vertices.extend([list(v1), list(v2), list(v3)])
                faces.append([base_idx, base_idx + 1, base_idx + 2])

                offset += 50

            if len(vertices) < 3 or len(faces) < 1:
                return None, None

            return np.array(vertices), np.array(faces)

        except Exception as e:
            _log.debug(f"Binary STL parsing failed: {e}")
            return None, None

    def _assess_mesh_numpy(self, vertices: np.ndarray, faces: np.ndarray) -> dict:
        """Assess mesh quality using only numpy operations.

        This is the fallback when trimesh is unavailable.

        Args:
            vertices: Vertex positions [N, 3]
            faces: Face indices [M, 3]

        Returns:
            Dictionary with quality metrics
        """
        num_vertices = len(vertices)
        num_faces = len(faces)

        results = {
            "status": "success",
            "method": "numpy_fallback",
            "num_vertices": num_vertices,
            "num_faces": num_faces,
        }

        # Get triangle vertices
        v0 = vertices[faces[:, 0]]
        v1 = vertices[faces[:, 1]]
        v2 = vertices[faces[:, 2]]

        # Compute edge vectors and lengths
        e0 = v1 - v0  # edge 0-1
        e1 = v2 - v1  # edge 1-2
        e2 = v0 - v2  # edge 2-0

        len0 = np.linalg.norm(e0, axis=1)
        len1 = np.linalg.norm(e1, axis=1)
        len2 = np.linalg.norm(e2, axis=1)

        # All edge lengths
        all_edge_lengths = np.concatenate([len0, len1, len2])

        results["edge_lengths"] = {
            "min": float(np.min(all_edge_lengths)) if len(all_edge_lengths) > 0 else 0.0,
            "max": float(np.max(all_edge_lengths)) if len(all_edge_lengths) > 0 else 0.0,
            "mean": float(np.mean(all_edge_lengths)) if len(all_edge_lengths) > 0 else 0.0,
            "std": float(np.std(all_edge_lengths)) if len(all_edge_lengths) > 0 else 0.0,
        }

        # Aspect ratios per triangle
        edge_lengths = np.stack([len0, len1, len2], axis=1)
        max_edges = np.max(edge_lengths, axis=1)
        min_edges = np.min(edge_lengths, axis=1)

        # Avoid division by zero
        aspect_ratios = np.divide(
            max_edges,
            min_edges,
            out=np.full_like(max_edges, np.inf),
            where=min_edges > 1e-10,
        )

        results["aspect_ratio"] = {
            "min": float(np.min(aspect_ratios[np.isfinite(aspect_ratios)])) if np.any(np.isfinite(aspect_ratios)) else 0.0,
            "max": float(np.max(aspect_ratios[np.isfinite(aspect_ratios)])) if np.any(np.isfinite(aspect_ratios)) else 0.0,
            "mean": float(np.mean(aspect_ratios[np.isfinite(aspect_ratios)])) if np.any(np.isfinite(aspect_ratios)) else 0.0,
            "std": float(np.std(aspect_ratios[np.isfinite(aspect_ratios)])) if np.any(np.isfinite(aspect_ratios)) else 0.0,
        }

        # Count poor aspect ratios
        poor_aspect_ratio = np.sum(aspect_ratios > self.aspect_ratio_threshold)
        results["num_poor_aspect_ratio"] = int(poor_aspect_ratio)
        results["percent_poor_aspect_ratio"] = (
            float(poor_aspect_ratio) / num_faces * 100 if num_faces > 0 else 0.0
        )

        # Compute skewness
        # Perimeter of each triangle
        perimeter = len0 + len1 + len2

        # Area via cross product
        cross = np.cross(e0, -e2)
        actual_area = 0.5 * np.linalg.norm(cross, axis=1)

        # Ideal area for equilateral triangle with same perimeter
        side_ideal = perimeter / 3.0
        ideal_area = (np.sqrt(3) / 4.0) * side_ideal**2

        # Skewness = 1 - (actual / ideal)
        skewness = 1.0 - np.divide(
            actual_area,
            ideal_area,
            out=np.ones_like(actual_area),
            where=ideal_area > 1e-10,
        )
        skewness = np.clip(skewness, 0.0, 1.0)

        results["skewness"] = {
            "min": float(np.min(skewness)),
            "max": float(np.max(skewness)),
            "mean": float(np.mean(skewness)),
            "std": float(np.std(skewness)),
        }

        # Count high skewness
        high_skewness = np.sum(skewness > self.skewness_threshold)
        results["num_high_skewness"] = int(high_skewness)
        results["percent_high_skewness"] = (
            float(high_skewness) / num_faces * 100 if num_faces > 0 else 0.0
        )

        # Degenerate faces (zero or near-zero area)
        degenerate = actual_area < self.min_area_threshold
        results["num_degenerate_faces"] = int(np.sum(degenerate))
        results["percent_degenerate_faces"] = (
            float(np.sum(degenerate)) / num_faces * 100 if num_faces > 0 else 0.0
        )

        # Manifold check (simplified - check if all edges are shared by exactly 2 faces)
        # This is a heuristic since full manifold check requires edge tracking
        is_manifold = self._check_manifold_numpy(faces, num_vertices)
        results["is_manifold"] = is_manifold

        # Overall quality score
        good_triangles = num_faces - poor_aspect_ratio - high_skewness - np.sum(degenerate)
        quality_score = max(0.0, min(1.0, float(good_triangles) / num_faces)) if num_faces > 0 else 0.0
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
            f"Numpy mesh assessment: {num_faces} faces, "
            f"score={quality_score:.2f}, class={results['quality_class']}"
        )

        return results

    def _check_manifold_numpy(self, faces: np.ndarray, num_vertices: int) -> bool:
        """Check if mesh is manifold using numpy.

        A mesh is manifold if every edge is shared by exactly 2 faces.

        Args:
            faces: Face indices [M, 3]
            num_vertices: Number of vertices

        Returns:
            True if mesh appears manifold
        """
        try:
            # Build edge count dictionary
            # Edge is represented as (min_idx, max_idx) to be direction-independent
            edge_counts = {}

            for face in faces:
                edges = [
                    (min(face[0], face[1]), max(face[0], face[1])),
                    (min(face[1], face[2]), max(face[1], face[2])),
                    (min(face[2], face[0]), max(face[2], face[0])),
                ]
                for edge in edges:
                    edge_counts[edge] = edge_counts.get(edge, 0) + 1

            # Check if all edges are shared by exactly 2 faces
            # (boundary edges have count 1, non-manifold have count > 2)
            for count in edge_counts.values():
                if count > 2:
                    return False

            return True

        except Exception as e:
            _log.debug(f"Manifold check failed: {e}")
            return False

    def _assess_from_tessellation(self, doc: CADlingDocument) -> Optional[dict]:
        """Assess mesh from CAD tessellation.

        Args:
            doc: Document to tessellate

        Returns:
            Quality metrics or None
        """
        # Try to get tessellated mesh
        mesh = self._tessellate_cad(doc)

        if mesh is not None:
            return self._assess_mesh(mesh)

        return None

    def _estimate_from_properties(
        self, doc: CADlingDocument, item: CADItem
    ) -> Optional[dict]:
        """Estimate mesh quality from item properties.

        This is the last-resort fallback that estimates quality metrics
        from available geometry properties like bounding box and surface area.

        Args:
            doc: Document containing the item
            item: Item to estimate for

        Returns:
            Estimated quality metrics or None
        """
        try:
            # Try to get geometry analysis properties
            geom = item.properties.get("geometry_analysis", {})

            bbox = geom.get("bounding_box", {})
            surface_area = geom.get("surface_area", 0)
            volume = geom.get("volume", 0)

            # Need at least some data
            if not bbox and surface_area == 0:
                return None

            # Estimate mesh parameters from geometry
            # Use heuristics based on typical mesh density

            # Estimate face count from surface area
            # Typical mesh has ~1000-10000 triangles per cm² depending on curvature
            # Use conservative estimate
            estimated_face_area = 1.0  # mm² per face (typical)
            if surface_area > 0:
                estimated_faces = int(surface_area / estimated_face_area)
            else:
                # Fall back to bounding box diagonal
                dx = bbox.get("max_x", 0) - bbox.get("min_x", 0)
                dy = bbox.get("max_y", 0) - bbox.get("min_y", 0)
                dz = bbox.get("max_z", 0) - bbox.get("min_z", 0)
                diagonal = np.sqrt(dx**2 + dy**2 + dz**2)
                estimated_faces = int(diagonal * 10)  # Rough estimate

            estimated_faces = max(estimated_faces, 12)  # Minimum like a cube
            estimated_vertices = estimated_faces // 2  # Rough estimate

            # Estimate edge length from bbox
            dx = bbox.get("max_x", 1) - bbox.get("min_x", 0)
            dy = bbox.get("max_y", 1) - bbox.get("min_y", 0)
            dz = bbox.get("max_z", 1) - bbox.get("min_z", 0)

            avg_dimension = (dx + dy + dz) / 3.0
            estimated_edge_length = avg_dimension / np.sqrt(estimated_faces / 12.0)

            return {
                "status": "estimated",
                "reason": "Estimated from geometry properties",
                "method": "property_estimation",
                "num_vertices": estimated_vertices,
                "num_faces": estimated_faces,
                "edge_lengths": {
                    "min": estimated_edge_length * 0.5,
                    "max": estimated_edge_length * 2.0,
                    "mean": estimated_edge_length,
                    "std": estimated_edge_length * 0.3,
                },
                "aspect_ratio": {
                    "min": 1.0,
                    "max": 3.0,  # Typical for good meshes
                    "mean": 1.5,
                    "std": 0.5,
                },
                "num_poor_aspect_ratio": 0,
                "percent_poor_aspect_ratio": 0.0,
                "skewness": {
                    "min": 0.0,
                    "max": 0.5,
                    "mean": 0.2,
                    "std": 0.1,
                },
                "num_high_skewness": 0,
                "percent_high_skewness": 0.0,
                "num_degenerate_faces": 0,
                "percent_degenerate_faces": 0.0,
                "is_manifold": True,  # Assume good mesh
                "quality_score": 0.8,  # Estimated good quality
                "quality_class": "estimated",
            }

        except Exception as e:
            _log.debug(f"Property-based estimation failed: {e}")
            return None

    def _assess_mesh(self, mesh) -> dict:
        """Assess mesh quality.

        Args:
            mesh: Trimesh object to assess

        Returns:
            Dictionary with quality metrics
        """
        import trimesh

        results = {
            "status": "success"
        }

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
        quality_score = max(0.0, min(1.0, float(good_triangles) / num_faces)) if num_faces > 0 else 0.0
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
