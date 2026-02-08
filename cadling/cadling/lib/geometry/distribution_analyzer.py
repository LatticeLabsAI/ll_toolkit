"""
Geometric Distribution Analysis Module

Analyzes geometric property distributions across CAD models:
- Dihedral angles between adjacent faces
- Curvature distributions (Gaussian and mean)
- Surface type distributions
- BRep topological hierarchy

Used for validating CAD data matches expected patterns for mechanical parts.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import numpy as np

try:
    from OCC.Core.TopoDS import TopoDS_Face
    HAS_OCC = True
except ImportError:
    HAS_OCC = False
    logging.warning("OpenCASCADE (pythonocc-core) not available.")


logger = logging.getLogger(__name__)


class DihedralAngleAnalyzer:
    """Compute dihedral angles between adjacent faces in a CAD model."""

    @staticmethod
    def compute_dihedral_angles(doc: Any) -> Dict[str, Any]:
        """
        Compute all dihedral angles in the model.

        Args:
            doc: CADlingDocument with topology and items

        Returns:
            Dictionary with:
            - 'angles': List[float] - angles in radians [0, π]
            - 'mean': float - mean angle in radians
            - 'std': float - standard deviation in radians
            - 'median': float - median angle in radians
            - 'histogram_bins': np.ndarray - histogram bin edges
            - 'histogram_counts': np.ndarray - histogram counts
        """
        from cadling.lib.graph.features import compute_dihedral_angle
        from cadling.lib.geometry.face_geometry import FaceGeometryExtractor

        angles = []

        # Check if document has OCC shape available
        if not hasattr(doc, '_occ_shape') or doc._occ_shape is None:
            logger.warning("Document does not have OCC shape - cannot compute dihedral angles")
            return {
                'angles': [],
                'mean': 0.0,
                'std': 0.0,
                'median': 0.0,
                'histogram_bins': [],
                'histogram_counts': []
            }

        if not HAS_OCC:
            logger.warning("OpenCASCADE not available - cannot compute dihedral angles")
            return {
                'angles': [],
                'mean': 0.0,
                'std': 0.0,
                'median': 0.0,
                'histogram_bins': [],
                'histogram_counts': []
            }

        # Extract OCC faces
        from OCC.Core.TopExp import TopExp_Explorer
        from OCC.Core.TopAbs import TopAbs_FACE, TopAbs_EDGE
        from OCC.Core.TopoDS import topods
        from OCC.Extend.TopologyUtils import TopologyExplorer

        try:
            topo = TopologyExplorer(doc._occ_shape)

            # Build face list
            faces = list(topo.faces())
            if len(faces) == 0:
                logger.warning("No faces found in OCC shape")
                return {
                    'angles': [],
                    'mean': 0.0,
                    'std': 0.0,
                    'median': 0.0,
                    'histogram_bins': [],
                    'histogram_counts': []
                }

            # Compute normals for all faces
            face_normals = {}
            extractor = FaceGeometryExtractor()
            for idx, face in enumerate(faces):
                try:
                    normal = extractor.compute_normal_at_center(face)
                    if normal is not None and np.linalg.norm(normal) > 1e-10:
                        face_normals[idx] = normal
                except Exception as e:
                    logger.debug(f"Failed to compute normal for face {idx}: {e}")

            # Find adjacent face pairs by iterating through edges
            for edge in topo.edges():
                try:
                    # Get faces adjacent to this edge
                    adjacent_faces = list(topo.faces_from_edge(edge))

                    if len(adjacent_faces) == 2:
                        # Find indices in our face list
                        face_indices = []
                        for adj_face in adjacent_faces:
                            for idx, face in enumerate(faces):
                                if face.IsSame(adj_face):
                                    face_indices.append(idx)
                                    break

                        if len(face_indices) == 2:
                            idx1, idx2 = face_indices
                            if idx1 in face_normals and idx2 in face_normals:
                                # Compute dihedral angle
                                angle = compute_dihedral_angle(
                                    face_normals[idx1],
                                    face_normals[idx2]
                                )
                                angles.append(angle)

                except Exception as e:
                    logger.debug(f"Failed to process edge for dihedral angles: {e}")

        except Exception as e:
            logger.warning(f"Failed to compute dihedral angles: {e}")

        if len(angles) == 0:
            logger.warning("No dihedral angles computed")
            return {
                'angles': [],
                'mean': 0.0,
                'std': 0.0,
                'median': 0.0,
                'histogram_bins': [],
                'histogram_counts': []
            }

        # Convert to numpy array
        angles = np.array(angles)

        # Compute statistics
        mean_angle = float(np.mean(angles))
        std_angle = float(np.std(angles))
        median_angle = float(np.median(angles))

        # Compute histogram (36 bins = 5° per bin)
        histogram_counts, histogram_bins = np.histogram(angles, bins=36, range=(0, np.pi))

        logger.info(f"Computed {len(angles)} dihedral angles: "
                   f"mean={np.degrees(mean_angle):.1f}°, "
                   f"std={np.degrees(std_angle):.1f}°, "
                   f"median={np.degrees(median_angle):.1f}°")

        return {
            'angles': angles.tolist(),
            'mean': mean_angle,
            'std': std_angle,
            'median': median_angle,
            'histogram_bins': histogram_bins.tolist(),
            'histogram_counts': histogram_counts.tolist()
        }


class CurvatureAnalyzer:
    """Extract curvature distributions from all faces in a CAD model."""

    @staticmethod
    def compute_curvature_distribution(occ_faces: List[TopoDS_Face]) -> Dict[str, Any]:
        """
        Compute Gaussian and mean curvature distributions.

        Args:
            occ_faces: List of OpenCASCADE TopoDS_Face objects

        Returns:
            Dictionary with:
            - 'gaussian': Dict with 'values', 'mean', 'std', 'histogram_bins', 'histogram_counts'
            - 'mean': Dict with same structure for mean curvature
        """
        from cadling.lib.geometry.face_geometry import FaceGeometryExtractor

        if not HAS_OCC:
            logger.warning("OpenCASCADE not available - cannot compute curvature")
            return {
                'gaussian': {
                    'values': [],
                    'mean': 0.0,
                    'std': 0.0,
                    'histogram_bins': [],
                    'histogram_counts': []
                },
                'mean': {
                    'values': [],
                    'mean': 0.0,
                    'std': 0.0,
                    'histogram_bins': [],
                    'histogram_counts': []
                }
            }

        gaussian_curvatures = []
        mean_curvatures = []

        extractor = FaceGeometryExtractor()
        for idx, face in enumerate(occ_faces):
            try:
                result = extractor.compute_curvature_at_center(face)
                if result is not None:
                    gaussian_k, mean_h = result
                    if gaussian_k is not None and np.isfinite(gaussian_k):
                        gaussian_curvatures.append(gaussian_k)
                    if mean_h is not None and np.isfinite(mean_h):
                        mean_curvatures.append(mean_h)
            except Exception as e:
                logger.debug(f"Failed to compute curvature for face {idx}: {e}")

        # Process Gaussian curvature
        gaussian_result = CurvatureAnalyzer._process_curvature_values(
            gaussian_curvatures, "Gaussian"
        )

        # Process mean curvature
        mean_result = CurvatureAnalyzer._process_curvature_values(
            mean_curvatures, "Mean"
        )

        return {
            'gaussian': gaussian_result,
            'mean': mean_result
        }

    @staticmethod
    def _process_curvature_values(values: List[float], curvature_type: str) -> Dict[str, Any]:
        """Process curvature values into statistics and histogram."""
        if len(values) == 0:
            logger.warning(f"No {curvature_type} curvature values computed")
            return {
                'values': [],
                'mean': 0.0,
                'std': 0.0,
                'histogram_bins': [],
                'histogram_counts': []
            }

        values_array = np.array(values)

        # Clip extreme values for histogram (keep within reasonable range)
        # Most mechanical parts have curvature in range [-100, 100]
        clipped_values = np.clip(values_array, -100, 100)

        # Compute statistics on original values
        mean_val = float(np.mean(values_array))
        std_val = float(np.std(values_array))

        # Compute histogram on clipped values
        histogram_counts, histogram_bins = np.histogram(clipped_values, bins=50)

        logger.info(f"{curvature_type} curvature: {len(values)} faces, "
                   f"mean={mean_val:.4f}, std={std_val:.4f}")

        return {
            'values': values_array.tolist(),
            'mean': mean_val,
            'std': std_val,
            'histogram_bins': histogram_bins.tolist(),
            'histogram_counts': histogram_counts.tolist()
        }


class SurfaceTypeAnalyzer:
    """Categorize and count surface types in a CAD model."""

    @staticmethod
    def analyze_surface_types(doc: Any) -> Dict[str, int]:
        """
        Count occurrences of each surface type.

        Args:
            doc: CADlingDocument with items

        Returns:
            Dictionary mapping surface type names to counts:
            {
                'PLANE': count,
                'CYLINDRICAL_SURFACE': count,
                'CONICAL_SURFACE': count,
                ...
            }
        """
        surface_counts = {}

        # Define surface type keywords to look for
        surface_types = [
            'PLANE',
            'CYLINDRICAL_SURFACE',
            'CONICAL_SURFACE',
            'SPHERICAL_SURFACE',
            'TOROIDAL_SURFACE',
            'B_SPLINE_SURFACE',
            'SURFACE_OF_REVOLUTION',
            'SURFACE_OF_LINEAR_EXTRUSION',
            'SWEPT_SURFACE',
            'ADVANCED_FACE',  # Generic face type
        ]

        # Initialize counts
        for surface_type in surface_types:
            surface_counts[surface_type] = 0

        # Count occurrences in document items
        for item in doc.items:
            entity_type = item.entity_type if hasattr(item, 'entity_type') else None
            if entity_type:
                # Check if this is a surface entity
                for surface_type in surface_types:
                    if surface_type in entity_type:
                        surface_counts[surface_type] += 1
                        break

        # Remove zero counts
        surface_counts = {k: v for k, v in surface_counts.items() if v > 0}

        # Add total
        total = sum(surface_counts.values())

        logger.info(f"Surface type distribution: {len(surface_counts)} types, "
                   f"{total} total surfaces")
        for surface_type, count in sorted(surface_counts.items(), key=lambda x: -x[1]):
            logger.info(f"  {surface_type}: {count} ({count/total*100:.1f}%)")

        return surface_counts


class BRepHierarchyAnalyzer:
    """Analyze BRep topological hierarchy in a CAD model."""

    @staticmethod
    def extract_hierarchy(doc: Any) -> Dict[str, Any]:
        """
        Extract shells → faces → edges → vertices hierarchy.

        Uses OCC topology traversal when available for accurate counts.
        Falls back to mesh metadata for STL documents or STEP entity types.

        Args:
            doc: CADlingDocument with topology and items

        Returns:
            Dictionary with:
            - 'num_shells': int
            - 'num_faces': int
            - 'num_edges': int
            - 'num_vertices': int
            - 'topology_type': str (e.g., 'brep', 'mesh')
            - 'euler_characteristic': int (V - E + F)
        """
        num_shells = 0
        num_faces = 0
        num_edges = 0
        num_vertices = 0

        # Get topology type from metadata
        topology_type = doc.metadata.get('topology', {}).get('type', 'unknown') \
                       if hasattr(doc, 'metadata') and doc.metadata else 'unknown'

        # Check for mesh document (STL) - extract from mesh metadata
        if hasattr(doc, 'mesh') and doc.mesh is not None:
            mesh = doc.mesh
            num_vertices = getattr(mesh, 'num_vertices', 0) or 0
            num_faces = getattr(mesh, 'num_facets', 0) or 0
            # For triangular mesh: edges = 3*F/2 (each edge shared by 2 triangles)
            # But for open meshes, some edges are only on one face
            # Estimate: E = 3*F/2 for closed, slightly more for open
            num_edges = int(num_faces * 3 / 2) if num_faces > 0 else 0
            num_shells = 1 if num_faces > 0 else 0
            topology_type = 'mesh'

            logger.debug(f"Mesh topology: {num_shells} shells, {num_faces} facets, "
                        f"{num_edges} edges (estimated), {num_vertices} vertices")

        # Prefer OCC topology traversal for accurate counts (B-Rep)
        elif HAS_OCC and hasattr(doc, '_occ_shape') and doc._occ_shape is not None:
            try:
                from OCC.Extend.TopologyUtils import TopologyExplorer

                topo = TopologyExplorer(doc._occ_shape)

                # Count actual B-Rep entities using OCC
                num_shells = len(list(topo.shells()))
                num_faces = len(list(topo.faces()))
                num_edges = len(list(topo.edges()))
                num_vertices = len(list(topo.vertices()))

                topology_type = 'brep'

                logger.debug(f"OCC topology: {num_shells} shells, {num_faces} faces, "
                           f"{num_edges} edges, {num_vertices} vertices")

            except Exception as e:
                logger.warning(f"Failed to extract OCC topology: {e}")
                # Fall back to counting by entity_type attribute
                num_shells, num_faces, num_edges, num_vertices = \
                    BRepHierarchyAnalyzer._count_by_entity_type(doc)
        else:
            # Fall back to counting by entity_type attribute
            num_shells, num_faces, num_edges, num_vertices = \
                BRepHierarchyAnalyzer._count_by_entity_type(doc)

        # Compute Euler characteristic: V - E + F
        # For a closed solid: V - E + F = 2 (Euler's formula)
        # For a surface/sheet: V - E + F = 1
        euler_characteristic = num_vertices - num_edges + num_faces

        logger.info(f"BRep hierarchy: {num_shells} shells, {num_faces} faces, "
                   f"{num_edges} edges, {num_vertices} vertices")
        logger.info(f"Euler characteristic: {euler_characteristic} "
                   f"(expected 2 for closed solid, 1 for open surface)")

        return {
            'num_shells': num_shells,
            'num_faces': num_faces,
            'num_edges': num_edges,
            'num_vertices': num_vertices,
            'topology_type': topology_type,
            'euler_characteristic': euler_characteristic
        }

    @staticmethod
    def _count_by_entity_type(doc: Any) -> tuple:
        """Fall back to counting by STEP entity_type attribute.

        Uses exact STEP B-Rep entity type names to avoid false positives
        from geometric entities that contain similar keywords.

        STEP B-Rep topology entity types:
        - Shells: CLOSED_SHELL, OPEN_SHELL
        - Faces: ADVANCED_FACE, FACE_SURFACE
        - Edges: EDGE_CURVE, ORIENTED_EDGE
        - Vertices: VERTEX_POINT
        """
        num_shells = 0
        num_faces = 0
        num_edges = 0
        num_vertices = 0

        # Exact STEP B-Rep entity types (topological, not geometric)
        shell_types = {'CLOSED_SHELL', 'OPEN_SHELL'}
        face_types = {'ADVANCED_FACE', 'FACE_SURFACE'}
        edge_types = {'EDGE_CURVE', 'ORIENTED_EDGE'}
        vertex_types = {'VERTEX_POINT'}

        for item in doc.items:
            entity_type = getattr(item, 'entity_type', '') or ''
            entity_type_upper = entity_type.upper()

            if entity_type_upper in shell_types:
                num_shells += 1
            elif entity_type_upper in face_types:
                num_faces += 1
            elif entity_type_upper in edge_types:
                num_edges += 1
            elif entity_type_upper in vertex_types:
                num_vertices += 1

        return num_shells, num_faces, num_edges, num_vertices
