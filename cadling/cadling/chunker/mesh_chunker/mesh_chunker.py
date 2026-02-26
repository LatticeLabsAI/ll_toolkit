"""Generic mesh chunking utilities.

This module provides format-agnostic mesh segmentation algorithms
that can be applied to various mesh representations (STL, OBJ, PLY, etc.).

Core algorithms:
    - Octree-based spatial partitioning
    - K-means clustering
    - Graph-based segmentation
    - Feature-based segmentation

Classes:
    MeshChunker: Generic mesh chunker with multiple algorithms
"""

from __future__ import annotations

import logging
import numpy as np
from collections import defaultdict, deque
from typing import TYPE_CHECKING, Iterator, Optional, List, Tuple, Any

from cadling.chunker.base_chunker import BaseCADChunker, CADChunk, CADChunkMeta

if TYPE_CHECKING:
    from cadling.datamodel.base_models import CADlingDocument

_log = logging.getLogger(__name__)


class MeshData:
    """Generic mesh data structure.

    Attributes:
        vertices: Nx3 array of vertex coordinates
        faces: Mx3 array of face indices
        normals: Mx3 array of face normals
        vertex_normals: Nx3 array of vertex normals (optional)
        colors: Nx3 or Mx3 array of colors (optional)
        properties: Additional mesh properties
    """

    def __init__(
        self,
        vertices: np.ndarray,
        faces: np.ndarray,
        normals: Optional[np.ndarray] = None,
        vertex_normals: Optional[np.ndarray] = None,
        colors: Optional[np.ndarray] = None,
    ):
        self.vertices = vertices
        self.faces = faces
        self.normals = normals
        self.vertex_normals = vertex_normals
        self.colors = colors
        self.properties: dict = {}

    @property
    def num_vertices(self) -> int:
        return len(self.vertices)

    @property
    def num_faces(self) -> int:
        return len(self.faces)


class OctreeNode:
    """Octree node for spatial partitioning.

    Attributes:
        bounds: (min_xyz, max_xyz) bounding box
        face_indices: Indices of faces in this node
        children: Child octree nodes (8 for octree)
        is_leaf: Whether this is a leaf node
    """

    def __init__(self, bounds: Tuple[np.ndarray, np.ndarray]):
        self.bounds = bounds
        self.face_indices: List[int] = []
        self.children: List[Optional[OctreeNode]] = [None] * 8
        self.is_leaf = True

    def subdivide(self):
        """Subdivide node into 8 children."""
        min_xyz, max_xyz = self.bounds
        mid_xyz = (min_xyz + max_xyz) / 2.0

        # Create 8 child nodes
        child_bounds = [
            # Bottom 4
            (min_xyz, mid_xyz),  # 0: ---
            (np.array([mid_xyz[0], min_xyz[1], min_xyz[2]]), np.array([max_xyz[0], mid_xyz[1], mid_xyz[2]])),  # 1: +--
            (np.array([min_xyz[0], mid_xyz[1], min_xyz[2]]), np.array([mid_xyz[0], max_xyz[1], mid_xyz[2]])),  # 2: -+-
            (np.array([mid_xyz[0], mid_xyz[1], min_xyz[2]]), np.array([max_xyz[0], max_xyz[1], mid_xyz[2]])),  # 3: ++-
            # Top 4
            (np.array([min_xyz[0], min_xyz[1], mid_xyz[2]]), np.array([mid_xyz[0], mid_xyz[1], max_xyz[2]])),  # 4: --+
            (np.array([mid_xyz[0], min_xyz[1], mid_xyz[2]]), np.array([max_xyz[0], mid_xyz[1], max_xyz[2]])),  # 5: +-+
            (np.array([min_xyz[0], mid_xyz[1], mid_xyz[2]]), np.array([mid_xyz[0], max_xyz[1], max_xyz[2]])),  # 6: -++
            (mid_xyz, max_xyz),  # 7: +++
        ]

        for i, bounds in enumerate(child_bounds):
            self.children[i] = OctreeNode(bounds)

        self.is_leaf = False


class MeshChunker(BaseCADChunker):
    """Generic mesh chunker with multiple segmentation algorithms.

    Supports various mesh formats and provides format-agnostic
    segmentation strategies.

    Attributes:
        strategy: Chunking strategy ("octree", "kmeans", "graph", "feature")
        max_faces_per_chunk: Maximum faces per chunk
        octree_max_depth: Maximum octree depth
        num_clusters: Number of clusters for k-means
    """

    def __init__(
        self,
        strategy: str = "octree",
        max_faces_per_chunk: int = 5000,
        max_tokens: int = 512,
        overlap_tokens: int = 50,
        octree_max_depth: int = 6,
        num_clusters: int = 8,
        seed: Optional[int] = None,
    ):
        """Initialize mesh chunker.

        Args:
            strategy: Segmentation strategy
            max_faces_per_chunk: Maximum faces per chunk
            max_tokens: Maximum tokens per chunk
            overlap_tokens: Overlap tokens
            octree_max_depth: Maximum octree depth
            num_clusters: Number of k-means clusters
            seed: Random seed for reproducibility
        """
        super().__init__(max_tokens, overlap_tokens)
        self.strategy = strategy
        self.max_faces_per_chunk = max_faces_per_chunk
        self.octree_max_depth = octree_max_depth
        self.num_clusters = num_clusters
        self.rng = np.random.default_rng(seed)  # Numpy RNG for reproducibility

    def chunk_mesh_data(self, mesh: MeshData, doc_name: str) -> Iterator[CADChunk]:
        """Chunk mesh data into segments.

        Args:
            mesh: Mesh data
            doc_name: Document name

        Yields:
            CADChunk objects
        """
        _log.info(f"Chunking mesh with {mesh.num_faces} faces using strategy: {self.strategy}")

        if self.strategy == "octree":
            segments = self._segment_by_octree(mesh)
        elif self.strategy == "kmeans":
            segments = self._segment_by_kmeans(mesh)
        elif self.strategy == "graph":
            segments = self._segment_by_graph(mesh)
        elif self.strategy == "feature":
            segments = self._segment_by_features(mesh)
        else:
            _log.warning(f"Unknown strategy: {self.strategy}, using octree")
            segments = self._segment_by_octree(mesh)

        _log.info(f"Created {len(segments)} mesh segments")

        for i, face_indices in enumerate(segments):
            yield self._create_chunk_from_faces(mesh, face_indices, doc_name, i)

    def chunk(self, doc: CADlingDocument) -> Iterator[CADChunk]:
        """Chunk CAD document (must contain mesh data).

        Args:
            doc: CADling document

        Yields:
            CADChunk objects
        """
        # Extract mesh data from document
        mesh = self._extract_mesh_from_doc(doc)

        if mesh is None:
            _log.warning("No mesh data found in document")
            return

        yield from self.chunk_mesh_data(mesh, doc.name)

    def _extract_mesh_from_doc(self, doc: CADlingDocument) -> Optional[MeshData]:
        """Extract mesh data from CADling document.

        Args:
            doc: CADling document

        Returns:
            MeshData or None
        """
        # Try to extract from STL document
        try:
            from cadling.datamodel.stl import STLDocument, STLFacet

            if isinstance(doc, STLDocument):
                facets = [item for item in doc.items if isinstance(item, STLFacet)]

                if not facets:
                    return None

                # Convert facets to mesh data
                vertices_list = []
                faces_list = []
                normals_list = []
                vertex_map = {}

                for facet in facets:
                    face_verts = []
                    for v in [facet.v1, facet.v2, facet.v3]:
                        v_key = tuple(v)
                        if v_key not in vertex_map:
                            vertex_map[v_key] = len(vertices_list)
                            vertices_list.append(v)
                        face_verts.append(vertex_map[v_key])

                    faces_list.append(face_verts)
                    normals_list.append(facet.normal)

                vertices = np.array(vertices_list)
                faces = np.array(faces_list)
                normals = np.array(normals_list)

                return MeshData(vertices, faces, normals)

        except ImportError as e:
            _log.warning(f"Failed to import STL datamodel: {e}. STL mesh extraction unavailable.")

        # Try to get mesh from generic backend
        if hasattr(doc, "_backend") and doc._backend is not None:
            backend = doc._backend

            # Try various patterns for getting mesh from backend
            mesh_obj = None

            # Pattern 1: backend.mesh
            if hasattr(backend, "mesh") and backend.mesh is not None:
                mesh_obj = backend.mesh
            # Pattern 2: backend._mesh
            elif hasattr(backend, "_mesh") and backend._mesh is not None:
                mesh_obj = backend._mesh
            # Pattern 3: backend.get_mesh()
            elif hasattr(backend, "get_mesh") and callable(backend.get_mesh):
                try:
                    mesh_obj = backend.get_mesh()
                except Exception as e:
                    _log.debug(f"backend.get_mesh() failed: {e}")
            # Pattern 4: backend.load_mesh()
            elif hasattr(backend, "load_mesh") and callable(backend.load_mesh):
                try:
                    mesh_obj = backend.load_mesh()
                except Exception as e:
                    _log.debug(f"backend.load_mesh() failed: {e}")

            if mesh_obj is not None:
                try:
                    # Check if it's a trimesh object
                    import trimesh
                    if isinstance(mesh_obj, trimesh.Trimesh):
                        vertices = np.array(mesh_obj.vertices)
                        faces = np.array(mesh_obj.faces)
                        normals = np.array(mesh_obj.face_normals) if hasattr(mesh_obj, "face_normals") else None
                        _log.debug(f"Extracted mesh from backend: {len(vertices)} vertices, {len(faces)} faces")
                        return MeshData(vertices, faces, normals)
                    # Check if it has vertices/faces attributes
                    elif hasattr(mesh_obj, "vertices") and hasattr(mesh_obj, "faces"):
                        vertices = np.array(mesh_obj.vertices)
                        faces = np.array(mesh_obj.faces)
                        normals = np.array(mesh_obj.normals) if hasattr(mesh_obj, "normals") else None
                        _log.debug(f"Extracted mesh from backend object: {len(vertices)} vertices, {len(faces)} faces")
                        return MeshData(vertices, faces, normals)
                except ImportError:
                    _log.debug("trimesh not available for mesh type checking")
                except Exception as e:
                    _log.warning(f"Failed to extract mesh from backend: {e}")

        return None

    def _segment_by_octree(self, mesh: MeshData) -> List[List[int]]:
        """Segment mesh using octree spatial partitioning.

        Args:
            mesh: Mesh data

        Returns:
            List of face index lists
        """
        # Compute mesh bounds
        min_xyz = mesh.vertices.min(axis=0)
        max_xyz = mesh.vertices.max(axis=0)

        # Build octree
        root = OctreeNode((min_xyz, max_xyz))
        self._build_octree(root, mesh, 0)

        # Collect leaf nodes
        segments = []
        self._collect_octree_leaves(root, segments)

        return segments

    def _build_octree(
        self,
        node: OctreeNode,
        mesh: MeshData,
        depth: int,
        candidate_faces: Optional[List[int]] = None,
    ):
        """Recursively build octree with proper face partitioning.

        Args:
            node: Current octree node
            mesh: Mesh data
            depth: Current depth
            candidate_faces: Face indices to consider (from parent). If None, consider all.
        """
        min_xyz, max_xyz = node.bounds

        # Determine which faces to consider
        if candidate_faces is None:
            # Root node: consider all faces
            faces_to_check = range(mesh.num_faces)
        else:
            # Child node: only consider faces from parent
            faces_to_check = candidate_faces

        # Assign faces whose centroids fall within this node's bounds
        for face_idx in faces_to_check:
            face = mesh.faces[face_idx]
            face_verts = mesh.vertices[face]
            face_center = face_verts.mean(axis=0)

            # Check if face center is in bounds
            if np.all(face_center >= min_xyz) and np.all(face_center <= max_xyz):
                node.face_indices.append(face_idx)

        # Stop at max depth - keep faces in this leaf
        if depth >= self.octree_max_depth:
            return

        # Subdivide if too many faces
        if len(node.face_indices) > self.max_faces_per_chunk:
            node.subdivide()

            mid_xyz = (min_xyz + max_xyz) / 2.0

            # Distribute parent's faces to children based on octant
            child_faces: List[List[int]] = [[] for _ in range(8)]

            for face_idx in node.face_indices:
                face = mesh.faces[face_idx]
                face_center = mesh.vertices[face].mean(axis=0)

                # Determine which octant the face belongs to
                octant = 0
                if face_center[0] >= mid_xyz[0]:
                    octant |= 1
                if face_center[1] >= mid_xyz[1]:
                    octant |= 2
                if face_center[2] >= mid_xyz[2]:
                    octant |= 4

                child_faces[octant].append(face_idx)

            # Recursively build children that have faces
            for i, child in enumerate(node.children):
                if child is not None and child_faces[i]:
                    # Pass only the faces assigned to this child
                    self._build_octree(child, mesh, depth + 1, child_faces[i])

            node.face_indices = []  # Clear parent's faces
            node.is_leaf = False

    def _collect_octree_leaves(self, node: OctreeNode, segments: List[List[int]]):
        """Collect face indices from octree leaves.

        Args:
            node: Current node
            segments: Output list of segments
        """
        if node.is_leaf:
            if node.face_indices:
                segments.append(node.face_indices)
        else:
            for child in node.children:
                if child is not None:
                    self._collect_octree_leaves(child, segments)

    def _segment_by_kmeans(self, mesh: MeshData) -> List[List[int]]:
        """Segment mesh using k-means clustering on face centroids.

        Args:
            mesh: Mesh data

        Returns:
            List of face index lists
        """
        # Compute face centroids
        centroids = np.zeros((mesh.num_faces, 3))
        for i, face in enumerate(mesh.faces):
            centroids[i] = mesh.vertices[face].mean(axis=0)

        # Simple k-means implementation
        k = min(self.num_clusters, mesh.num_faces)
        labels = self._kmeans(centroids, k)

        # Group faces by cluster
        clusters = defaultdict(list)
        for i, label in enumerate(labels):
            clusters[label].append(i)

        return list(clusters.values())

    def _kmeans(self, points: np.ndarray, k: int, max_iters: int = 100) -> np.ndarray:
        """Simple k-means clustering.

        Args:
            points: Nx3 array of points
            k: Number of clusters
            max_iters: Maximum iterations

        Returns:
            Array of cluster labels
        """
        # Initialize centroids randomly (use seeded RNG for reproducibility)
        indices = self.rng.choice(len(points), k, replace=False)
        centroids = points[indices].copy()

        labels = np.zeros(len(points), dtype=int)

        for _ in range(max_iters):
            # Assign points to nearest centroid
            distances = np.linalg.norm(points[:, None] - centroids[None, :], axis=2)
            new_labels = np.argmin(distances, axis=1)

            if np.all(new_labels == labels):
                break

            labels = new_labels

            # Update centroids
            for i in range(k):
                cluster_points = points[labels == i]
                if len(cluster_points) > 0:
                    centroids[i] = cluster_points.mean(axis=0)

        return labels

    def _segment_by_graph(self, mesh: MeshData) -> List[List[int]]:
        """Segment mesh using graph-based methods.

        Uses normalized cuts approximation. Ensures no faces are dropped:
        faces that remain in the BFS queue when a segment reaches
        max_faces_per_chunk are revisited as seeds for subsequent segments.

        Args:
            mesh: Mesh data

        Returns:
            List of face index lists
        """
        # Build adjacency graph
        adjacency = self._build_face_adjacency(mesh)

        # Simple connected components with size limit
        visited = set()
        segments = []

        for start_idx in range(mesh.num_faces):
            if start_idx in visited:
                continue

            # BFS using deque for O(1) popleft
            component = []
            queue = deque([start_idx])
            visited.add(start_idx)

            while queue:
                if len(component) >= self.max_faces_per_chunk:
                    # Emit the current segment but do NOT discard
                    # remaining queued faces -- unmark them so they
                    # become seeds for the next segment.
                    for remaining in queue:
                        visited.discard(remaining)
                    break

                current = queue.popleft()  # O(1) instead of O(n)
                component.append(current)

                for neighbor in adjacency.get(current, []):
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append(neighbor)

            if component:
                segments.append(component)

        return segments

    def _segment_by_features(self, mesh: MeshData) -> List[List[int]]:
        """Segment mesh using geometric features.

        Combines normal similarity, curvature, and spatial proximity.
        Faces that fail the normal similarity test against all their
        neighbors are still assigned to their own single-face segments
        so that no faces are lost.

        Args:
            mesh: Mesh data

        Returns:
            List of face index lists
        """
        if mesh.normals is None:
            _log.warning("No normals available, falling back to octree")
            return self._segment_by_octree(mesh)

        # Cluster by normal similarity
        adjacency = self._build_face_adjacency(mesh)
        visited = set()
        segments = []

        normal_threshold = 0.9  # Cosine similarity threshold

        for seed_idx in range(mesh.num_faces):
            if seed_idx in visited:
                continue

            seed_normal = mesh.normals[seed_idx]
            region = []
            queue = deque([seed_idx])  # Use deque for O(1) popleft
            visited.add(seed_idx)

            while queue:
                if len(region) >= self.max_faces_per_chunk:
                    # Unmark remaining queued faces so they are not lost
                    for remaining in queue:
                        visited.discard(remaining)
                    break

                current_idx = queue.popleft()  # O(1) instead of O(n)
                region.append(current_idx)

                for neighbor_idx in adjacency.get(current_idx, []):
                    if neighbor_idx in visited:
                        continue

                    # Check normal similarity
                    neighbor_normal = mesh.normals[neighbor_idx]
                    similarity = np.dot(seed_normal, neighbor_normal)

                    if similarity > normal_threshold:
                        visited.add(neighbor_idx)
                        queue.append(neighbor_idx)

            if region:
                segments.append(region)

        # Ensure no faces are dropped: any face not yet visited gets its own segment
        all_face_indices = set(range(mesh.num_faces))
        assigned = set()
        for seg in segments:
            assigned.update(seg)
        orphans = all_face_indices - assigned
        if orphans:
            _log.debug("Assigning %d orphan faces to individual segments", len(orphans))
            for face_idx in sorted(orphans):
                segments.append([face_idx])

        return segments

    def _build_face_adjacency(self, mesh: MeshData) -> dict:
        """Build face adjacency graph.

        Args:
            mesh: Mesh data

        Returns:
            Dictionary mapping face index to adjacent face indices
        """
        adjacency = defaultdict(set)
        edge_to_faces = defaultdict(set)

        # Build edge to face mapping
        for face_idx, face in enumerate(mesh.faces):
            edges = [
                tuple(sorted([face[0], face[1]])),
                tuple(sorted([face[1], face[2]])),
                tuple(sorted([face[2], face[0]])),
            ]

            for edge in edges:
                edge_to_faces[edge].add(face_idx)

        # Build adjacency from shared edges
        for edge, face_set in edge_to_faces.items():
            face_list = list(face_set)
            for i, f1 in enumerate(face_list):
                for f2 in face_list[i+1:]:
                    adjacency[f1].add(f2)
                    adjacency[f2].add(f1)

        return adjacency

    def _create_chunk_from_faces(
        self,
        mesh: MeshData,
        face_indices: List[int],
        doc_name: str,
        segment_id: int,
    ) -> CADChunk:
        """Create CADChunk from mesh faces.

        Args:
            mesh: Mesh data
            face_indices: Indices of faces in segment
            doc_name: Document name
            segment_id: Segment identifier

        Returns:
            CADChunk
        """
        # Extract vertices for this segment
        segment_vertices = set()
        for face_idx in face_indices:
            face = mesh.faces[face_idx]
            segment_vertices.update(face)

        segment_verts_array = mesh.vertices[list(segment_vertices)]

        # Compute bounds
        min_bounds = segment_verts_array.min(axis=0)
        max_bounds = segment_verts_array.max(axis=0)

        # Compute surface area
        total_area = 0.0
        for face_idx in face_indices:
            face = mesh.faces[face_idx]
            verts = mesh.vertices[face]
            area = 0.5 * np.linalg.norm(np.cross(verts[1] - verts[0], verts[2] - verts[0]))
            total_area += area

        properties = {
            "face_indices": face_indices,  # CRITICAL: Store for mesh segmentation!
            "num_faces": len(face_indices),
            "num_vertices": len(segment_vertices),
            "surface_area": float(total_area),
            "bounds_min": min_bounds.tolist(),
            "bounds_max": max_bounds.tolist(),
        }

        # Generate text representation
        text_lines = [f"Mesh Segment {segment_id}"]
        text_lines.append(f"Faces: {len(face_indices)}, Vertices: {len(segment_vertices)}")
        text_lines.append(f"Bounds: {min_bounds.tolist()} to {max_bounds.tolist()}")
        text_lines.append(f"Surface Area: {total_area:.6f}")

        text = "\n".join(text_lines)

        meta = CADChunkMeta(properties=properties)

        return CADChunk(
            text=text,
            meta=meta,
            chunk_id=f"{doc_name}_mesh_segment_{segment_id}",
            doc_name=doc_name,
        )


# Convenience aliases
def OctreeChunker(**kwargs) -> MeshChunker:
    """Create a MeshChunker with octree strategy."""
    return MeshChunker(strategy="octree", **kwargs)


def KMeansChunker(**kwargs) -> MeshChunker:
    """Create a MeshChunker with kmeans strategy."""
    return MeshChunker(strategy="kmeans", **kwargs)


def GraphChunker(**kwargs) -> MeshChunker:
    """Create a MeshChunker with graph strategy."""
    return MeshChunker(strategy="graph", **kwargs)


def FeatureChunker(**kwargs) -> MeshChunker:
    """Create a MeshChunker with feature strategy."""
    return MeshChunker(strategy="feature", **kwargs)
