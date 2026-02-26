"""Vertex clustering and merging for CAD mesh generation.

Groups predicted vertices that are within a merge distance and replaces
each cluster with a single representative point.  After merging, face
indices are remapped so the mesh remains consistent.

Three clustering strategies are provided:

- ``"kdtree"``: KDTree ``query_ball_point`` grouping (fastest, recommended).
- ``"hierarchical"``: Complete-linkage agglomerative clustering via scipy.
- ``"dbscan"``: DBSCAN density-based clustering via scipy.

All strategies are accessible through a single ``VertexClusterer`` class.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

import numpy as np

_log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result data classes
# ---------------------------------------------------------------------------


@dataclass
class ClusteringResult:
    """Result of vertex clustering."""

    labels: np.ndarray  # (N,) cluster ID per vertex (0-indexed)
    centers: np.ndarray  # (K, 3) cluster representative positions
    merge_map: Dict[int, int]  # old_vertex_idx → new_cluster_idx
    num_clusters: int  # K
    num_merged: int  # N - K (vertices removed by merging)
    method: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# VertexClusterer
# ---------------------------------------------------------------------------


class VertexClusterer:
    """Cluster vertices by proximity and identify duplicates.

    Given a set of predicted vertex positions, groups those that fall
    within ``merge_distance`` of each other into clusters.  Each
    cluster is represented by its centroid (or weighted centroid if
    confidence weights are provided).

    Args:
        merge_distance: Maximum Euclidean distance for two vertices
            to be merged into the same cluster.
        method: Clustering algorithm — ``"kdtree"`` (default),
            ``"hierarchical"``, or ``"dbscan"``.

    Example::

        clusterer = VertexClusterer(merge_distance=0.01)
        result = clusterer.cluster(vertices)
        merged_verts, new_faces = VertexMerger.merge(vertices, faces, result)
    """

    def __init__(
        self,
        merge_distance: float = 1e-3,
        method: str = "kdtree",
    ) -> None:
        if method not in ("kdtree", "hierarchical", "dbscan"):
            raise ValueError(
                f"Unknown clustering method '{method}'. "
                "Choose from 'kdtree', 'hierarchical', 'dbscan'."
            )
        self.merge_distance = merge_distance
        self.method = method

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def cluster(
        self,
        vertices: np.ndarray,
        weights: Optional[np.ndarray] = None,
    ) -> ClusteringResult:
        """Cluster vertices and compute merged representative positions.

        Args:
            vertices: Vertex positions ``(N, 3)`` float.
            weights: Optional per-vertex confidence scores ``(N,)`` float.
                When provided, cluster centers are weighted centroids.

        Returns:
            ``ClusteringResult`` with labels, centers, and merge map.
        """
        if vertices.ndim != 2 or vertices.shape[1] != 3:
            raise ValueError(
                f"vertices must be (N, 3), got shape {vertices.shape}"
            )

        if len(vertices) == 0:
            return ClusteringResult(
                labels=np.array([], dtype=np.int64),
                centers=np.empty((0, 3), dtype=np.float32),
                merge_map={},
                num_clusters=0,
                num_merged=0,
                method=self.method,
            )

        if len(vertices) == 1:
            return ClusteringResult(
                labels=np.array([0], dtype=np.int64),
                centers=vertices.copy().astype(np.float32),
                merge_map={0: 0},
                num_clusters=1,
                num_merged=0,
                method=self.method,
            )

        if self.method == "kdtree":
            return self._cluster_kdtree(vertices, weights)
        elif self.method == "hierarchical":
            return self._cluster_hierarchical(vertices, weights)
        elif self.method == "dbscan":
            return self._cluster_dbscan(vertices, weights)
        else:
            raise ValueError(f"Unknown method: {self.method}")

    # ------------------------------------------------------------------
    # KDTree clustering (recommended)
    # ------------------------------------------------------------------

    def _cluster_kdtree(
        self,
        vertices: np.ndarray,
        weights: Optional[np.ndarray],
    ) -> ClusteringResult:
        """Cluster using KDTree query_ball_point for merge groups.

        Performs connected-component grouping on the proximity graph
        defined by vertices within ``merge_distance``.  ``O(n log n)``
        average case.
        """
        from scipy.spatial import KDTree

        tree = KDTree(vertices)

        # Find all pairs within merge distance
        neighbors = tree.query_ball_point(vertices, r=self.merge_distance)

        # Union-Find to group connected components
        parent = list(range(len(vertices)))

        def find(x: int) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]  # path compression
                x = parent[x]
            return x

        def union(a: int, b: int) -> None:
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[rb] = ra

        for i, nbrs in enumerate(neighbors):
            for j in nbrs:
                if j != i:
                    union(i, j)

        # Assign cluster labels
        root_to_label: dict[int, int] = {}
        labels = np.zeros(len(vertices), dtype=np.int64)
        for i in range(len(vertices)):
            root = find(i)
            if root not in root_to_label:
                root_to_label[root] = len(root_to_label)
            labels[i] = root_to_label[root]

        return self._finalize_clusters(vertices, labels, weights, "kdtree")

    # ------------------------------------------------------------------
    # Hierarchical clustering
    # ------------------------------------------------------------------

    def _cluster_hierarchical(
        self,
        vertices: np.ndarray,
        weights: Optional[np.ndarray],
    ) -> ClusteringResult:
        """Cluster using complete-linkage agglomerative clustering.

        Note:
            Complete linkage uses the maximum distance between points
            in two clusters as the merge criterion, so the
            ``merge_distance`` threshold passed to ``fcluster`` directly
            corresponds to the maximum Euclidean diameter of any cluster.
        """
        from scipy.cluster.hierarchy import fcluster, linkage

        Z = linkage(vertices, method="complete")
        raw_labels = fcluster(Z, t=self.merge_distance, criterion="distance")

        # fcluster labels are 1-indexed; shift to 0-indexed
        labels = (raw_labels - 1).astype(np.int64)

        return self._finalize_clusters(vertices, labels, weights, "hierarchical")

    # ------------------------------------------------------------------
    # DBSCAN clustering
    # ------------------------------------------------------------------

    def _cluster_dbscan(
        self,
        vertices: np.ndarray,
        weights: Optional[np.ndarray],
    ) -> ClusteringResult:
        """Cluster using DBSCAN with eps = merge_distance."""
        try:
            from sklearn.cluster import DBSCAN
        except ImportError:
            _log.warning(
                "sklearn not available, falling back to KDTree clustering"
            )
            return self._cluster_kdtree(vertices, weights)

        clusterer = DBSCAN(eps=self.merge_distance, min_samples=1)
        labels = clusterer.fit_predict(vertices).astype(np.int64)

        return self._finalize_clusters(vertices, labels, weights, "dbscan")

    # ------------------------------------------------------------------
    # Shared finalization
    # ------------------------------------------------------------------

    def _finalize_clusters(
        self,
        vertices: np.ndarray,
        labels: np.ndarray,
        weights: Optional[np.ndarray],
        method: str,
    ) -> ClusteringResult:
        """Compute cluster centers and build merge map from labels.

        Args:
            vertices: ``(N, 3)`` vertex positions.
            labels: ``(N,)`` cluster IDs (0-indexed).
            weights: Optional ``(N,)`` confidence weights.
            method: Name of clustering method used.

        Returns:
            ``ClusteringResult`` with centers and merge map.
        """
        if len(labels) == 0:
            return ClusteringResult(
                labels=labels,
                centers=np.empty((0, 3), dtype=np.float32),
                merge_map={},
                num_clusters=0,
                num_merged=0,
                method=method,
            )

        # Handle DBSCAN noise points (label == -1): assign each noise
        # point to the nearest valid cluster center via KDTree query.
        # If no valid clusters exist, assign all points to cluster 0.
        noise_mask = labels == -1
        if noise_mask.any():
            valid_mask = ~noise_mask
            if valid_mask.any():
                from scipy.spatial import KDTree

                # Compute centers of valid clusters first
                valid_labels = labels[valid_mask]
                unique_valid = np.unique(valid_labels)
                temp_centers = np.array(
                    [vertices[labels == c].mean(axis=0) for c in unique_valid]
                )
                tree = KDTree(temp_centers)
                noise_positions = vertices[noise_mask]
                _, nearest_idx = tree.query(noise_positions)
                labels[noise_mask] = unique_valid[nearest_idx]
            else:
                # All points are noise — assign everything to cluster 0
                labels[:] = 0

        num_clusters = int(labels.max()) + 1
        centers = np.zeros((num_clusters, 3), dtype=np.float32)

        for c in range(num_clusters):
            mask = labels == c
            if weights is not None:
                w = weights[mask]
                w_sum = w.sum()
                if w_sum > 0:
                    centers[c] = np.average(
                        vertices[mask], axis=0, weights=w
                    )
                else:
                    centers[c] = vertices[mask].mean(axis=0)
            else:
                centers[c] = vertices[mask].mean(axis=0)

        merge_map = {int(i): int(labels[i]) for i in range(len(vertices))}

        result = ClusteringResult(
            labels=labels,
            centers=centers,
            merge_map=merge_map,
            num_clusters=num_clusters,
            num_merged=len(vertices) - num_clusters,
            method=method,
        )

        _log.debug(
            "Clustered %d vertices → %d clusters (%d merged) via %s",
            len(vertices), num_clusters, result.num_merged, method,
        )
        return result


# ---------------------------------------------------------------------------
# VertexMerger
# ---------------------------------------------------------------------------


class VertexMerger:
    """Merge clustered vertices and update face indices.

    After ``VertexClusterer`` assigns cluster labels, this class
    replaces the vertex array with cluster centers and remaps face
    indices accordingly.  Degenerate faces (where merging causes two
    or more corner indices to become identical) are removed.

    Example::

        clustering = clusterer.cluster(vertices)
        merged_verts, clean_faces = VertexMerger.merge(vertices, faces, clustering)
    """

    @staticmethod
    def merge(
        vertices: np.ndarray,
        faces: np.ndarray,
        clustering: ClusteringResult,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Merge vertices by cluster assignment and remap faces.

        Args:
            vertices: Original vertices ``(N, 3)``.
            faces: Original faces ``(F, 3)`` int.
            clustering: Result from ``VertexClusterer.cluster()``.

        Returns:
            Tuple of ``(merged_vertices, remapped_faces)`` where
            ``merged_vertices`` is ``(K, 3)`` and ``remapped_faces``
            is ``(F', 3)`` with degenerate faces removed.
        """
        merged_vertices = clustering.centers.copy()

        if faces.size == 0:
            return merged_vertices, faces

        # Remap face indices through the merge map
        merge_map = clustering.merge_map
        unique_indices = np.unique(faces)
        missing = [int(idx) for idx in unique_indices if int(idx) not in merge_map]
        if missing:
            raise ValueError(
                f"Face indices {missing} not found in merge_map. "
                f"merge_map covers indices 0..{max(merge_map.keys()) if merge_map else 'N/A'}."
            )
        remapped = np.vectorize(merge_map.__getitem__)(faces)

        # Remove degenerate faces (two or more identical vertex indices)
        clean_faces = VertexMerger.remove_degenerate_faces(remapped)

        _log.debug(
            "Merged: %d → %d vertices, %d → %d faces",
            len(vertices), len(merged_vertices),
            len(faces), len(clean_faces),
        )
        return merged_vertices, clean_faces

    @staticmethod
    def remove_degenerate_faces(faces: np.ndarray) -> np.ndarray:
        """Remove faces where two or more vertex indices are identical.

        This commonly happens after vertex merging when previously
        distinct vertices collapse to the same cluster.

        Args:
            faces: ``(F, 3)`` int face indices (possibly with degenerates).

        Returns:
            ``(F', 3)`` int face indices with degenerates removed.
        """
        if faces.size == 0:
            return faces

        # A face is degenerate if any two of its three indices are equal
        v0, v1, v2 = faces[:, 0], faces[:, 1], faces[:, 2]
        non_degenerate = (v0 != v1) & (v1 != v2) & (v0 != v2)

        clean = faces[non_degenerate]

        removed = len(faces) - len(clean)
        if removed > 0:
            _log.debug("Removed %d degenerate faces after merging", removed)

        return clean
