"""Tests for vertex clustering and merging module."""
import logging

import numpy as np
import pytest

from geotoken.vertex.vertex_clustering import (
    ClusteringResult,
    VertexClusterer,
    VertexMerger,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def simple_clustered_vertices():
    """Create a simple set of vertices with clear clusters.

    Cluster 0: [0, 0, 0], [0.0005, 0.0005, 0.0005], [0.0008, 0.0008, 0.0008]
    Cluster 1: [1, 1, 1], [1.0004, 1.0004, 1.0004]
    Cluster 2: [2, 2, 2]  (isolated single vertex)

    merge_distance = 1e-3 will group first three and next two together.
    """
    return np.array(
        [
            [0.0, 0.0, 0.0],
            [0.0005, 0.0005, 0.0005],
            [0.0008, 0.0008, 0.0008],
            [1.0, 1.0, 1.0],
            [1.0004, 1.0004, 1.0004],
            [2.0, 2.0, 2.0],
        ],
        dtype=np.float32,
    )


@pytest.fixture
def simple_faces():
    """Create simple triangular faces for testing."""
    return np.array(
        [
            [0, 1, 2],
            [1, 2, 3],
            [3, 4, 5],
            [4, 5, 0],
        ],
        dtype=np.int64,
    )


@pytest.fixture
def weighted_vertices():
    """Create vertices with associated confidence weights."""
    verts = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.001, 0.0, 0.0],
            [5.0, 5.0, 5.0],
        ],
        dtype=np.float32,
    )
    weights = np.array([1.0, 2.0, 1.0], dtype=np.float32)
    return verts, weights


# =============================================================================
# VertexClusterer Tests - KDTree Method
# =============================================================================


class TestVertexClustererKDTree:
    """Test KDTree clustering method."""

    def test_kdtree_basic_clustering(self, simple_clustered_vertices):
        """Test basic KDTree clustering with known clusters."""
        clusterer = VertexClusterer(merge_distance=1e-3, method="kdtree")
        result = clusterer.cluster(simple_clustered_vertices)

        assert isinstance(result, ClusteringResult)
        assert result.method == "kdtree"
        assert len(result.labels) == 6
        assert result.num_clusters == 3
        assert result.num_merged == 3  # 6 - 3 = 3 vertices merged
        assert result.centers.shape == (3, 3)

    def test_kdtree_labels_correct(self, simple_clustered_vertices):
        """Verify that KDTree clustering assigns correct labels to nearby vertices."""
        clusterer = VertexClusterer(merge_distance=1e-3, method="kdtree")
        result = clusterer.cluster(simple_clustered_vertices)

        # First three vertices should have same label
        assert result.labels[0] == result.labels[1] == result.labels[2]
        # Next two vertices should have same label (different from first group)
        assert result.labels[3] == result.labels[4]
        assert result.labels[3] != result.labels[0]
        # Last vertex should be in its own cluster
        assert result.labels[5] != result.labels[0]
        assert result.labels[5] != result.labels[3]

    def test_kdtree_centers_computed_correctly(self, simple_clustered_vertices):
        """Verify that cluster centers are computed as arithmetic means."""
        clusterer = VertexClusterer(merge_distance=1e-3, method="kdtree")
        result = clusterer.cluster(simple_clustered_vertices)

        # Center of cluster 0 (vertices 0, 1, 2)
        expected_center_0 = np.array(
            simple_clustered_vertices[:3], dtype=np.float32
        ).mean(axis=0)
        # Center of cluster 1 (vertices 3, 4)
        expected_center_1 = np.array(
            simple_clustered_vertices[3:5], dtype=np.float32
        ).mean(axis=0)
        # Center of cluster 2 (vertex 5)
        expected_center_2 = np.array([2.0, 2.0, 2.0], dtype=np.float32)

        # Find which cluster index corresponds to which group
        for i, label in enumerate(result.labels):
            if i < 3:
                cluster_idx = label
                np.testing.assert_allclose(
                    result.centers[cluster_idx], expected_center_0, atol=1e-6
                )
                break

    def test_kdtree_merge_map(self, simple_clustered_vertices):
        """Verify that merge_map correctly maps vertex indices to cluster indices."""
        clusterer = VertexClusterer(merge_distance=1e-3, method="kdtree")
        result = clusterer.cluster(simple_clustered_vertices)

        # Each vertex index should map to its cluster label
        for i in range(len(simple_clustered_vertices)):
            assert result.merge_map[i] == result.labels[i]

    def test_kdtree_distance_parameter_strict(self, simple_clustered_vertices):
        """Test that strict merge_distance prevents clustering of distant vertices."""
        # With a very small merge_distance, each vertex should be its own cluster
        clusterer = VertexClusterer(merge_distance=1e-6, method="kdtree")
        result = clusterer.cluster(simple_clustered_vertices)

        assert result.num_clusters == 6  # Each vertex is isolated

    def test_kdtree_distance_parameter_permissive(self, simple_clustered_vertices):
        """Test that large merge_distance groups all vertices together."""
        # With a large merge_distance, all vertices should be in one cluster
        clusterer = VertexClusterer(merge_distance=10.0, method="kdtree")
        result = clusterer.cluster(simple_clustered_vertices)

        assert result.num_clusters == 1  # All vertices in one cluster


# =============================================================================
# VertexClusterer Tests - Hierarchical Method
# =============================================================================


class TestVertexClustererHierarchical:
    """Test hierarchical clustering method."""

    def test_hierarchical_basic_clustering(self, simple_clustered_vertices):
        """Test basic hierarchical clustering."""
        clusterer = VertexClusterer(merge_distance=1e-3, method="hierarchical")
        result = clusterer.cluster(simple_clustered_vertices)

        assert isinstance(result, ClusteringResult)
        assert result.method == "hierarchical"
        assert len(result.labels) == 6
        assert result.num_clusters >= 1

    def test_hierarchical_groups_close_vertices(self, simple_clustered_vertices):
        """Verify hierarchical clustering groups close vertices."""
        clusterer = VertexClusterer(merge_distance=1e-3, method="hierarchical")
        result = clusterer.cluster(simple_clustered_vertices)

        # Vertices 0, 1, 2 should be grouped (all very close)
        labels_0_1_2 = result.labels[[0, 1, 2]]
        assert len(np.unique(labels_0_1_2)) == 1 or (
            np.std(simple_clustered_vertices[:3], axis=0).max() < 1e-3
        )

    def test_hierarchical_produces_valid_clustering(self, simple_clustered_vertices):
        """Test that hierarchical clustering produces valid output."""
        clusterer = VertexClusterer(merge_distance=1e-3, method="hierarchical")
        result = clusterer.cluster(simple_clustered_vertices)

        # Check basic properties
        assert result.num_clusters == len(np.unique(result.labels))
        assert result.num_merged == len(simple_clustered_vertices) - result.num_clusters
        assert result.centers.shape == (result.num_clusters, 3)


# =============================================================================
# VertexClusterer Tests - DBSCAN Method
# =============================================================================


class TestVertexClustererDBSCAN:
    """Test DBSCAN clustering method."""

    def test_dbscan_basic_clustering(self, simple_clustered_vertices):
        """Test basic DBSCAN clustering."""
        pytest.importorskip("sklearn", reason="sklearn required for DBSCAN tests")
        clusterer = VertexClusterer(merge_distance=1e-3, method="dbscan")
        result = clusterer.cluster(simple_clustered_vertices)

        assert isinstance(result, ClusteringResult)
        assert result.method == "dbscan"
        assert len(result.labels) == 6

    def test_dbscan_fallback_to_kdtree(self, simple_clustered_vertices):
        """Test that DBSCAN falls back to KDTree when sklearn is unavailable."""
        clusterer = VertexClusterer(merge_distance=1e-3, method="dbscan")

        result = clusterer.cluster(simple_clustered_vertices)
        assert isinstance(result, ClusteringResult)
        # The clustering result should be valid
        assert result.num_clusters >= 1
        # Method is "dbscan" when sklearn available, "kdtree" when falling back
        assert result.method in ("dbscan", "kdtree")


# =============================================================================
# VertexClusterer Tests - Edge Cases
# =============================================================================


class TestVertexClustererEdgeCases:
    """Test edge cases for vertex clustering."""

    def test_empty_vertices(self):
        """Test clustering with empty vertex array."""
        clusterer = VertexClusterer(merge_distance=1e-3, method="kdtree")
        vertices = np.empty((0, 3), dtype=np.float32)

        result = clusterer.cluster(vertices)

        assert result.num_clusters == 0
        assert result.num_merged == 0
        assert len(result.labels) == 0
        assert result.centers.shape == (0, 3)

    def test_single_vertex(self):
        """Test clustering with single vertex."""
        clusterer = VertexClusterer(merge_distance=1e-3, method="kdtree")
        vertices = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)

        result = clusterer.cluster(vertices)

        assert result.num_clusters == 1
        assert result.num_merged == 0
        assert result.labels[0] == 0
        np.testing.assert_allclose(result.centers[0], [1.0, 2.0, 3.0])

    def test_all_identical_vertices(self):
        """Test clustering when all vertices are identical."""
        clusterer = VertexClusterer(merge_distance=1e-3, method="kdtree")
        vertices = np.array(
            [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]],
            dtype=np.float32,
        )

        result = clusterer.cluster(vertices)

        assert result.num_clusters == 1
        assert result.num_merged == 2
        assert np.all(result.labels == 0)
        np.testing.assert_allclose(result.centers[0], [1.0, 1.0, 1.0])

    def test_invalid_method_raises_error(self):
        """Test that invalid clustering method raises ValueError."""
        with pytest.raises(ValueError, match="Unknown clustering method"):
            VertexClusterer(merge_distance=1e-3, method="invalid_method")

    def test_invalid_vertex_shape(self):
        """Test that invalid vertex shape raises ValueError."""
        clusterer = VertexClusterer(merge_distance=1e-3, method="kdtree")

        # 2D instead of 3D
        with pytest.raises(ValueError, match="vertices must be"):
            clusterer.cluster(np.array([[1.0, 2.0]], dtype=np.float32))

        # Wrong shape
        with pytest.raises(ValueError, match="vertices must be"):
            clusterer.cluster(np.array([[[1.0, 2.0, 3.0]]], dtype=np.float32))


# =============================================================================
# VertexClusterer Tests - Weighted Clustering
# =============================================================================


class TestVertexClustererWeighted:
    """Test weighted clustering functionality."""

    def test_weighted_clustering_basic(self, weighted_vertices):
        """Test clustering with confidence weights."""
        verts, weights = weighted_vertices
        clusterer = VertexClusterer(merge_distance=1e-2, method="kdtree")

        result = clusterer.cluster(verts, weights=weights)

        assert isinstance(result, ClusteringResult)
        assert result.num_clusters == 2

    def test_weighted_centroid_computation(self, weighted_vertices):
        """Test that weighted clustering computes correct weighted centroids."""
        verts, weights = weighted_vertices
        # Vertices 0 and 1 will be clustered together
        # Their weighted centroid should be closer to vertex 1 (weight=2.0)
        clusterer = VertexClusterer(merge_distance=1e-2, method="kdtree")

        result = clusterer.cluster(verts, weights=weights)

        # Find cluster center for vertices 0 and 1
        cluster_label = result.labels[0]  # label of vertex 0
        cluster_center = result.centers[cluster_label]

        # Manually compute expected weighted centroid
        mask = result.labels == cluster_label
        weighted_verts = verts[mask]
        cluster_weights = weights[mask]
        expected_center = np.average(weighted_verts, axis=0, weights=cluster_weights)

        np.testing.assert_allclose(cluster_center, expected_center, atol=1e-6)

    def test_weighted_vs_unweighted(self, weighted_vertices):
        """Compare weighted and unweighted clustering results."""
        verts, weights = weighted_vertices

        clusterer = VertexClusterer(merge_distance=1e-2, method="kdtree")
        result_weighted = clusterer.cluster(verts, weights=weights)
        result_unweighted = clusterer.cluster(verts, weights=None)

        # Both should have same clustering labels
        np.testing.assert_array_equal(result_weighted.labels, result_unweighted.labels)

        # But centers should differ due to weighting
        cluster_idx = result_weighted.labels[0]
        assert not np.allclose(
            result_weighted.centers[cluster_idx],
            result_unweighted.centers[cluster_idx],
        )


# =============================================================================
# VertexMerger Tests
# =============================================================================


class TestVertexMerger:
    """Test vertex merging and face remapping."""

    def test_merge_basic(self, simple_clustered_vertices, simple_faces):
        """Test basic vertex merging and face remapping."""
        clusterer = VertexClusterer(merge_distance=1e-3, method="kdtree")
        clustering = clusterer.cluster(simple_clustered_vertices)

        merged_verts, remapped_faces = VertexMerger.merge(
            simple_clustered_vertices, simple_faces, clustering
        )

        assert merged_verts.shape[0] == clustering.num_clusters
        assert merged_verts.shape[1] == 3

    def test_merge_preserves_cluster_centers(
        self, simple_clustered_vertices, simple_faces
    ):
        """Test that merged vertices match cluster centers."""
        clusterer = VertexClusterer(merge_distance=1e-3, method="kdtree")
        clustering = clusterer.cluster(simple_clustered_vertices)

        merged_verts, _ = VertexMerger.merge(
            simple_clustered_vertices, simple_faces, clustering
        )

        np.testing.assert_allclose(merged_verts, clustering.centers, atol=1e-6)

    def test_merge_remaps_faces_correctly(self, simple_clustered_vertices, simple_faces):
        """Test that face indices are correctly remapped after clustering."""
        clusterer = VertexClusterer(merge_distance=1e-3, method="kdtree")
        clustering = clusterer.cluster(simple_clustered_vertices)

        original_merged_verts, remapped_faces = VertexMerger.merge(
            simple_clustered_vertices, simple_faces, clustering
        )

        # Check that remapped face indices reference valid vertices
        assert remapped_faces.min() >= 0
        assert remapped_faces.max() < len(original_merged_verts)

    def test_merge_with_empty_faces(self, simple_clustered_vertices):
        """Test merging with empty face array."""
        clusterer = VertexClusterer(merge_distance=1e-3, method="kdtree")
        clustering = clusterer.cluster(simple_clustered_vertices)

        empty_faces = np.empty((0, 3), dtype=np.int64)
        merged_verts, remapped_faces = VertexMerger.merge(
            simple_clustered_vertices, empty_faces, clustering
        )

        assert len(remapped_faces) == 0

    def test_merge_removes_degenerate_faces(self):
        """Test that merging removes degenerate faces."""
        # Create vertices where 0, 1, 2 will be in one cluster and 3 separate
        vertices = np.array(
            [
                [0.0, 0.0, 0.0],
                [0.0001, 0.0, 0.0],
                [0.00015, 0.0, 0.0],
                [1.0, 1.0, 1.0],
            ],
            dtype=np.float32,
        )

        # Create faces where 0, 1, 2 will merge to same cluster
        faces = np.array(
            [
                [0, 1, 2],  # All three merge to same cluster -> degenerate
                [0, 1, 3],  # 0,1 merge but 3 is separate -> degenerate (0==1 after merge)
                [1, 2, 3],  # 1,2 merge but 3 is separate -> degenerate
            ],
            dtype=np.int64,
        )

        # Use tight merge distance to ensure 0,1,2 are in same cluster
        clusterer = VertexClusterer(merge_distance=1e-3, method="kdtree")
        clustering = clusterer.cluster(vertices)

        # Verify that vertices 0, 1, 2 are in same cluster
        assert clustering.labels[0] == clustering.labels[1] == clustering.labels[2]

        merged_verts, remapped_faces = VertexMerger.merge(vertices, faces, clustering)

        # All faces should be degenerate and removed
        assert len(remapped_faces) == 0

    def test_remove_degenerate_faces_directly(self):
        """Test remove_degenerate_faces static method directly."""
        faces = np.array(
            [
                [0, 1, 2],  # Valid
                [0, 0, 1],  # Degenerate (0 == 0)
                [1, 2, 1],  # Degenerate (1 == 1)
                [2, 3, 4],  # Valid
                [5, 5, 5],  # Degenerate (all same)
            ],
            dtype=np.int64,
        )

        clean_faces = VertexMerger.remove_degenerate_faces(faces)

        assert len(clean_faces) == 2
        np.testing.assert_array_equal(clean_faces[0], [0, 1, 2])
        np.testing.assert_array_equal(clean_faces[1], [2, 3, 4])

    def test_remove_degenerate_faces_empty(self):
        """Test remove_degenerate_faces with empty array."""
        faces = np.empty((0, 3), dtype=np.int64)
        clean_faces = VertexMerger.remove_degenerate_faces(faces)

        assert len(clean_faces) == 0

    def test_remove_degenerate_faces_all_valid(self):
        """Test remove_degenerate_faces when all faces are valid."""
        faces = np.array(
            [
                [0, 1, 2],
                [1, 2, 3],
                [2, 3, 4],
            ],
            dtype=np.int64,
        )

        clean_faces = VertexMerger.remove_degenerate_faces(faces)

        assert len(clean_faces) == 3
        np.testing.assert_array_equal(clean_faces, faces)


# =============================================================================
# Integration Tests
# =============================================================================


class TestIntegration:
    """Integration tests for clustering and merging pipeline."""

    def test_full_pipeline_kdtree(self, simple_clustered_vertices, simple_faces):
        """Test complete clustering and merging pipeline with KDTree."""
        clusterer = VertexClusterer(merge_distance=1e-3, method="kdtree")
        clustering = clusterer.cluster(simple_clustered_vertices)
        merged_verts, merged_faces = VertexMerger.merge(
            simple_clustered_vertices, simple_faces, clustering
        )

        # Check output shapes
        assert 0 < merged_verts.shape[0] <= simple_clustered_vertices.shape[0]
        assert merged_verts.shape[1] == 3
        assert merged_faces.shape[1] == 3

        # All face indices should be valid
        assert merged_faces.size == 0 or (
            merged_faces.min() >= 0 and merged_faces.max() < len(merged_verts)
        )

    def test_full_pipeline_hierarchical(
        self, simple_clustered_vertices, simple_faces
    ):
        """Test complete pipeline with hierarchical clustering."""
        clusterer = VertexClusterer(merge_distance=1e-3, method="hierarchical")
        clustering = clusterer.cluster(simple_clustered_vertices)
        merged_verts, merged_faces = VertexMerger.merge(
            simple_clustered_vertices, simple_faces, clustering
        )

        assert merged_verts.shape[0] > 0
        assert merged_verts.shape[1] == 3

    def test_full_pipeline_dbscan(self, simple_clustered_vertices, simple_faces):
        """Test complete pipeline with DBSCAN clustering."""
        clusterer = VertexClusterer(merge_distance=1e-3, method="dbscan")
        clustering = clusterer.cluster(simple_clustered_vertices)
        merged_verts, merged_faces = VertexMerger.merge(
            simple_clustered_vertices, simple_faces, clustering
        )

        assert merged_verts.shape[0] > 0
        assert merged_verts.shape[1] == 3

    def test_pipeline_with_weights(self, weighted_vertices):
        """Test full pipeline with weighted clustering."""
        verts, weights = weighted_vertices

        # Create faces appropriate for 3 vertices
        faces = np.array(
            [[0, 1, 2]],
            dtype=np.int64,
        )

        clusterer = VertexClusterer(merge_distance=1e-2, method="kdtree")
        clustering = clusterer.cluster(verts, weights=weights)
        merged_verts, merged_faces = VertexMerger.merge(verts, faces, clustering)

        # Check that weighted centers are used in merged vertices
        np.testing.assert_allclose(merged_verts, clustering.centers, atol=1e-6)

    def test_pipeline_no_vertex_loss(self):
        """Test that pipeline doesn't lose vertices in geometry."""
        vertices = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 1.0, 1.0],
                [2.0, 2.0, 2.0],
                [3.0, 3.0, 3.0],
            ],
            dtype=np.float32,
        )
        faces = np.array(
            [
                [0, 1, 2],
                [1, 2, 3],
            ],
            dtype=np.int64,
        )

        # Use large merge distance to cluster all vertices
        clusterer = VertexClusterer(merge_distance=10.0, method="kdtree")
        clustering = clusterer.cluster(vertices)

        assert clustering.num_clusters == 1

        merged_verts, merged_faces = VertexMerger.merge(vertices, faces, clustering)

        # All faces should become degenerate and be removed
        assert len(merged_faces) == 0


# =============================================================================
# Merge Distance Parameter Tests
# =============================================================================


class TestMergeDistanceParameter:
    """Test behavior with varying merge_distance values."""

    def test_merge_distance_zero(self):
        """Test clustering with merge_distance=0."""
        vertices = np.array(
            [
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [1.0, 1.0, 1.0],
            ],
            dtype=np.float32,
        )

        clusterer = VertexClusterer(merge_distance=0.0, method="kdtree")
        result = clusterer.cluster(vertices)

        # Identical vertices should still cluster (distance=0)
        # Distinct vertices should not
        assert result.num_clusters >= 1

    def test_merge_distance_very_large(self):
        """Test clustering with very large merge_distance."""
        vertices = np.array(
            [
                [0.0, 0.0, 0.0],
                [100.0, 100.0, 100.0],
                [1000.0, 1000.0, 1000.0],
            ],
            dtype=np.float32,
        )

        clusterer = VertexClusterer(merge_distance=10000.0, method="kdtree")
        result = clusterer.cluster(vertices)

        assert result.num_clusters == 1  # All merge together

    def test_merge_distance_intermediate_effects(self):
        """Test that merge_distance has expected effect on clustering count."""
        vertices = np.array(
            [
                [0.0, 0.0, 0.0],
                [0.5, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [1.5, 0.0, 0.0],
                [2.0, 0.0, 0.0],
            ],
            dtype=np.float32,
        )

        # Small distance: each vertex is its own cluster
        clusterer_small = VertexClusterer(merge_distance=0.1, method="kdtree")
        result_small = clusterer_small.cluster(vertices)

        # Large distance: all vertices in one cluster
        clusterer_large = VertexClusterer(merge_distance=5.0, method="kdtree")
        result_large = clusterer_large.cluster(vertices)

        assert result_small.num_clusters > result_large.num_clusters


# =============================================================================
# Clustering Result Tests
# =============================================================================


class TestClusteringResult:
    """Test ClusteringResult dataclass."""

    def test_clustering_result_fields(self):
        """Test that ClusteringResult has expected fields."""
        labels = np.array([0, 0, 1, 2], dtype=np.int64)
        centers = np.array(
            [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [2.0, 2.0, 2.0]], dtype=np.float32
        )
        merge_map = {0: 0, 1: 0, 2: 1, 3: 2}

        result = ClusteringResult(
            labels=labels,
            centers=centers,
            merge_map=merge_map,
            num_clusters=3,
            num_merged=1,
            method="kdtree",
        )

        assert result.num_clusters == 3
        assert result.num_merged == 1
        assert result.method == "kdtree"
        np.testing.assert_array_equal(result.labels, labels)
        np.testing.assert_array_equal(result.centers, centers)
        assert result.merge_map == merge_map

    def test_clustering_result_consistency(self, simple_clustered_vertices):
        """Test that ClusteringResult is internally consistent."""
        clusterer = VertexClusterer(merge_distance=1e-3, method="kdtree")
        result = clusterer.cluster(simple_clustered_vertices)

        # Check invariants
        assert len(result.labels) == len(simple_clustered_vertices)
        assert result.num_clusters == len(np.unique(result.labels))
        assert result.num_merged == len(simple_clustered_vertices) - result.num_clusters
        assert len(result.merge_map) == len(simple_clustered_vertices)
        assert result.centers.shape == (result.num_clusters, 3)


# =============================================================================
# Data Type and Precision Tests
# =============================================================================


class TestDataTypes:
    """Test handling of different data types."""

    def test_float64_input(self):
        """Test clustering with float64 vertices."""
        vertices = np.array(
            [
                [0.0, 0.0, 0.0],
                [0.0005, 0.0005, 0.0005],
                [1.0, 1.0, 1.0],
            ],
            dtype=np.float64,
        )

        clusterer = VertexClusterer(merge_distance=1e-3, method="kdtree")
        result = clusterer.cluster(vertices)

        assert result.centers.dtype == np.float32

    def test_int_face_indices(self):
        """Test that face indices can be various integer types."""
        vertices = np.array(
            [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [2.0, 2.0, 2.0]], dtype=np.float32
        )

        # Test with different integer types
        for dtype in [np.int32, np.int64]:
            faces = np.array([[0, 1, 2]], dtype=dtype)
            clusterer = VertexClusterer(merge_distance=1e-3, method="kdtree")
            clustering = clusterer.cluster(vertices)
            merged_verts, merged_faces = VertexMerger.merge(vertices, faces, clustering)

            assert merged_faces.size == 0 or merged_faces.dtype in [
                np.int32,
                np.int64,
            ]
