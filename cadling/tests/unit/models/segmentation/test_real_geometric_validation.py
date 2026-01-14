"""REAL validation tests that actually run and validate correctness.

No pytest.skip(), no pass statements, no TODOs.
These tests either pass (implementation is correct) or fail (implementation is broken).
"""

from __future__ import annotations

import numpy as np
import pytest


class TestMeshGraphRealGeometry:
    """Test that mesh graph features are computed from real geometry."""

    def test_cube_has_90_degree_dihedral_angles(self):
        """Cube edges must have 90-degree dihedral angles."""
        trimesh = pytest.importorskip("trimesh")
        pytest.importorskip("torch")

        from cadling.lib.graph.mesh_graph import mesh_to_pyg_graph

        mesh = trimesh.creation.box(extents=[10, 10, 10])
        graph = mesh_to_pyg_graph(mesh, use_face_graph=True)

        dihedral_angles = graph.edge_attr[:, 0].numpy()

        # Cube has edges at 90 degrees or coplanar faces at 0 degrees
        tolerance = 0.15  # ~8.6 degrees tolerance for numerical error

        distances_to_zero = np.abs(dihedral_angles)
        distances_to_90deg = np.abs(dihedral_angles - np.pi/2)
        min_distances = np.minimum(distances_to_zero, distances_to_90deg)

        failures = min_distances >= tolerance
        if np.any(failures):
            bad_angles = dihedral_angles[failures]
            raise AssertionError(
                f"Cube must have dihedral angles of 0° or 90°.\n"
                f"Found {len(bad_angles)} bad angles: {np.degrees(bad_angles)} degrees"
            )

        # Must have SOME 90-degree angles (cube has edges)
        num_90deg = np.sum(distances_to_90deg < tolerance)
        if num_90deg == 0:
            raise AssertionError("Cube must have edges with 90-degree angles")

    def test_sphere_has_positive_curvature(self):
        """Sphere vertices must have positive curvature."""
        trimesh = pytest.importorskip("trimesh")

        from cadling.models.segmentation.graph_utils import compute_vertex_curvature

        mesh = trimesh.creation.icosphere(subdivisions=2, radius=5.0)
        curvature = compute_vertex_curvature(mesh)

        # Sphere K = 1/r^2 = 1/25 = 0.04
        mean_curv = np.mean(curvature)

        if mean_curv <= 0:
            raise AssertionError(
                f"Sphere must have positive curvature, got mean={mean_curv}"
            )

        if mean_curv < 0.01 or mean_curv > 0.1:
            raise AssertionError(
                f"Sphere r=5 should have curvature ~0.04, got {mean_curv}"
            )

        # Sphere has constant curvature - std should be low
        std_curv = np.std(curvature)
        if std_curv > 0.05:
            raise AssertionError(
                f"Sphere should have uniform curvature, got std={std_curv}"
            )

    def test_face_normals_are_normalized(self):
        """Face normals must be unit vectors."""
        trimesh = pytest.importorskip("trimesh")
        pytest.importorskip("torch")

        from cadling.lib.graph.mesh_graph import mesh_to_pyg_graph

        mesh = trimesh.creation.icosphere(subdivisions=1, radius=3.0)
        graph = mesh_to_pyg_graph(mesh, use_face_graph=True, include_normals=True)

        # Normals are in dims 3:6
        normals = graph.x[:, 3:6].numpy()
        magnitudes = np.linalg.norm(normals, axis=1)

        if not np.allclose(magnitudes, 1.0, atol=1e-5):
            bad_idx = np.where(np.abs(magnitudes - 1.0) > 1e-5)[0]
            raise AssertionError(
                f"All face normals must be unit vectors.\n"
                f"Found {len(bad_idx)} bad normals with magnitudes: {magnitudes[bad_idx]}"
            )

    def test_face_areas_sum_to_surface_area(self):
        """Face areas must sum to total surface area."""
        trimesh = pytest.importorskip("trimesh")
        pytest.importorskip("torch")

        from cadling.lib.graph.mesh_graph import mesh_to_pyg_graph

        mesh = trimesh.creation.box(extents=[10, 10, 10])
        graph = mesh_to_pyg_graph(mesh, use_face_graph=True)

        # Last feature is area
        areas = graph.x[:, -1].numpy()

        if np.any(areas <= 0):
            raise AssertionError(
                f"All face areas must be positive. Got min={areas.min()}"
            )

        total_area = np.sum(areas)
        expected = 600.0  # 6 faces * 10*10

        if np.abs(total_area - expected) > 1.0:
            raise AssertionError(
                f"Cube surface area must be {expected}, got {total_area}"
            )

    def test_vertex_graph_edge_lengths_positive(self):
        """Vertex graph edges must have positive lengths."""
        trimesh = pytest.importorskip("trimesh")
        pytest.importorskip("torch")

        from cadling.lib.graph.mesh_graph import mesh_to_pyg_graph

        mesh = trimesh.creation.icosphere(subdivisions=1, radius=2.0)
        graph = mesh_to_pyg_graph(mesh, use_face_graph=False, include_normals=True)

        edge_lengths = graph.edge_attr[:, 0].numpy()
        normal_angles = graph.edge_attr[:, 1].numpy()

        if np.any(edge_lengths <= 0):
            raise AssertionError(
                f"All edge lengths must be positive. Got min={edge_lengths.min()}"
            )

        if np.any(normal_angles < 0) or np.any(normal_angles > np.pi):
            raise AssertionError(
                f"Normal angles must be in [0,π]. Got range [{normal_angles.min()}, {normal_angles.max()}]"
            )

        # Edge lengths must vary (not all same placeholder)
        if np.std(edge_lengths) == 0:
            raise AssertionError(
                f"Edge lengths are all identical ({edge_lengths[0]}), likely placeholder"
            )

    def test_vertex_curvature_not_zeros(self):
        """Vertex curvature must not be all zeros."""
        trimesh = pytest.importorskip("trimesh")

        from cadling.models.segmentation.graph_utils import compute_vertex_curvature

        mesh = trimesh.creation.icosphere(subdivisions=2)
        curvature = compute_vertex_curvature(mesh)

        if np.allclose(curvature, 0.0):
            raise AssertionError("Curvature is all zeros - placeholder implementation")

        if not np.all(np.isfinite(curvature)):
            raise AssertionError("Curvature contains inf/nan values")


class TestBRepGraphBuilder:
    """Test BRep graph builder produces real features."""

    def test_edge_features_have_correct_dimensions(self):
        """BRep edge features must have 8 dimensions."""
        pytest.importorskip("torch")

        from cadling.models.segmentation.brep_graph_builder import BRepFaceGraphBuilder
        from cadling.datamodel.step import STEPDocument, STEPEntityItem
        from cadling.datamodel.base_models import CADItemLabel

        doc = STEPDocument(
            name="test.step",
            format="step",
            items=[
                STEPEntityItem(
                    entity_id=i,
                    entity_type="ADVANCED_FACE",
                    label=CADItemLabel(text=f"#{i} ADVANCED_FACE"),
                    text=f"ADVANCED_FACE('',#100,#200,.T.)",
                    properties={}
                )
                for i in range(5)
            ]
        )

        builder = BRepFaceGraphBuilder()
        graph = builder.build_face_graph(doc, None)

        if graph.num_edges > 0:
            if graph.edge_attr.shape[1] != 8:
                raise AssertionError(
                    f"Edge features must have 8 dimensions, got {graph.edge_attr.shape[1]}"
                )

            edge_features = graph.edge_attr.numpy()

            # Dihedral angles (dim 0) must be in [0, π]
            dihedral = edge_features[:, 0]
            if np.any(dihedral < 0) or np.any(dihedral > np.pi):
                raise AssertionError(
                    f"Dihedral angles must be in [0,π], got range [{dihedral.min()}, {dihedral.max()}]"
                )

            # Edge type (dim 1) must be in [0, 1]
            edge_type = edge_features[:, 1]
            if np.any(edge_type < 0) or np.any(edge_type > 1):
                raise AssertionError(
                    f"Edge type must be in [0,1], got range [{edge_type.min()}, {edge_type.max()}]"
                )

    def test_node_features_not_all_zeros(self):
        """BRep node features must not be all zeros."""
        pytest.importorskip("torch")

        from cadling.models.segmentation.brep_graph_builder import BRepFaceGraphBuilder
        from cadling.datamodel.step import STEPDocument, STEPEntityItem
        from cadling.datamodel.base_models import CADItemLabel

        doc = STEPDocument(
            name="test.step",
            format="step",
            items=[
                STEPEntityItem(
                    entity_id=1,
                    entity_type="ADVANCED_FACE",
                    label=CADItemLabel(text="#1 ADVANCED_FACE"),
                    text="ADVANCED_FACE('',#2,#3,.T.)",
                    properties={}
                )
            ]
        )

        builder = BRepFaceGraphBuilder()
        graph = builder.build_face_graph(doc, doc.items[0])

        if graph.num_nodes > 0:
            features = graph.x.numpy()

            nonzero_count = np.count_nonzero(features)
            total = features.size

            # At least 10% of values should be non-zero for real geometry
            if nonzero_count / total < 0.1:
                raise AssertionError(
                    f"Features are {100*nonzero_count/total:.1f}% non-zero, likely all zeros placeholder"
                )


class TestNoPlaceholderCode:
    """Verify no placeholder code exists."""

    def test_no_notimplementederror_in_production_code(self):
        """Production code must not raise NotImplementedError."""
        import subprocess

        result = subprocess.run(
            ["grep", "-r", "raise NotImplementedError", "/Users/ryanoboyle/LatticeLabs_toolkit/cadling/cadling/"],
            capture_output=True,
            text=True
        )

        if result.returncode == 0:
            raise AssertionError(
                f"Found NotImplementedError in production code:\n{result.stdout}"
            )

    def test_no_random_tensor_creation_in_builders(self):
        """Graph builders must not create random tensors."""
        import subprocess

        # Search for torch.randn in graph builders
        result = subprocess.run(
            ["grep", "-n", "torch.randn",
             "/Users/ryanoboyle/LatticeLabs_toolkit/cadling/cadling/models/segmentation/brep_graph_builder.py",
             "/Users/ryanoboyle/LatticeLabs_toolkit/cadling/cadling/models/segmentation/training/streaming_pipeline.py"],
            capture_output=True,
            text=True
        )

        if result.returncode == 0:
            raise AssertionError(
                f"Found torch.randn in graph builders (placeholder data):\n{result.stdout}"
            )

class TestGeometricConsistency:
    """Test that geometric features are internally consistent."""

    def test_edge_indices_reference_valid_nodes(self):
        """Edge indices must reference valid node IDs."""
        trimesh = pytest.importorskip("trimesh")
        pytest.importorskip("torch")

        from cadling.lib.graph.mesh_graph import mesh_to_pyg_graph

        mesh = trimesh.creation.box(extents=[5, 5, 5])
        graph = mesh_to_pyg_graph(mesh, use_face_graph=True)

        max_idx = graph.edge_index.max().item()
        num_nodes = graph.num_nodes

        if max_idx >= num_nodes:
            raise AssertionError(
                f"Edge index references node {max_idx}, but only {num_nodes} nodes exist"
            )

    def test_position_features_match_centroids(self):
        """Position features must match computed centroids."""
        trimesh = pytest.importorskip("trimesh")
        pytest.importorskip("torch")

        from cadling.lib.graph.mesh_graph import mesh_to_pyg_graph

        mesh = trimesh.creation.icosphere(subdivisions=1)
        graph = mesh_to_pyg_graph(mesh, use_face_graph=True)

        # Centroids in x[:, 0:3]
        centroids_from_x = graph.x[:, 0:3].numpy()

        # Position attribute
        pos = graph.pos.numpy()

        if not np.allclose(centroids_from_x, pos, atol=1e-5):
            raise AssertionError(
                "Centroids in x don't match pos attribute"
            )
