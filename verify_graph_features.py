#!/usr/bin/env python
"""Verification script for graph feature determinism and dimensions.

This script verifies:
1. Graph features are deterministic (same input -> same output)
2. Feature dimensions are correct
   - Mesh: node features [N, 7], edge features [E, 2]
   - B-Rep: node features [N, 24], edge features [E, 8]
"""

import numpy as np
import torch
import trimesh

from cadling.lib.graph import mesh_to_pyg_graph, brep_to_pyg_graph

def verify_mesh_graph_determinism():
    """Verify mesh graph features are deterministic."""
    print("=" * 70)
    print("VERIFYING MESH GRAPH DETERMINISM")
    print("=" * 70)

    # Create a simple test mesh (cube)
    vertices = np.array([
        [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
        [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]
    ], dtype=np.float32)

    faces = np.array([
        [0, 1, 2], [0, 2, 3],  # Bottom
        [4, 5, 6], [4, 6, 7],  # Top
        [0, 1, 5], [0, 5, 4],  # Front
        [2, 3, 7], [2, 7, 6],  # Back
        [0, 3, 7], [0, 7, 4],  # Left
        [1, 2, 6], [1, 6, 5],  # Right
    ], dtype=np.int32)

    mesh = trimesh.Trimesh(vertices=vertices, faces=faces)

    # Build graph twice
    graph1 = mesh_to_pyg_graph(mesh, use_face_graph=True)
    graph2 = mesh_to_pyg_graph(mesh, use_face_graph=True)

    # Verify determinism
    print(f"\nGraph 1 - Nodes: {graph1.num_nodes}, Edges: {graph1.num_edges}")
    print(f"Graph 2 - Nodes: {graph2.num_nodes}, Edges: {graph2.num_edges}")

    node_features_match = torch.allclose(graph1.x, graph2.x, atol=1e-6)
    edge_index_match = torch.equal(graph1.edge_index, graph2.edge_index)
    edge_attr_match = torch.allclose(graph1.edge_attr, graph2.edge_attr, atol=1e-6)

    print(f"\n✓ Node features match: {node_features_match}")
    print(f"✓ Edge index match: {edge_index_match}")
    print(f"✓ Edge attributes match: {edge_attr_match}")

    # Verify dimensions
    print(f"\n✓ Node feature dimensions: {graph1.x.shape} (expected [N, 7])")
    print(f"✓ Edge feature dimensions: {graph1.edge_attr.shape} (expected [E, 2])")

    assert node_features_match, "❌ Node features are NOT deterministic!"
    assert edge_index_match, "❌ Edge index is NOT deterministic!"
    assert edge_attr_match, "❌ Edge attributes are NOT deterministic!"
    assert graph1.x.shape[1] == 7, f"❌ Node features should be [N, 7], got {graph1.x.shape}"
    assert graph1.edge_attr.shape[1] == 2, f"❌ Edge features should be [E, 2], got {graph1.edge_attr.shape}"

    print("\n✅ MESH GRAPH DETERMINISM VERIFIED!")
    return True


def verify_brep_graph_determinism():
    """Verify B-Rep graph features are deterministic."""
    print("\n" + "=" * 70)
    print("VERIFYING B-REP GRAPH DETERMINISM")
    print("=" * 70)

    # Create mock STEP entities for testing
    # This simulates what STEPParser.parse() would return
    entities = {
        100: {
            "type": "ADVANCED_FACE",
            "surface_type": "PLANE",
            "attributes": {
                "area": 10.0,
                "centroid": [0.0, 0.0, 5.0],
                "normal": [0.0, 0.0, 1.0]
            }
        },
        101: {
            "type": "ADVANCED_FACE",
            "surface_type": "CYLINDRICAL_SURFACE",
            "attributes": {
                "area": 15.0,
                "centroid": [5.0, 0.0, 0.0],
                "normal": [1.0, 0.0, 0.0]
            }
        },
        102: {
            "type": "ADVANCED_FACE",
            "surface_type": "PLANE",
            "attributes": {
                "area": 10.0,
                "centroid": [0.0, 5.0, 0.0],
                "normal": [0.0, 1.0, 0.0]
            }
        },
        200: {
            "type": "EDGE_CURVE",
            "references": [100, 101]
        },
        201: {
            "type": "EDGE_CURVE",
            "references": [101, 102]
        }
    }

    face_labels = np.array([0, 1, 2], dtype=np.int32)

    # Build graph twice
    graph1 = brep_to_pyg_graph(entities, face_labels=face_labels)
    graph2 = brep_to_pyg_graph(entities, face_labels=face_labels)

    # Verify determinism
    print(f"\nGraph 1 - Nodes: {graph1.num_nodes}, Edges: {graph1.num_edges}")
    print(f"Graph 2 - Nodes: {graph2.num_nodes}, Edges: {graph2.num_edges}")

    if graph1.num_nodes > 0:
        node_features_match = torch.allclose(graph1.x, graph2.x, atol=1e-6)
        print(f"\n✓ Node features match: {node_features_match}")

        # Verify dimensions
        print(f"✓ Node feature dimensions: {graph1.x.shape} (expected [N, 24])")

        assert node_features_match, "❌ Node features are NOT deterministic!"
        assert graph1.x.shape[1] == 24, f"❌ Node features should be [N, 24], got {graph1.x.shape}"

    if graph1.num_edges > 0:
        edge_index_match = torch.equal(graph1.edge_index, graph2.edge_index)
        edge_attr_match = torch.allclose(graph1.edge_attr, graph2.edge_attr, atol=1e-6)

        print(f"✓ Edge index match: {edge_index_match}")
        print(f"✓ Edge attributes match: {edge_attr_match}")
        print(f"✓ Edge feature dimensions: {graph1.edge_attr.shape} (expected [E, 8])")

        assert edge_index_match, "❌ Edge index is NOT deterministic!"
        assert edge_attr_match, "❌ Edge attributes are NOT deterministic!"
        assert graph1.edge_attr.shape[1] == 8, f"❌ Edge features should be [E, 8], got {graph1.edge_attr.shape}"

    if graph1.y is not None:
        labels_match = torch.equal(graph1.y, graph2.y)
        print(f"✓ Labels match: {labels_match}")
        assert labels_match, "❌ Labels are NOT deterministic!"

    print("\n✅ B-REP GRAPH DETERMINISM VERIFIED!")
    return True


def verify_feature_values_not_random():
    """Verify that feature values are NOT random (i.e., not all different)."""
    print("\n" + "=" * 70)
    print("VERIFYING FEATURES ARE NOT RANDOM")
    print("=" * 70)

    # Create test mesh
    vertices = np.array([
        [0, 0, 0], [1, 0, 0], [0, 1, 0]
    ], dtype=np.float32)

    faces = np.array([[0, 1, 2]], dtype=np.int32)

    mesh = trimesh.Trimesh(vertices=vertices, faces=faces)

    # Build graphs multiple times
    graphs = [mesh_to_pyg_graph(mesh, use_face_graph=True) for _ in range(5)]

    # If features were random, they would be different each time
    all_match = all(
        torch.allclose(graphs[0].x, g.x, atol=1e-6) for g in graphs[1:]
    )

    print(f"\n✓ All 5 graphs have identical features: {all_match}")
    assert all_match, "❌ Features are random (different each time)!"

    # Verify features are not all zeros (real computation)
    has_nonzero = torch.any(graphs[0].x != 0.0).item()
    print(f"✓ Features contain non-zero values: {has_nonzero}")
    assert has_nonzero, "❌ Features are all zeros!"

    print("\n✅ FEATURES ARE REAL (NOT RANDOM)!")
    return True


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("GRAPH FEATURE VERIFICATION")
    print("=" * 70)

    try:
        # Run all verifications
        verify_mesh_graph_determinism()
        verify_brep_graph_determinism()
        verify_feature_values_not_random()

        print("\n" + "=" * 70)
        print("✅ ALL VERIFICATIONS PASSED!")
        print("=" * 70)
        print("\nSummary:")
        print("  ✓ Mesh graphs are deterministic")
        print("  ✓ B-Rep graphs are deterministic")
        print("  ✓ Features are real (not random)")
        print("  ✓ Mesh dimensions correct: node [N, 7], edge [E, 2]")
        print("  ✓ B-Rep dimensions correct: node [N, 24], edge [E, 8]")
        print("\n")

    except Exception as e:
        print(f"\n❌ VERIFICATION FAILED: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
