"""Regression tests for B-Rep edge reconstruction producing real geometry.

Guards the fix where ``GraphReconstructor._reconstruct_edge`` previously decoded
edge parameters but returned a primitive with **no OCC shape** ("For now,
return without OCC shape"). It now builds a real OCC edge between the centroids
of the two faces the edge connects (recovered from the edge index + per-face
centroids), and honestly returns no shape when those endpoints are unavailable.
"""

from __future__ import annotations

import numpy as np
import pytest


def _reconstructor():
    from cadling.generation.reconstruction.graph_reconstructor import (
        GraphReconstructor,
    )

    return GraphReconstructor()


def _node_features():
    # 3 faces with distinct centroids stored at dims [16:19].
    nf = np.zeros((3, 20), dtype=np.float32)
    nf[0, 16:19] = [0.0, 0.0, 0.0]
    nf[1, 16:19] = [10.0, 0.0, 0.0]
    nf[2, 16:19] = [0.0, 5.0, 0.0]
    return nf


def _edge_features():
    ef = np.zeros((2, 12), dtype=np.float32)
    ef[:, 0] = 1.0  # curve-type one-hot -> LINE
    ef[:, 6] = 10.0  # length
    return ef


@pytest.mark.requires_pythonocc
def test_edge_reconstruction_builds_real_occ_geometry():
    import cadling.generation.reconstruction.graph_reconstructor as gr

    if not gr.HAS_OCC:
        pytest.skip("pythonocc not available")

    rec = _reconstructor()
    node_features = _node_features()
    edge_index = np.array([[0, 0], [1, 2]])  # edges (0,1) and (0,2)
    edge_features = _edge_features()

    prim = rec._reconstruct_edge(0, edge_features[0], edge_index, 0, node_features)

    assert prim.occ_shape is not None
    assert prim.success
    assert prim.parameters["start_point"] == [0.0, 0.0, 0.0]
    assert prim.parameters["end_point"] == [10.0, 0.0, 0.0]


def test_edge_reconstruction_no_endpoints_returns_no_shape():
    """Without node features the endpoints are unrecoverable -> honest no-shape
    (not a fabricated primitive)."""
    rec = _reconstructor()
    edge_index = np.array([[0], [1]])
    edge_features = _edge_features()

    prim = rec._reconstruct_edge(0, edge_features[0], edge_index, 0, None)

    assert prim.occ_shape is None
    assert not prim.success
    # Parameters are still decoded (curve type, length, etc.).
    assert prim.parameters["curve_type"] in {"LINE", "OTHER"}
