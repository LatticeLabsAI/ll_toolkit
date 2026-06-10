"""Tests for the JSON topology front-end (no pythonocc required)."""

from __future__ import annotations

import numpy as np
import pytest

from ll_brepnet.pipelines.extract_brepnet_data_from_json import BRepJsonExtractor


def _valid_topology():
    # Two faces, one shared edge -> two coedges (mates of each other).
    return {
        "coedge_to_next": [0, 1],
        "coedge_to_prev": [0, 1],
        "coedge_to_mate": [1, 0],
        "coedge_to_face": [0, 1],
        "coedge_to_edge": [0, 0],
        "face_features": [[1.0] * 8, [0.0] * 8],
        "edge_features": [[0.5] * 7],
    }


def test_json_extractor_builds_arrays():
    arrays = BRepJsonExtractor(_valid_topology()).extract_arrays()
    assert int(arrays["num_faces"]) == 2
    assert int(arrays["num_edges"]) == 1
    assert int(arrays["num_coedges"]) == 2
    # Grids default to zeros with the right shape when absent.
    assert arrays["face_point_grids"].shape == (2, 7, 10, 10)
    assert arrays["edge_point_grids"].shape == (1, 6, 10)
    assert np.array_equal(arrays["coedge_to_mate"], np.array([1, 0]))


def test_json_extractor_missing_key_raises():
    topo = _valid_topology()
    del topo["face_features"]
    with pytest.raises(ValueError, match="face_features"):
        BRepJsonExtractor(topo).extract_arrays()


def test_json_extractor_out_of_range_index_raises():
    topo = _valid_topology()
    topo["coedge_to_face"] = [0, 5]  # only 2 faces exist
    with pytest.raises(ValueError, match="coedge_to_face"):
        BRepJsonExtractor(topo).extract_arrays()


def test_json_extractor_process_writes_npz(tmp_path):
    out = tmp_path / "solid.npz"
    BRepJsonExtractor(_valid_topology()).process(out)
    assert out.exists()
    with np.load(out) as data:
        assert int(data["num_coedges"]) == 2
