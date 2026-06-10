"""Tests for STEP -> coedge-graph + geometry extraction (on real fixtures)."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("OCC")
pytest.importorskip("occwl")

from ll_brepnet.pipelines.extract_brepnet_data_from_step import (  # noqa: E402
    NUM_EDGE_FEATURES,
    NUM_FACE_FEATURES,
    BRepDataExtractor,
    load_step_shape,
    scale_shape_to_unit_box,
)


@pytest.mark.requires_pythonocc
def test_extract_arrays_shapes_and_topology(one_step_fixture):
    arrays = BRepDataExtractor(one_step_fixture).extract_arrays()
    f = int(arrays["num_faces"])
    e = int(arrays["num_edges"])
    c = int(arrays["num_coedges"])
    assert f > 0 and e > 0 and c > 0

    assert arrays["face_features"].shape == (f, NUM_FACE_FEATURES)
    assert arrays["face_point_grids"].shape == (f, 7, 10, 10)
    assert arrays["edge_features"].shape == (e, NUM_EDGE_FEATURES)
    assert arrays["edge_point_grids"].shape == (e, 6, 10)
    for key in (
        "coedge_to_next",
        "coedge_to_prev",
        "coedge_to_mate",
        "coedge_to_face",
        "coedge_to_edge",
    ):
        assert arrays[key].shape == (c,)


@pytest.mark.requires_pythonocc
def test_mate_is_an_involution(one_step_fixture):
    arrays = BRepDataExtractor(one_step_fixture).extract_arrays()
    mate = arrays["coedge_to_mate"]
    c = int(arrays["num_coedges"])
    # mate(mate(c)) == c for every coedge (boundary coedges are their own mate).
    assert np.array_equal(mate[mate], np.arange(c))


@pytest.mark.requires_pythonocc
def test_incidence_indices_in_range(one_step_fixture):
    arrays = BRepDataExtractor(one_step_fixture).extract_arrays()
    f, e = int(arrays["num_faces"]), int(arrays["num_edges"])
    assert arrays["coedge_to_face"].min() >= 0 and arrays["coedge_to_face"].max() < f
    assert arrays["coedge_to_edge"].min() >= 0 and arrays["coedge_to_edge"].max() < e
    # Every face is referenced by at least one coedge.
    assert set(arrays["coedge_to_face"].tolist()) == set(range(f))


@pytest.mark.requires_pythonocc
def test_face_onehot_and_grids_have_material(one_step_fixture):
    arrays = BRepDataExtractor(one_step_fixture).extract_arrays()
    # Surface-type one-hot rows sum to exactly 1.
    onehot = arrays["face_features"][:, :7]
    assert np.allclose(onehot.sum(axis=1), 1.0)
    # At least one face sampled real material (non-empty trimming mask).
    trim = arrays["face_point_grids"][:, 6]
    assert (trim.sum(axis=(1, 2)) > 0).any()


@pytest.mark.requires_pythonocc
def test_scale_to_unit_box(one_step_fixture):
    from OCC.Core.Bnd import Bnd_Box
    from OCC.Core.BRepBndLib import brepbndlib

    shape = scale_shape_to_unit_box(load_step_shape(one_step_fixture))
    box = Bnd_Box()
    brepbndlib.Add(shape, box)
    xmin, ymin, zmin, xmax, ymax, zmax = box.Get()
    half = max(xmax - xmin, ymax - ymin, zmax - zmin) / 2.0
    # Largest half-extent normalised to ~1.
    assert half == pytest.approx(1.0, abs=1e-3)
