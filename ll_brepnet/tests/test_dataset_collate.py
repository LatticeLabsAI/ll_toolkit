"""Tests for the dataset, offset-aware collation and the face-count sampler."""

from __future__ import annotations

import pytest

pytest.importorskip("torch")
pytest.importorskip("OCC")

import torch  # noqa: E402

from ll_brepnet.dataloaders.brep_dataset import (  # noqa: E402
    IGNORE_INDEX,
    BRepDataset,
    brep_collate_fn,
)
from ll_brepnet.dataloaders.max_num_faces_loader import MaxNumFacesSampler  # noqa: E402


def _load_all(prepared_dataset):
    npz_dir, manifest = prepared_dataset
    ds = BRepDataset(manifest, npz_dir, "training_set", label_dir=npz_dir)
    if len(ds) == 0:
        pytest.skip("empty training split")
    return ds, [ds[i] for i in range(len(ds))]


@pytest.mark.requires_pythonocc
def test_collate_offsets_and_mate_survive(prepared_dataset):
    ds, samples = _load_all(prepared_dataset)
    batch = brep_collate_fn(samples)
    f, e, c = batch.num_faces, int(batch.edge_features.shape[0]), batch.num_coedges

    # Concatenated counts equal the sum over solids.
    assert f == sum(int(s["face_features"].shape[0]) for s in samples)
    assert c == sum(int(s["coedge_to_next"].shape[0]) for s in samples)

    # Offsets keep every index valid in the merged graph.
    assert int(batch.coedge_to_face.max()) < f
    assert int(batch.coedge_to_edge.max()) < e
    assert int(batch.coedge_to_next.max()) < c

    # Mate involution survives the offsetting.
    mate = batch.coedge_to_mate
    assert torch.equal(mate[mate], torch.arange(c))


@pytest.mark.requires_pythonocc
def test_split_batch_recovers_per_solid_labels(prepared_dataset):
    ds, samples = _load_all(prepared_dataset)
    batch = brep_collate_fn(samples)
    assert len(batch.split_batch) == len(samples)
    for si, (start, end) in enumerate(batch.split_batch):
        assert torch.equal(batch.labels[start:end], samples[si]["labels"])


@pytest.mark.requires_pythonocc
def test_standardization_applied(prepared_dataset):
    npz_dir, manifest = prepared_dataset
    ds_std = BRepDataset(manifest, npz_dir, "training_set", label_dir=npz_dir, standardize=True)
    ds_raw = BRepDataset(manifest, npz_dir, "training_set", label_dir=npz_dir, standardize=False)
    if len(ds_std) == 0:
        pytest.skip("empty training split")
    # The area column (index 7) is continuous, so standardization changes it.
    std_area = ds_std[0]["face_features"][:, 7]
    raw_area = ds_raw[0]["face_features"][:, 7]
    assert not torch.allclose(std_area, raw_area)


@pytest.mark.requires_pythonocc
def test_missing_labels_are_ignore_index(prepared_dataset):
    npz_dir, manifest = prepared_dataset
    # A split with no .seg files present -> all labels IGNORE_INDEX.
    ds = BRepDataset(manifest, npz_dir, "training_set", label_dir="/nonexistent")
    if len(ds) == 0:
        pytest.skip("empty training split")
    assert int((ds[0]["labels"] != IGNORE_INDEX).sum()) == 0


@pytest.mark.requires_pythonocc
def test_max_num_faces_sampler_packs_within_cap(prepared_dataset):
    ds, _ = _load_all(prepared_dataset)
    cap = max(ds.get_num_faces(i) for i in range(len(ds)))  # at least the largest solid
    sampler = MaxNumFacesSampler(ds, max_num_faces_per_batch=cap, shuffle=False)
    batches = list(sampler)
    assert len(batches) == len(sampler)
    for batch in batches:
        total = sum(ds.get_num_faces(i) for i in batch)
        assert total <= cap
    # Every (within-cap) solid appears exactly once across all batches.
    assert sorted(i for b in batches for i in b) == list(range(len(ds)))
