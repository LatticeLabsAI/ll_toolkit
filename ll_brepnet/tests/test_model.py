"""Tests for the LLBRepNet model: forward shape, finiteness, learning."""

from __future__ import annotations

import pytest

pytest.importorskip("torch")
pytest.importorskip("OCC")

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from ll_brepnet.dataloaders.brep_dataset import BRepDataset, brep_collate_fn  # noqa: E402
from ll_brepnet.models.ll_brepnet import LLBRepNet  # noqa: E402
from ll_brepnet.models.uvnet_encoders import UVNetCurveEncoder, UVNetSurfaceEncoder  # noqa: E402


def _batch(prepared_dataset):
    npz_dir, manifest = prepared_dataset
    ds = BRepDataset(manifest, npz_dir, "training_set", label_dir=npz_dir)
    if len(ds) == 0:
        pytest.skip("empty training split")
    return brep_collate_fn([ds[i] for i in range(len(ds))])


def test_uvnet_encoder_shapes():
    surf = UVNetSurfaceEncoder(in_channels=7, out_dim=16)
    curve = UVNetCurveEncoder(in_channels=6, out_dim=16)
    assert surf(torch.randn(5, 7, 10, 10)).shape == (5, 16)
    assert curve(torch.randn(5, 6, 10)).shape == (5, 16)
    # Empty inputs are handled.
    assert surf(torch.zeros(0, 7, 10, 10)).shape == (0, 16)


@pytest.mark.requires_pythonocc
def test_forward_shape_and_finite(prepared_dataset):
    batch = _batch(prepared_dataset)
    model = LLBRepNet(
        num_classes=7,
        num_layers=2,
        hidden_dim=32,
        entity_hidden=16,
        surf_emb_dim=16,
        curve_emb_dim=16,
    )
    model.eval()
    logits = model(batch)
    assert logits.shape == (batch.num_faces, 7)
    assert torch.isfinite(logits).all()


@pytest.mark.requires_pythonocc
def test_one_step_decreases_loss(prepared_dataset):
    torch.manual_seed(0)
    batch = _batch(prepared_dataset)
    model = LLBRepNet(
        num_classes=7,
        num_layers=2,
        hidden_dim=32,
        entity_hidden=16,
        surf_emb_dim=16,
        curve_emb_dim=16,
    )
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    first = last = None
    for _step in range(12):
        opt.zero_grad()
        loss = F.cross_entropy(model(batch), batch.labels, ignore_index=-1)
        loss.backward()
        opt.step()
        if first is None:
            first = float(loss)
        last = float(loss)
    assert last < first


@pytest.mark.requires_pythonocc
def test_predict_logits_per_solid(prepared_dataset):
    batch = _batch(prepared_dataset)
    model = LLBRepNet(
        num_classes=7,
        num_layers=2,
        hidden_dim=32,
        entity_hidden=16,
        surf_emb_dim=16,
        curve_emb_dim=16,
    )
    out = model.predict_logits(batch)
    assert len(out) == batch.num_solids
    for (_stem, probs), (start, end) in zip(out, batch.split_batch):
        assert probs.shape == (end - start, 7)
        assert torch.allclose(probs.sum(dim=1), torch.ones(end - start), atol=1e-4)
