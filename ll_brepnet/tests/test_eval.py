"""Tests for the evaluation / inference path."""

from __future__ import annotations

import json

import numpy as np
import pytest

pytest.importorskip("torch")

import torch  # noqa: E402

from ll_brepnet.eval.evaluate import _inference_manifest, write_logits  # noqa: E402


def test_write_logits_shape(tmp_path):
    probs = torch.tensor([[0.1, 0.9], [0.7, 0.3], [0.5, 0.5]])
    out = write_logits("solidA", probs, tmp_path)
    assert out.name == "solidA.logits"
    arr = np.loadtxt(out, ndmin=2)
    assert arr.shape == (3, 2)
    assert np.allclose(arr.sum(axis=1), 1.0, atol=1e-4)


def test_inference_manifest_reuses_training_stats(tmp_path):
    train_manifest = tmp_path / "dataset.json"
    train_manifest.write_text(
        json.dumps(
            {
                "training_set": ["x"],
                "feature_standardization": {
                    "face_features": [{"mean": 1.0, "standard_deviation": 2.0}]
                },
                "num_classes": 5,
                "class_names": ["a", "b", "c", "d", "e"],
            }
        )
    )
    manifest = _inference_manifest(train_manifest, ["s1", "s2"])
    assert manifest["test_set"] == ["s1", "s2"]
    assert manifest["training_set"] == []
    assert manifest["num_classes"] == 5
    assert manifest["feature_standardization"]["face_features"][0]["mean"] == 1.0


@pytest.mark.slow
@pytest.mark.requires_pythonocc
def test_evaluate_folder_end_to_end(prepared_dataset, tmp_path, step_fixture_files):
    """Train one epoch, then segment a folder of real STEP files to logits."""
    import shutil

    import pytorch_lightning as pl

    from ll_brepnet.dataloaders.brep_dataset import BRepDataModule
    from ll_brepnet.eval.evaluate import evaluate_folder
    from ll_brepnet.models.ll_brepnet import LLBRepNet

    npz_dir, manifest = prepared_dataset
    dm = BRepDataModule(manifest, npz_dir, label_dir=npz_dir, max_num_faces_per_batch=4096)
    dm.setup()
    if dm.train_dataset is None or len(dm.train_dataset) == 0:
        pytest.skip("empty training split")

    model = LLBRepNet(
        num_classes=7,
        num_layers=2,
        hidden_dim=32,
        entity_hidden=16,
        surf_emb_dim=16,
        curve_emb_dim=16,
    )
    trainer = pl.Trainer(
        max_epochs=1,
        accelerator="cpu",
        devices=1,
        enable_progress_bar=False,
        logger=False,
        enable_checkpointing=False,
    )
    trainer.fit(model, datamodule=dm)
    ckpt = tmp_path / "model.ckpt"
    trainer.save_checkpoint(ckpt)

    step_dir = tmp_path / "steps"
    step_dir.mkdir()
    for f in step_fixture_files[:2]:
        shutil.copy(f, step_dir / f.name)

    written = evaluate_folder(step_dir, ckpt, manifest, tmp_path / "preds", num_workers=1)
    assert len(written) >= 1
    for p in written:
        arr = np.loadtxt(p, ndmin=2)
        assert arr.shape[1] == 7  # one column per class
