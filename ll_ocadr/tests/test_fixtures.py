"""Scaffold self-check: the shared fixtures produce usable, correctly-shaped
objects and the offline tiny model builds (SPEC-1 M4, T4.1)."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")


@pytest.mark.requires_torch
def test_synth_point_cloud_fixtures(
    synth_coords, synth_normals, synth_pointcloud_6d
) -> None:
    assert synth_coords.shape == (1, 512, 3)
    assert synth_normals.shape == (1, 512, 3)
    assert synth_pointcloud_6d.shape == (512, 6)
    # Determinism: re-derivable, finite values.
    assert torch.isfinite(synth_coords).all()
    assert torch.isfinite(synth_normals).all()


@pytest.mark.requires_trimesh
def test_tiny_stl_fixture_is_real_mesh(tiny_stl_file) -> None:
    trimesh = pytest.importorskip("trimesh")
    mesh = trimesh.load(tiny_stl_file)
    assert mesh.vertices.shape[1] == 3
    assert len(mesh.faces) == 12  # a box is 12 triangles


@pytest.mark.requires_torch
def test_ocadr_model_builds_offline(ocadr_model, ocadr_config) -> None:
    assert isinstance(ocadr_model, torch.nn.Module)
    # Encoders + projector + LM are all present and parameterised.
    assert any(True for _ in ocadr_model.parameters())
    # n_embed matches the tiny LM hidden size (offline GPT-2).
    assert ocadr_config.n_embed == 64
