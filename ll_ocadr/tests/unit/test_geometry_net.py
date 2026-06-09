"""Unit tests for GeometryNet (PointNet++ local encoder) — SPEC-1 M4, T4.2."""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from ll_ocadr.vllm.lattice_encoder.geometry_net import (  # noqa: E402
    _farthest_point_sample_python,
    _has_torch_cluster,
    build_geometry_net,
    farthest_point_sample,
)


@pytest.mark.requires_torch
@pytest.mark.unit
class TestGeometryNetForward:
    def test_output_shape(self, synth_coords, synth_normals) -> None:
        """GeometryNet returns [B, 128, 256] (128 sampled points, 256-dim feats)."""
        net = build_geometry_net().eval()
        with torch.no_grad():
            out = net(synth_coords, synth_normals)
        assert out.shape == (1, 128, 256)
        assert torch.isfinite(out).all()

    def test_gradients_flow(self, synth_coords, synth_normals) -> None:
        """Backprop reaches the encoder parameters (FPS indices are not a barrier)."""
        net = build_geometry_net().train()
        out = net(synth_coords, synth_normals)
        out.sum().backward()
        grads = [p.grad for p in net.parameters() if p.grad is not None]
        assert grads, "no parameter received a gradient"
        assert any(g.abs().sum().item() > 0 for g in grads), "all gradients are zero"

    def test_small_point_cloud_below_nsample(self) -> None:
        """Regression: a cloud with fewer points than the ball-query nsample must
        not crash query_ball_point.

        A simple STEP part's B-rep vertices give very few points (part.step ->
        ~8); the live run on it hit an IndexError in query_ball_point because
        group_first was repeated to nsample while group_idx held only N columns.
        """
        net = build_geometry_net().eval()
        coords = torch.randn(1, 8, 3)
        normals = torch.randn(1, 8, 3)
        with torch.no_grad():
            out = net(coords, normals)
        assert out.shape == (1, 128, 256)
        assert torch.isfinite(out).all()


@pytest.mark.requires_torch
@pytest.mark.unit
class TestFarthestPointSample:
    def test_python_fallback_shape_and_range(self) -> None:
        """The pure-Python FPS returns in-range indices of the requested count."""
        gen = torch.Generator().manual_seed(0)
        xyz = torch.randn(2, 256, 3, generator=gen)
        idx = _farthest_point_sample_python(xyz, npoint=64)
        assert idx.shape == (2, 64)
        assert idx.dtype == torch.long
        assert int(idx.min()) >= 0 and int(idx.max()) < 256

    def test_public_fps_uses_fallback_when_no_torch_cluster(self) -> None:
        """Without torch_cluster, the public wrapper runs the Python path."""
        gen = torch.Generator().manual_seed(1)
        xyz = torch.randn(1, 128, 3, generator=gen)
        idx = farthest_point_sample(xyz, npoint=32)
        assert idx.shape == (1, 32)
        assert int(idx.min()) >= 0 and int(idx.max()) < 128

    @pytest.mark.skipif(not _has_torch_cluster, reason="torch_cluster not installed")
    def test_fast_path_shape(self) -> None:
        """When torch_cluster IS present, the fused kernel returns the same shape."""
        gen = torch.Generator().manual_seed(2)
        xyz = torch.randn(1, 128, 3, generator=gen)
        idx = farthest_point_sample(xyz, npoint=32)
        assert idx.shape == (1, 32)
        assert int(idx.min()) >= 0 and int(idx.max()) < 128
