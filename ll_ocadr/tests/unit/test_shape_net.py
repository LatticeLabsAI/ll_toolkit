"""Unit tests for ShapeNet (ViT global encoder) — SPEC-1 M4, T4.2."""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from ll_ocadr.vllm.lattice_encoder.shape_net import (  # noqa: E402
    PointPatchEmbedding,
    build_shape_net,
)


@pytest.mark.requires_torch
@pytest.mark.unit
class TestShapeNetForward:
    def test_output_shape_cls_plus_patches(self, synth_coords, synth_normals) -> None:
        """ShapeNet returns [B, 257, embed_dim] = CLS token + 256 patch tokens."""
        net = build_shape_net(embed_dim=768, depth=1, num_heads=8).eval()
        with torch.no_grad():
            out = net(synth_coords, synth_normals)
        assert out.shape == (1, 257, 768)
        assert torch.isfinite(out).all()

    def test_cls_token_differs_from_patches(self, synth_coords, synth_normals) -> None:
        """The CLS token (index 0) is a distinct learned summary, not a copy."""
        net = build_shape_net(embed_dim=768, depth=1, num_heads=8).eval()
        with torch.no_grad():
            out = net(synth_coords, synth_normals)
        cls_tok = out[:, 0]
        first_patch = out[:, 1]
        assert not torch.allclose(cls_tok, first_patch)


@pytest.mark.requires_torch
@pytest.mark.unit
class TestPointPatchEmbedding:
    def test_emits_256_patches_when_evenly_divisible(self, synth_coords, synth_normals) -> None:
        """N=512 divides into exactly 256 patches (no remainder), embed_dim wide."""
        embed = PointPatchEmbedding(embed_dim=768).eval()
        with torch.no_grad():
            tokens = embed(synth_coords, synth_normals)  # N=512 -> 256 patches
        assert tokens.shape == (1, 256, 768)

    def test_constructor_patch_size_is_ignored_for_grouping(
        self, synth_coords, synth_normals
    ) -> None:
        """num_patches is hardcoded to 256; the ``patch_size`` ctor arg does not
        change the patch count (documents the known shape_net.py behaviour)."""
        a = PointPatchEmbedding(patch_size=4, embed_dim=768).eval()
        b = PointPatchEmbedding(patch_size=64, embed_dim=768).eval()
        with torch.no_grad():
            ta = a(synth_coords, synth_normals)
            tb = b(synth_coords, synth_normals)
        # Same patch count regardless of the ignored patch_size arg.
        assert ta.shape == tb.shape == (1, 256, 768)

    def test_partial_tail_patch_appended_when_not_divisible(self) -> None:
        """N not divisible by 256 yields one extra tail patch -> 257."""
        gen = torch.Generator().manual_seed(3)
        n = 513  # 513 // 256 = 2, remainder 1 -> tail patch appended
        coords = torch.randn(1, n, 3, generator=gen)
        normals = torch.randn(1, n, 3, generator=gen)
        embed = PointPatchEmbedding(embed_dim=768).eval()
        with torch.no_grad():
            tokens = embed(coords, normals)
        assert tokens.shape == (1, 257, 768)

    def test_remainder_larger_than_patch_size_does_not_crash(self) -> None:
        """Regression: when remainder > patch_size (e.g. N=642 -> patch_size=2,
        remainder=130) the old zero-pad path built a negative-sized tensor and
        crashed. The tail must be max-pooled directly into one extra patch."""
        gen = torch.Generator().manual_seed(4)
        n = 642  # 642 // 256 = 2, remainder 130 > patch_size 2
        coords = torch.randn(1, n, 3, generator=gen)
        normals = torch.randn(1, n, 3, generator=gen)
        embed = PointPatchEmbedding(embed_dim=768).eval()
        with torch.no_grad():
            tokens = embed(coords, normals)
        assert tokens.shape == (1, 257, 768)
        assert torch.isfinite(tokens).all()
