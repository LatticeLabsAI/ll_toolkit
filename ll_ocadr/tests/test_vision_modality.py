"""Tests for the LL-OCADR rendered-image vision modality.

Covers the two SDPA image encoders (``clip_sdpa``, ``sam_vary_sdpa``), the dual
branch ``VisionTower``, and the end-to-end wiring into
``LatticelabsOCADRForCausalLM`` (image tokens spliced into the LLM input
alongside the 3D mesh tokens). The model's LLM is stubbed so no checkpoint /
network is required.
"""

from __future__ import annotations

import types

import pytest

torch = pytest.importorskip("torch")

from ll_ocadr.vllm.lattice_encoder.clip_sdpa import CLIPVisionConfig, CLIPVisionSDPA
from ll_ocadr.vllm.lattice_encoder.sam_vary_sdpa import (
    SAMVaryConfig,
    SAMVaryViTSDPA,
)
from ll_ocadr.vllm.lattice_encoder.vision_tower import VisionTower


def _tiny_clip_cfg():
    return CLIPVisionConfig(
        image_size=224, patch_size=14, embed_dim=32, depth=2, num_heads=4
    )


def _tiny_sam_cfg():
    return SAMVaryConfig(
        image_size=224,
        patch_size=16,
        embed_dim=24,
        depth=2,
        num_heads=4,
        out_chans=16,
        window_size=7,
        global_attn_indexes=(1,),
    )


@pytest.mark.requires_torch
class TestEncoders:
    def test_clip_output_shape_and_grad(self):
        clip = CLIPVisionSDPA(_tiny_clip_cfg())
        out = clip(torch.randn(2, 3, 224, 224))
        assert out.shape == (2, 1 + 16 * 16, 32)  # class token + 16x16 patches
        out.mean().backward()
        assert all(
            p.grad is not None for p in clip.parameters() if p.requires_grad
        )

    def test_sam_output_shape_and_grad(self):
        sam = SAMVaryViTSDPA(_tiny_sam_cfg())
        out = sam(torch.randn(2, 3, 224, 224))
        assert out.shape == (2, 16, 14, 14)  # out_chans x (224/16)^2 grid
        out.mean().backward()
        assert all(p.grad is not None for p in sam.parameters() if p.requires_grad)

    def test_resolution_flexibility(self):
        """Positional embeddings interpolate, so non-default sizes still work."""
        clip = CLIPVisionSDPA(_tiny_clip_cfg())
        sam = SAMVaryViTSDPA(_tiny_sam_cfg())
        assert clip(torch.randn(1, 3, 196, 196)).shape[0] == 1
        assert sam(torch.randn(1, 3, 256, 256)).shape[-1] == 16


@pytest.mark.requires_torch
class TestVisionTower:
    def test_tower_token_count_and_grad(self):
        tower = VisionTower(
            n_embed=16,
            clip_config=_tiny_clip_cfg(),
            sam_config=_tiny_sam_cfg(),
            sam_compress_stride=2,
        )
        out = tower(torch.randn(2, 3, 224, 224))
        # CLIP patches 16x16=256 + SAM 14x14 compressed (stride 2) -> 7x7=49.
        assert out.shape == (2, 256 + 49, 16)
        out.mean().backward()
        assert all(
            p.grad is not None for p in tower.parameters() if p.requires_grad
        )


@pytest.mark.requires_torch
class TestModelWiring:
    """End-to-end: rendered-image tokens are spliced into the LLM input."""

    def _config(self, n_embed=32):
        return types.SimpleNamespace(
            projector_type="linear",
            input_dim=16 + 256,  # shape_embed_dim + geometry dim (unused here)
            n_embed=n_embed,
            language_model_name="stub",
            mesh_token_id=1,
            shape_embed_dim=16,
            shape_depth=1,
            shape_num_heads=2,
            use_vision=True,
            image_token_id=2,
            vision_image_size=224,
            vision_clip_embed_dim=32,
            vision_clip_depth=2,
            vision_clip_num_heads=4,
            vision_sam_embed_dim=24,
            vision_sam_depth=2,
            vision_sam_num_heads=4,
            vision_sam_out_chans=16,
        )

    def _build_model(self, monkeypatch, n_embed=32):
        import ll_ocadr.vllm.latticelabs_ocadr as mod

        captured = {}

        class _StubLLM(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.embed = torch.nn.Embedding(50, n_embed)
                # Real HF models expose .config.hidden_size; the model derives
                # n_embed from it so the projector/splice match the LLM width.
                self.config = types.SimpleNamespace(
                    hidden_size=n_embed, vocab_size=50
                )

            def get_input_embeddings(self):
                return self.embed

            def forward(self, inputs_embeds=None, attention_mask=None, **kwargs):
                captured["inputs_embeds"] = inputs_embeds
                return types.SimpleNamespace(logits=inputs_embeds.sum())

        monkeypatch.setattr(
            mod.AutoModelForCausalLM,
            "from_pretrained",
            staticmethod(lambda *a, **k: _StubLLM()),
        )
        model = mod.LatticelabsOCADRForCausalLM(self._config(n_embed))
        model.eval()
        return model, captured

    def test_vision_tower_built_when_enabled(self, monkeypatch):
        model, _ = self._build_model(monkeypatch)
        assert hasattr(model, "vision_tower")
        assert model.use_vision is True

    def test_image_tokens_spliced_into_llm_input(self, monkeypatch):
        n_embed = 32
        model, captured = self._build_model(monkeypatch, n_embed)

        # input_ids: text + image_token_id (2) placeholders.
        input_ids = torch.tensor([[5, 2, 2, 2, 2, 6]])
        pixel_values = torch.randn(1, 3, 224, 224)

        # Pure-text embeddings for comparison.
        text_only = model.language_model.get_input_embeddings()(input_ids).clone()

        model.forward(input_ids=input_ids, pixel_values=pixel_values)
        merged = captured["inputs_embeds"]

        assert merged.shape == (1, 6, n_embed)
        # Image-token positions (1..4) must have been replaced (differ from text);
        # text positions (0, 5) must be unchanged.
        assert not torch.allclose(merged[0, 1:5], text_only[0, 1:5])
        assert torch.allclose(merged[0, 0], text_only[0, 0])
        assert torch.allclose(merged[0, 5], text_only[0, 5])

    def test_no_vision_when_pixel_values_absent(self, monkeypatch):
        """Without pixel_values the path is the original text/mesh behavior."""
        model, captured = self._build_model(monkeypatch)
        input_ids = torch.tensor([[5, 2, 2, 6]])
        text_only = model.language_model.get_input_embeddings()(input_ids).clone()
        model.forward(input_ids=input_ids)  # no images
        # image_token positions stay as text embeddings (nothing spliced).
        assert torch.allclose(captured["inputs_embeds"], text_only)
