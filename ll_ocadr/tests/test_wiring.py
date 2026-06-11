"""Regression: the LatticelabsOCADR multimodal wiring must run end to end.

The configured model could not run before three fixes:

1. ``config.n_embed`` was hardcoded (1280) and matched no real base LLM
   (Qwen2-0.5B=896, 1.5B=1536, 7B=3584), so the projected mesh tokens (n_embed)
   could not be spliced into the LLM input embeddings (hidden_size) — shape
   mismatch. ``n_embed`` is now derived from the loaded LLM's hidden size.
2. ``get_config_for_model`` referenced non-existent Qwen2 sizes; it now uses
   real sizes ("0.5b"/"1.5b"/"7b").
3. ``_splice_tokens`` assigned fp32 modality embeddings into a bf16 LLM
   embedding tensor — ``Index put requires ... dtypes match``. It now casts to
   the destination dtype/device.

These tests mock the base LLM so they do not download Qwen, exercising the
ll_ocadr-side wiring (encoders -> projector -> splice -> LLM) directly.
"""
from __future__ import annotations

import types
from unittest.mock import patch

import pytest

torch = pytest.importorskip("torch")
import torch.nn as nn  # noqa: E402


class _FakeLLM(nn.Module):
    """Stand-in base LLM with a known hidden size, loaded in bfloat16 (as real
    Qwen2 checkpoints are), exposing the get_input_embeddings/forward surface
    LatticelabsOCADR depends on."""

    def __init__(self, hidden: int = 64, vocab: int = 128) -> None:
        super().__init__()
        self.config = types.SimpleNamespace(hidden_size=hidden, vocab_size=vocab)
        self._embed = nn.Embedding(vocab, hidden)
        self._head = nn.Linear(hidden, vocab)
        self.to(torch.bfloat16)  # emulate a bf16-loaded LLM

    def get_input_embeddings(self):
        return self._embed

    def forward(self, inputs_embeds=None, attention_mask=None, **kwargs):
        return types.SimpleNamespace(logits=self._head(inputs_embeds))


def _build_model(hidden: int = 64, vocab: int = 128):
    from ll_ocadr.vllm.config import get_config_for_model
    from ll_ocadr.vllm import latticelabs_ocadr as M

    cfg = get_config_for_model("0.5b")
    cfg.mesh_token_id = 7
    with patch.object(M, "AutoModelForCausalLM") as auto:
        auto.from_pretrained.return_value = _FakeLLM(hidden, vocab)
        model = M.LatticelabsOCADRForCausalLM(cfg).eval()
    return cfg, model


def test_n_embed_is_derived_from_llm():
    """n_embed must equal the LLM hidden size, not the hardcoded 1280."""
    cfg, model = _build_model(hidden=64)
    assert cfg.n_embed == 64
    assert model.language_model.config.hidden_size == 64
    # the projector output layer must map into the LLM hidden dim
    assert model.projector(torch.randn(1, 3, cfg.input_dim)).shape[-1] == 64


def test_forward_runs_and_splices_mesh_across_dtypes():
    """Full forward: mesh -> encoders -> projector -> bf16 splice -> LLM logits."""
    cfg, model = _build_model(hidden=64, vocab=128)
    vc = torch.randn(1, 256, 3)
    vn = torch.randn(1, 256, 3)
    mesh = model._mesh_to_embedding(vc, vn)
    n = mesh[0].shape[0]
    assert mesh[0].shape[-1] == 64  # projected to LLM hidden dim

    input_ids = torch.cat(
        [torch.tensor([[1, 2]]), torch.full((1, n), 7), torch.tensor([[3]])], dim=1
    )
    out = model.forward(
        input_ids=input_ids,
        attention_mask=torch.ones_like(input_ids),
        vertex_coords=vc,
        vertex_normals=vn,
    )
    assert out.logits.shape == (1, input_ids.shape[1], 128)
    assert torch.isfinite(out.logits.float()).all()
