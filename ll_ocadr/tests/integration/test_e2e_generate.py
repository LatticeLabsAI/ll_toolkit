"""End-to-end HF-native forward + generate smoke test (SPEC-1 M4, T4.4).

Builds the full LatticelabsOCADRForCausalLM around a tiny offline GPT-2, feeds a
synthetic mesh + a prompt containing <mesh> placeholder tokens, and verifies the
real pipeline runs: 3D encoders -> projector -> merge into LM embeddings ->
language_model forward/generate. No network, no vLLM.
"""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")


def _mesh_prompt_ids(mesh_token_id: int, num_mesh_tokens: int = 4):
    """input_ids = [bos, text, <mesh>*k, text] with the mesh placeholders."""
    ids = [1, 2] + [mesh_token_id] * num_mesh_tokens + [3, 4]
    return torch.tensor([ids], dtype=torch.long)


@pytest.mark.requires_torch
@pytest.mark.integration
@pytest.mark.slow
class TestEndToEndGenerate:
    def test_forward_produces_logits(self, ocadr_model, ocadr_config, synth_coords, synth_normals) -> None:
        input_ids = _mesh_prompt_ids(ocadr_config.mesh_token_id)
        attention_mask = torch.ones_like(input_ids)

        with torch.no_grad():
            out = ocadr_model.forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                vertex_coords=synth_coords,
                vertex_normals=synth_normals,
            )

        # GPT-2 LM head logits: [batch, seq_len, vocab_size].
        assert out.logits.shape[0] == 1
        assert out.logits.shape[1] == input_ids.shape[1]
        assert out.logits.shape[2] == 512  # tiny LM vocab
        assert torch.isfinite(out.logits).all()

    def test_generate_emits_tokens(self, ocadr_model, ocadr_config, synth_coords, synth_normals) -> None:
        input_ids = _mesh_prompt_ids(ocadr_config.mesh_token_id)
        attention_mask = torch.ones_like(input_ids)

        gen = ocadr_model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            vertex_coords=synth_coords,
            vertex_normals=synth_normals,
            max_new_tokens=4,
            do_sample=False,
        )

        # generate() feeds inputs_embeds, so it returns only the newly generated
        # token ids (not the prompt) — assert we got the requested 4.
        assert gen.shape[0] == 1
        assert gen.shape[1] == 4
        assert gen.dtype == torch.long

    def test_forward_runs_without_mesh(self, ocadr_model, ocadr_config) -> None:
        """Text-only path (no mesh tensors) must also run end-to-end."""
        input_ids = torch.tensor([[1, 2, 3, 4, 5]], dtype=torch.long)
        with torch.no_grad():
            out = ocadr_model.forward(input_ids=input_ids)
        assert out.logits.shape == (1, 5, 512)
