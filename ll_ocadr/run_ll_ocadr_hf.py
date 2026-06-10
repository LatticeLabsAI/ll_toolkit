#!/usr/bin/env python
"""HF-native inference for LatticeLabs OCADR (no vLLM).

Loads a CAD/mesh file, runs it through the 3D encoders + projector, merges the
result into a HuggingFace language model's embeddings, and generates a textual
description. This is the supported inference path; the vLLM integration in
``ll_ocadr/vllm/`` is experimental/future work and is NOT used here.

Usage:
    python run_ll_ocadr_hf.py \
        --model Qwen/Qwen2-1.8B \
        --mesh part.step \
        --prompt "Describe this part: <mesh>" \
        --max-new-tokens 64

The prompt may contain one ``<mesh>`` placeholder per mesh; it is expanded to
the appropriate number of mesh tokens automatically.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence

import torch
from transformers import AutoConfig, AutoTokenizer

from ll_ocadr.vllm.config import LLOCADRConfig
from ll_ocadr.vllm.latticelabs_ocadr import LatticelabsOCADRForCausalLM
from ll_ocadr.vllm.process.mesh_process import LLOCADRProcessor


def _derive_hidden_size(lm_config) -> int:
    """Return the language model's hidden/embedding size across architectures."""
    # Cover the common attribute names across architectures:
    #   hidden_size -> Llama/Qwen/Mistral/BERT-style
    #   n_embd      -> GPT-2/GPT-Neo-style
    #   d_model     -> T5/encoder-decoder/BART-style
    for attr in ("hidden_size", "n_embd", "d_model"):
        value = getattr(lm_config, attr, None)
        if value is not None:
            return int(value)
    raise ValueError(
        f"Could not determine the language model's hidden size for config "
        f"'{type(lm_config).__name__}': none of hidden_size / n_embd / d_model "
        f"are set. Pass a model whose config exposes one of these, or set "
        f"LLOCADRConfig.n_embed explicitly to the LM's embedding dimension."
    )


def build_model_and_tokenizer(
    model_name: str,
    device: str = "cpu",
    shape_depth: int | None = None,
) -> tuple[LatticelabsOCADRForCausalLM, AutoTokenizer, LLOCADRConfig, LLOCADRProcessor]:
    """Build the OCADR model, tokenizer, config, and preprocessor for HF inference.

    ``n_embed`` is derived from the chosen language model so the mesh embeddings
    line up with the LM's embedding space. The ``<mesh>`` token is registered and
    the LM's token-embedding matrix is resized to match.
    """
    lm_config = AutoConfig.from_pretrained(model_name)
    n_embed = _derive_hidden_size(lm_config)

    config = LLOCADRConfig(language_model_name=model_name, n_embed=n_embed)
    if shape_depth is not None:
        config.shape_depth = shape_depth

    model = LatticelabsOCADRForCausalLM(config).to(device).eval()

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    mesh_tokens_added = 0
    if config.mesh_token not in tokenizer.get_vocab():
        mesh_tokens_added = tokenizer.add_tokens([config.mesh_token])
    config.mesh_token_id = tokenizer.convert_tokens_to_ids(config.mesh_token)

    # The model caches the mesh token id (config.mesh_token_id was None at build
    # time); update it. Resize the LM embeddings only when we actually added a
    # new token — resizing reinitializes the matrix, so skip it otherwise.
    model.mesh_token_id = config.mesh_token_id
    if mesh_tokens_added > 0:
        model.language_model.resize_token_embeddings(len(tokenizer))

    processor = LLOCADRProcessor(
        tokenizer=tokenizer,
        mesh_token_id=config.mesh_token_id,
        min_chunk_size=config.min_chunk_size,
        max_chunks=config.max_chunks,
        target_global_faces=config.target_global_faces,
    )
    return model, tokenizer, config, processor


def run_inference(
    model: LatticelabsOCADRForCausalLM,
    processor: LLOCADRProcessor,
    tokenizer,
    mesh_file: str | Sequence[str],
    prompt: str,
    max_new_tokens: int = 64,
    cropping: bool = True,
    do_sample: bool = False,
) -> str:
    """Run the full mesh-file -> text pipeline and return the decoded output.

    ``mesh_file`` may be a single path or a sequence of paths. The number of
    ``<mesh>`` placeholders in ``prompt`` must equal the number of mesh files;
    as a convenience, a single mesh with no placeholder gets one appended.

    The target device is taken from the model itself (single source of truth),
    so the inputs always land on the same device as the model.
    """
    model_device = next(model.parameters()).device
    mesh_files = [mesh_file] if isinstance(mesh_file, str) else list(mesh_file)

    placeholder_count = prompt.count("<mesh>")
    if placeholder_count == 0 and len(mesh_files) == 1:
        prompt = f"{prompt} <mesh>"
        placeholder_count = 1
    if placeholder_count != len(mesh_files):
        raise ValueError(
            f"prompt contains {placeholder_count} '<mesh>' placeholder(s) but "
            f"{len(mesh_files)} mesh file(s) were provided; they must match "
            f"one-to-one."
        )

    inputs = processor.tokenize_with_meshes(
        mesh_files=mesh_files,
        conversation=prompt,
        cropping=cropping,
    )
    # num_mesh_tokens is metadata, not a generate() argument.
    inputs.pop("num_mesh_tokens", None)
    tensors = {
        key: (value.to(model_device) if isinstance(value, torch.Tensor) else value)
        for key, value in inputs.items()
    }

    gen_kwargs = {"max_new_tokens": max_new_tokens, "do_sample": do_sample}
    if tokenizer.pad_token_id is not None:
        gen_kwargs["pad_token_id"] = tokenizer.pad_token_id
    elif tokenizer.eos_token_id is not None:
        gen_kwargs["pad_token_id"] = tokenizer.eos_token_id

    with torch.no_grad():
        generated = model.generate(**tensors, **gen_kwargs)

    return tokenizer.decode(generated[0], skip_special_tokens=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="HF-native LatticeLabs OCADR inference (no vLLM)."
    )
    parser.add_argument("--model", required=True, help="HF model name or local path.")
    parser.add_argument(
        "--mesh", required=True, help="Path to a STEP/STL/OBJ/PLY file."
    )
    parser.add_argument(
        "--prompt",
        default="Describe this CAD part: <mesh>",
        help="Prompt text; include a <mesh> placeholder.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--shape-depth",
        type=int,
        default=None,
        help="Override ShapeNet transformer depth (smaller = faster).",
    )
    parser.add_argument(
        "--no-cropping",
        dest="cropping",
        action="store_false",
        help="Disable spatial chunking (global view only).",
    )
    parser.add_argument("--do-sample", action="store_true")
    args = parser.parse_args()

    model, tokenizer, _config, processor = build_model_and_tokenizer(
        args.model, device=args.device, shape_depth=args.shape_depth
    )
    text = run_inference(
        model=model,
        processor=processor,
        tokenizer=tokenizer,
        mesh_file=args.mesh,
        prompt=args.prompt,
        max_new_tokens=args.max_new_tokens,
        cropping=args.cropping,
        do_sample=args.do_sample,
    )
    print(text)


if __name__ == "__main__":
    main()
