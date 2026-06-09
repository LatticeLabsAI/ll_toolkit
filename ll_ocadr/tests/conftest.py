"""Shared pytest fixtures for the ll_ocadr suite (SPEC-1 M4).

OpenMP safety (macOS): torch is imported FIRST, before anything else, and
``OMP_NUM_THREADS`` is pinned to 1 to avoid the libomp double-initialisation
crash documented in the repo CLAUDE.md.

All fixtures are fully offline and deterministic — the "tiny LM" is a small
GPT-2 constructed and saved to a temp dir, so the end-to-end tests need no
network access or model download.
"""

from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")

# Import torch first for OpenMP protection (stdlib imports above are fine).
import pytest  # noqa: E402
import torch  # noqa: E402

# A real point cloud size: must be >= 256 so ShapeNet's fixed 256-patch
# embedding and GeometryNet's set-abstraction both have enough points.
_NUM_POINTS = 512


@pytest.fixture
def synth_coords() -> torch.Tensor:
    """Deterministic [1, N, 3] vertex coordinates."""
    gen = torch.Generator().manual_seed(0)
    return torch.randn(1, _NUM_POINTS, 3, generator=gen)


@pytest.fixture
def synth_normals() -> torch.Tensor:
    """Deterministic [1, N, 3] unit-ish vertex normals."""
    gen = torch.Generator().manual_seed(1)
    n = torch.randn(1, _NUM_POINTS, 3, generator=gen)
    return torch.nn.functional.normalize(n, dim=-1)


@pytest.fixture
def synth_pointcloud_6d(synth_coords, synth_normals) -> torch.Tensor:
    """[N, 6] coord+normal point cloud (single mesh, batch dim removed)."""
    return torch.cat([synth_coords[0], synth_normals[0]], dim=-1)


@pytest.fixture
def tiny_stl_file(tmp_path):
    """Write a small binary STL (a unit box, 12 triangles) to a temp path.

    Skips if trimesh is unavailable.
    """
    trimesh = pytest.importorskip("trimesh")
    mesh = trimesh.creation.box(extents=(1.0, 1.0, 1.0))
    path = tmp_path / "box.stl"
    mesh.export(str(path))
    return str(path)


@pytest.fixture(scope="session")
def tiny_lm_dir(tmp_path_factory) -> str:
    """Build and save a tiny offline GPT-2 LM; return its directory.

    Session-scoped: the (already small) model is constructed once. n_embd is
    kept at 64 so the OCADR config's ``n_embed`` can match it exactly.
    """
    from transformers import GPT2Config, GPT2LMHeadModel

    d = tmp_path_factory.mktemp("tiny_lm")
    lm_config = GPT2Config(
        vocab_size=512,
        n_positions=128,
        n_embd=64,
        n_layer=2,
        n_head=2,
    )
    GPT2LMHeadModel(lm_config).save_pretrained(str(d))
    return str(d)


@pytest.fixture
def ocadr_config(tiny_lm_dir):
    """An LLOCADRConfig wired to the tiny offline LM.

    ``n_embed`` matches the tiny GPT-2 hidden size (64); the encoder output
    dims (geometry 256 + shape 768 = input_dim 1024) are kept at their real
    values since GeometryNet's output width is fixed.
    """
    from ll_ocadr.vllm.config import LLOCADRConfig

    return LLOCADRConfig(
        language_model_name=tiny_lm_dir,
        n_embed=64,
        shape_embed_dim=768,
        shape_depth=1,
        shape_num_heads=8,
        input_dim=1024,
        mesh_token_id=5,
        projector_type="linear",
    )


@pytest.fixture
def ocadr_model(ocadr_config):
    """A fully-built LatticelabsOCADRForCausalLM on CPU, in eval mode."""
    from ll_ocadr.vllm.latticelabs_ocadr import LatticelabsOCADRForCausalLM

    return LatticelabsOCADRForCausalLM(ocadr_config).eval()


@pytest.fixture(scope="session")
def tiny_lm_with_tokenizer_dir(tmp_path_factory) -> str:
    """A tiny offline GPT-2 saved WITH a real tokenizer, for the CLI script test.

    The script path (``run_ll_ocadr_hf.py``) loads the LM, tokenizer, and config
    via ``from_pretrained`` on this dir, so both must be present — and fully
    offline. n_positions is 512 so the ~385 global mesh tokens fit the context.
    """
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast

    d = tmp_path_factory.mktemp("tiny_lm_tok")

    words = [
        "[PAD]",
        "[UNK]",
        "[BOS]",
        "[EOS]",
        "<mesh>",
        "describe",
        "this",
        "cad",
        "part",
        "a",
        "box",
        "sphere",
    ]
    vocab = {w: i for i, w in enumerate(words)}
    tk = Tokenizer(models.WordLevel(vocab=vocab, unk_token="[UNK]"))
    tk.pre_tokenizer = pre_tokenizers.Whitespace()
    fast = PreTrainedTokenizerFast(
        tokenizer_object=tk,
        unk_token="[UNK]",
        pad_token="[PAD]",
        bos_token="[BOS]",
        eos_token="[EOS]",
    )

    GPT2LMHeadModel(
        GPT2Config(
            vocab_size=len(words), n_positions=512, n_embd=64, n_layer=2, n_head=2
        )
    ).save_pretrained(str(d))
    fast.save_pretrained(str(d))
    return str(d)


@pytest.fixture
def sphere_stl_file(tmp_path) -> str:
    """A denser STL (icosphere, 642 vertices) for non-degenerate encoder input."""
    trimesh = pytest.importorskip("trimesh")
    mesh = trimesh.creation.icosphere(subdivisions=3, radius=1.0)
    path = tmp_path / "sphere.stl"
    mesh.export(str(path))
    return str(path)
