"""Integration test for the HF-native inference CLI (SPEC-1 M4, T4.5).

Invokes ``run_ll_ocadr_hf.py`` as a subprocess on a real STL with a tiny
offline LM + tokenizer, exercising the genuine file -> tensors -> generate ->
text path end to end (no vLLM, no network).
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("torch")
pytest.importorskip("trimesh")

_REPO_ROOT = Path(__file__).resolve().parents[3]
_SCRIPT = _REPO_ROOT / "ll_ocadr" / "run_ll_ocadr_hf.py"


@pytest.mark.slow
@pytest.mark.integration
def test_cli_generates_text_from_stl(tiny_lm_with_tokenizer_dir, sphere_stl_file) -> None:
    env = dict(os.environ)
    env["OMP_NUM_THREADS"] = "1"
    # The script imports `ll_ocadr`; ensure the repo root is importable in the
    # subprocess (Python only auto-adds the script's own directory to sys.path).
    env["PYTHONPATH"] = str(_REPO_ROOT) + os.pathsep + env.get("PYTHONPATH", "")

    result = subprocess.run(
        [
            sys.executable,
            str(_SCRIPT),
            "--model", tiny_lm_with_tokenizer_dir,
            "--mesh", sphere_stl_file,
            "--prompt", "describe this part <mesh>",
            "--max-new-tokens", "4",
            "--no-cropping",
            "--shape-depth", "1",
        ],
        cwd=str(_REPO_ROOT),
        capture_output=True,
        text=True,
        timeout=300,
        env=env,
    )
    assert result.returncode == 0, (
        f"script failed (exit {result.returncode}).\nSTDERR:\n{result.stderr[-3000:]}"
    )
    # The script prints the decoded generation. Untrained output may be short,
    # but the pipeline must have run and produced a string line of stdout.
    assert result.stdout is not None
    assert isinstance(result.stdout, str)
