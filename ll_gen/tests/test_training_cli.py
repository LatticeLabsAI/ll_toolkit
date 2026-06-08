"""Integration test for the RL training CLI (SPEC-1 M2, T2.4).

Invokes ``python -m ll_gen.training.run`` as a subprocess on a tiny local
prompts file and verifies it runs an epoch end to end and writes a checkpoint.
Disposal degrades gracefully without pythonocc (the reward signal still gives
partial credit), so real gradient steps occur and the run completes.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("torch")

_REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.slow
@pytest.mark.integration
def test_training_cli_runs_and_saves_checkpoint(tmp_path) -> None:
    prompts = tmp_path / "prompts.jsonl"
    prompts.write_text('{"prompt": "a 20mm cube"}\n{"prompt": "a bracket"}\n')
    checkpoint = tmp_path / "vae.pt"

    env = dict(os.environ)
    env["OMP_NUM_THREADS"] = "1"
    env["PYTHONPATH"] = str(_REPO_ROOT) + os.pathsep + env.get("PYTHONPATH", "")

    result = subprocess.run(
        [
            sys.executable, "-m", "ll_gen.training.run",
            "--generator", "vae",
            "--prompts-file", str(prompts),
            "--max-samples", "2",
            "--epochs", "1",
            "--seed", "0",
            "--save", str(checkpoint),
            "--output-dir", str(tmp_path / "out"),
            "--log-level", "ERROR",
        ],
        cwd=str(_REPO_ROOT),
        capture_output=True,
        text=True,
        timeout=600,
        env=env,
    )
    assert result.returncode == 0, f"CLI failed:\n{result.stderr[-3000:]}"
    assert checkpoint.exists(), "training CLI did not write a checkpoint"

    # The CLI prints the final epoch metrics as JSON on stdout.
    last_line = [ln for ln in result.stdout.splitlines() if ln.strip()][-1]
    metrics = json.loads(last_line)
    assert {"mean_reward", "mean_loss", "validity_rate", "epoch_time_ms"} <= set(metrics)
