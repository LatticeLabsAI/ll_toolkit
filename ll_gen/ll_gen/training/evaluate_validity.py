"""Validity-evaluation harness for ll_gen neural generators (SPEC-1 M3, T3.2).

Runs ``propose → dispose`` over a held-out prompt set and reports the
disposal-validity rate (``valid_rate = #valid / #total``).  This rate is the
go/no-go metric for M3's acceptance gate: a trained checkpoint must beat the
random-init baseline by a recorded margin.

The neural generators sample stochastically from their latent prior, so a
prompt is conditioning/trace metadata; ``n_samples`` independent draws are
taken per prompt to estimate the validity rate with low variance.  The
evaluation deliberately mirrors the no-gradient path used by
``RLAlignmentTrainer.evaluate`` (``generate() → dispose(export=False) →
is_valid``) so baseline and trained numbers are produced by the same code.

The disposal step is the **real** OCC/cadquery geometry kernel, so meaningful
numbers require the conda ``cadling`` environment (pythonocc-core present).
Run in the uv ``.venv`` and every sample will error in disposal — the harness
counts those as invalid and warns loudly rather than reporting a vacuous 0%.

CLI::

    python -m ll_gen.training.evaluate_validity --generator vqvae \\
        --checkpoint checkpoints/vqvae.pt --prompts eval/heldout.jsonl \\
        --n-samples 64 --seed 0
"""

from __future__ import annotations

import argparse
import contextlib
import json
import logging
import random
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from ll_gen.config import DisposalConfig, FeedbackConfig
from ll_gen.feedback.reward_signal import compute_reward
from ll_gen.generators.base import BaseNeuralGenerator
from ll_gen.proposals.base import BaseProposal
from ll_gen.proposals.disposal_result import DisposalResult
from ll_gen.training.metrics import GenerationMetrics, MetricsComputer

_log = logging.getLogger(__name__)

# A dispose function maps a proposal to its disposal result.  The production
# implementation is ``DisposalEngine.dispose``; tests inject a deterministic
# stand-in so the harness's aggregation logic is verifiable without OCC.
DisposeFn = Callable[[BaseProposal], DisposalResult]

PromptRecord = dict[str, Any]


def _normalize_prompts(prompts: Sequence[Any]) -> list[PromptRecord]:
    """Normalize a prompt list into ``{prompt, target_dimensions?}`` records.

    Accepts raw strings or dicts (``prompt``/``caption`` keys, optional
    ``target_dimensions``), matching the record shape used by
    ``ll_gen.training.run``.

    Args:
        prompts: Sequence of strings and/or dicts.

    Returns:
        List of normalized records.

    Raises:
        TypeError: If an element is neither a string nor a dict.
    """
    records: list[PromptRecord] = []
    for item in prompts:
        if isinstance(item, str):
            records.append({"prompt": item})
        elif isinstance(item, dict):
            prompt = item.get("prompt") or item.get("caption") or ""
            record: PromptRecord = {"prompt": str(prompt)}
            if item.get("target_dimensions") is not None:
                record["target_dimensions"] = tuple(item["target_dimensions"])
            records.append(record)
        else:
            raise TypeError(
                f"Prompt records must be str or dict, got {type(item).__name__}"
            )
    return records


def _seed_everything(seed: int) -> None:
    """Seed Python, NumPy, and torch RNGs for reproducible sampling.

    Identical seeding for baseline and trained runs keeps the validity
    comparison apples-to-apples rather than sampling noise.
    """
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def _maybe_no_grad(model: Any) -> Any:
    """Return ``torch.no_grad()`` when a torch model is present, else a no-op."""
    if model is None:
        return contextlib.nullcontext()
    try:
        import torch

        return torch.no_grad()
    except ImportError:
        return contextlib.nullcontext()


def _extract_state_dict(checkpoint: Any) -> Any:
    """Pull a flat model ``state_dict`` out of a saved checkpoint.

    ``RLAlignmentTrainer.save_checkpoint`` writes a nested dict
    ``{"model_state_dict": ..., "optimizer_state_dict": ..., ...}`` while
    ``BaseNeuralGenerator.load_checkpoint`` expects a flat state dict.  This
    bridges both: nested checkpoints are unwrapped, flat ones pass through.

    Args:
        checkpoint: The object returned by ``torch.load``.

    Returns:
        A flat state dict suitable for ``model.load_state_dict``.
    """
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        return checkpoint["model_state_dict"]
    return checkpoint


def load_generator_checkpoint(
    generator: BaseNeuralGenerator, path: str | Path
) -> dict[str, Any]:
    """Load a trained checkpoint into a generator's model.

    Handles both the trainer's nested checkpoint format and a bare model
    ``state_dict``.  Initializes the generator's model first if needed so the
    weights have a destination.

    Args:
        generator: The neural generator to load weights into.
        path: Checkpoint file path.

    Returns:
        The raw loaded checkpoint dict (carries ``step_count``/``baseline``
        metadata for trainer checkpoints; empty dict for a bare state dict).

    Raises:
        FileNotFoundError: If the checkpoint does not exist.
        RuntimeError: If the model cannot be initialized.
        ImportError: If torch is not installed.
    """
    try:
        import torch
    except ImportError:
        raise ImportError(
            "torch is required to load checkpoints; install via conda-forge"
        ) from None

    if getattr(generator, "_model", None) is None and hasattr(generator, "_init_model"):
        generator._init_model()
    if generator._model is None:
        raise RuntimeError(
            "generator._model is None after _init_model(); cannot load checkpoint"
        )

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    try:
        checkpoint = torch.load(path, map_location=generator.device, weights_only=True)
    except Exception:
        # Older torch, or a checkpoint carrying non-tensor metadata that
        # weights_only rejects — fall back to the full unpickler.
        checkpoint = torch.load(path, map_location=generator.device, weights_only=False)

    state_dict = _extract_state_dict(checkpoint)
    generator._model.load_state_dict(state_dict)
    generator._model = generator._model.to(generator.device)
    _log.info("Loaded checkpoint %s into %s", path, type(generator).__name__)

    return checkpoint if isinstance(checkpoint, dict) else {}


def _score_proposal_sequence(
    generator: BaseNeuralGenerator, proposal: BaseProposal
) -> float | None:
    """Teacher-forcing log-prob of a proposal's token sequence under the policy.

    Scores the proposal from its *own* latent (``proposal.latent_vector``) so
    the result is a deterministic reconstruction-likelihood, not a noisy prior
    sample (see :meth:`BaseNeuralGenerator.score_token_sequence`). Returns
    ``None`` when the generator emits no command-token sequence (e.g. diffusion)
    or scoring is unavailable — those samples are simply excluded from the mean.
    """
    scorer = getattr(generator, "score_token_sequence", None)
    token_ids = getattr(proposal, "token_ids", None)
    if scorer is None or not token_ids:
        return None
    try:
        log_prob, _entropy = scorer(
            token_ids, latent=getattr(proposal, "latent_vector", None)
        )
    except Exception as exc:  # diagnostic metric must never break evaluation
        _log.debug("sequence scoring failed: %s", exc)
        return None
    if log_prob is None:
        return None
    return float(log_prob.detach().cpu())


def _make_sampler(generator: BaseNeuralGenerator, decode_mode: str):
    """Return a ``prompt -> proposal`` callable for the requested decode path.

    - ``"inference"``: ``generator.generate`` — the deployment path (e.g. the
      pipeline decode used by ``GenerationOrchestrator``).
    - ``"training"``: ``generator.generate_for_training`` — the direct sampling
      path that the REINFORCE loss optimizes. Measuring this is what makes a
      before/after RL comparison reflect what training actually changed.
    """
    if decode_mode == "inference":
        return generator.generate
    if decode_mode == "training":
        return generator.generate_for_training
    raise ValueError(
        f"decode_mode must be 'inference' or 'training', got {decode_mode!r}"
    )


def evaluate_validity(
    generator: BaseNeuralGenerator,
    prompts: Sequence[Any],
    *,
    n_samples: int = 1,
    dispose_fn: DisposeFn | None = None,
    feedback_config: FeedbackConfig | None = None,
    disposal_config: DisposalConfig | None = None,
    output_dir: str | Path = "eval_output",
    seed: int | None = None,
    decode_mode: str = "inference",
) -> GenerationMetrics:
    """Measure a generator's disposal-validity rate over a prompt set.

    For each prompt, draws ``n_samples`` proposals, disposes each through the
    deterministic engine, and aggregates validity / compile / reward metrics.
    A sample whose proposal or disposal raises is recorded as an **invalid**
    result (reward −1.0) rather than dropped, so the denominator is honest and
    a broken environment surfaces as 0% + a warning, not a divide-by-zero.

    Args:
        generator: Neural generator to evaluate (model auto-initialized).
        prompts: Held-out prompts (strings or ``{prompt, target_dimensions?}``).
        n_samples: Independent draws per prompt (use 50+ for a stable rate).
        dispose_fn: Optional disposal override; defaults to a real
            ``DisposalEngine``.  Tests inject a deterministic stand-in.
        feedback_config: Reward-signal config (defaults to ``FeedbackConfig``).
        disposal_config: Disposal/validation config (defaults to defaults).
        output_dir: Working directory for the disposal engine.
        seed: Optional RNG seed for reproducible sampling.
        decode_mode: ``"inference"`` (``generate``, the deployment path) or
            ``"training"`` (``generate_for_training``, the path RL optimizes).

    Returns:
        Populated ``GenerationMetrics`` (``validity_rate`` is the M3 gate).

    Raises:
        ValueError: If ``n_samples < 1``, ``prompts`` is empty, or
            ``decode_mode`` is unknown.
    """
    if n_samples < 1:
        raise ValueError(f"n_samples must be >= 1, got {n_samples}")

    sampler = _make_sampler(generator, decode_mode)

    records = _normalize_prompts(prompts)
    if not records:
        raise ValueError("prompts is empty; provide at least one prompt")

    feedback_config = feedback_config or FeedbackConfig()

    if seed is not None:
        _seed_everything(seed)

    # Ensure the model exists so generate() has weights to sample from.
    if getattr(generator, "_model", None) is None and hasattr(generator, "_init_model"):
        generator._init_model()

    # Build a real disposal engine unless the caller injected one.
    if dispose_fn is None:
        from ll_gen.disposal.engine import DisposalEngine

        engine = DisposalEngine(
            disposal_config=disposal_config or DisposalConfig(),
            feedback_config=feedback_config,
            output_dir=str(Path(output_dir) / "disposed"),
        )

        def dispose_fn(proposal: BaseProposal) -> DisposalResult:
            return engine.dispose(proposal, export=False)

    model = getattr(generator, "_model", None)
    results: list[DisposalResult] = []
    seq_log_probs: list[float] = []
    n_errors = 0

    with _maybe_no_grad(model):
        if model is not None and hasattr(model, "eval"):
            model.eval()

        for record in records:
            prompt = record["prompt"]
            target_dims = record.get("target_dimensions")
            for _ in range(n_samples):
                try:
                    proposal = sampler(prompt)
                    seq_lp = _score_proposal_sequence(generator, proposal)
                    if seq_lp is not None:
                        seq_log_probs.append(seq_lp)
                    result = dispose_fn(proposal)
                    result.reward_signal = compute_reward(
                        result,
                        config=feedback_config,
                        target_dimensions=target_dims,
                    )
                    results.append(result)
                except Exception as exc:
                    n_errors += 1
                    _log.error(
                        "Evaluation sample failed (prompt=%r): %s",
                        prompt[:60],
                        exc,
                    )
                    results.append(
                        DisposalResult(
                            is_valid=False,
                            reward_signal=-1.0,
                            error_message=str(exc),
                        )
                    )

    if n_errors == len(results):
        _log.warning(
            "ALL %d evaluation samples errored in propose/dispose. validity_rate "
            "is 0%% for the wrong reason — the disposal kernel likely failed "
            "(run in the conda 'cadling' env with pythonocc-core installed).",
            n_errors,
        )
    elif n_errors:
        _log.warning(
            "%d/%d samples errored and counted as invalid", n_errors, len(results)
        )

    metrics = MetricsComputer().compute_all(results)
    if seq_log_probs:
        metrics.mean_sequence_log_prob = float(sum(seq_log_probs) / len(seq_log_probs))
    _log.info(
        "validity_rate=%.4f compile_rate=%.4f mean_reward=%.4f "
        "mean_seq_log_prob=%.4f (n=%d, errors=%d, scored=%d)",
        metrics.validity_rate,
        metrics.compile_rate,
        metrics.mean_reward,
        metrics.mean_sequence_log_prob,
        metrics.num_samples,
        n_errors,
        len(seq_log_probs),
    )
    return metrics


def _load_prompts_file(path: str | Path) -> list[Any]:
    """Read a prompts file: one JSON object or raw string per non-blank line."""
    prompts: list[Any] = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                prompts.append(json.loads(line))
            except json.JSONDecodeError:
                prompts.append(line)
    if not prompts:
        raise ValueError(f"No prompts found in {path!r}")
    return prompts


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate ll_gen generator disposal-validity rate (M3 gate)."
    )
    parser.add_argument(
        "--generator", required=True, choices=["vae", "vqvae", "diffusion"]
    )
    parser.add_argument(
        "--checkpoint", default=None, help="Trained checkpoint to load (optional)."
    )
    parser.add_argument(
        "--prompts", required=True, help="JSONL/text file of held-out prompts."
    )
    parser.add_argument("--n-samples", type=int, default=64)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", default="eval_output")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))

    # Reuse the trainer CLI's generator factory so eval and train build models
    # identically (same config, same _init_model()).
    from ll_gen.training.run import build_generator

    generator = build_generator(args.generator, args.device)
    if args.checkpoint:
        load_generator_checkpoint(generator, args.checkpoint)

    prompts = _load_prompts_file(args.prompts)
    _log.info("Loaded %d prompts; %d samples each", len(prompts), args.n_samples)

    metrics = evaluate_validity(
        generator,
        prompts,
        n_samples=args.n_samples,
        seed=args.seed,
        output_dir=args.output_dir,
    )

    print(json.dumps(metrics.summary()))


if __name__ == "__main__":
    main()
