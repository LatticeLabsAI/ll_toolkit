"""Proof-of-life RL run: same model, prior-sampling validity before vs after RL.

SPEC-1 M3 T3.4 + T3.5 combined into one process so the comparison is reproducible
and fair: a single generator is built, its prior-sampling validity is measured
(baseline B), the REINFORCE dispose-reward loop is run for a few epochs, and the
*same* model's validity is measured again (trained T) with the identical eval
seed and prompt set. Because random-init weights depend on ambient RNG, comparing
a trained checkpoint against a separately-built random-init is noisy — measuring
one model before and after removes that confound entirely.

The M3 gate is prior-sampling validity (z ~ N(0,1) -> decode -> dispose). We
report ``validity_rate`` AND ``num_distinct_valid`` for both points plus the
per-epoch ``train_epoch`` curve, so a validity gain from mode collapse (one valid
shape repeated) is visible rather than mistaken for success.

CLI::

    python -m ll_gen.training.proof_of_life --generator vae \\
        --prompts eval/heldout.jsonl --epochs 5 --steps-per-epoch 80 \\
        --n-eval-samples 100 --seed 0 --save checkpoints/vae_rl.pt \\
        --results results/proof_of_life_vae.json
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

_log = logging.getLogger(__name__)


def _seed_weights(seed: int) -> None:
    """Seed torch/numpy/random before model construction for reproducible init."""
    import random

    import torch

    random.seed(seed)
    torch.manual_seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass


def _eval_summary(metrics) -> dict[str, Any]:
    return {
        "validity_rate": metrics.validity_rate,
        "compile_rate": metrics.compile_rate,
        "num_valid": metrics.num_valid,
        "num_distinct_valid": metrics.num_distinct_valid,
        "mean_reward": metrics.mean_reward,
        "num_samples": metrics.num_samples,
    }


def proof_of_life(
    generator_name: str,
    prompts: list[Any],
    epochs: int,
    steps_per_epoch: int,
    learning_rate: float,
    device: str,
    seed: int,
    eval_seed: int,
    n_eval_samples: int,
    save_path: str | None,
    output_dir: str,
) -> dict[str, Any]:
    """Build, baseline-eval, RL-train, re-eval one generator; return the report.

    Args:
        generator_name: "vae" / "vqvae" / "diffusion".
        prompts: Held-out prompts for evaluation and (tiled) for training records.
        epochs: RL epochs.
        steps_per_epoch: train_steps per epoch (prompts are tiled to this length).
        learning_rate: RL optimizer LR.
        device: Torch device.
        seed: Seed for weight init + trainer RNG.
        eval_seed: Seed for both prior-sampling evals (identical draws B vs T).
        n_eval_samples: Prior samples per prompt at eval.
        save_path: Optional checkpoint path for the trained model.
        output_dir: Disposal working directory for eval/train.

    Returns:
        Report dict with ``baseline``, ``trained``, ``delta``, ``curve``.
    """
    from ll_gen.training.evaluate_validity import evaluate_validity
    from ll_gen.training.rl_trainer import RLAlignmentTrainer
    from ll_gen.training.run import build_generator

    def _eval_both(tag: str) -> dict[str, Any]:
        # Measure both decode paths: "training" (generate_for_training — the path
        # the REINFORCE loss optimizes, and the M3 gate) and "inference"
        # (generate — the deployment/pipeline path). Reporting both surfaces any
        # divergence between what RL optimizes and what deployment uses.
        out = {}
        for mode in ("training", "inference"):
            metrics = evaluate_validity(
                generator,
                prompts,
                n_samples=n_eval_samples,
                seed=eval_seed,
                decode_mode=mode,
                output_dir=str(Path(output_dir) / f"{tag}_{mode}"),
            )
            out[mode] = _eval_summary(metrics)
        return out

    _seed_weights(seed)
    generator = build_generator(generator_name, device)

    # --- Baseline: prior-sampling validity of the untrained model ---
    baseline = _eval_both("baseline")
    _log.info("Baseline: %s", baseline)

    # --- RL training on the same model ---
    trainer = RLAlignmentTrainer(
        generator,
        learning_rate=learning_rate,
        device=device,
        output_dir=str(Path(output_dir) / "rl"),
        seed=seed,
    )
    trainer._init_training()
    if generator._model is not None:
        generator._model.train()

    # Tile prompts to the requested steps-per-epoch (prompts are non-conditioning
    # trace metadata for the dispose-reward loop; this just sets the step count).
    records = [
        {"prompt": _prompt_text(prompts[i % len(prompts)])}
        for i in range(steps_per_epoch)
    ]

    curve: list[dict[str, float]] = []
    for epoch in range(epochs):
        metrics = trainer.train_epoch(records)
        curve.append(metrics)
        _log.info("RL epoch %d/%d: %s", epoch + 1, epochs, metrics)

    # --- Trained: same model, same eval seed/prompts, both decode paths ---
    if generator._model is not None:
        generator._model.eval()
    trained = _eval_both("trained")
    _log.info("Trained: %s", trained)

    if save_path:
        trainer.save_checkpoint(Path(save_path))
        _log.info("Saved trained checkpoint to %s", save_path)

    # Gate metric is the RL-optimized "training" decode path.
    delta = {
        mode: {
            "validity_rate": round(
                trained[mode]["validity_rate"] - baseline[mode]["validity_rate"], 4
            ),
            "num_distinct_valid": trained[mode]["num_distinct_valid"]
            - baseline[mode]["num_distinct_valid"],
        }
        for mode in ("training", "inference")
    }
    return {
        "generator": generator_name,
        "epochs": epochs,
        "steps_per_epoch": steps_per_epoch,
        "n_eval_samples": n_eval_samples,
        "seed": seed,
        "eval_seed": eval_seed,
        "gate_decode_mode": "training",
        "baseline": baseline,
        "trained": trained,
        "delta": delta,
        "curve": curve,
    }


def _prompt_text(item: Any) -> str:
    if isinstance(item, str):
        return item
    if isinstance(item, dict):
        return str(item.get("prompt") or item.get("caption") or "")
    return str(item)


def _load_prompts(path: str) -> list[Any]:
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
        raise ValueError(f"No prompts in {path!r}")
    return prompts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--generator", default="vae", choices=["vae", "vqvae", "diffusion"]
    )
    parser.add_argument("--prompts", required=True)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--steps-per-epoch", type=int, default=80)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--eval-seed", type=int, default=12345)
    parser.add_argument("--n-eval-samples", type=int, default=100)
    parser.add_argument("--save", default=None)
    parser.add_argument("--results", default=None)
    parser.add_argument("--output-dir", default="proof_of_life_out")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))

    prompts = _load_prompts(args.prompts)
    report = proof_of_life(
        generator_name=args.generator,
        prompts=prompts,
        epochs=args.epochs,
        steps_per_epoch=args.steps_per_epoch,
        learning_rate=args.lr,
        device=args.device,
        seed=args.seed,
        eval_seed=args.eval_seed,
        n_eval_samples=args.n_eval_samples,
        save_path=args.save,
        output_dir=args.output_dir,
    )

    if args.results:
        results_path = Path(args.results)
        results_path.parent.mkdir(parents=True, exist_ok=True)
        results_path.write_text(json.dumps(report, indent=2))
        _log.info("Wrote results to %s", results_path)

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
