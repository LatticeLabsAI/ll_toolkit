"""Dimension-conditioning experiment for the VAE generator (M3 follow-up B).

Warm-starts from the solid-gated checkpoint, then RL-trains the (zero-init)
dimension encoder with a DENSE dimensional reward across a small discrete set of
target boxes. Success is measured the honest way the advisor specified:
**does the achieved bounding box track the requested target?** — not generic
"diversity". We report, per target, the mean achieved bbox of valid solids, and
whether bigger targets yield bigger solids (rank correlation of requested vs
achieved overall size).

This is open-ended ML on a tiny CPU model; a flat/partial result is a legitimate
honest outcome, reported as such.

Usage::

    python -m scripts.condition_experiment --warm-start checkpoints/vae_rl_solid.pt \\
        --epochs 8 --steps-per-target 20 --eval-samples 40 --seed 0
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

_log = logging.getLogger(__name__)

# Three well-separated target boxes spanning the model's achievable bbox range.
TARGETS = {
    "small": (0.3, 0.3, 0.3),
    "medium": (1.0, 1.5, 2.0),
    "large": (2.5, 3.5, 3.5),
}


def _bbox_sum(dims) -> float:
    return float(sum(dims)) if dims else 0.0


def run_experiment(
    warm_start: str | None,
    epochs: int,
    steps_per_target: int,
    eval_samples: int,
    lr: float,
    seed: int,
    reward_scale: float,
    output_dir: str,
) -> dict:
    import numpy as np
    import torch

    from ll_gen.config import DisposalConfig, FeedbackConfig
    from ll_gen.disposal.engine import DisposalEngine
    from ll_gen.generators.neural_vae import NeuralVAEGenerator
    from ll_gen.training.rl_trainer import RLAlignmentTrainer

    torch.manual_seed(seed)
    np.random.seed(seed)

    generator = NeuralVAEGenerator(
        checkpoint_path=Path(warm_start) if warm_start else None, device="cpu"
    )
    generator._init_model()

    feedback = FeedbackConfig(
        dense_dimension_reward=True, dimension_reward_scale=reward_scale
    )
    trainer = RLAlignmentTrainer(
        generator,
        learning_rate=lr,
        device="cpu",
        output_dir=str(Path(output_dir) / "rl"),
        feedback_config=feedback,
        seed=seed,
    )
    trainer._init_training()
    generator._model.train()

    # Records cycle through the targets; train_step threads target_dimensions to
    # BOTH the conditioner (generate_for_training) and the dense reward.
    target_items = list(TARGETS.items())
    records = []
    for _ in range(steps_per_target):
        for name, dims in target_items:
            records.append({"prompt": f"a {name} part", "target_dimensions": dims})

    curve = []
    for epoch in range(epochs):
        metrics = trainer.train_epoch(records)
        curve.append({k: round(float(v), 4) for k, v in metrics.items()})
        _log.info("epoch %d/%d: %s", epoch + 1, epochs, metrics)

    # --- Measurement: per target, mean achieved bbox of valid solids ---
    generator._model.eval()
    eng = DisposalEngine(
        disposal_config=DisposalConfig(),
        feedback_config=feedback,
        output_dir=str(Path(output_dir) / "eval"),
    )
    results = {}
    with torch.no_grad():
        for name, dims in target_items:
            sums, achieved = [], []
            for _ in range(eval_samples):
                # Measure the inference path (generate), not the RL training
                # path — same decoder, but no log-prob/entropy plumbing, so this
                # reflects deployment behavior.
                prop = generator.generate(f"a {name} part", target_dimensions=dims)
                r = eng.dispose(prop, export=False)
                if (
                    r.is_valid
                    and r.geometry_report
                    and r.geometry_report.solid_count >= 1
                ):
                    bd = r.geometry_report.bbox_dimensions
                    if bd:
                        achieved.append(sorted(bd))
                        sums.append(_bbox_sum(bd))
            results[name] = {
                "requested_dims": dims,
                "requested_sum": round(float(sum(dims)), 3),
                "n_valid_solids": len(sums),
                "mean_achieved_bbox_sum": (
                    round(float(np.mean(sums)), 3) if sums else None
                ),
                "mean_achieved_dims": (
                    [round(float(x), 3) for x in np.mean(achieved, axis=0)]
                    if achieved
                    else None
                ),
            }

    # Does achieved size track requested size across the 3 targets?
    ordered = [
        results[n]["mean_achieved_bbox_sum"] for n in ("small", "medium", "large")
    ]
    tracks = (
        all(v is not None for v in ordered) and ordered[0] < ordered[1] < ordered[2]
    )

    return {
        "warm_start": warm_start,
        "epochs": epochs,
        "reward_scale": reward_scale,
        "targets": TARGETS,
        "per_target": results,
        "achieved_sum_small_medium_large": ordered,
        "achieved_tracks_requested": tracks,
        "curve": curve,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--warm-start", default="checkpoints/vae_rl_solid.pt")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--steps-per-target", type=int, default=20)
    parser.add_argument("--eval-samples", type=int, default=40)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--reward-scale", type=float, default=1.5)
    parser.add_argument("--output-dir", default="condition_out")
    parser.add_argument("--results", default=None)
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))

    report = run_experiment(
        warm_start=args.warm_start or None,
        epochs=args.epochs,
        steps_per_target=args.steps_per_target,
        eval_samples=args.eval_samples,
        lr=args.lr,
        seed=args.seed,
        reward_scale=args.reward_scale,
        output_dir=args.output_dir,
    )
    if args.results:
        rp = Path(args.results)
        rp.parent.mkdir(parents=True, exist_ok=True)
        rp.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
