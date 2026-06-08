"""Training entry point for ll_gen RL alignment (SPEC-1 M2, T2.4).

Wires a prompt dataset -> neural generator -> ``RLAlignmentTrainer.train_epoch``
-> checkpoint, runnable as ``python -m ll_gen.training.run``.

The training dataset is a list of ``{"prompt": str, "target_dimensions"?: tuple}``
records. Prompts come either from a local JSONL file (``--prompts-file``) or by
adapting a CAD dataset (``--dataset {deepcad,abc} --data-path ...``); the neural
generators sample from their latent prior, so the prompt is conditioning/trace
metadata for the disposal-reward RL loop.

Examples:
    python -m ll_gen.training.run --generator vae \
        --prompts-file prompts.jsonl --epochs 2 --save checkpoints/vae.pt

    python -m ll_gen.training.run --generator vqvae \
        --dataset deepcad --data-path latticelabs/deepcad \
        --max-samples 256 --epochs 1 --save checkpoints/vqvae.pt
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

from ll_gen.generators.base import BaseNeuralGenerator
from ll_gen.training.rl_trainer import RLAlignmentTrainer

_log = logging.getLogger(__name__)

_DEFAULT_PROMPT = "Generate a valid CAD model."


def build_generator(name: str, device: str) -> BaseNeuralGenerator:
    """Instantiate a neural generator by name and initialize its model."""
    name = name.lower()
    if name == "vae":
        from ll_gen.generators.neural_vae import NeuralVAEGenerator

        generator: BaseNeuralGenerator = NeuralVAEGenerator(device=device)
    elif name == "vqvae":
        from ll_gen.generators.neural_vqvae import NeuralVQVAEGenerator

        generator = NeuralVQVAEGenerator(device=device)
    elif name == "diffusion":
        from ll_gen.generators.neural_diffusion import NeuralDiffusionGenerator

        generator = NeuralDiffusionGenerator(device=device)
    else:
        raise ValueError(
            f"Unknown generator {name!r}; expected one of vae / vqvae / diffusion."
        )

    # The trainer requires generator._model to be set before the first step.
    generator._init_model()
    return generator


def _sample_to_record(sample: Any, idx: int) -> dict[str, Any]:
    """Adapt a dataset sample into a {prompt, target_dimensions?} record."""
    if isinstance(sample, dict):
        prompt = sample.get("prompt") or sample.get("caption") or _DEFAULT_PROMPT
        record: dict[str, Any] = {"prompt": str(prompt)}
        if sample.get("target_dimensions") is not None:
            record["target_dimensions"] = tuple(sample["target_dimensions"])
        return record
    return {"prompt": _DEFAULT_PROMPT}


def load_dataset_records(
    prompts_file: str | None,
    dataset: str | None,
    data_path: str | None,
    max_samples: int | None,
) -> list[dict[str, Any]]:
    """Build the list of training records from a prompts file or a CAD dataset."""
    if prompts_file:
        records = []
        with open(prompts_file) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                records.append(_sample_to_record(obj, len(records)))
                if max_samples is not None and len(records) >= max_samples:
                    break
        if not records:
            raise ValueError(f"No prompt records found in {prompts_file!r}")
        return records

    if dataset == "deepcad":
        from ll_gen.datasets.deepcad_loader import load_deepcad

        data = load_deepcad(path=data_path or "latticelabs/deepcad", max_samples=max_samples)
    elif dataset == "abc":
        from ll_gen.datasets.abc_loader import load_abc

        data = load_abc(path=data_path or "latticelabs/abc", max_samples=max_samples)
    else:
        raise ValueError("Provide either --prompts-file or --dataset {deepcad,abc}.")

    records = [_sample_to_record(data[i], i) for i in range(len(data))]
    if max_samples is not None:
        records = records[:max_samples]
    return records


def train(
    generator_name: str,
    records: list[dict[str, Any]],
    epochs: int,
    learning_rate: float,
    device: str,
    save_path: str | None,
    resume_path: str | None,
    seed: int | None,
    output_dir: str,
) -> dict[str, float]:
    """Run the RL training loop and optionally save a checkpoint."""
    generator = build_generator(generator_name, device)
    trainer = RLAlignmentTrainer(
        generator,
        learning_rate=learning_rate,
        device=device,
        output_dir=output_dir,
        seed=seed,
    )
    trainer._init_training()
    if resume_path:
        trainer.load_checkpoint(Path(resume_path))

    metrics: dict[str, float] = {}
    for epoch in range(epochs):
        metrics = trainer.train_epoch(records)
        _log.info("Epoch %d/%d: %s", epoch + 1, epochs, metrics)

    if save_path:
        trainer.save_checkpoint(Path(save_path))
        _log.info("Saved checkpoint to %s", save_path)

    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="ll_gen RL alignment training.")
    parser.add_argument("--generator", required=True, choices=["vae", "vqvae", "diffusion"])
    parser.add_argument("--prompts-file", default=None, help="JSONL file of {prompt,...} records.")
    parser.add_argument("--dataset", default=None, choices=["deepcad", "abc"])
    parser.add_argument("--data-path", default=None, help="Dataset path/HF id.")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--save", default=None, help="Checkpoint output path.")
    parser.add_argument("--resume", default=None, help="Checkpoint to resume from.")
    parser.add_argument("--output-dir", default="training_output")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))

    records = load_dataset_records(
        args.prompts_file, args.dataset, args.data_path, args.max_samples
    )
    _log.info("Loaded %d training records", len(records))

    metrics = train(
        generator_name=args.generator,
        records=records,
        epochs=args.epochs,
        learning_rate=args.lr,
        device=args.device,
        save_path=args.save,
        resume_path=args.resume,
        seed=args.seed,
        output_dir=args.output_dir,
    )
    print(json.dumps(metrics))


if __name__ == "__main__":
    main()
