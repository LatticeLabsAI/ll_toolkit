"""Supervised warm-start of the STEPVAE on real DeepCAD sequences (SPEC-1 M3, T3.3).

Trains the VAE generator's STEPVAE to *reconstruct* DeepCAD command sequences
before the RL fine-tune (T3.4).  Reconstruction alone does not directly optimize
the M3 gate — which measures validity of shapes decoded from **prior** samples
``z ~ N(0, 1)`` — but it gives the decoder a head start at emitting well-formed
command structures, which the RL stage then aligns toward dispose-validity.

STEPVAE's own ``forward(..., param_targets=...)`` divides by every one of its 16
parameter heads, but real commands use at most 6 slots (ARC), so the unused
heads receive all-ignored targets and poison ``recon_loss`` with NaN.  This
trainer therefore computes the reconstruction loss directly from the returned
logits, masking empty slots.

Sequence representation (per-command, S = number of commands, padded to
``model.max_seq_len``):
    token_ids[s]        = command-type token id of command s (PAD 0)
    command_targets[s]  = command-type class index (token id - 6), -1 for pad
    param_targets[s, j] = quantized level of param j, -1 for inactive/pad

CLI::

    python -m ll_gen.training.warm_start --data-dir data/deepcad_dsl --split train \\
        --epochs 5 --batch-size 32 --max-samples 2000 --save checkpoints/vae_warm.pt
"""

from __future__ import annotations

import argparse
import json
import logging
import random
from pathlib import Path
from typing import Any

from ll_gen.datasets._tokenization import COMMAND_TYPE_IDS

_log = logging.getLogger(__name__)

# Command-type class index = token id - (SOL token id).  STEPVAE's command_head
# emits classes 0..5 for SOL/LINE/ARC/CIRCLE/EXTRUDE/EOS (see base.CMD_TOKEN_MAP).
_CMD_CLASS_OFFSET = COMMAND_TYPE_IDS["SOL"]  # == 6


def build_targets(
    command_tokens: list[dict[str, Any]],
    max_commands: int,
    num_param_slots: int,
    num_command_types: int,
) -> tuple[list[int], list[int], list[int], list[list[int]]]:
    """Build per-command encoder input + reconstruction targets for one sample.

    Args:
        command_tokens: Quantized command dicts from ``DeepCADDataset``.
        max_commands: Sequence length to pad/truncate to (``model.max_seq_len``).
        num_param_slots: Number of parameter heads (16).
        num_command_types: Number of command classes (6).

    Returns:
        ``(token_ids, attention_mask, command_targets, param_targets)`` lists,
        each padded to ``max_commands``.
    """
    token_ids = [0] * max_commands
    attention_mask = [0] * max_commands
    command_targets = [-1] * max_commands
    param_targets = [[-1] * num_param_slots for _ in range(max_commands)]

    for i, cmd in enumerate(command_tokens[:max_commands]):
        type_id = int(cmd["command_type"])
        cls = type_id - _CMD_CLASS_OFFSET
        if cls < 0 or cls >= num_command_types:
            # Unknown/special token — leave as padding.
            continue
        token_ids[i] = type_id
        attention_mask[i] = 1
        command_targets[i] = cls
        for j, level in enumerate(cmd.get("parameters", [])[:num_param_slots]):
            param_targets[i][j] = int(level)

    return token_ids, attention_mask, command_targets, param_targets


def _collate(batch_samples, model, torch):
    """Stack per-sample targets into batched tensors on the model's device."""
    max_commands = model.max_seq_len
    num_param_slots = len(model.param_heads)
    num_command_types = model.num_command_types
    device = next(model.parameters()).device

    tok, amask, cmd_t, par_t = [], [], [], []
    for sample in batch_samples:
        t, a, c, p = build_targets(
            sample["command_tokens"], max_commands, num_param_slots, num_command_types
        )
        tok.append(t)
        amask.append(a)
        cmd_t.append(c)
        par_t.append(p)

    return (
        torch.tensor(tok, dtype=torch.long, device=device),
        torch.tensor(amask, dtype=torch.long, device=device),
        torch.tensor(cmd_t, dtype=torch.long, device=device),
        torch.tensor(par_t, dtype=torch.long, device=device),
    )


def reconstruction_loss(model, outputs, command_targets, param_targets, torch):
    """Masked command + parameter cross-entropy from STEPVAE logits.

    Computes the parameter loss only over slots that have at least one
    non-ignored target in the batch, avoiding the all-ignored NaN that
    STEPVAE.forward's built-in param loss produces.

    Returns:
        ``(recon_loss, cmd_loss, param_loss)`` scalar tensors.
    """
    import torch.nn.functional as functional

    command_logits = outputs["command_logits"]  # [B, S, C]
    cmd_loss = functional.cross_entropy(
        command_logits.reshape(-1, model.num_command_types),
        command_targets.reshape(-1),
        ignore_index=-1,
    )

    param_logits = outputs["param_logits"]  # 16 x [B, S, P]
    param_loss = torch.zeros((), device=command_logits.device)
    active = 0
    for i, head_logits in enumerate(param_logits):
        target = param_targets[..., i].reshape(-1)
        if bool((target >= 0).any()):
            param_loss = param_loss + functional.cross_entropy(
                head_logits.reshape(-1, model.num_param_levels),
                target,
                ignore_index=-1,
            )
            active += 1
    if active:
        param_loss = param_loss / active

    return cmd_loss + param_loss, cmd_loss, param_loss


def _seed_everything(seed: int, torch) -> None:
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass
    torch.manual_seed(seed)


def warm_start(
    generator_name: str = "vae",
    data_dir: str = "data/deepcad_dsl",
    split: str = "train",
    epochs: int = 5,
    batch_size: int = 32,
    learning_rate: float = 1e-3,
    max_samples: int | None = None,
    device: str = "cpu",
    save_path: str | None = None,
    seed: int | None = 0,
    kl_warmup_epochs: int = 3,
) -> dict[str, float]:
    """Run supervised reconstruction warm-start and optionally save a checkpoint.

    Args:
        generator_name: Neural generator to warm-start (currently "vae").
        data_dir: Root of the local DeepCAD subset (``<dir>/<split>/*.json``).
        split: Dataset split directory name.
        epochs: Number of passes over the subset.
        batch_size: Samples per optimizer step.
        learning_rate: Adam learning rate.
        max_samples: Cap on samples loaded (None = all).
        device: Torch device.
        save_path: If set, save the model state_dict here (flat, loadable via
            ``evaluate_validity.load_generator_checkpoint``).
        seed: RNG seed.
        kl_warmup_epochs: Epochs over which the KL weight warms 0 -> kl_weight.

    Returns:
        Final-epoch metrics: ``recon_loss``, ``cmd_loss``, ``param_loss``,
        ``kl_loss``, ``total_loss``, ``num_samples``, ``epochs``.
    """
    import torch

    from ll_gen.datasets.deepcad_loader import DeepCADDataset
    from ll_gen.training.run import build_generator

    if seed is not None:
        _seed_everything(seed, torch)

    generator = build_generator(generator_name, device)
    model = generator._model
    model.set_kl_warmup_epochs(kl_warmup_epochs)
    model.train()

    dataset = DeepCADDataset(
        data_dir=data_dir,
        split=split,
        max_commands=model.max_seq_len,
        max_samples=max_samples,
    )
    n = len(dataset)
    if n == 0:
        raise ValueError(f"No samples found under {data_dir}/{split}")

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    metrics: dict[str, float] = {}
    for epoch in range(epochs):
        model.set_epoch(epoch)
        order = list(range(n))
        random.shuffle(order)

        sums = {"recon": 0.0, "cmd": 0.0, "param": 0.0, "kl": 0.0, "total": 0.0}
        steps = 0
        for start in range(0, n, batch_size):
            idxs = order[start : start + batch_size]
            samples = [dataset[i] for i in idxs]
            tok, amask, cmd_t, par_t = _collate(samples, model, torch)

            outputs = model(tok, attention_mask=amask)
            recon, cmd_loss, param_loss = reconstruction_loss(
                model, outputs, cmd_t, par_t, torch
            )
            kl_loss = outputs["kl_loss"]
            total = recon + model.beta * kl_loss

            optimizer.zero_grad()
            total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            sums["recon"] += recon.item()
            sums["cmd"] += cmd_loss.item()
            sums["param"] += float(param_loss.item())
            sums["kl"] += kl_loss.item()
            sums["total"] += total.item()
            steps += 1

        metrics = {
            "recon_loss": sums["recon"] / steps,
            "cmd_loss": sums["cmd"] / steps,
            "param_loss": sums["param"] / steps,
            "kl_loss": sums["kl"] / steps,
            "total_loss": sums["total"] / steps,
            "num_samples": float(n),
            "epochs": float(epochs),
        }
        _log.info("Warm-start epoch %d/%d: %s", epoch + 1, epochs, metrics)

    if save_path:
        path = Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), path)
        _log.info("Saved warm-start checkpoint to %s", path)

    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="STEPVAE supervised warm-start.")
    parser.add_argument("--generator", default="vae", choices=["vae"])
    parser.add_argument("--data-dir", default="data/deepcad_dsl")
    parser.add_argument("--split", default="train")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--kl-warmup-epochs", type=int, default=3)
    parser.add_argument("--save", default=None)
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))

    metrics = warm_start(
        generator_name=args.generator,
        data_dir=args.data_dir,
        split=args.split,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        max_samples=args.max_samples,
        device=args.device,
        save_path=args.save,
        seed=args.seed,
        kl_warmup_epochs=args.kl_warmup_epochs,
    )
    print(json.dumps(metrics))


if __name__ == "__main__":
    main()
