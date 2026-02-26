"""Shared utilities for streaming trainers.

Extracts common functionality used across StreamingVAETrainer,
StreamingDiffusionTrainer, and StreamingGANTrainer to reduce
code duplication.
"""
from __future__ import annotations

import logging
import math
from typing import Any

_log = logging.getLogger(__name__)

try:
    import torch
    from torch.optim.lr_scheduler import LambdaLR

    _has_torch = True
except ImportError:
    _has_torch = False


def build_dataset_from_config(dataset_config: Any) -> Any:
    """Build a streaming dataset from a StreamingCadlingConfig.

    Lazy-imports ``cadling.data.streaming.CADStreamingDataset`` and
    ``cadling.data.streaming.CADStreamingConfig`` to construct a
    streaming data pipeline from the provided ll_stepnet
    ``StreamingCadlingConfig``.

    Args:
        dataset_config: A ``StreamingCadlingConfig`` instance.

    Returns:
        A ``CADStreamingDataset`` ready for iteration.
    """
    try:
        from cadling.data.streaming import (
            CADStreamingDataset,
            CADStreamingConfig,
        )
    except ImportError as exc:
        raise ImportError(
            "cadling is required for dataset_config support. "
            "Install via: pip install cadling"
        ) from exc

    # Map StreamingCadlingConfig fields -> CADStreamingConfig
    cadling_cfg = CADStreamingConfig(
        dataset_id=dataset_config.dataset_id,
        split=dataset_config.split,
        streaming=dataset_config.streaming,
        batch_size=dataset_config.batch_size,
        shuffle=dataset_config.shuffle,
        shuffle_buffer_size=dataset_config.shuffle_buffer_size,
        max_samples=dataset_config.max_samples,
    )
    ds = CADStreamingDataset(cadling_cfg)

    _log.info(
        "Built CADStreamingDataset from config: dataset_id=%s, split=%s",
        dataset_config.dataset_id,
        dataset_config.split,
    )
    return ds


def create_cosine_scheduler(
    optimizer: Any,
    warmup_steps: int,
    total_steps: int,
) -> "LambdaLR":
    """Create a learning rate scheduler with linear warmup and cosine decay.

    Args:
        optimizer: The optimizer to schedule.
        warmup_steps: Number of warmup steps for linear ramp.
        total_steps: Total training steps for cosine decay.

    Returns:
        A ``LambdaLR`` scheduler instance.
    """

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            # Linear warmup
            return step / max(warmup_steps, 1)
        else:
            # Cosine decay
            progress = (step - warmup_steps) / max(
                total_steps - warmup_steps, 1
            )
            return 0.5 * (1.0 + torch.cos(torch.tensor(progress * math.pi)).item())

    return LambdaLR(optimizer, lr_lambda)


__all__ = ["build_dataset_from_config", "create_cosine_scheduler"]
