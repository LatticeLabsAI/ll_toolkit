"""Batch sampler that bounds the number of faces per batch.

B-Rep solids vary widely in face count, so fixed solid-count batches produce
very uneven memory/compute. :class:`MaxNumFacesSampler` instead greedily packs
solids into batches whose **total face count** stays under a cap, which keeps
the merged coedge graph (and the per-face segmentation head) a roughly constant
size. Solids that individually exceed the cap are logged and skipped rather than
silently dropped or truncated.
"""

from __future__ import annotations

import logging
import random
from collections.abc import Iterator
from typing import Protocol

from torch.utils.data import Sampler

_log = logging.getLogger(__name__)


class _FaceCounted(Protocol):
    """Minimal dataset interface the sampler needs."""

    def __len__(self) -> int: ...

    def get_num_faces(self, idx: int) -> int: ...


def _pack(order: list[int], face_counts: list[int], cap: int) -> list[list[int]]:
    """Greedily pack ``order`` into batches with total face count <= ``cap``."""
    batches: list[list[int]] = []
    batch: list[int] = []
    total = 0
    skipped = 0
    for idx in order:
        nf = face_counts[idx]
        if nf > cap:
            skipped += 1
            continue
        if batch and total + nf > cap:
            batches.append(batch)
            batch, total = [], 0
        batch.append(idx)
        total += nf
    if batch:
        batches.append(batch)
    if skipped:
        _log.warning(
            "MaxNumFacesSampler skipped %d solid(s) exceeding the %d-face cap",
            skipped,
            cap,
        )
    return batches


class MaxNumFacesSampler(Sampler[list[int]]):
    """Yield lists of dataset indices whose total face count is under a cap.

    Args:
        dataset: A dataset exposing ``__len__`` and ``get_num_faces(idx)``.
        max_num_faces_per_batch: Per-batch face-count cap.
        shuffle: Reshuffle solid order each epoch (deterministically from
            ``seed`` + an epoch counter).
        seed: Base RNG seed for shuffling.
    """

    def __init__(
        self,
        dataset: _FaceCounted,
        max_num_faces_per_batch: int = 4096,
        shuffle: bool = True,
        seed: int = 0,
    ):
        self.dataset = dataset
        self.cap = int(max_num_faces_per_batch)
        self.shuffle = shuffle
        self.seed = seed
        self._epoch = 0
        self._face_counts = [dataset.get_num_faces(i) for i in range(len(dataset))]
        # Stable batch-count estimate (natural order) for ``__len__``.
        self._num_batches = len(_pack(list(range(len(dataset))), self._face_counts, self.cap))

    def set_epoch(self, epoch: int) -> None:
        """Set the epoch so each epoch reshuffles differently but reproducibly."""
        self._epoch = epoch

    def __iter__(self) -> Iterator[list[int]]:
        order = list(range(len(self.dataset)))
        if self.shuffle:
            random.Random(self.seed + self._epoch).shuffle(order)
            self._epoch += 1
        yield from _pack(order, self._face_counts, self.cap)

    def __len__(self) -> int:
        return self._num_batches
