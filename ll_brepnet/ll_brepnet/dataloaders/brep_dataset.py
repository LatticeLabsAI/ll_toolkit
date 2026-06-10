"""Dataset, batch container and collation for BRepNet ``.npz`` records.

A :class:`BRepDataset` reads the per-solid ``.npz`` records produced by
:mod:`ll_brepnet.pipelines.extract_brepnet_data_from_step`, applies the
training-set feature standardization from the dataset manifest, attaches the
per-face segment labels (``<stem>.seg``), and returns per-solid tensors.

:func:`brep_collate_fn` packs several solids into one disjoint graph by
concatenating their entity tensors and offsetting every index array
(``coedge_to_*``) so the references stay valid in the merged batch. The
:class:`BRepBatch` it returns keeps a ``split_batch`` so per-solid faces can be
recovered after the model runs (e.g. to write one logits file per solid).

The per-coedge *input features* the message-passing model consumes are not
materialised here -- the model gathers them from ``face_features`` /
``edge_features`` via ``coedge_to_face`` / ``coedge_to_edge`` and fuses the
UV-grid geometry, so the dataset stays geometry/representation agnostic.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset

from .max_num_faces_loader import MaxNumFacesSampler

_log = logging.getLogger(__name__)

#: Label value for faces without a ground-truth segment (ignored by the loss).
IGNORE_INDEX = -1

_SPLIT_KEYS = {
    "training_set": "training_set",
    "train": "training_set",
    "validation_set": "validation_set",
    "val": "validation_set",
    "test_set": "test_set",
    "test": "test_set",
}


@dataclass
class BRepBatch:
    """A batch of solids merged into one disjoint coedge graph.

    All index arrays are already offset into the concatenated entity tensors.
    """

    face_features: torch.Tensor  # [F, Df]
    face_point_grids: torch.Tensor  # [F, 7, U, V]
    edge_features: torch.Tensor  # [E, De]
    edge_point_grids: torch.Tensor  # [E, 6, U]
    coedge_to_next: torch.Tensor  # [C] -> [0, C)
    coedge_to_prev: torch.Tensor  # [C] -> [0, C)
    coedge_to_mate: torch.Tensor  # [C] -> [0, C)
    coedge_to_face: torch.Tensor  # [C] -> [0, F)
    coedge_to_edge: torch.Tensor  # [C] -> [0, E)
    coedge_reversed: torch.Tensor  # [C, 1]
    labels: torch.Tensor  # [F]
    face_batch_index: torch.Tensor  # [F] -> solid index within the batch
    split_batch: list[tuple[int, int]]  # per-solid (face_start, face_end)
    file_stems: list[str]

    def to(self, device: torch.device | str) -> BRepBatch:
        """Move every tensor field to ``device`` (returns ``self``)."""
        for name in (
            "face_features",
            "face_point_grids",
            "edge_features",
            "edge_point_grids",
            "coedge_to_next",
            "coedge_to_prev",
            "coedge_to_mate",
            "coedge_to_face",
            "coedge_to_edge",
            "coedge_reversed",
            "labels",
            "face_batch_index",
        ):
            setattr(self, name, getattr(self, name).to(device))
        return self

    @property
    def num_solids(self) -> int:
        return len(self.file_stems)

    @property
    def num_faces(self) -> int:
        return int(self.face_features.shape[0])

    @property
    def num_coedges(self) -> int:
        return int(self.coedge_to_next.shape[0])


class BRepDataset(Dataset):
    """A split of a BRepNet dataset.

    Args:
        dataset_file: Path to the ``dataset.json`` manifest.
        dataset_dir: Directory containing the ``<stem>.npz`` records.
        split: One of ``training_set`` / ``validation_set`` / ``test_set``
            (aliases ``train`` / ``val`` / ``test`` accepted).
        label_dir: Directory with ``<stem>.seg`` per-face label files. Defaults
            to ``dataset_dir``. Missing label files yield ``IGNORE_INDEX``
            labels (so inference-only data is supported).
        standardize: Apply the manifest's per-column z-scoring to face/edge
            features.
    """

    def __init__(
        self,
        dataset_file: Path | str,
        dataset_dir: Path | str,
        split: str,
        label_dir: Path | str | None = None,
        standardize: bool = True,
    ):
        self.dataset_dir = Path(dataset_dir)
        self.label_dir = Path(label_dir) if label_dir is not None else self.dataset_dir
        self.standardize = standardize

        manifest = json.loads(Path(dataset_file).read_text())
        split_key = _SPLIT_KEYS.get(split)
        if split_key is None:
            raise ValueError(f"Unknown split {split!r}; expected one of {set(_SPLIT_KEYS)}")
        self.split = split_key
        self.file_stems: list[str] = list(manifest.get(split_key, []))

        self.num_classes: int | None = manifest.get("num_classes")
        self.class_names: list[str] = manifest.get("class_names", [])

        self._mean_std = self._build_standardization(manifest.get("feature_standardization", {}))
        self._num_faces_cache: dict[int, int] = {}

    @staticmethod
    def _build_standardization(
        std_block: dict,
    ) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        out: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for key, cols in std_block.items():
            if not cols:
                continue
            mean = np.array([c["mean"] for c in cols], dtype=np.float32)
            std = np.array([c["standard_deviation"] for c in cols], dtype=np.float32)
            std[std < 1e-6] = 1.0
            out[key] = (mean, std)
        return out

    def __len__(self) -> int:
        return len(self.file_stems)

    def _apply_std(self, key: str, arr: np.ndarray) -> np.ndarray:
        if self.standardize and key in self._mean_std and arr.shape[0] > 0:
            mean, std = self._mean_std[key]
            if arr.shape[1] == mean.shape[0]:
                return (arr - mean) / std
        return arr

    def _load_labels(self, stem: str, num_faces: int) -> np.ndarray:
        label_path = self.label_dir / f"{stem}.seg"
        if not label_path.exists():
            return np.full(num_faces, IGNORE_INDEX, dtype=np.int64)
        labels = np.loadtxt(label_path, dtype=np.int64, ndmin=1)
        if labels.shape[0] != num_faces:
            raise ValueError(f"Label count {labels.shape[0]} != num_faces {num_faces} for {stem}")
        return labels.astype(np.int64)

    def get_num_faces(self, idx: int) -> int:
        """Number of faces in solid ``idx`` (cached; reads only metadata)."""
        if idx not in self._num_faces_cache:
            stem = self.file_stems[idx]
            with np.load(self.dataset_dir / f"{stem}.npz") as data:
                self._num_faces_cache[idx] = int(data["num_faces"])
        return self._num_faces_cache[idx]

    def __getitem__(self, idx: int) -> dict:
        stem = self.file_stems[idx]
        with np.load(self.dataset_dir / f"{stem}.npz") as data:
            face_features = self._apply_std(
                "face_features", data["face_features"].astype(np.float32)
            )
            edge_features = self._apply_std(
                "edge_features", data["edge_features"].astype(np.float32)
            )
            num_faces = int(data["num_faces"])
            self._num_faces_cache[idx] = num_faces
            sample = {
                "face_features": torch.from_numpy(np.ascontiguousarray(face_features)),
                "face_point_grids": torch.from_numpy(
                    np.ascontiguousarray(data["face_point_grids"].astype(np.float32))
                ),
                "edge_features": torch.from_numpy(np.ascontiguousarray(edge_features)),
                "edge_point_grids": torch.from_numpy(
                    np.ascontiguousarray(data["edge_point_grids"].astype(np.float32))
                ),
                "coedge_to_next": torch.from_numpy(data["coedge_to_next"].astype(np.int64)),
                "coedge_to_prev": torch.from_numpy(data["coedge_to_prev"].astype(np.int64)),
                "coedge_to_mate": torch.from_numpy(data["coedge_to_mate"].astype(np.int64)),
                "coedge_to_face": torch.from_numpy(data["coedge_to_face"].astype(np.int64)),
                "coedge_to_edge": torch.from_numpy(data["coedge_to_edge"].astype(np.int64)),
                "coedge_reversed": torch.from_numpy(
                    data["coedge_reversed"].astype(np.float32)
                ).unsqueeze(1),
                "labels": torch.from_numpy(self._load_labels(stem, num_faces)),
                "file_stem": stem,
            }
        return sample


def brep_collate_fn(samples: list[dict]) -> BRepBatch:
    """Merge per-solid samples into one disjoint-graph :class:`BRepBatch`.

    Each solid's coedge index arrays are shifted by the running entity offsets
    so they keep pointing at the right rows of the concatenated tensors.
    """
    face_features, face_grids = [], []
    edge_features, edge_grids = [], []
    nxt, prv, mate, c2f, c2e, rev = [], [], [], [], [], []
    labels, face_batch_index = [], []
    split_batch: list[tuple[int, int]] = []
    file_stems: list[str] = []

    face_offset = edge_offset = coedge_offset = 0
    for solid_idx, s in enumerate(samples):
        nf = int(s["face_features"].shape[0])
        ne = int(s["edge_features"].shape[0])
        nc = int(s["coedge_to_next"].shape[0])

        face_features.append(s["face_features"])
        face_grids.append(s["face_point_grids"])
        edge_features.append(s["edge_features"])
        edge_grids.append(s["edge_point_grids"])

        nxt.append(s["coedge_to_next"] + coedge_offset)
        prv.append(s["coedge_to_prev"] + coedge_offset)
        mate.append(s["coedge_to_mate"] + coedge_offset)
        c2f.append(s["coedge_to_face"] + face_offset)
        c2e.append(s["coedge_to_edge"] + edge_offset)
        rev.append(s["coedge_reversed"])

        labels.append(s["labels"])
        face_batch_index.append(torch.full((nf,), solid_idx, dtype=torch.int64))
        split_batch.append((face_offset, face_offset + nf))
        file_stems.append(s["file_stem"])

        face_offset += nf
        edge_offset += ne
        coedge_offset += nc

    return BRepBatch(
        face_features=torch.cat(face_features, dim=0),
        face_point_grids=torch.cat(face_grids, dim=0),
        edge_features=torch.cat(edge_features, dim=0),
        edge_point_grids=torch.cat(edge_grids, dim=0),
        coedge_to_next=torch.cat(nxt, dim=0),
        coedge_to_prev=torch.cat(prv, dim=0),
        coedge_to_mate=torch.cat(mate, dim=0),
        coedge_to_face=torch.cat(c2f, dim=0),
        coedge_to_edge=torch.cat(c2e, dim=0),
        coedge_reversed=torch.cat(rev, dim=0),
        labels=torch.cat(labels, dim=0),
        face_batch_index=torch.cat(face_batch_index, dim=0),
        split_batch=split_batch,
        file_stems=file_stems,
    )


class BRepDataModule(pl.LightningDataModule):
    """LightningDataModule wiring :class:`BRepDataset` + face-count batching.

    Args:
        dataset_file: Path to the ``dataset.json`` manifest.
        dataset_dir: Directory of ``<stem>.npz`` records.
        label_dir: Directory of ``<stem>.seg`` label files (defaults to
            ``dataset_dir``).
        max_num_faces_per_batch: Per-batch face-count cap for the sampler
            (used when ``batch_size`` is ``None``).
        batch_size: Fixed number of solids per batch. When set it overrides the
            face-count sampler.
        num_workers: DataLoader worker processes.
        shuffle_train: Shuffle the training order each epoch.
        standardize: Apply the manifest's feature standardization.
        seed: Base seed for the training shuffle.
    """

    def __init__(
        self,
        dataset_file: Path | str,
        dataset_dir: Path | str,
        label_dir: Path | str | None = None,
        max_num_faces_per_batch: int = 4096,
        batch_size: int | None = None,
        num_workers: int = 0,
        shuffle_train: bool = True,
        standardize: bool = True,
        seed: int = 0,
    ):
        super().__init__()
        self.dataset_file = dataset_file
        self.dataset_dir = dataset_dir
        self.label_dir = label_dir
        self.max_num_faces_per_batch = max_num_faces_per_batch
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.shuffle_train = shuffle_train
        self.standardize = standardize
        self.seed = seed

        self.train_dataset: BRepDataset | None = None
        self.val_dataset: BRepDataset | None = None
        self.test_dataset: BRepDataset | None = None
        self.num_classes: int | None = None
        self.class_names: list[str] = []

    def setup(self, stage: str | None = None) -> None:
        common = {
            "dataset_file": self.dataset_file,
            "dataset_dir": self.dataset_dir,
            "label_dir": self.label_dir,
            "standardize": self.standardize,
        }
        self.train_dataset = BRepDataset(split="training_set", **common)
        self.val_dataset = BRepDataset(split="validation_set", **common)
        self.test_dataset = BRepDataset(split="test_set", **common)
        self.num_classes = self.train_dataset.num_classes
        self.class_names = self.train_dataset.class_names

    def _loader(self, dataset: BRepDataset | None, shuffle: bool) -> DataLoader | None:
        if dataset is None or len(dataset) == 0:
            return None
        if self.batch_size is not None:
            return DataLoader(
                dataset,
                batch_size=self.batch_size,
                shuffle=shuffle,
                collate_fn=brep_collate_fn,
                num_workers=self.num_workers,
            )
        sampler = MaxNumFacesSampler(
            dataset,
            max_num_faces_per_batch=self.max_num_faces_per_batch,
            shuffle=shuffle,
            seed=self.seed,
        )
        return DataLoader(
            dataset,
            batch_sampler=sampler,
            collate_fn=brep_collate_fn,
            num_workers=self.num_workers,
        )

    def train_dataloader(self) -> DataLoader | None:
        return self._loader(self.train_dataset, self.shuffle_train)

    def val_dataloader(self) -> DataLoader | None:
        return self._loader(self.val_dataset, False)

    def test_dataloader(self) -> DataLoader | None:
        return self._loader(self.test_dataset, False)
