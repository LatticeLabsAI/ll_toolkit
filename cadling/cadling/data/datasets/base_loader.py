"""Base class for CAD research dataset loaders."""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Callable, Iterator, Optional

_log = logging.getLogger(__name__)


class BaseCADDataset(ABC):
    """Abstract base class for CAD dataset loaders.

    Provides a common interface for loading, caching, and iterating
    over CAD research datasets. Subclasses implement format-specific
    parsing and preprocessing.

    Supports two modes:
    1. Local mode: Load from local files in root_dir
    2. Hub mode: Stream from HuggingFace Hub using hub_id

    Args:
        root_dir: Root directory for dataset storage (optional if hub_id provided).
        split: Dataset split ('train', 'val', 'test').
        transform: Optional transform applied to each sample.
        download: Whether to download the dataset if not present.
        hub_id: HuggingFace Hub dataset ID (e.g., "latticelabs/deepcad").
        streaming: Whether to stream from Hub (default True for Hub mode).
    """

    def __init__(
        self,
        root_dir: Optional[str] = None,
        split: str = "train",
        transform: Optional[Callable] = None,
        download: bool = False,
        hub_id: Optional[str] = None,
        streaming: bool = True,
    ) -> None:
        self.split = split
        self.transform = transform
        self.hub_id = hub_id
        self.streaming = streaming
        self._use_hub = hub_id is not None
        self._hf_dataset = None
        self._hf_iterator = None

        if self._use_hub:
            # Hub mode - root_dir is optional (for caching)
            self.root_dir = Path(root_dir) if root_dir else None
            _log.info(
                "Initializing %s in Hub mode: %s (split=%s, streaming=%s)",
                self.__class__.__name__, hub_id, split, streaming,
            )
            self._init_hub_dataset()
        else:
            # Local mode - root_dir is required
            if root_dir is None:
                raise ValueError("root_dir is required when hub_id is not provided")
            self.root_dir = Path(root_dir)
            self.root_dir.mkdir(parents=True, exist_ok=True)

            if download and not self._verify_integrity():
                _log.info("Dataset not found at %s, downloading...", self.root_dir)
                self.download()

            if self._verify_integrity():
                _log.info(
                    "Loaded %s split '%s' from %s (%d samples)",
                    self.__class__.__name__, split, self.root_dir, len(self),
                )
            else:
                _log.warning(
                    "Dataset %s not found at %s. Call download() or set download=True.",
                    self.__class__.__name__, self.root_dir,
                )

    def _init_hub_dataset(self) -> None:
        """Initialize HuggingFace Hub dataset."""
        try:
            from datasets import load_dataset

            load_kwargs = {
                "path": self.hub_id,
                "split": self.split,
                "streaming": self.streaming,
            }

            if self.root_dir:
                load_kwargs["cache_dir"] = str(self.root_dir / "cache")

            self._hf_dataset = load_dataset(**load_kwargs)
            _log.info("Loaded Hub dataset: %s", self.hub_id)

        except ImportError:
            raise ImportError(
                "datasets library required for Hub mode. "
                "Install via: pip install datasets>=2.16.0"
            )
        except Exception as e:
            _log.error("Failed to load Hub dataset %s: %s", self.hub_id, e)
            raise

    def __iter__(self) -> Iterator[dict[str, Any]]:
        """Iterate over samples (works for both local and Hub mode)."""
        if self._use_hub:
            for sample in self._hf_dataset:
                processed = self._process_hub_sample(sample)
                if self.transform:
                    processed = self.transform(processed)
                yield processed
        else:
            for i in range(len(self)):
                yield self[i]

    def _process_hub_sample(self, sample: dict[str, Any]) -> dict[str, Any]:
        """Process a sample from Hub format to loader format.

        Override in subclasses for dataset-specific processing.
        Default implementation returns sample as-is.
        """
        return sample

    @abstractmethod
    def __len__(self) -> int:
        """Return the total number of samples in this split."""

    @abstractmethod
    def __getitem__(self, idx: int) -> dict[str, Any]:
        """Return a single sample as a dictionary.

        The dict should contain at minimum a representation of the CAD
        model (tokens, graph, etc.) suitable for the target task.
        """

    @abstractmethod
    def download(self) -> None:
        """Download and prepare the dataset."""

    @property
    def _cache_dir(self) -> Path:
        """Directory for cached preprocessed data."""
        cache = self.root_dir / "cache" / self.split
        cache.mkdir(parents=True, exist_ok=True)
        return cache

    def _verify_integrity(self) -> bool:
        """Check whether the dataset is present and intact.

        Default implementation checks for a split-specific marker file.
        Override for dataset-specific checks.
        """
        marker = self.root_dir / f".{self.split}_ready"
        return marker.exists()

    def _mark_ready(self) -> None:
        """Create a marker file indicating the split is prepared."""
        marker = self.root_dir / f".{self.split}_ready"
        marker.touch()
