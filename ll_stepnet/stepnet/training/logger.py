"""Logging utilities for training infrastructure.

Provides logging backends for experiment tracking during model training.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

_log = logging.getLogger(__name__)


@dataclass
class WandbLogger:
    """Weights & Biases experiment logger.

    Wraps the wandb library for experiment tracking during training.
    Handles lazy initialization to avoid import overhead when not used.

    Args:
        project: W&B project name.
        entity: W&B entity (team or username).
        name: Run name (auto-generated if None).
        config: Configuration dict to log.
        tags: List of tags for the run.
        notes: Notes/description for the run.
        dir: Local directory for wandb files.
        mode: One of 'online', 'offline', 'disabled'.
    """

    project: str = "stepnet"
    entity: Optional[str] = None
    name: Optional[str] = None
    config: Dict[str, Any] = field(default_factory=dict)
    tags: list[str] = field(default_factory=list)
    notes: Optional[str] = None
    dir: Optional[str] = None
    mode: str = "online"

    _run: Any = field(default=None, init=False, repr=False)
    _wandb: Any = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        """Initialize the logger (lazy - doesn't connect until init() called)."""
        pass

    def init(self) -> None:
        """Initialize W&B run. Call this before logging."""
        try:
            import wandb

            self._wandb = wandb
            self._run = wandb.init(
                project=self.project,
                entity=self.entity,
                name=self.name,
                config=self.config,
                tags=self.tags,
                notes=self.notes,
                dir=self.dir,
                mode=self.mode,
            )
            _log.info("Initialized W&B run: %s", self._run.name if self._run else "None")
        except ImportError:
            _log.warning("wandb not installed, logging disabled")
            self._wandb = None
            self._run = None
        except Exception as e:
            _log.error("Failed to initialize W&B: %s", e)
            self._wandb = None
            self._run = None

    def log(self, metrics: Dict[str, Any], step: Optional[int] = None) -> None:
        """Log metrics to W&B.

        Args:
            metrics: Dictionary of metric name -> value.
            step: Global step number (optional).
        """
        if self._wandb is None or self._run is None:
            return

        self._wandb.log(metrics, step=step)

    def log_image(
        self,
        key: str,
        image: Any,
        caption: Optional[str] = None,
        step: Optional[int] = None,
    ) -> None:
        """Log an image to W&B.

        Args:
            key: Metric key for the image.
            image: Image data (numpy array, PIL Image, or path).
            caption: Optional caption.
            step: Global step number.
        """
        if self._wandb is None or self._run is None:
            return

        if isinstance(image, (str, Path)):
            img = self._wandb.Image(str(image), caption=caption)
        else:
            img = self._wandb.Image(image, caption=caption)

        self._wandb.log({key: img}, step=step)

    def log_artifact(
        self,
        name: str,
        artifact_type: str,
        path: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Log an artifact (model checkpoint, dataset, etc.) to W&B.

        Args:
            name: Artifact name.
            artifact_type: Type (e.g., 'model', 'dataset').
            path: Path to file or directory.
            metadata: Optional metadata dict.
        """
        if self._wandb is None or self._run is None:
            return

        artifact = self._wandb.Artifact(name, type=artifact_type, metadata=metadata)
        artifact.add_file(path) if Path(path).is_file() else artifact.add_dir(path)
        self._run.log_artifact(artifact)

    def watch(self, model: Any, log: str = "gradients", log_freq: int = 100) -> None:
        """Watch a model for gradient/parameter logging.

        Args:
            model: PyTorch model to watch.
            log: What to log ('gradients', 'parameters', 'all').
            log_freq: Logging frequency.
        """
        if self._wandb is None or self._run is None:
            return

        self._wandb.watch(model, log=log, log_freq=log_freq)

    def finish(self) -> None:
        """Finish the W&B run."""
        if self._run is not None:
            self._run.finish()
            _log.info("Finished W&B run")

    @property
    def run(self) -> Any:
        """Get the underlying W&B run object."""
        return self._run

    def __enter__(self) -> "WandbLogger":
        """Context manager entry."""
        self.init()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Context manager exit."""
        self.finish()


@dataclass
class TensorBoardLogger:
    """TensorBoard experiment logger.

    Alternative to W&B for local experiment tracking.

    Args:
        log_dir: Directory for TensorBoard logs.
        comment: Comment appended to log directory.
    """

    log_dir: str = "runs"
    comment: str = ""

    _writer: Any = field(default=None, init=False, repr=False)

    def init(self) -> None:
        """Initialize TensorBoard writer."""
        try:
            from torch.utils.tensorboard import SummaryWriter

            self._writer = SummaryWriter(log_dir=self.log_dir, comment=self.comment)
            _log.info("Initialized TensorBoard at: %s", self.log_dir)
        except ImportError:
            _log.warning("tensorboard not installed, logging disabled")
            self._writer = None

    def log(self, metrics: Dict[str, Any], step: int) -> None:
        """Log scalar metrics."""
        if self._writer is None:
            return

        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                self._writer.add_scalar(key, value, step)

    def finish(self) -> None:
        """Close the writer."""
        if self._writer is not None:
            self._writer.close()

    def __enter__(self) -> "TensorBoardLogger":
        """Context manager entry."""
        self.init()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Context manager exit."""
        self.finish()
