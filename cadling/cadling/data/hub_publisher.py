"""Hub publishing utilities for CAD datasets.

Provides CADDatasetPublisher for uploading CAD datasets to
HuggingFace Hub with automatic sharding and compression.
"""
from __future__ import annotations

import json
import logging
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Generator, Iterator, List, Optional, Union

_log = logging.getLogger(__name__)

# Lazy imports
_pa = None
_pq = None
_hf_api = None


def _ensure_pyarrow():
    """Lazily import PyArrow."""
    global _pa, _pq
    if _pa is None:
        try:
            import pyarrow as pa
            import pyarrow.parquet as pq
            _pa = pa
            _pq = pq
        except ImportError:
            raise ImportError(
                "PyArrow is required for Hub publishing. "
                "Install via: pip install pyarrow>=14.0.0"
            )
    return _pa, _pq


def _ensure_hf_api():
    """Lazily import HuggingFace Hub API."""
    global _hf_api
    if _hf_api is None:
        try:
            from huggingface_hub import HfApi
            _hf_api = HfApi()
        except ImportError:
            raise ImportError(
                "huggingface-hub is required for Hub publishing. "
                "Install via: pip install huggingface-hub>=0.20.0"
            )
    return _hf_api


@dataclass
class PublishConfig:
    """Configuration for dataset publishing.

    Attributes:
        repo_id: HuggingFace repository ID (e.g., 'username/dataset-name').
        private: Whether to make the repository private.
        token: HuggingFace API token (uses HF_TOKEN env var if None).
        shard_size_mb: Target size per Parquet shard in MB.
        compression: Parquet compression codec ('zstd', 'snappy', 'gzip').
        compression_level: Compression level (1-22 for zstd).
        row_group_size: Rows per row group (affects streaming performance).
        commit_message: Default commit message for uploads.
        create_readme: Whether to auto-generate a dataset card.
        tags: Dataset tags for discoverability.
        license: Dataset license identifier.
    """

    repo_id: str
    private: bool = False
    token: Optional[str] = None
    shard_size_mb: int = 500
    compression: str = "zstd"
    compression_level: int = 3
    row_group_size: int = 10000
    commit_message: str = "Upload CAD dataset"
    create_readme: bool = True
    tags: List[str] = field(default_factory=lambda: ["cad", "3d", "engineering"])
    license: str = "apache-2.0"


class CADDatasetPublisher:
    """Publisher for uploading CAD datasets to HuggingFace Hub.

    Handles automatic sharding, compression, and dataset card generation
    for efficient hosting of large CAD datasets on HuggingFace Hub.

    Features:
    - Automatic Parquet sharding (default 500MB per shard)
    - Zstd compression for optimal size/speed tradeoff
    - Incremental uploads via HuggingFace Hub API
    - Auto-generated dataset cards with usage examples

    Usage:
        >>> publisher = CADDatasetPublisher(
        ...     config=PublishConfig(repo_id="latticelabs/deepcad-sequences")
        ... )
        >>>
        >>> # Publish from a generator
        >>> publisher.publish_from_generator(
        ...     samples=sample_generator(),
        ...     schema=COMMAND_SEQUENCE_SCHEMA(),
        ...     split="train",
        ... )
        >>>
        >>> # Or publish from local files
        >>> publisher.publish_from_directory(
        ...     source_dir="/path/to/parquet/files"
        ... )

    Args:
        config: Publishing configuration.
    """

    def __init__(self, config: PublishConfig) -> None:
        self.config = config
        self._token = config.token or os.environ.get("HF_TOKEN")

    def _get_api(self) -> "HfApi":
        """Get authenticated HuggingFace API instance."""
        api = _ensure_hf_api()
        return api

    def _ensure_repo_exists(self) -> str:
        """Ensure the repository exists, create if needed."""
        api = self._get_api()

        try:
            api.repo_info(
                repo_id=self.config.repo_id,
                repo_type="dataset",
                token=self._token,
            )
            _log.info("Repository exists: %s", self.config.repo_id)
        except Exception:
            _log.info("Creating repository: %s", self.config.repo_id)
            api.create_repo(
                repo_id=self.config.repo_id,
                repo_type="dataset",
                private=self.config.private,
                token=self._token,
            )

        return self.config.repo_id

    def publish_from_generator(
        self,
        samples: Union[Generator[Dict[str, Any], None, None], Iterator[Dict[str, Any]]],
        schema: "pa.Schema",
        split: str = "train",
        total_samples: Optional[int] = None,
    ) -> str:
        """Publish a dataset from a sample generator.

        Streams samples into Parquet shards and uploads to Hub
        incrementally. Memory-efficient for large datasets.

        Args:
            samples: Generator or iterator yielding sample dicts.
            schema: PyArrow schema for the data.
            split: Dataset split name.
            total_samples: Optional total count for progress tracking.

        Returns:
            URL of the published dataset.
        """
        pa, pq = _ensure_pyarrow()
        from .schemas import samples_to_table

        self._ensure_repo_exists()
        api = self._get_api()

        # Target bytes per shard
        target_bytes = self.config.shard_size_mb * 1024 * 1024

        shard_idx = 0
        current_batch: List[Dict[str, Any]] = []
        current_bytes = 0
        total_rows = 0

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)

            for sample in samples:
                current_batch.append(sample)
                # Estimate sample size (rough)
                current_bytes += self._estimate_sample_size(sample)
                total_rows += 1

                # Check if we should write a shard
                if current_bytes >= target_bytes:
                    self._write_and_upload_shard(
                        current_batch, schema, split, shard_idx, tmp_path, api
                    )
                    shard_idx += 1
                    current_batch = []
                    current_bytes = 0
                    _log.info(
                        "Published shard %d (%d total rows so far)",
                        shard_idx - 1, total_rows
                    )

            # Write final shard if any samples remain
            if current_batch:
                self._write_and_upload_shard(
                    current_batch, schema, split, shard_idx, tmp_path, api
                )
                _log.info(
                    "Published final shard %d (%d total rows)",
                    shard_idx, total_rows
                )

        # Generate and upload dataset card
        if self.config.create_readme:
            self.publish_dataset_card(
                num_samples=total_rows,
                splits=[split],
            )

        return f"https://huggingface.co/datasets/{self.config.repo_id}"

    def _estimate_sample_size(self, sample: Dict[str, Any]) -> int:
        """Estimate the size of a sample in bytes."""
        size = 0
        for value in sample.values():
            if value is None:
                size += 4
            elif isinstance(value, str):
                size += len(value.encode("utf-8"))
            elif isinstance(value, bytes):
                size += len(value)
            elif isinstance(value, (list, tuple)):
                # Rough estimate: 4 bytes per element
                size += len(value) * 4
            elif hasattr(value, "nbytes"):
                size += value.nbytes
            else:
                size += 8

        return size

    def _write_and_upload_shard(
        self,
        samples: List[Dict[str, Any]],
        schema: "pa.Schema",
        split: str,
        shard_idx: int,
        tmp_path: Path,
        api: "HfApi",
    ) -> None:
        """Write samples to a Parquet shard and upload to Hub."""
        pa, pq = _ensure_pyarrow()
        from .schemas import samples_to_table

        # Convert to Arrow table
        table = samples_to_table(samples, schema)

        # Determine shard filename
        shard_name = f"{split}-{shard_idx:05d}.parquet"
        local_path = tmp_path / shard_name

        # Write with compression
        pq.write_table(
            table,
            local_path,
            compression=self.config.compression,
            compression_level=self.config.compression_level,
            row_group_size=self.config.row_group_size,
        )

        # Upload to Hub
        api.upload_file(
            path_or_fileobj=str(local_path),
            path_in_repo=f"data/{shard_name}",
            repo_id=self.config.repo_id,
            repo_type="dataset",
            token=self._token,
            commit_message=f"Upload {shard_name}",
        )

        # Clean up local file
        local_path.unlink()

    def publish_from_directory(
        self,
        source_dir: str,
        pattern: str = "*.parquet",
        split_mapping: Optional[Dict[str, str]] = None,
    ) -> str:
        """Publish Parquet files from a local directory.

        Args:
            source_dir: Directory containing Parquet files.
            pattern: Glob pattern for finding files.
            split_mapping: Optional mapping from filename patterns to splits.

        Returns:
            URL of the published dataset.
        """
        self._ensure_repo_exists()
        api = self._get_api()

        source_path = Path(source_dir)
        files = list(source_path.glob(pattern))

        if not files:
            raise ValueError(f"No files matching '{pattern}' found in {source_dir}")

        _log.info("Publishing %d files from %s", len(files), source_dir)

        for file_path in files:
            # Determine target path in repo
            if split_mapping:
                # Try to match filename to a split
                for pattern, split in split_mapping.items():
                    if pattern in file_path.name:
                        repo_path = f"data/{split}/{file_path.name}"
                        break
                else:
                    repo_path = f"data/{file_path.name}"
            else:
                repo_path = f"data/{file_path.name}"

            _log.info("Uploading %s to %s", file_path.name, repo_path)

            api.upload_file(
                path_or_fileobj=str(file_path),
                path_in_repo=repo_path,
                repo_id=self.config.repo_id,
                repo_type="dataset",
                token=self._token,
                commit_message=f"Upload {file_path.name}",
            )

        # Generate dataset card
        if self.config.create_readme:
            self.publish_dataset_card(
                num_files=len(files),
            )

        return f"https://huggingface.co/datasets/{self.config.repo_id}"

    def publish_dataset_card(
        self,
        num_samples: Optional[int] = None,
        num_files: Optional[int] = None,
        splits: Optional[List[str]] = None,
        description: Optional[str] = None,
    ) -> None:
        """Generate and upload a dataset card (README.md).

        Args:
            num_samples: Total number of samples.
            num_files: Number of Parquet files.
            splits: List of available splits.
            description: Custom description text.
        """
        api = self._get_api()

        # Generate YAML frontmatter
        yaml_content = [
            "---",
            "license: " + self.config.license,
            "tags:",
        ]
        for tag in self.config.tags:
            yaml_content.append(f"  - {tag}")

        yaml_content.extend([
            "task_categories:",
            "  - unconditional-generation",
            "  - text-to-3d",
            "size_categories:",
        ])

        if num_samples:
            if num_samples < 1000:
                yaml_content.append("  - n<1K")
            elif num_samples < 10000:
                yaml_content.append("  - 1K<n<10K")
            elif num_samples < 100000:
                yaml_content.append("  - 10K<n<100K")
            elif num_samples < 1000000:
                yaml_content.append("  - 100K<n<1M")
            else:
                yaml_content.append("  - n>1M")
        else:
            yaml_content.append("  - unknown")

        yaml_content.append("---")

        # Generate markdown content
        md_content = [
            "",
            f"# {self.config.repo_id.split('/')[-1]}",
            "",
            description or "A CAD dataset for generative modeling and machine learning research.",
            "",
            "## Dataset Details",
            "",
        ]

        if num_samples:
            md_content.append(f"- **Total samples**: {num_samples:,}")
        if num_files:
            md_content.append(f"- **Parquet files**: {num_files}")
        if splits:
            md_content.append(f"- **Splits**: {', '.join(splits)}")

        md_content.extend([
            "",
            "## Usage",
            "",
            "### Streaming (recommended for large datasets)",
            "",
            "```python",
            "from datasets import load_dataset",
            "",
            f'dataset = load_dataset("{self.config.repo_id}", streaming=True)',
            "",
            "for sample in dataset['train']:",
            "    print(sample['command_types'])",
            "    break",
            "```",
            "",
            "### Download",
            "",
            "```python",
            "from datasets import load_dataset",
            "",
            f'dataset = load_dataset("{self.config.repo_id}")',
            "print(dataset)",
            "```",
            "",
            "### With CADling",
            "",
            "```python",
            "from cadling.data.streaming import CADStreamingDataset, CADStreamingConfig",
            "",
            "config = CADStreamingConfig(",
            f'    dataset_id="{self.config.repo_id}",',
            "    batch_size=8,",
            "    streaming=True,",
            ")",
            "dataset = CADStreamingDataset(config)",
            "",
            "for batch in dataset.batch_iter():",
            "    print(batch)",
            "    break",
            "```",
            "",
            "## Schema",
            "",
            "This dataset contains CAD command sequences with the following fields:",
            "",
            "| Field | Type | Description |",
            "|-------|------|-------------|",
            "| `sample_id` | string | Unique sample identifier |",
            "| `command_types` | list[int] | Command type indices |",
            "| `parameters` | list[int] | Quantized command parameters |",
            "| `mask` | list[float] | Valid command mask |",
            "| `num_commands` | int | Number of valid commands |",
            "",
            "## Citation",
            "",
            "If you use this dataset, please cite the original data sources.",
            "",
            "## License",
            "",
            f"This dataset is released under the {self.config.license} license.",
            "",
        ])

        readme_content = "\n".join(yaml_content + md_content)

        # Upload README
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".md", delete=False
        ) as f:
            f.write(readme_content)
            f.flush()

            api.upload_file(
                path_or_fileobj=f.name,
                path_in_repo="README.md",
                repo_id=self.config.repo_id,
                repo_type="dataset",
                token=self._token,
                commit_message="Update dataset card",
            )

            os.unlink(f.name)

        _log.info("Published dataset card")


def publish_dataset(
    samples: Union[Generator, Iterator, List[Dict[str, Any]]],
    repo_id: str,
    schema: Optional["pa.Schema"] = None,
    split: str = "train",
    private: bool = False,
    token: Optional[str] = None,
    **kwargs: Any,
) -> str:
    """Convenience function to publish a dataset to HuggingFace Hub.

    Args:
        samples: Samples to publish.
        repo_id: HuggingFace repository ID.
        schema: PyArrow schema (will infer if None).
        split: Dataset split name.
        private: Whether to make private.
        token: HuggingFace API token.
        **kwargs: Additional PublishConfig options.

    Returns:
        Dataset URL.
    """
    # Convert list to generator
    if isinstance(samples, list):
        samples = iter(samples)

    # Infer schema if not provided
    if schema is None:
        from .schemas import infer_schema_from_sample

        # Peek at first sample
        first_sample = next(samples, None)
        if first_sample is None:
            raise ValueError("No samples to publish")

        schema = infer_schema_from_sample(first_sample)

        # Re-add first sample to generator
        def _with_first():
            yield first_sample
            yield from samples

        samples = _with_first()

    config = PublishConfig(
        repo_id=repo_id,
        private=private,
        token=token,
        **kwargs,
    )

    publisher = CADDatasetPublisher(config)
    return publisher.publish_from_generator(samples, schema, split)


__all__ = [
    "PublishConfig",
    "CADDatasetPublisher",
    "publish_dataset",
]
