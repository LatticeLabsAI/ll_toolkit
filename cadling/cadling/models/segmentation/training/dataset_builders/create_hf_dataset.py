"""Script to convert local CAD datasets to HuggingFace format.

Usage:
    python create_hf_dataset.py \
        --input_dir /path/to/MFCAD \
        --output_dir ./mfcad_hf \
        --dataset_name mfcad-plus-plus \
        --dataset_type mfcad

Then upload to HuggingFace Hub:
    huggingface-cli upload <username>/mfcad-plus-plus ./mfcad_hf
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    from datasets import Dataset, DatasetDict, Features, Value, Sequence
    from huggingface_hub import HfApi
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False
    logger.error("HuggingFace libraries not available. Install: pip install datasets huggingface_hub")


def create_hf_dataset_from_local(
    input_dir: Path,
    output_dir: Path,
    dataset_type: str = "mfcad",
    dataset_name: Optional[str] = None,
    upload: bool = False,
    hf_username: Optional[str] = None,
) -> None:
    """Convert local CAD dataset to HuggingFace format.

    Args:
        input_dir: Input directory with STEP files and annotations
        output_dir: Output directory for HF dataset
        dataset_type: Dataset type ('mfcad', 'mfinstseg', 'custom')
        dataset_name: Dataset name (for upload)
        upload: Whether to upload to HuggingFace Hub
        hf_username: HuggingFace username (for upload)
    """
    if not HF_AVAILABLE:
        raise ImportError("HuggingFace libraries required. Install: pip install datasets huggingface_hub")

    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    logger.info(f"Converting dataset from {input_dir} to {output_dir}")
    logger.info(f"Dataset type: {dataset_type}")

    # Load dataset using appropriate builder
    if dataset_type == "mfcad":
        from .mfcad_builder import MFCADDatasetBuilder

        # Create dataset from splits
        splits = {}

        for split_name in ["train", "val", "test"]:
            split_dir = input_dir / split_name

            if not split_dir.exists():
                logger.warning(f"Split directory not found: {split_dir}")
                continue

            logger.info(f"Processing {split_name} split...")

            # Generate examples
            examples = []
            builder = MFCADDatasetBuilder()

            for idx, sample in builder._generate_examples(split_dir):
                examples.append(sample)

            if examples:
                splits[split_name] = Dataset.from_list(examples)
                logger.info(f"  {split_name}: {len(examples)} samples")

        if not splits:
            logger.error("No splits created. Check input directory structure.")
            return

        # Create DatasetDict
        dataset_dict = DatasetDict(splits)

    elif dataset_type == "mfinstseg":
        logger.info("MFInstSeg dataset builder not yet implemented")
        logger.info("Using generic builder...")
        dataset_dict = _create_generic_dataset(input_dir)

    elif dataset_type == "custom":
        logger.info("Creating custom dataset...")
        dataset_dict = _create_generic_dataset(input_dir)

    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")

    # Save to disk
    logger.info(f"Saving dataset to {output_dir}")
    dataset_dict.save_to_disk(str(output_dir))
    logger.info(f"Dataset saved successfully!")

    # Print statistics
    for split_name, split_dataset in dataset_dict.items():
        logger.info(f"  {split_name}: {len(split_dataset)} samples")

    # Upload to HuggingFace Hub
    if upload:
        if not dataset_name:
            raise ValueError("dataset_name required for upload")
        if not hf_username:
            raise ValueError("hf_username required for upload")

        repo_id = f"{hf_username}/{dataset_name}"
        logger.info(f"Uploading to HuggingFace Hub: {repo_id}")

        try:
            dataset_dict.push_to_hub(repo_id)
            logger.info(f"Dataset uploaded successfully to {repo_id}")
            logger.info(f"Access with: load_dataset('{repo_id}', streaming=True)")
        except Exception as e:
            logger.error(f"Upload failed: {e}")
            logger.info("You can upload manually with:")
            logger.info(f"  huggingface-cli upload {repo_id} {output_dir}")


def _create_generic_dataset(input_dir: Path) -> DatasetDict:
    """Create generic dataset from STEP files and JSON annotations.

    Args:
        input_dir: Directory with train/val/test subdirectories

    Returns:
        DatasetDict with train/val/test splits
    """
    splits = {}

    for split_name in ["train", "val", "test"]:
        split_dir = input_dir / split_name

        if not split_dir.exists():
            logger.warning(f"Split directory not found: {split_dir}")
            continue

        logger.info(f"Processing {split_name} split...")

        # Find annotation files
        annotation_files = list(split_dir.glob("*.json"))

        if not annotation_files:
            logger.warning(f"No annotation files found in {split_dir}")
            continue

        examples = []

        for ann_file in annotation_files:
            try:
                # Load annotation
                with open(ann_file, "r") as f:
                    annotation = json.load(f)

                # Load STEP file
                step_file_name = annotation.get("file_name", ann_file.stem + ".step")
                step_file = split_dir / step_file_name

                if not step_file.exists():
                    logger.warning(f"STEP file not found: {step_file}")
                    continue

                with open(step_file, "r", encoding="utf-8", errors="ignore") as f:
                    step_content = f.read()

                # Create sample
                sample = {
                    "file_name": step_file_name,
                    "step_content": step_content,
                    "annotation": json.dumps(annotation),
                }

                examples.append(sample)

            except Exception as e:
                logger.error(f"Error processing {ann_file}: {e}")
                continue

        if examples:
            splits[split_name] = Dataset.from_list(examples)
            logger.info(f"  {split_name}: {len(examples)} samples")

    return DatasetDict(splits)


def main():
    """Command-line interface."""
    parser = argparse.ArgumentParser(
        description="Convert local CAD dataset to HuggingFace format"
    )

    parser.add_argument(
        "--input_dir",
        type=Path,
        required=True,
        help="Input directory with STEP files and annotations",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        required=True,
        help="Output directory for HF dataset",
    )
    parser.add_argument(
        "--dataset_type",
        type=str,
        default="mfcad",
        choices=["mfcad", "mfinstseg", "custom"],
        help="Dataset type",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        help="Dataset name (for upload to HF Hub)",
    )
    parser.add_argument(
        "--upload",
        action="store_true",
        help="Upload to HuggingFace Hub after creation",
    )
    parser.add_argument(
        "--hf_username",
        type=str,
        help="HuggingFace username (required for upload)",
    )

    args = parser.parse_args()

    create_hf_dataset_from_local(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        dataset_type=args.dataset_type,
        dataset_name=args.dataset_name,
        upload=args.upload,
        hf_username=args.hf_username,
    )


if __name__ == "__main__":
    main()
