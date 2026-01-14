"""HuggingFace dataset builder for MFCAD++ dataset.

Creates a HuggingFace dataset from local MFCAD++ STEP files and annotations.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Iterator

logger = logging.getLogger(__name__)

try:
    import datasets
    from datasets import (
        GeneratorBasedBuilder,
        Features,
        Value,
        Sequence,
        DatasetInfo,
        SplitGenerator,
        Split,
    )
    HF_DATASETS_AVAILABLE = True
except ImportError:
    HF_DATASETS_AVAILABLE = False
    logger.warning("HuggingFace datasets not available. Install: pip install datasets")


class MFCADDatasetBuilder(GeneratorBasedBuilder):
    """MFCAD++ dataset builder for HuggingFace.

    Dataset Structure:
    - STEP files with B-Rep geometry
    - JSON annotations with face-level labels
    - 24 manufacturing feature classes
    - 15,488 total samples (12,390 train / 1,549 val / 1,549 test)

    Usage:
        >>> from datasets import load_dataset
        >>>
        >>> # Load from builder script
        >>> dataset = load_dataset(
        ...     "path/to/mfcad_builder.py",
        ...     data_dir="/path/to/MFCAD",
        ...     streaming=True
        ... )
        >>>
        >>> for sample in dataset['train'].take(10):
        ...     step_content = sample['step_content']
        ...     face_labels = sample['faces']
    """

    VERSION = datasets.Version("1.0.0")

    BUILDER_CONFIGS = [
        datasets.BuilderConfig(
            name="default",
            version=VERSION,
            description="MFCAD++ manufacturing feature recognition dataset",
        )
    ]

    # Feature classes (24 total)
    FEATURE_CLASSES = [
        "base", "stock", "boss", "rib", "protrusion", "circular_boss",
        "rectangular_boss", "hex_boss", "pocket", "hole", "slot",
        "chamfer", "fillet", "groove", "through_hole", "blind_hole",
        "countersink", "counterbore", "round_pocket", "rectangular_pocket",
        "thread", "keyway", "dovetail", "t_slot", "o_ring_groove"
    ]

    def _info(self) -> DatasetInfo:
        """Dataset metadata and feature schema."""
        return DatasetInfo(
            description=(
                "MFCAD++ manufacturing feature recognition dataset. "
                "Contains 15,488 STEP files with face-level annotations "
                "for 24 manufacturing feature classes."
            ),
            features=Features({
                "file_name": Value("string"),
                "step_content": Value("string"),
                "faces": Sequence({
                    "face_id": Value("int32"),
                    "label": Value("string"),
                    "label_id": Value("int32"),
                    "instance_id": Value("int32"),
                    "is_bottom_face": Value("bool"),
                    "parameters": {
                        "diameter": Value("float32"),
                        "depth": Value("float32"),
                        "radius": Value("float32"),
                        "width": Value("float32"),
                        "length": Value("float32"),
                        "angle": Value("float32"),
                    },
                }),
                "num_faces": Value("int32"),
                "num_instances": Value("int32"),
            }),
            supervised_keys=("step_content", "faces"),
            homepage="https://github.com/hducg/MFCAD",
            citation=(
                "@article{MFCAD2023,\n"
                "  title={MFCAD: Towards a Benchmark for Manufacturing Feature Recognition},\n"
                "  author={...},\n"
                "  journal={...},\n"
                "  year={2023}\n"
                "}"
            ),
        )

    def _split_generators(self, dl_manager) -> list[SplitGenerator]:
        """Define dataset splits.

        Expects data_dir to contain:
        - train/ - Training STEP files and annotations
        - val/ - Validation STEP files and annotations
        - test/ - Test STEP files and annotations
        """
        # Get data directory from config
        data_dir = Path(self.config.data_dir) if hasattr(self.config, 'data_dir') else Path("data/MFCAD")

        if not data_dir.exists():
            raise FileNotFoundError(
                f"Data directory not found: {data_dir}\n"
                f"Please download MFCAD++ dataset and specify data_dir"
            )

        return [
            SplitGenerator(
                name=Split.TRAIN,
                gen_kwargs={"data_dir": data_dir / "train"},
            ),
            SplitGenerator(
                name=Split.VALIDATION,
                gen_kwargs={"data_dir": data_dir / "val"},
            ),
            SplitGenerator(
                name=Split.TEST,
                gen_kwargs={"data_dir": data_dir / "test"},
            ),
        ]

    def _generate_examples(self, data_dir: Path) -> Iterator[tuple[int, dict[str, Any]]]:
        """Generate dataset examples from data directory.

        Args:
            data_dir: Directory containing STEP files and JSON annotations

        Yields:
            (idx, sample) tuples
        """
        data_dir = Path(data_dir)

        # Find all annotation files
        annotation_files = sorted(data_dir.glob("*.json"))

        if not annotation_files:
            logger.warning(f"No annotation files found in {data_dir}")
            return

        logger.info(f"Found {len(annotation_files)} annotation files in {data_dir}")

        for idx, ann_file in enumerate(annotation_files):
            try:
                # Load annotation
                with open(ann_file, "r") as f:
                    annotation = json.load(f)

                # Load corresponding STEP file
                step_file_name = annotation.get("file_name", ann_file.stem + ".step")
                step_file = data_dir / step_file_name

                if not step_file.exists():
                    logger.warning(f"STEP file not found: {step_file}")
                    continue

                # Read STEP content
                with open(step_file, "r", encoding="utf-8", errors="ignore") as f:
                    step_content = f.read()

                # Process face annotations
                faces = []
                instance_ids = set()

                for face_ann in annotation.get("faces", []):
                    label_name = face_ann.get("label", "base")
                    label_id = (
                        self.FEATURE_CLASSES.index(label_name)
                        if label_name in self.FEATURE_CLASSES
                        else 0
                    )

                    instance_id = face_ann.get("instance_id", 0)
                    instance_ids.add(instance_id)

                    # Extract parameters (with defaults)
                    params = face_ann.get("parameters", {})

                    faces.append({
                        "face_id": face_ann.get("face_id", 0),
                        "label": label_name,
                        "label_id": label_id,
                        "instance_id": instance_id,
                        "is_bottom_face": face_ann.get("is_bottom_face", False),
                        "parameters": {
                            "diameter": params.get("diameter", 0.0),
                            "depth": params.get("depth", 0.0),
                            "radius": params.get("radius", 0.0),
                            "width": params.get("width", 0.0),
                            "length": params.get("length", 0.0),
                            "angle": params.get("angle", 0.0),
                        },
                    })

                # Create sample
                sample = {
                    "file_name": step_file_name,
                    "step_content": step_content,
                    "faces": faces,
                    "num_faces": len(faces),
                    "num_instances": len(instance_ids),
                }

                yield idx, sample

            except Exception as e:
                logger.error(f"Error processing {ann_file}: {e}")
                continue


# Example usage
if __name__ == "__main__":
    # Build dataset
    from datasets import load_dataset

    # Load from builder
    dataset = load_dataset(
        __file__,
        data_dir="/path/to/MFCAD",  # Update this path
        streaming=True,
        split="train",
    )

    # Print first sample
    for sample in dataset.take(1):
        print(f"File: {sample['file_name']}")
        print(f"Faces: {sample['num_faces']}")
        print(f"Instances: {sample['num_instances']}")
        print(f"STEP content length: {len(sample['step_content'])} chars")
        print(f"Face labels: {[f['label'] for f in sample['faces'][:5]]}")
