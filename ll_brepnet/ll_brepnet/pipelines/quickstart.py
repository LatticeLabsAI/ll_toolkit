"""End-to-end dataset preparation for the Fusion 360 Gallery segmentation set.

Given an unzipped ``s2.0.0`` directory (layout: ``breps/step/*.stp``,
``breps/seg/*.seg``, ``train_test.json``, ``segment_names.json``), this:

1. selects the solids to use (the full official split, or a deterministic
   subset of it),
2. extracts each STEP file to a BRepNet ``.npz`` record (in parallel),
3. copies the matching ``.seg`` per-face labels next to the records, and
4. writes a ``dataset.json`` manifest with the train/val/test split, the
   train-only feature standardization, and the official class names.

The result is directly consumable by ``ll_brepnet.train`` and
``ll_brepnet.eval``.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import shutil
from pathlib import Path

from sklearn.model_selection import train_test_split

from .build_dataset_file import compute_standardization
from .extract_brepnet_data_from_step import extract_step_files

_log = logging.getLogger(__name__)


def _resolve_layout(dataset_dir: Path) -> tuple[Path, Path, Path, Path]:
    dataset_dir = Path(dataset_dir)
    step_dir = dataset_dir / "breps" / "step"
    seg_dir = dataset_dir / "breps" / "seg"
    train_test = dataset_dir / "train_test.json"
    segment_names = dataset_dir / "segment_names.json"
    for p in (step_dir, seg_dir, train_test, segment_names):
        if not p.exists():
            raise FileNotFoundError(f"Expected Fusion 360 layout file/dir missing: {p}")
    return step_dir, seg_dir, train_test, segment_names


def _step_path(step_dir: Path, stem: str) -> Path | None:
    for ext in (".stp", ".step", ".STP", ".STEP"):
        p = step_dir / f"{stem}{ext}"
        if p.exists():
            return p
    return None


def prepare_fusion360(
    dataset_dir: Path,
    output_dir: Path,
    num_workers: int = 1,
    train_subset: int | None = None,
    test_subset: int | None = None,
    validation_split: float = 0.2,
    seed: int = 42,
) -> Path:
    """Prepare the dataset and return the path to the written ``dataset.json``.

    Args:
        dataset_dir: Unzipped ``s2.0.0`` directory.
        output_dir: Where to write ``.npz`` records, ``.seg`` labels, and the
            manifest.
        num_workers: Parallel STEP-extraction workers.
        train_subset: If set, randomly use this many solids from the official
            ``train`` split (deterministic from ``seed``); otherwise use all.
        test_subset: Same, for the official ``test`` split.
        validation_split: Fraction of the (selected) training solids held out
            for validation.
        seed: RNG seed for subset sampling and the train/val split.
    """
    step_dir, seg_dir, train_test_json, segment_names_json = _resolve_layout(dataset_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    class_names = json.loads(segment_names_json.read_text())
    split = json.loads(train_test_json.read_text())
    train_all = list(split.get("train", []))
    test_all = list(split.get("test", []))

    rng = random.Random(seed)
    if train_subset is not None and train_subset < len(train_all):
        train_all = rng.sample(train_all, train_subset)
    if test_subset is not None and test_subset < len(test_all):
        test_all = rng.sample(test_all, test_subset)

    selected = train_all + test_all
    step_files = [p for s in selected if (p := _step_path(step_dir, s)) is not None]
    _log.info(
        "Extracting %d STEP files (%d train + %d test) with %d worker(s)",
        len(step_files),
        len(train_all),
        len(test_all),
        num_workers,
    )
    written = extract_step_files(step_files, output_dir, num_workers=num_workers)
    extracted = {p.stem for p in written}
    _log.info("Extracted %d/%d records", len(extracted), len(step_files))

    # Copy the matching per-face labels next to the records.
    n_labels = 0
    for stem in extracted:
        seg = seg_dir / f"{stem}.seg"
        if seg.exists():
            shutil.copy(seg, output_dir / f"{stem}.seg")
            n_labels += 1
    _log.info("Copied %d .seg label files", n_labels)

    # Splits restricted to what actually extracted, preserving the official
    # train/test partition; validation is held out from the training solids.
    train_stems = [s for s in train_all if s in extracted]
    test_stems = sorted(s for s in test_all if s in extracted)
    if validation_split and validation_split > 0.0 and len(train_stems) > 2:
        train_stems, val_stems = train_test_split(
            train_stems, test_size=validation_split, random_state=seed
        )
    else:
        val_stems = []
    train_stems, val_stems = sorted(train_stems), sorted(val_stems)

    standardization = compute_standardization(output_dir, train_stems)

    manifest = {
        "training_set": train_stems,
        "validation_set": val_stems,
        "test_set": test_stems,
        "feature_standardization": standardization,
        "num_classes": len(class_names),
        "class_names": class_names,
    }
    manifest_path = output_dir / "dataset.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    _log.info(
        "Wrote %s: train=%d val=%d test=%d, %d classes",
        manifest_path,
        len(train_stems),
        len(val_stems),
        len(test_stems),
        len(class_names),
    )
    return manifest_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Prepare the Fusion 360 segmentation dataset.")
    parser.add_argument("--dataset-dir", required=True, help="Unzipped s2.0.0 directory")
    parser.add_argument(
        "--output-dir", required=True, help="Output directory for records + manifest"
    )
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--train-subset", type=int, default=None)
    parser.add_argument("--test-subset", type=int, default=None)
    parser.add_argument("--validation-split", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    manifest = prepare_fusion360(
        Path(args.dataset_dir),
        Path(args.output_dir),
        num_workers=args.num_workers,
        train_subset=args.train_subset,
        test_subset=args.test_subset,
        validation_split=args.validation_split,
        seed=args.seed,
    )
    print(f"Wrote manifest: {manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
