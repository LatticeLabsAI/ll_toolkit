"""Build the dataset manifest (``dataset.json``) for a folder of ``.npz`` records.

The manifest defines the train / validation / test split and the per-feature
standardisation statistics (mean + standard deviation) computed over the
**training split only**. The dataset loader applies these to z-score the
continuous features at load time, so statistics never leak from val/test.

Manifest schema::

    {
      "training_set":   ["<stem>", ...],
      "validation_set": ["<stem>", ...],
      "test_set":       ["<stem>", ...],
      "feature_standardization": {
        "face_features": [{"mean": m, "standard_deviation": s}, ...],
        "edge_features": [{"mean": m, "standard_deviation": s}, ...]
      },
      "num_classes": <int or null>,
      "class_names": [<str>, ...]   # optional
    }
"""

from __future__ import annotations

import argparse
import json
import logging
from collections.abc import Sequence
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split

_log = logging.getLogger(__name__)

# Features whose per-column mean/std the loader will use to z-score.
STANDARDIZED_FEATURES = ("face_features", "edge_features")


def _stems_in(npz_dir: Path) -> list[str]:
    return sorted(p.stem for p in Path(npz_dir).glob("*.npz"))


def compute_standardization(
    npz_dir: Path,
    train_stems: Sequence[str],
    feature_keys: Sequence[str] = STANDARDIZED_FEATURES,
) -> dict[str, list[dict[str, float]]]:
    """Compute per-column mean/std for ``feature_keys`` over the training set.

    Uses a single streaming pass (sum + sum-of-squares) so it scales to large
    datasets. A near-zero standard deviation is floored to 1.0 so the loader's
    division is well-defined for constant columns (e.g. an always-off one-hot).
    """
    npz_dir = Path(npz_dir)
    accum: dict[str, dict[str, np.ndarray]] = {}

    for stem in train_stems:
        path = npz_dir / f"{stem}.npz"
        if not path.exists():
            _log.warning("Skipping missing npz for standardization: %s", path.name)
            continue
        with np.load(path) as data:
            for key in feature_keys:
                if key not in data:
                    continue
                arr = np.asarray(data[key], dtype=np.float64)
                if arr.ndim != 2 or arr.shape[0] == 0:
                    continue
                if key not in accum:
                    ncol = arr.shape[1]
                    accum[key] = {
                        "sum": np.zeros(ncol, dtype=np.float64),
                        "sumsq": np.zeros(ncol, dtype=np.float64),
                        "count": np.zeros(1, dtype=np.float64),
                    }
                accum[key]["sum"] += arr.sum(axis=0)
                accum[key]["sumsq"] += (arr**2).sum(axis=0)
                accum[key]["count"][0] += arr.shape[0]

    standardization: dict[str, list[dict[str, float]]] = {}
    for key, acc in accum.items():
        count = max(float(acc["count"][0]), 1.0)
        mean = acc["sum"] / count
        var = np.maximum(acc["sumsq"] / count - mean**2, 0.0)
        std = np.sqrt(var)
        std[std < 1e-6] = 1.0
        standardization[key] = [
            {"mean": float(m), "standard_deviation": float(s)} for m, s in zip(mean, std)
        ]
    return standardization


def build_dataset_file(
    npz_dir: Path,
    output_file: Path,
    validation_split: float = 0.2,
    test_split: float = 0.15,
    train_test_file: Path | None = None,
    class_names: Sequence[str] | None = None,
    seed: int = 42,
) -> dict:
    """Create train/val/test splits + standardization and write ``dataset.json``.

    Args:
        npz_dir: Directory of ``.npz`` records.
        output_file: Path to write the manifest JSON to.
        validation_split: Fraction of the (post-test) training pool held out for
            validation.
        test_split: Fraction of all solids held out for test (ignored when
            ``train_test_file`` is given).
        train_test_file: Optional JSON ``{"train": [...], "test": [...]}`` of
            stems to use instead of a random test split.
        class_names: Optional ordered list of segment class names.
        seed: RNG seed for the random splits (deterministic).

    Returns:
        The manifest dictionary that was written.
    """
    npz_dir = Path(npz_dir)
    all_stems = _stems_in(npz_dir)
    if not all_stems:
        raise ValueError(f"No .npz records found in {npz_dir}")

    if train_test_file is not None:
        split = json.loads(Path(train_test_file).read_text())
        test_stems = [s for s in split.get("test", []) if s in set(all_stems)]
        train_pool = [s for s in split.get("train", []) if s in set(all_stems)]
        if not train_pool:
            train_pool = [s for s in all_stems if s not in set(test_stems)]
    elif test_split and test_split > 0.0 and len(all_stems) > 2:
        train_pool, test_stems = train_test_split(
            all_stems, test_size=test_split, random_state=seed
        )
    else:
        train_pool, test_stems = all_stems, []

    if validation_split and validation_split > 0.0 and len(train_pool) > 2:
        train_stems, val_stems = train_test_split(
            train_pool, test_size=validation_split, random_state=seed
        )
    else:
        train_stems, val_stems = train_pool, []

    train_stems, val_stems, test_stems = (
        sorted(train_stems),
        sorted(val_stems),
        sorted(test_stems),
    )

    standardization = compute_standardization(npz_dir, train_stems)

    num_classes: int | None = len(class_names) if class_names else None

    manifest: dict = {
        "training_set": train_stems,
        "validation_set": val_stems,
        "test_set": test_stems,
        "feature_standardization": standardization,
        "num_classes": num_classes,
        "class_names": list(class_names) if class_names else [],
    }

    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(json.dumps(manifest, indent=2))
    _log.info(
        "Wrote %s: train=%d val=%d test=%d",
        output_file,
        len(train_stems),
        len(val_stems),
        len(test_stems),
    )
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build a BRepNet dataset manifest.")
    parser.add_argument("--npz-dir", required=True, help="Directory of .npz records")
    parser.add_argument("--output", required=True, help="Path to write dataset.json")
    parser.add_argument("--validation-split", type=float, default=0.2)
    parser.add_argument("--test-split", type=float, default=0.15)
    parser.add_argument("--train-test-file", default=None)
    parser.add_argument(
        "--class-names-file",
        default=None,
        help="Optional JSON list of segment class names",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    class_names = None
    if args.class_names_file:
        class_names = json.loads(Path(args.class_names_file).read_text())

    manifest = build_dataset_file(
        Path(args.npz_dir),
        Path(args.output),
        validation_split=args.validation_split,
        test_split=args.test_split,
        train_test_file=Path(args.train_test_file) if args.train_test_file else None,
        class_names=class_names,
        seed=args.seed,
    )
    print(
        f"train={len(manifest['training_set'])} "
        f"val={len(manifest['validation_set'])} "
        f"test={len(manifest['test_set'])}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
