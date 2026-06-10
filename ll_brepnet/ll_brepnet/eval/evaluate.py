"""Run a trained ``ll_brepnet`` checkpoint over a folder of STEP files.

For every ``.step`` / ``.stp`` file in the input folder this:

1. extracts the BRepNet ``.npz`` record (same pipeline used for training),
2. runs the model to produce per-face class probabilities, and
3. writes a ``<stem>.logits`` text file (one row per face, one column per
   class) so the predicted segment of face *i* is ``argmax`` of row *i*.

The training ``dataset.json`` is required so inference uses exactly the feature
standardization and class set the model was trained with.

Usage::

    python -m ll_brepnet.eval.evaluate \
        --step-dir ./example_files/step_examples \
        --model ./logs/ll_brepnet/version_0/checkpoints/best.ckpt \
        --dataset-file ./processed/dataset.json \
        --output-dir ./predictions
"""

from __future__ import annotations

import argparse
import json
import logging
import tempfile
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from ..dataloaders.brep_dataset import BRepDataset, brep_collate_fn
from ..dataloaders.max_num_faces_loader import MaxNumFacesSampler
from ..models.ll_brepnet import LLBRepNet
from ..pipelines.extract_brepnet_data_from_step import extract_brepnet_data_from_step

_log = logging.getLogger(__name__)


def write_logits(stem: str, probs: torch.Tensor, output_dir: Path) -> Path:
    """Write per-face class probabilities to ``<stem>.logits`` (one row/face)."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{stem}.logits"
    np.savetxt(out_path, probs.numpy(), fmt="%.6f")
    return out_path


def _inference_manifest(training_manifest: Path, stems: list[str]) -> dict:
    """Build an inference manifest reusing the training stats + class set."""
    train = json.loads(Path(training_manifest).read_text())
    return {
        "training_set": [],
        "validation_set": [],
        "test_set": sorted(stems),
        "feature_standardization": train.get("feature_standardization", {}),
        "num_classes": train.get("num_classes"),
        "class_names": train.get("class_names", []),
    }


def evaluate_folder(
    step_dir: Path,
    model_ckpt: Path,
    dataset_file: Path,
    output_dir: Path,
    num_workers: int = 1,
    max_num_faces_per_batch: int = 4096,
    npz_dir: Path | None = None,
) -> list[Path]:
    """Segment every STEP file in ``step_dir`` and write per-face logits.

    Args:
        step_dir: Folder of ``.step`` / ``.stp`` files.
        model_ckpt: Trained ``LLBRepNet`` checkpoint.
        dataset_file: Training ``dataset.json`` (for standardization + classes).
        output_dir: Where to write the ``<stem>.logits`` files.
        num_workers: Parallel STEP-extraction workers.
        max_num_faces_per_batch: Inference batch face-count cap.
        npz_dir: Where to cache the intermediate ``.npz`` (a temp dir if None).

    Returns:
        The list of written ``.logits`` paths.
    """
    output_dir = Path(output_dir)
    tmp_ctx: tempfile.TemporaryDirectory | None = None
    if npz_dir is None:
        tmp_ctx = tempfile.TemporaryDirectory(prefix="ll_brepnet_eval_")
        npz_dir = Path(tmp_ctx.name)
    else:
        npz_dir = Path(npz_dir)

    try:
        written_npz = extract_brepnet_data_from_step(
            Path(step_dir), npz_dir, num_workers=num_workers
        )
        stems = [p.stem for p in written_npz]
        if not stems:
            _log.warning("No STEP files extracted from %s", step_dir)
            return []

        manifest_path = npz_dir / "inference_dataset.json"
        manifest_path.write_text(json.dumps(_inference_manifest(dataset_file, stems), indent=2))

        model = LLBRepNet.load_from_checkpoint(model_ckpt, map_location="cpu")
        model.eval()

        dataset = BRepDataset(manifest_path, npz_dir, "test_set", standardize=True)
        sampler = MaxNumFacesSampler(
            dataset, max_num_faces_per_batch=max_num_faces_per_batch, shuffle=False
        )
        loader = DataLoader(dataset, batch_sampler=sampler, collate_fn=brep_collate_fn)

        written: list[Path] = []
        with torch.no_grad():
            for batch in loader:
                for stem, probs in model.predict_logits(batch):
                    written.append(write_logits(stem, probs, output_dir))
        _log.info("Wrote %d .logits file(s) to %s", len(written), output_dir)
        return written
    finally:
        if tmp_ctx is not None:
            tmp_ctx.cleanup()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Segment a folder of STEP files with a trained ll_brepnet model."
    )
    parser.add_argument("--step-dir", required=True, help="Folder of STEP files")
    parser.add_argument("--model", required=True, help="Trained checkpoint (.ckpt)")
    parser.add_argument("--dataset-file", required=True, help="Training dataset.json")
    parser.add_argument("--output-dir", required=True, help="Where to write .logits files")
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--max-num-faces-per-batch", type=int, default=4096)
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    written = evaluate_folder(
        Path(args.step_dir),
        Path(args.model),
        Path(args.dataset_file),
        Path(args.output_dir),
        num_workers=args.num_workers,
        max_num_faces_per_batch=args.max_num_faces_per_batch,
    )
    print(f"Wrote {len(written)} .logits file(s) to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
