"""PyTorch-Lightning training entry point for ``ll_brepnet``.

Usage::

    python -m ll_brepnet.train \
        --dataset-file /path/to/processed/dataset.json \
        --dataset-dir  /path/to/processed/ \
        --label-dir    /path/to/labels/ \
        --max-epochs 200

Hyperparameters for the model are added by ``LLBRepNet.add_model_specific_args``;
run with ``--help`` to see them all. Checkpoints (best by validation mIoU, or by
training loss when there is no validation split) and TensorBoard logs are written
under ``--log-dir``.
"""

from __future__ import annotations

import argparse
import logging

import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger

from .dataloaders.brep_dataset import IGNORE_INDEX, BRepDataModule, BRepDataset
from .models.ll_brepnet import LLBRepNet

_log = logging.getLogger(__name__)


def infer_num_classes(dataset: BRepDataset) -> int:
    """Infer the class count as ``max(label) + 1`` over the dataset's labels."""
    max_label = -1
    for idx in range(len(dataset)):
        stem = dataset.file_stems[idx]
        label_path = dataset.label_dir / f"{stem}.seg"
        if not label_path.exists():
            continue
        labels = np.loadtxt(label_path, dtype=np.int64, ndmin=1)
        valid = labels[labels != IGNORE_INDEX]
        if valid.size:
            max_label = max(max_label, int(valid.max()))
    if max_label < 0:
        raise ValueError(
            "Could not infer num_classes: no labels found. Pass --num-classes "
            "or provide .seg label files / a manifest with num_classes."
        )
    return max_label + 1


def build_model(args: argparse.Namespace, num_classes: int) -> LLBRepNet:
    return LLBRepNet(
        num_classes=num_classes,
        surf_emb_dim=args.surf_emb_dim,
        curve_emb_dim=args.curve_emb_dim,
        entity_hidden=args.entity_hidden,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        learning_rate=args.learning_rate,
        use_face_grids=not args.no_face_grids,
        use_edge_grids=not args.no_edge_grids,
    )


def do_training(
    args: argparse.Namespace,
    extra_callbacks: list | None = None,
) -> pl.Trainer:
    """Run training (and a final test pass when a test split exists).

    Args:
        args: Parsed CLI arguments.
        extra_callbacks: Optional extra Lightning callbacks (e.g. for testing).

    Returns:
        The fitted :class:`pytorch_lightning.Trainer`.
    """
    pl.seed_everything(args.seed, workers=True)

    datamodule = BRepDataModule(
        dataset_file=args.dataset_file,
        dataset_dir=args.dataset_dir,
        label_dir=args.label_dir,
        max_num_faces_per_batch=args.max_num_faces_per_batch,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        standardize=not args.no_standardize,
        seed=args.seed,
    )
    datamodule.setup()

    num_classes = args.num_classes or datamodule.num_classes
    if not num_classes:
        num_classes = infer_num_classes(datamodule.train_dataset)
    _log.info("Training with num_classes=%d", num_classes)

    model = build_model(args, num_classes)

    has_val = datamodule.val_dataset is not None and len(datamodule.val_dataset) > 0
    monitor, mode = ("val_miou", "max") if has_val else ("train_loss", "min")
    checkpoint = ModelCheckpoint(monitor=monitor, mode=mode, save_top_k=1, save_last=True)
    callbacks = [checkpoint]
    if extra_callbacks:
        callbacks.extend(extra_callbacks)

    try:
        logger = TensorBoardLogger(save_dir=args.log_dir, name="ll_brepnet")
    except ModuleNotFoundError:
        _log.warning("tensorboard not installed; logging to CSV instead")
        logger = CSVLogger(save_dir=args.log_dir, name="ll_brepnet")
    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        accelerator=args.accelerator,
        devices=args.devices,
        logger=logger,
        callbacks=callbacks,
        log_every_n_steps=1,
        enable_progress_bar=not args.no_progress_bar,
    )

    trainer.fit(model, datamodule=datamodule)

    if datamodule.test_dataset is not None and len(datamodule.test_dataset) > 0:
        trainer.test(
            model, datamodule=datamodule, ckpt_path="best" if checkpoint.best_model_path else None
        )

    return trainer


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train ll_brepnet face segmentation.")
    parser.add_argument("--dataset-file", required=True, help="Path to dataset.json")
    parser.add_argument("--dataset-dir", required=True, help="Directory of .npz records")
    parser.add_argument("--label-dir", default=None, help="Directory of .seg label files")
    parser.add_argument("--num-classes", type=int, default=None)
    parser.add_argument("--max-epochs", type=int, default=100)
    parser.add_argument("--accelerator", default="auto")
    parser.add_argument("--devices", default="auto")
    parser.add_argument("--max-num-faces-per-batch", type=int, default=4096)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--no-standardize", action="store_true")
    parser.add_argument("--log-dir", default="logs")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-progress-bar", action="store_true")
    LLBRepNet.add_model_specific_args(parser)
    return parser


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args = build_arg_parser().parse_args(argv)
    trainer = do_training(args)
    metrics = {k: float(v) for k, v in trainer.callback_metrics.items()}
    print("Final metrics:", metrics)
    print("Best checkpoint:", trainer.checkpoint_callback.best_model_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
