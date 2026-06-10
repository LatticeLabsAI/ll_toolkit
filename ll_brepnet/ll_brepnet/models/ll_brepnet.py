"""``LLBRepNet`` -- a LightningModule for per-face B-Rep segmentation.

Data flow for one (batched) solid::

    face_features ─┐                         (UV-grid)  face_point_grids
                   ├─► face_repr ◄─ UVNetSurfaceEncoder ◄─┘
    edge_features ─┐                         (UV-grid)  edge_point_grids
                   ├─► edge_repr ◄─ UVNetCurveEncoder ◄──┘
                   │
    coedge input  Xc = [ face_repr[coedge_to_face] ,
                         edge_repr[coedge_to_edge] ,
                         coedge_reversed ]
                   │
                   ▼
            BRepNetEncoder  (coedge message passing + coedge->face mean pool)
                   ▼
            per-face embeddings ──► Linear seg head ──► [num_faces, num_classes]

The coedge message-passing encoder (``BRepNetEncoder`` + ``CoedgeConvLayer``) is
reused from ``cadling``'s MIT-licensed segmentation architectures; the geometry
encoders, feature fusion, segmentation head and the Lightning training/eval
logic are implemented here.
"""

from __future__ import annotations

import argparse

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from cadling.models.segmentation.architectures.brep_net import BRepNetEncoder, CoedgeData
from torch import nn
from torchmetrics.classification import MulticlassAccuracy, MulticlassJaccardIndex

from ..dataloaders.brep_dataset import IGNORE_INDEX, BRepBatch
from .uvnet_encoders import UVNetCurveEncoder, UVNetSurfaceEncoder


class LLBRepNet(pl.LightningModule):
    """B-Rep face-segmentation network.

    Args:
        num_classes: Number of segment classes.
        face_feature_dim: Width of the per-face scalar feature vector.
        edge_feature_dim: Width of the per-edge scalar feature vector.
        surf_emb_dim: Face UV-grid embedding width.
        curve_emb_dim: Edge UV-grid embedding width.
        entity_hidden: Width of the projected face / edge representations.
        hidden_dim: Coedge message-passing hidden width.
        num_layers: Number of coedge convolution layers.
        dropout: Dropout before the segmentation head.
        learning_rate: Adam learning rate.
        use_face_grids: Fuse the face UV-grid geometry.
        use_edge_grids: Fuse the edge UV-grid geometry.
        ignore_index: Label value excluded from loss/metrics.
    """

    def __init__(
        self,
        num_classes: int,
        face_feature_dim: int = 8,
        edge_feature_dim: int = 7,
        surf_emb_dim: int = 64,
        curve_emb_dim: int = 64,
        entity_hidden: int = 64,
        hidden_dim: int = 128,
        num_layers: int = 6,
        dropout: float = 0.0,
        learning_rate: float = 1e-3,
        use_face_grids: bool = True,
        use_edge_grids: bool = True,
        ignore_index: int = IGNORE_INDEX,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.learning_rate = learning_rate

        self.surface_encoder = (
            UVNetSurfaceEncoder(in_channels=7, out_dim=surf_emb_dim) if use_face_grids else None
        )
        self.curve_encoder = (
            UVNetCurveEncoder(in_channels=6, out_dim=curve_emb_dim) if use_edge_grids else None
        )

        face_in = face_feature_dim + (surf_emb_dim if use_face_grids else 0)
        edge_in = edge_feature_dim + (curve_emb_dim if use_edge_grids else 0)
        self.face_proj = nn.Sequential(nn.Linear(face_in, entity_hidden), nn.ReLU(inplace=True))
        self.edge_proj = nn.Sequential(nn.Linear(edge_in, entity_hidden), nn.ReLU(inplace=True))

        coedge_in = 2 * entity_hidden + 1  # face_repr + edge_repr + reversed flag
        self.encoder = BRepNetEncoder(
            input_dim=coedge_in,
            hidden_dim=hidden_dim,
            output_dim=hidden_dim,
            num_layers=num_layers,
        )
        self.dropout = nn.Dropout(dropout)
        self.seg_head = nn.Linear(hidden_dim, num_classes)

        metric_kwargs = {"num_classes": num_classes, "ignore_index": ignore_index}
        self.val_iou = MulticlassJaccardIndex(average="macro", **metric_kwargs)
        self.val_acc = MulticlassAccuracy(average="micro", **metric_kwargs)
        self.test_iou = MulticlassJaccardIndex(average="macro", **metric_kwargs)
        self.test_acc = MulticlassAccuracy(average="micro", **metric_kwargs)

    # -- forward ----------------------------------------------------------
    def forward(self, batch: BRepBatch) -> torch.Tensor:
        """Return per-face class logits ``[num_faces, num_classes]``."""
        face_in = batch.face_features
        if self.surface_encoder is not None:
            face_geo = self.surface_encoder(batch.face_point_grids)
            face_in = torch.cat([face_in, face_geo], dim=1)
        face_repr = self.face_proj(face_in)

        edge_in = batch.edge_features
        if self.curve_encoder is not None:
            edge_geo = self.curve_encoder(batch.edge_point_grids)
            edge_in = torch.cat([edge_in, edge_geo], dim=1)
        edge_repr = self.edge_proj(edge_in)

        coedge_feats = torch.cat(
            [
                face_repr[batch.coedge_to_face],
                edge_repr[batch.coedge_to_edge],
                batch.coedge_reversed,
            ],
            dim=1,
        )
        coedge_data = CoedgeData(
            features=coedge_feats,
            next_indices=batch.coedge_to_next,
            prev_indices=batch.coedge_to_prev,
            mate_indices=batch.coedge_to_mate,
            face_indices=batch.coedge_to_face,
        )
        face_embeddings, _ = self.encoder(coedge_data)
        return self.seg_head(self.dropout(face_embeddings))

    # -- shared step ------------------------------------------------------
    def _compute_loss(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        if int((labels != self.ignore_index).sum()) == 0:
            return logits.new_zeros((), requires_grad=True)
        return F.cross_entropy(logits, labels, ignore_index=self.ignore_index)

    def training_step(self, batch: BRepBatch, batch_idx: int) -> torch.Tensor:
        logits = self(batch)
        loss = self._compute_loss(logits, batch.labels)
        self.log("train_loss", loss, batch_size=batch.num_faces, prog_bar=True)
        return loss

    def validation_step(self, batch: BRepBatch, batch_idx: int) -> torch.Tensor:
        logits = self(batch)
        loss = self._compute_loss(logits, batch.labels)
        self.val_iou.update(logits, batch.labels)
        self.val_acc.update(logits, batch.labels)
        self.log("val_loss", loss, batch_size=batch.num_faces, prog_bar=True)
        return loss

    def on_validation_epoch_end(self) -> None:
        self.log("val_miou", self.val_iou.compute(), prog_bar=True)
        self.log("val_acc", self.val_acc.compute(), prog_bar=True)
        self.val_iou.reset()
        self.val_acc.reset()

    def test_step(self, batch: BRepBatch, batch_idx: int) -> torch.Tensor:
        logits = self(batch)
        loss = self._compute_loss(logits, batch.labels)
        self.test_iou.update(logits, batch.labels)
        self.test_acc.update(logits, batch.labels)
        self.log("test_loss", loss, batch_size=batch.num_faces)
        return loss

    def on_test_epoch_end(self) -> None:
        self.log("test_miou", self.test_iou.compute())
        self.log("test_acc", self.test_acc.compute())
        self.test_iou.reset()
        self.test_acc.reset()

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)

    # -- inference helper -------------------------------------------------
    @torch.no_grad()
    def predict_logits(self, batch: BRepBatch) -> list[tuple[str, torch.Tensor]]:
        """Return ``(file_stem, per-face softmax probabilities)`` per solid."""
        self.eval()
        probs = F.softmax(self(batch), dim=1)
        out: list[tuple[str, torch.Tensor]] = []
        for stem, (start, end) in zip(batch.file_stems, batch.split_batch):
            out.append((stem, probs[start:end].cpu()))
        return out

    @staticmethod
    def add_model_specific_args(parent_parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        """Add ``LLBRepNet`` hyperparameters to an argparse parser."""
        parser = parent_parser.add_argument_group("LLBRepNet")
        parser.add_argument("--surf-emb-dim", type=int, default=64)
        parser.add_argument("--curve-emb-dim", type=int, default=64)
        parser.add_argument("--entity-hidden", type=int, default=64)
        parser.add_argument("--hidden-dim", type=int, default=128)
        parser.add_argument("--num-layers", type=int, default=6)
        parser.add_argument("--dropout", type=float, default=0.0)
        parser.add_argument("--learning-rate", type=float, default=1e-3)
        parser.add_argument("--no-face-grids", action="store_true")
        parser.add_argument("--no-edge-grids", action="store_true")
        return parent_parser
