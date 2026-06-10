# ll_brepnet

A B-Rep **face-graph neural network** for CAD solid-model segmentation, in the
UV-Net / BRepNet lineage. `ll_brepnet` operates directly on the boundary
representation of a solid — faces and edges connected through oriented *coedges*
(carrying next / previous / mate / parent-face / parent-edge adjacency) — and
fuses that topology with UV-grid surface and curve geometry to predict a
semantic segment label for every face.

It is an **independent, MIT-licensed** package built on the LatticeLabs
toolkit's own B-Rep machinery (`cadling`). It is *inspired by* BRepNet
([arXiv:2104.00706](https://arxiv.org/abs/2104.00706)) and UV-Net but contains
no code from those projects — see [`ATTRIBUTION.md`](ATTRIBUTION.md).

> **Status — under active implementation.** This package is being built out
> milestone-by-milestone per
> [`docs/plans/2026-06-10-implement-ll-brepnet.md`](../docs/plans/2026-06-10-implement-ll-brepnet.md).
> The package installs and imports today (M0 complete). Extraction, dataset,
> model, training and evaluation land in subsequent milestones; this README and
> the documentation site are updated as each milestone is verified, and will
> never claim a capability that is not yet backed by working, tested code.

## Package layout

| Subpackage | Purpose |
|---|---|
| `ll_brepnet.pipelines` | STEP/JSON → coedge graph + geometry → `.npz`; dataset-manifest building |
| `ll_brepnet.dataloaders` | `BRepDataset` / `BRepDataModule`, multi-solid collation, face-count sampler |
| `ll_brepnet.models` | UV-Net geometry encoders + the `LLBRepNet` LightningModule + per-face seg head |
| `ll_brepnet.eval` | folder/checkpoint inference → per-face logits |

## Installation

`ll_brepnet` needs PyTorch, `pythonocc-core` and `occwl`, which on macOS must
come from conda (see the repo `CLAUDE.md` for the OpenMP rationale). The simplest
path is to reuse the existing `cadling` conda environment, which already
provides them, and add `pytorch-lightning`:

```bash
conda activate cadling
pip install pytorch-lightning
pip install -e ./ll_brepnet
```

Or create a standalone environment:

```bash
conda env create -f ll_brepnet/environment.yaml
conda activate ll-brepnet
```

## License

MIT — see [`LICENSE`](LICENSE) and [`ATTRIBUTION.md`](ATTRIBUTION.md).
