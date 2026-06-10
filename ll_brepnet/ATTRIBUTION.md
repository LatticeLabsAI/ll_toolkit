# Attribution & provenance

`ll_brepnet` is an **independent, MIT-licensed** implementation. It is *inspired
by* prior research on neural networks for boundary-representation (B-Rep) CAD
models, but it **does not include, copy, or adapt source code** from those
projects.

## Inspiration (ideas, not code)

- **BRepNet: A Topological Message Passing System for Solid Models** —
  Lambourne, Willis, Jayaraman, Sanghi, Meltzer, Shayani. CVPR 2021.
  [arXiv:2104.00706](https://arxiv.org/abs/2104.00706).
  The Autodesk reference implementation (`AutodeskAILab/BRepNet`) is licensed
  **CC BY-NC-SA 4.0** (NonCommercial + ShareAlike). To keep this package MIT and
  free of NonCommercial restrictions, `ll_brepnet` deliberately does **not**
  reproduce that code or its distinctive expression — in particular it does
  **not** implement BRepNet's configurable *kernel*-based topological
  convolution (the `kernels/*.json` winged-edge walk machinery). It uses only
  the published *ideas* (coedge adjacency, message passing over the B-Rep,
  fusing UV-grid geometry) as a guide.

- **UV-Net: Learning from Boundary Representations** — Jayaraman et al. The
  UV-grid sampling + 1D/2D CNN surface/curve encoder idea informs our geometry
  encoders, which are written from scratch as standard convolutional stacks.

## What this package is actually built on

The implementation is composed from the **LatticeLabs toolkit's own MIT-licensed
B-Rep machinery**, principally from `cadling`:

- coedge incidence extraction — `cadling.lib.topology.coedge_extractor`
- B-Rep face-graph + face/edge features — `cadling.lib.topology.brep_face_graph`
- UV-grid sampling — `cadling.lib.geometry.uv_grid_extractor`
- UV-Net-style encoders and the simple coedge message-passing encoder —
  `cadling.models.segmentation.architectures.{uv_net,brep_net}`

Geometry I/O uses the public APIs of `pythonocc-core` and `occwl` (separate,
independently-licensed libraries consumed as dependencies — not vendored).

## Consequence

Because no CC BY-NC-SA code is copied, `ll_brepnet` is distributed under the
**MIT License** (see `LICENSE`), consistent with the rest of the toolkit, and
carries no NonCommercial or ShareAlike obligations. The trade-off is that it is
**not a faithful reproduction** of the BRepNet paper's architecture and is not
expected to match its published accuracy numbers.
