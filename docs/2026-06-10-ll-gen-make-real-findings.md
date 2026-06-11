# ll_gen "make-real" findings (2026-06-10)

Goal: make `ll_gen`'s generators genuinely produce valid CAD with **honest**
metrics (no faked success), using the canonical DeepCAD dataset
(`data.tar` → `resources/DeepCAD/data/cad_vec/`, 179,133 real construction
sequences).

`ll_gen` has **two** generation paths, and they fail/succeed for different
reasons:

| Path | Representation | Executor | Status |
|---|---|---|---|
| command-VAE (`STEPVAE`) | command sequence (SOL/LINE/ARC/CIRCLE/EXTRUDE/EOS) | `command_executor` (sketch→extrude) | real but **primitive-limited** |
| diffusion (`StructuredDiffusion`) | per-face UV-grids + per-edge polylines | `surface_executor` (B-spline fit → sew) | codec **real**; denoiser convergence is compute-bound |

---

## 1. Command-VAE: real, but architecturally primitive-limited

**The existing 95%-valid checkpoint generates cylinders.** Inspecting
`ll_gen/checkpoints/vae_rl_solid.pt` (58/60 "valid"): the valid shapes are
dominated by **extruded circles** (bbox dims `(small, X, X)` — two equal dims),
and several "valid" shapes have **volume = 0** (flat faces OCC accepts but aren't
solids), inflating the rate. "88 distinct" is diversity of *size*, not *kind*.

**Root cause (architecture, not tuning).** `STEPVAE` is a *global-latent,
non-autoregressive* model: `encode()` mean-pools the sequence into one orderless
latent (`vae.py:168`); `decode(z, seq_len)` emits every position's params
independently. Nothing ties LINE_i's end to LINE_{i+1}'s start, so multi-line
sketches almost never close → only self-closing primitives (circles) validate.
It reached 95% only because **RL** directly rewarded valid circles.

**Supervised training confirms the limit from the other side.** Trained on real
DeepCAD `cad_vec` (command-CE + param-CE + KL warmup), prior-sample validity is
**0%** and the model **posterior-collapses to all-SOL** (60 SOL tokens, zero
curves; cmd_acc 0.191 = the SOL frequency). Under the KL pressure generation
requires, the orderless mean-pool latent is ignored. The VAE can be
informative-latent XOR good-prior-sampling, never both.

### Fixes shipped (command path)

- **Decode enum mismatch** (`generation_pipeline.py`, main `5fbf909`) — int→str
  CommandType crash made ALL generation 0% valid; regression test.
- **Resilient checkpoint load** (`rl_trainer.py`, main `25cc091`) — strict load
  rejected the real M3 checkpoint (`dim_encoder` drift); regression test.
- **NaN param loss** (`vae.py`) — `STEPVAE.forward`'s param loss averaged over
  all 16 heads, but slots 8–15 are never active in the 6-command schema →
  `F.cross_entropy(mean)` over zero elements = NaN → poisoned `recon_loss`,
  silently blocking the supervised forward. Now skips all-ignored heads.
  Regression: `ll_stepnet/tests/test_vae_sparse_param_loss.py`.
- **Closure-aware decode** (`command_executor.py::_build_sketch_face`) — builds
  sketch loops by **threading curve endpoints + auto-closing**, so multi-line
  polygons close *by construction* regardless of decoder endpoint alignment.
  Verified: a deliberately non-connecting square → valid solid (vol 1.01);
  triangle/pentagon/hexagon/octagon all valid; the 95% checkpoint now emits some
  genuinely **non-cylindrical** solids (dims like `(1.97, 2.23, 2.7)`) with no
  regression (59/60). Tests: `TestClosureAwareSketch` (3) in
  `ll_gen/tests/test_command_executor.py` (43 pass). This removes the
  wire-closure limit for **any** non-collapsed command generator — it cannot
  rescue the posterior-collapsed VAE (no curves to close), but it is a real
  prerequisite a better decoder needs.

**Conclusion:** the command-VAE is a dead end for *diverse* generation by
architecture. Diverse valid CAD needs either an autoregressive/closure-aware
*decoder* or the diffusion path.

---

## 2. Diffusion: the genuine path to diverse CAD

`StructuredDiffusion` decodes per-face UV-grids + per-edge polylines and
`surface_executor` fits B-splines and **sews** them — no posterior collapse, no
polyline-closure limit.

### Built + verified

- **Geometry extraction** (`resources/ll_gen_proof/diffusion_codec_train.py`):
  `cad_vec → solid` (validated translation, 30/30) → sample each face as an
  8×8×3 UV grid and each edge as a 12×3 polyline, normalized to the unit cube,
  padded to `num_faces=8 / num_edges=12` with masks. Avg 4.3 faces / 8.5 edges.
- **GeometryCodec trained on real geometry**: reconstruction MSE
  **0.40 (untrained) → 0.0003 (trained, 60 epochs / 4000 solids)** — RMS error
  ~1.7% of the unit cube. The latent↔geometry map is now real. Checkpoint:
  `resources/ll_gen_proof/diffusion_codec.pt`.

### Bugs fixed (diffusion path)

- **MPS backward crash** (`diffusion.py` `encode_faces`/`encode_edges`) —
  `permute(...)` then Conv2d/Conv1d; the conv backward calls `.view()` on the
  non-contiguous tensor and raises. Added `.contiguous()`. Also fixes the
  existing RL diffusion path on MPS.
- **Surface-fitter signature** (`surface_executor.py::_fit_bspline_surface`) —
  called cadling's `BSplineSurfaceFitter.fit_surface(grid, tolerance=...)`, but
  that method takes only `point_grid` and returns a `dict` (not a face).
  `TypeError` → **every** face silently failed → nothing ever sewed. Fixed to
  use the constructor tolerance + extract `result["face"]`, falling back to the
  direct OCC fit. Verified `_fit_bspline_surface` now returns a face.

- **Topology-merge crash** (`surface_executor.py::execute_latent_proposal`) —
  the merge step called `TopologyMerger.merge_edges(edges)`, which does not
  exist (real API is `merge(faces) -> {shape, valid, ...}`, purpose-built to
  mate independently-generated faces and sew a watertight solid). The
  `AttributeError` was uncaught, so **every** sew crashed — the diffusion path
  could never produce a shape, training or not. Fixed: call `merge(faces)` as
  the primary watertight path, fall back to the built-in edge dedup + sew.

### The sewing pipeline now works on in-distribution geometry

With the surface-fit + merge fixes, **7/15 real DeepCAD solids sew into closed,
non-zero-volume shells** (the ≥4-face prismatic models). The 3-face failures are
cylinders whose periodic lateral surface the UV-grid doesn't seam — a known
sampling limitation. Regression test:
`ll_gen/tests/test_surface_executor.py::TestWatertightSew` (unit cube → closed
volume). So the path was *completely broken by three real bugs*, not by physics;
it is now functional for in-distribution geometry.

### Honest remaining bottleneck: the denoiser does not converge

The remaining step is **denoiser convergence**, and here is an honest negative:
the 4-stage denoisers **plateau at the trivial "predict-zero-noise" solution**
(denoiser-only loss ≈ 1.0 per stage = `mean(noise²)` for unit-variance noise —
i.e. the network outputs ≈0 instead of the noise). It drops only 5.28→4.03 then
flattens. Latent normalization (scaling the sub-unit ~0.42-std codec latents to
~unit variance — the standard latent-diffusion trick) was tried and did **not**
move the plateau, so it was reverted. The `CADDenoiser` architecture is
structurally sound (input_proj → sinusoidal time-embed → 12-layer transformer →
output_proj), so this is a genuine training-convergence problem (LR schedule /
architecture / per-token-vs-set conditioning), not a one-line bug — it needs
real diffusion-training experimentation.

**Net for the diffusion path:** codec is real, the sewing pipeline is fixed and
works on in-distribution geometry, but the generator **cannot yet produce valid
CAD because the denoiser does not learn to denoise** — stated plainly, no number
claimed. This is the honest frontier; closing it is a focused follow-on.

---

## Reusable asset

`cad_vec → executor-schema` translation (cadlib `CADSequence.from_vector` →
SOL/LINE-abs/ARC-start-end-center/CIRCLE/EXTRUDE → quantize to the executor's
symmetric [0,255]↔[-2,2]). Validated 30/30 real models → valid solids. Transfers
to any target consuming the executor schema.

## Artifacts

- Scripts: `resources/ll_gen_proof/{deepcad_supervised_train,
  diffusion_codec_train, diffusion_full_train, rl_refine_from_supervised}.py`
- Results: `resources/ll_gen_proof/{DEEPCAD_SUPERVISED, DIFFUSION_CODEC,
  DIFFUSION_FULL}.json`, `diffusion_codec.pt`
- DeepCAD data: `resources/DeepCAD/data/cad_vec/` (179k h5, gitignored)
