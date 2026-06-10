# Partial & Deceptive Code Audit — LatticeLabs Toolkit

**Date:** 2026-06-09
**Scope:** Full monorepo (~575 `.py` files): `cadling/`, `ll_gen/`, `ll_stepnet/`, `geotoken/`, `ll_ocadr/`, `ll_clouds/`. (`ll_brepnet/` and the top-level `lib/` named in CLAUDE.md contain **0 `.py` files** — empty/aspirational — so there was nothing to audit there. The audited `lib/` code lives under `cadling/lib/`.)
**Method:** Six parallel read-only audit passes; every finding read at the source. The three HIGH-impact claims below were independently re-verified against the code after the passes completed. No files were modified.

**Definitions used**
- **PARTIAL** — incomplete code: `NotImplementedError`, empty/`pass`/`...` bodies, TODO, or returns `None/[]/0` where real computed logic is expected.
- **DECEPTIVE** — code that *looks* complete/production but isn't: hardcoded/fabricated values dressed as computed results, docstrings claiming behavior the body doesn't perform, fail-open error paths returning fake success, "simplified/approximation" passed off as the real algorithm.

---

## Executive Summary

The codebase is **mostly genuinely implemented** — the recognition cores (PointNet++/Point-BERT in `ll_ocadr`, segmentation in `ll_clouds`), the STEP parsing/topology stack in `cadling/backend`, the geotoken quantizers, and the ll_stepnet model paths are real. The legacy `cadling/docs/RequiredToBeCorrected.md` ("~200 methods") is **substantially stale**: 22 of 24 verified high-priority items are now implemented.

The real risk is concentrated in a small number of **load-bearing deceptions** — code that silently fabricates data while presenting itself as working. The single most important pattern: **fabricated B-Rep graph topology baked into ML training data**, and an **RL training mode that updates zero parameters**.

**Tally:** 5 HIGH · 11 MEDIUM · ~16 LOW.

---

## HIGH — silently fake in a load-bearing path

### H1. Diffusion "RL training" trains zero model parameters — DECEPTIVE → **FIXED 2026-06-09**
`ll_gen/ll_gen/generators/neural_diffusion.py` — `generate_for_training`
The REINFORCE `log_prob` was a function only of `noise` — a fresh `torch.randn(..., requires_grad=True)` leaf. The actual geometry came from `self._model.sample(...)` under `torch.no_grad()` with its *own* internal noise. So `log_probs` was connected to **0 model parameters**. In `rl_trainer.py`, `total_loss.backward()` + `optimizer.step()` therefore updated nothing while logging a finite loss. `diffusion` is a fully selectable generator in `training/run.py` and `training/proof_of_life.py`.

**Remediation (DDPO).** Added `StructuredDiffusion.sample_with_log_prob` (`ll_stepnet/stepnet/diffusion.py`) — a stochastic DDIM reverse process run **with gradients**, where each step's transition mean is produced by the denoiser and the per-step Gaussian log-prob (the sampled action detached, so `∇ log π` flows through the mean only) accumulates into a trajectory log-prob connected to the model. The DDIM `eta` sampler + Gaussian log-prob math is grounded in the reference implementations in `resources/autodesk_models` (Make-a-Shape `gaussian_diffusion.ddim_sample` + `normal_kl`; identical in `brepdiff`/`diff3d`). `generate_for_training` now calls this path (no `no_grad`, `eta` coerced to a stochastic value). Verified empirically: the trajectory log-prob reaches **142/142** trainable tensors of the test model, and **one `RLAlignmentTrainer.train_step` changes all 142 parameter tensors** (was 0). Regression tests: `ll_gen/tests/test_diffusion_ddpo.py` (4); existing `test_log_prob_scorer`/`test_training`/`test_generators` (147) still pass.

**Adjacent gap (decoder) — FIXED 2026-06-09.** `StructuredDiffusion.sample()` previously emitted a flat per-stage latent `[B, latent_dim]` with **no latent→geometry decoder**, so the raw latent was surfaced as a "face grid" (the `Unexpected tensor shape` warning + B-spline fit failures). Now built: the diffusion operates on **per-primitive token sets** (one token per face/edge), and a real trainable `GeometryCodec` (`ll_stepnet/stepnet/diffusion.py`) maps latents ↔ B-Rep geometry — a UV-Net-style Conv encoder (Conv2d over face UV grids, Conv1d over edge polylines) + mirrored MLP decoder + masked-MSE reconstruction loss, grounded in `resources` (BrepDiff `uvgrid.py`/`brepdiff.py` representation+loss; UV-Net `encoders.py`). `sample()`/`sample_with_log_prob()` decode the final tokens into `face_grids [B,N_faces,U,V,3]` + `edge_points [B,N_edges,M,3]`; `forward_train(geometry=…)` trains the codec (recon) and the denoisers in the codec's latent space. Verified: codec overfits a sample to ~0 loss; `sample()` emits correct-shape geometry; the **surface executor fits B-spline surfaces/curves** to the decoded grids (warnings gone); and the **DDPO RL gradient still reaches 136/136 denoiser params** (codec is trained separately by reconstruction, not RL). Tests: `ll_gen/tests/test_diffusion_ddpo.py` (`TestGeometryDecoder`, 5).

### H2–H4. Fabricated B-Rep face adjacency in three ML dataset builders — DECEPTIVE → **FIXED 2026-06-09**
- `cadling/cadling/data/hf_builders/brep_graph_builder.py` — `_process_step_file_pythonocc`
- `cadling/cadling/data/hf_builders/arrow_brep_builder.py` — `ArrowBRepGraphBuilder._process_step_pythonocc`
- `cadling/cadling/data/webdataset.py` — `STEPWebDataset._step_bytes_to_graph`

All three connected each face to the *next 4 faces by array index* (`for j in range(i+1, min(i+5, num_faces))`) instead of by shared edges — `# For now, create a simple complete graph on faces (placeholder)`. Node/edge geometric features were zeroed (`normal: [0,0,1]`, `curvatures: [0,0]`, `convexity: 0.0`, `dihedral_angle: 0.0`). These builders emit **GNN training data** (Parquet / HuggingFace / WebDataset), so the fake topology was invisible once serialized and silently corrupted any model trained on it. This was the highest-impact systemic issue.

**Remediation:** added a single shared helper `cadling/cadling/lib/topology/brep_face_graph.py::build_brep_face_graph(shape)` that derives real face-to-face adjacency from B-Rep topology via OCC `MapShapesAndAncestors` (two faces adjacent iff they share an edge) and computes real per-face outward normals/curvature (`FaceGeometryExtractor`, orientation-corrected from the in-solid face) and per-edge dihedral angle + **signed convexity** via the coedge test `sign((nA × tA) · nB)` where `tA` is the edge tangent oriented by its FORWARD coedge in face A (`s<0` convex, `s>0` concave). All three builders now delegate to it. Verified across planar **and curved** solids: box (12 convex), corner-notch (3 concave), cylinder (caps convex), through-hole (rim convex — the case a centroid heuristic gets wrong), and blind pocket (floor ring concave); all with symmetric, self-loop-free adjacency. Regression tests: `cadling/tests/unit/lib/topology/test_brep_face_graph.py` (incl. curved-face convexity coverage and static guards that the `min(i+5, …)` placeholder never returns); full topology suite passes, no regressions.

### H5. Hole depth is always fabricated — DECEPTIVE
`cadling/cadling/models/segmentation/geometry_extractors.py:174,267` — `HoleGeometryExtractor._extract_from_step_text` / `_extract_from_occ_faces`
Diameter/location/orientation are computed from real OCC geometry (`cylinder.Radius()`/`Axis()`), but `depth` is always `diameter * 2.0` (text path: `* 2.0 if diameter else 20.0`) — never measured, even when full OCC faces are available. Returned with `confidence: 0.9` (OCC) / up to `1.0` (text). Consumed by `feature_recognition.py:476-507` and logged as `depth={…}mm`, i.e. a fabricated measurement presented as a computed feature parameter. (The Pocket/Boss extractors in the same file *do* compute real depth, so Hole is the outlier.)

---

## MEDIUM — disclosed-but-misleading, or fake only on a secondary path

### ll_gen
- **Coverage/MMD/JSD were always 0.0 in every shipped path** — `ll_gen/ll_gen/training/metrics.py` `compute_all`. — **FIXED 2026-06-09.** The metrics now default to **`None`** (undefined without a reference set) instead of `0.0`, so a missing reference is never reported as a computed zero (`GenerationMetrics.coverage/mmd/jsd` are `float | None`). And `generated_points` are now **real surface points tessellated from each result's `TopoDS_Shape`** (`_sample_shape_points` via `BRepMesh_IncrementalMesh`), replacing the 8 bbox corners. Verified: no-reference → `None`; with a real box shape + reference → metrics computed in range. Both production callers (`evaluate_validity`, `rl_trainer.evaluate`) only read validity/compile/reward, so `None` is safe. Tests: `ll_gen/tests/test_training.py` (74 pass).
- **Fail-open VLM verifier** — `ll_gen/ll_gen/pipeline/verification.py` — **FIXED 2026-06-09.** All six error / missing-dep / no-render / unknown-backend paths returned `{"matches": True}`, which `verify()` counted as a *passed* VLM method (inflating confidence). They now return `{"verified": False, "matches": None}`; `verify()` only counts the VLM (and applies its verdict) when `verified is True`, and records `VerificationResult.vlm_verified`. An unavailable verifier no longer masquerades as a passed check. Tests: `ll_gen/tests/test_verification.py::TestVlmFailsClosed` (34 pass).

### cadling
- **Chamfer distance hardcoded** — `geometry_extractors.py` `ChamferGeometryExtractor._extract_from_occ_geometry` — **FIXED 2026-06-09.** `distance = 2.0` is replaced by the **measured chamfer face width** (mean short UV-extent of the planar chamfer faces; a plane's U/V are real lengths), with a `distance_measured` flag. Verified: setback 2.0→2.828, 5.0→7.071 (scales with the chamfer). Test: `test_geometry_extractors_depth.py::TestMeasuredChamferDistance`.
- **Surface area fabricated under the canonical key** — `geometry_analysis.py` `_analyze_from_step_text` — **FIXED 2026-06-09.** The bbox-derived estimate now lands in `surface_area_estimate` (mirroring `volume_estimate`), NOT the canonical `surface_area`; the ratio is `surface_to_volume_ratio_estimate`. No consumer reads `surface_area` from this path. 
- **"Normalized cuts" is plain BFS** — `mesh_chunker.py` `_segment_by_graph` — **FIXED 2026-06-09 (docstring).** Docstring now truthfully describes connected-components BFS region-growing with a size cap, explicitly "NOT a spectral normalized-cuts partition" (the algorithm was already correct; only the claim was false).
- **Symmetry constraint without the geometry check** — `geometric_constraint_model.py` `_extract_symmetry_constraints` — **FIXED 2026-06-09.** Now performs a real **centroid-reflection symmetry test** on the hole locations (every hole must have a mirror partner; confidence scales with the matched fraction) and only emits `SYMMETRIC` when the holes actually mirror-match. Verified: symmetric square → emitted (frac 1.0); asymmetric trio → none. Test: `test_geometric_constraint_model.py`.
- **UV-grid trimming mask always 1.0** — `occ_wrapper.py` `_uv_grid_pythonocc` — **FIXED 2026-06-09.** The mask channel is now set by a real 2D face classifier (`BRepTopAdaptor_FClass2d`): 0.0 for samples outside the trimmed boundary (e.g. inside a hole), 1.0 on material. Verified on a holed face (11% masked out).
- **Toy surface classifier feeding "machining features"** — `threaded_geometry_vlm_pipeline.py` Stage-1 — **FIXED 2026-06-09 (made real).** Replaced the curvature-threshold toy classifier *and* the broken `extract_from_face(...)` calls (a method that never existed, so parameters were always `{}`) *and* the unreachable `planar_recessed` branch. Stage-1 now reads each face's **real parsed surface type** from the 24-dim node-feature one-hot, detects holes from cylindrical faces with **measured** parameters (diameter from mean curvature `d=1/|H|`, depth from lateral area `h=A/(πd)`, location from centroid), and detects pockets only when a planar face is **geometrically recessed** (centroid-projection-along-normal test) with measured width/length/depth. Confidence derives from the real signal. Tests: `tests/unit/experimental/pipeline/test_threaded_geometry_vlm_pipeline.py::TestStage1RealDetection`.
- **Edge reconstruction produces no geometry** — `graph_reconstructor.py` `_reconstruct_edge` — **FIXED 2026-06-09.** Now builds a **real OCC edge** (`BRepBuilderAPI_MakeEdge`) between the centroids of the two faces the edge connects (recovered from the edge index + per-face centroids), so the primitive carries actual geometry (`occ_shape` set, `success=True`); when the adjacent-face endpoints are unavailable it honestly returns no shape. Tests: `tests/unit/generation/test_edge_reconstruction.py`.

### geotoken
- **Synthetic XYZ in UV-grid quantizer** — `geotoken/geotoken/quantization/uv_grid_quantizer.py` `quantize_from_topology` — **FIXED 2026-06-09.** The synthesized-xyz path still sets `is_approximated=True` per token, but now also emits a **WARNING** clearly stating the tokens are synthesized from feature statistics (not B-Rep surface evaluation) and pointing to `quantize_surface_samples` for exact tokens — so the approximation is visible at runtime, not just via a flag no consumer inspected.

### ll_ocadr
- **Two empty encoder modules** — `ll_ocadr/vllm/lattice_encoder/clip_sdpa.py` and `sam_vary_sdpa.py` were **0 bytes**, never imported — **FIXED 2026-06-09 (implemented as a full rendered-image modality).** Both are now real SDPA vision encoders: `CLIPVisionSDPA` (patch-embed + class token + interpolable pos-embed + pre-LN SDPA transformer) and `SAMVaryViTSDPA` (ViTDet windowed/global SDPA attention + conv neck). A new `vision_tower.py` composes them (dual SAM + CLIP branch → LLM-dim tokens, DeepSeek-OCR DeepEncoder style), and `latticelabs_ocadr.py` is wired to accept `pixel_values` (single or multi-view), encode them, and splice the image tokens into the LLM input at `image_token_id` alongside the 3D mesh tokens (guarded by `config.use_vision`; 3D-only behavior unchanged when no images are supplied). Note: the model encodes 3D meshes; this adds an *optional* rendered-image modality the empty files were originally meant for. Tests: `ll_ocadr/tests/test_vision_modality.py` (7, incl. end-to-end token-splice with a stubbed LLM).

---

## LOW — honestly-labeled approximations, fallbacks, or cosmetic

**LOW remediation 2026-06-09 (genuine bugs/deceptions among the LOWs — FIXED):**
- `generation_metrics.py::_is_valid_shape` now **fails closed** (returns False + WARNING) when pythonocc can't validate a TopoDS_Shape, instead of "assume valid" (which inflated `validity_rate`).
- Non-deterministic `hash()` → **deterministic** `stable_hash` (new `cadling/lib/hashing.py`, BLAKE2b) at every site that writes token ids / feature values into data: `stepnet_integration.py` (entity-type feature), `chunker/tokenizer/tokenizer.py`, `sdg/qa/sequence_annotator.py` (4 sites). Verified reproducible across `PYTHONHASHSEED`.
- `brep_graph_builder._compute_edge_features` convexity is now a **real signed** centroid-plane test (1.0 convex / 0.0 concave / 0.5 tangent), not angle-magnitude (which cannot sign a 90° edge).
- `ll_clouds/registration.py` `inlier_rmse` now computed over **inliers only** (matching its docstring), not all correspondences.
- `uv_net._sample_face_placeholder` is now called with a loud WARNING (synthetic grid no longer enters the CNN silently).
- Misleading comments corrected: `geometry_extractors.py` "Placeholder classes" (above real Pocket/Boss extractors) and `feature_recognition.py` "For now, classify as generic" (above a real hole/boss classifier); `graph_utils._compute_dihedral_angles` comment now states it returns unsigned angles by design (consistent with the trimesh path).

Still open (DISCLOSED-honest — labeled approximations/fallbacks, NOT deceptions, so left as-is): `_encode_fallback` (tagged `hash_fallback`), `constraint_predictor` empty-when-untrained, `gan_trainer.fid_approx` (named `*_approx`), `text_cad_annotator` placeholder renders (labeled+logged), `ll_ocadr` vLLM `EXPERIMENTAL / NOT WIRED` block + inert `get_num_mesh_tokens` divergence, BOM grouping-by-name (`assembly_hierarchy_pipeline.py`, whose tests pre-existingly hang).

Disclosed/contained (logged warnings, error tags, or last-resort fallbacks):
- `ll_gen` conditioning `_encode_fallback` returns hash-seeded random vectors when `ll_stepnet` absent — tagged `source_model="hash_fallback"` (`text_encoder.py:205`, `image_encoder.py:190`).
- `ll_gen/conditioning/constraint_predictor.py:272` — `predict_from_embeddings` returns `[]` when the (never-trained) learned MLP is unset; rule-based path is the real default.
- `cadling` non-determinism / approximations: `stepnet_integration.py:813` (`hash()` node feature, PYTHONHASHSEED-salted), `graph_utils.py:219` & `brep_graph_builder.py:420` (unsigned concave/convex from angle magnitude only), `uv_net.py:196` (`_sample_face_placeholder` random grid for unmatched faces), `sdg/qa/text_cad_annotator.py:550` (gray placeholder render views, labeled), `sequence_annotator.py:636` (hash-based fallback tokenizer), `generation_metrics.py:117` (`_is_valid_shape` returns True when OCC unavailable — inflates validity_rate), `evaluation/generation_metrics.py`.
- `ll_stepnet`: `gan_trainer.py:416` / `streaming_gan_trainer.py:447` `fid_approx` is a diagonal-moment proxy (named `*_approx`); `tasks.py:74,398` `forward` raises NotImplementedError by design (use `generate()`).
- `ll_ocadr`: `latticelabs_ocadr.py:488,519` & `run_ll_ocadr.py:198` — vLLM serving path returns plain dicts/strings instead of real vLLM objects, but the whole block is labeled `EXPERIMENTAL / NOT WIRED`; `get_num_mesh_tokens` token-count formula diverges from the processor (latent bug, inert until wired).
- `ll_clouds/registration.py:84-91` — `icp` `inlier_rmse` is computed over *all* correspondences despite the "inlier" docstring (cosmetic naming).
- `cadling` BOM grouping by name only — `assembly_hierarchy_pipeline.py:868` (`should use geometry hash`).

Cosmetic stale comments above working code (delete to avoid future confusion):
- `geometry_extractors.py:357` — "Placeholder classes for other feature extractors" sits above fully-implemented Pocket/Boss extractors.
- `feature_recognition.py:323` — "For now, classify as generic cylindrical feature" sits above a real hole/boss classifier.

---

## Test honesty (deceptive tests / coverage of the findings)

- **A test enshrines the always-zero metrics (M-metrics).** `ll_gen/tests/test_training.py:42-44` asserts `metrics.coverage == 0.0`, `metrics.mmd == 0.0`, `metrics.jsd == 0.0` for the no-reference path. It's testing the *default*, but it codifies the shipped behavior (gate always emits 0.0) as correct rather than flagging that production never supplies reference points. The same file's `:67` asserts `coverage == 0.75` on the with-reference path, so real computation *is* covered — just never exercised by the actual gate callers.
- **The new scorer test documents H1's gradient decoupling as intended.** `ll_gen/tests/test_log_prob_scorer.py` is a *good* regression test for the VAE/VQ-VAE scorer (`_connected_to_params` asserts the gradient reaches model params). But its own docstring states the diffusion RL gradient "stays on `proposal.log_probs` from `generate_for_training`" and that diffusion's scorer returns `(None, 0.0)` as "honest not-applicable." So the suite **codifies H1 as a deliberate deferral** rather than catching the inert-training-mode footgun — there is no test asserting that `diffusion`'s `generate_for_training` RL signal connects to parameters (because it doesn't).
- No test was found that asserts the fabricated hole-depth (`*2.0`) or the fail-open `matches: True` as correct — those deceptions are simply untested.

## Note on `cadling/docs/RequiredToBeCorrected.md`

The doc is **substantially stale**. Of 24 verified high-priority items, **22 are now implemented** (geometry analysis, face adjacency in `models/`, mate detection, overlap/interference, surface curvature, mesh quality, watershed, topology validation), 0 are genuinely still-partial code, and the 2 "remaining" are the cosmetic stale comments above. Its line numbers are all stale (files have grown); method names still resolve. **Do not trust its "~200 methods" headline count.** Caveat: this was a HIGH-priority-weighted sample, not a full sweep of its long tail.

---

## Open Questions (not determinable from code alone)
- Is the `diffusion` generator actually used in any real training run, or only `vae`/`vqvae`? (Determines whether H1 has shipped impact.)
- Have any GNN datasets already been generated and published from the three fabricated-adjacency builders **before** the 2026-06-09 fix (H2–H4)? If so, those artifacts are corrupted and should be regenerated with the fixed builders. (The `cadling hub build` CLI never routed to the B-Rep builder before the fix, so any corrupted dataset would have come from direct Python-API use.)
- Are the `evaluate_validity` / `rl_trainer.evaluate` gates ever expected to receive `reference_points`? If not, coverage/MMD/JSD should be removed from `summary()` rather than reported as 0.0.

---

## Remediation summary (2026-06-09)
Fixed during this session (H2–H4 + CLI wiring):
- New shared helper `cadling/cadling/lib/topology/brep_face_graph.py` — real shared-edge face adjacency + real face normals/curvature + edge dihedral/signed-convexity (planar **and** curved).
- `brep_graph_builder.py`, `arrow_brep_builder.py`, `webdataset.py` delegate to it (fabricated `min(i+5,…)` adjacency removed).
- `cadling hub build --type brep_graphs` wired to `BRepGraphBuilder` (`cli/hub.py`) so real B-Rep graph datasets build end-to-end from the CLI.
- Tests: `tests/unit/lib/topology/test_brep_face_graph.py` (13), `tests/unit/cli/test_hub_build.py` (3); full topology suite green; new files lint-clean.

Still open from this audit (not yet addressed): H1 (diffusion inert RL), H5 (hole-depth fabrication), and the MEDIUM/LOW findings above.

---

## Addendum (2026-06-10) — STEP tokenizer silently parsed zero entities — DECEPTIVE → **FIXED**

Discovered while fixing the two assembly test hangs (Mock shapes passing `is not None` guards into OCC C++): fixing the hangs let the full cadling suite run to completion and surfaced ~10 STEP integration failures (`test_step_pipeline`, `test_step_end_to_end`) plus tokenizer unit failures, all root-caused to a real broken-parser bug.

`cadling/cadling/backend/step/tokenizer.py::parse_step_file` (the **basic** STEP parser the backend falls back to when OCC produces no shape, `parsing_method: "basic"`) called `self._normalize_whitespace(content)` — which correctly collapses newlines (STEP is whitespace-insensitive outside string literals) — and then `content.split('\n')` and matched sections with `line.startswith('DATA;')`. After normalization there are **no newlines**, so the whole file became one "line", the `DATA;`/`HEADER;` section markers never matched, and `_parse_entities([])` returned **0 entities** for every file. The backend then produced a document with **0 CAD items** while reporting success — a parser that silently yields nothing.

Two coupled bugs, both fixed:
1. **Section/statement splitting** — `parse_step_file` now splits the normalized content into STEP *statements* (terminated by `;`) via a new string-literal-aware `_split_statements` helper (a `;` inside a quoted value such as `FILE_DESCRIPTION(...,'2;1')` does **not** terminate a statement), and classifies `HEADER`/`DATA`/`ENDSEC`/`ISO-10303` markers from the statement keywords rather than from `\n`.
2. **Empty-string literal in the multiline collector** — `_collect_multiline_entity` tracked string boundaries with a `prev_char == "'"` check that treated the **empty string `''`** (open-quote immediately followed by close-quote, ubiquitous in `CARTESIAN_POINT('',(…))`) as an *escaped* quote, leaving `in_string` stuck open so the terminating `;` was never seen and consecutive entities were glued together. Replaced with the correct lookahead rule (a doubled `''` is an escape **only when already inside a string**), matching `_split_statements`. This also fixes a latent bug for genuine multiline STEP files containing empty-string literals.

Verified: `parse_step_file` returns the correct 3 entities for a HEADER+DATA file (string-aware split confirmed on the `'2;1'` literal); `tests/unit/backend/step/test_tokenizer.py` + `tests/integration/test_step_pipeline.py` + `tests/integration/test_step_end_to_end.py` go from 11 failed → 0 (91 passed). Full cadling unit+integration suite: **45 → 26 failures** (1485 → 1497 passed); all 26 remaining failures are in subsystems this work did not modify (`git diff --name-only` confirms), so the fix is a clean net improvement with no new breakage.

### Coda — `_parse_single_param` numeric contract (resolved)
The above work surfaced one more pre-existing failure: `test_parse_single_param_number_string` asserted numbers were **kept as strings**, while `STEPTokenizer._parse_single_param` coerces numeric tokens to `int`/`float`. Determined the canonical contract from the code (not the test): **coerced numbers**. Every consumer treats numeric params as numbers on its *primary* path and only falls back to string-reparse defensively — `feature_extractor._extract_numeric_features`/`_extract_coordinates` branch on `isinstance(param,(int,float))` first (`feature_extractor.py:447-451`, `501-506`), `stepnet_integration._tokenize_param` quantizes numeric params as `[NUM_*]` tokens but routes strings through the enum vocabulary (`stepnet_integration.py:401-414`), and `stepnet_integration._parse_step_param` performs the identical int/float coercion (`377-383`). A number left as a string would be mis-tokenized as an unknown enum. The implementation was therefore correct and load-bearing; the **test was stale/deceptive** (its "kept as strings" docstring actively misrepresented the design). Aligned to the canonical contract: corrected the test (now `test_parse_single_param_number_coerced`, asserts `int`/`float` results + the non-numeric→`str` boundary) and tightened the `_parse_single_param` docstring to document the numeric contract explicitly. Suite: 27 → 26 failures (1497 passed).
