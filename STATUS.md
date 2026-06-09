# LatticeLabs Toolkit — Viability & Status Report

> Reality-based assessment of the codebase against the real-world state of the art (2024–2026).
> Generated 2026-06-08. Findings verified by reading/running the actual code and by web research.
> **Updated 2026-06-08** after milestones M1, M2, M4, M5 (see the "Update" section) — the
> original audit's "partially unwired / empty ll_clouds" findings are now resolved; the
> *untrained* finding stands (training is M3).

---

## One-sentence verdict

The code is impressively **real** and the ideas are mostly **grounded in current research**; the neural-propose track and training loop now **wire up and run end-to-end**, but the models remain **untrained** (so outputs are noise until M3) and the project is **aimed at a market dominated by funded teams** — "runs end-to-end" is still a step short of "viable product."

---

## Update — milestones since the initial audit (2026-06-08)

The four "critical wiring defects" below were the original audit's findings; they have been
addressed. Status of each spec milestone (`docs/specs/SPEC-1-...md`):

| Milestone | Scope | Status | Evidence |
|---|---|---|---|
| **M1** | Fix the dead `ll_gen` neural-propose imports + surfaced API bugs | ✅ merged (PR #4) | VAE/diffusion/VQ-VAE now construct, propose, and reach dispose on CPU; `ll_gen/tests/test_neural_imports.py`, `test_orchestrator_neural.py`. Defect #1 below is **resolved**. |
| **M2** | `ll_gen` RL training loop runnable | ✅ open (PR #7) | Live-graph log-probs (VAE/VQ-VAE), real `train_step` gradient update, `python -m ll_gen.training.run` CLI, checkpoint round-trip; `test_generate_for_training.py`, `test_rl_trainer.py`, `test_training_cli.py`, `test_checkpoint_roundtrip.py`. |
| **M4** | `ll_ocadr` HF-native inference + pytest suite | ✅ merged (PR #5) | `run_ll_ocadr_hf.py` runs mesh→text on a tiny offline LM; 21 passed / 2 skipped. vLLM now documented as future (defect #3 **resolved/clarified**). |
| **M5** | `ll_clouds` core point-cloud library | ✅ merged (PR #6) | Built from the empty scaffold: I/O, preprocessing, features, ICP, RANSAC/clustering, lazy bridges; **74 tests, 93% coverage**, ruff/black/mypy clean. Defect #4 below is **resolved**. |
| **M3** | Train proof-of-life checkpoints | ✅ proof-of-life achieved (real closed solids) | RL on the **real OCC/cadquery dispose reward** drives **actual closed-solid generation 3% → 66%** on the deployment `generate()` path (BRepCheck-validity 4% → 96%; per-epoch validity 0.13 → 0.94). **The honesty path that got here:** the first reward reward-hacked — with no `target_dimensions` it was pure topological validity, which accepts a lone planar face, so ~90% of "valid" outputs were non-solid stacks of circles (actual solids only 3% → 7%). Fixed by gating `validity_reward` on `solid_count ≥ 1` (0.1 partial credit for valid non-solids); retraining then produced real solids two-thirds of the time. **Honest limits:** the solids are **predominantly cylinders** (circle+extrude, low structural variety). A **dimension-conditioning follow-up** (dense dimensional reward + a zero-init latent dimension-encoder, threaded through training) was attempted and is a **documented NEGATIVE**: across small/medium/large target boxes the achieved bbox stayed ~constant (didn't track the request) in a 480-step CPU budget — the conditioning architecture is correct and verified, but REINFORCE-through-discrete-sampling gives too weak a dimensional gradient at this scale (future work: GPU/more steps, heavier dim-reward, or param-token conditioning). Real bugs fixed en route (all tested): executor schema (was 0% for *any* sequence), decode-path unification (`_decode_and_sample`, now used by all four VAE paths), FR-G5 nested-checkpoint loading. Real DeepCAD subset wired (`palapav/DeepCAD-DSL`); supervised warm-start + dimension-conditioning = documented **negatives**. Full record: `docs/plans/2026-06-08-m3-llgen-checkpoints.md`. |
| **M6** | Cross-cutting quality gate + this truth-up | ✅ done (2026-06-09) | All three packages **ruff + black + mypy clean** (`ruff check ll_gen/ll_gen ll_ocadr ll_clouds/ll_clouds`, `black --check`, `mypy ll_gen/ll_gen ll_ocadr/vllm ll_clouds/ll_clouds` → 0 errors); stub scan = 0 (the `NotImplementedError` scorer + diffusion zero-grid placeholder were replaced with real implementations); suites green (ll_gen 1322 / ll_ocadr 23 / ll_clouds 74). ll_ocadr got a package-local ruff config matching its siblings; dynamic torch/OCC/numpy modules use scoped mypy overrides (repo's "start lenient" stance). |

**Net change to the verdict:** the original "two of five headline paths don't execute as wired"
and "ll_clouds is an empty scaffold" are no longer true. What has **not** changed: no trained
checkpoints exist (every neural output is still noise until M3), and the commercial reality is
unchanged.

---

## "Viable" splits into four questions — they get different answers

| Axis | Verdict | Evidence |
|---|---|---|
| **Is it real code, not stubs?** | ✅ Largely yes | Real GCN message-passing, WGAN-GP / DDPM / REINFORCE loops, PointNet++ / ViT encoders, sandboxed subprocess CadQuery execution, OCC-based geometry analysis. `cadling/docs/RequiredToBeCorrected.md` placeholders are genuinely built out; **0** `raise NotImplementedError`, **~1** TODO tree-wide. geotoken passes **357 tests at 82.7% coverage**. |
| **Does it run end-to-end today?** | ◐ Wires + runs, but untrained | **Now wired:** the neural propose→dispose path, the RL `train_step`/`train_epoch` loop, the `ll_ocadr` HF-native mesh→text path, and the `ll_clouds` library all run end-to-end (M1/M2/M4/M5). **Still untrained:** no checkpoints exist, so every neural *output* is noise until the loops are run on real data (M3). The original "two paths don't execute / ll_clouds empty" defects are resolved. |
| **Are the approaches grounded in reality?** | ✅ 4 of 5 | See Technical Bets below. |
| **Is it commercially winnable solo?** | ⚠️ Hard on the strongest axes | Science is open and reproducible, but the moat (proprietary CAD datasets, kernel access, platform distribution, trust) favors Autodesk / Onshape-PTC / Zoo.dev. Industry's own 2026 verdict: *"most AI CAD tools solve problems engineers don't actually have."* ([Leo AI](https://www.getleo.ai/blog/ai-cad-design-2026-whats-real)) |

---

## Critical wiring defects (verified by hand, not inferred)

> **Historical (original audit).** Defects #1, #3, #4 below are **RESOLVED** (M1, M4, M5);
> #2 (no trained weights) **still stands** — training is M3. Kept here for the record and to
> document what the fixes addressed.

### 1. ll_gen neural-propose / RL track imports nonexistent modules — entire track is dead code  — ✅ RESOLVED (M1)

- `ll_gen/ll_gen/generators/neural_vae.py:430` → `from ll_stepnet.stepnet.models import STEPVAE`
- `ll_gen/ll_gen/generators/neural_diffusion.py:418` → `from ll_stepnet.stepnet.models import StructuredDiffusion`
- `ll_gen/ll_gen/generators/neural_vqvae.py:424` → `from ll_stepnet.stepnet.models import VQVAEModel`
- `ll_gen/ll_gen/generators/neural_vae.py:431` / `neural_vqvae.py:425` → `from ll_stepnet.stepnet.pipeline import CADGenerationPipeline`

**Reality:**
- `ll_stepnet/stepnet/models.py` **does not exist** (no module, no `models/` package, no re-export shim).
- `ll_stepnet/stepnet/pipeline.py` **does not exist** — the file is `ll_stepnet/stepnet/generation_pipeline.py`.
- The real classes live in `ll_stepnet/stepnet/vae.py`, `diffusion.py`, `vqvae.py`.
- The `ll_stepnet.` prefix is **also** wrong: the package directory is `stepnet/` and the project's own README uses `from stepnet.encoder import StepNetEncoder`, so the importable top-level name is `stepnet`, not `ll_stepnet.stepnet`.

**Consequence:** `generate_for_training()` → `_init_model()` raises `ImportError`, so the REINFORCE loop in `ll_gen/ll_gen/training/rl_trainer.py` has nothing to train. This is a stale refactor (classes moved out of an aggregator that was never replaced) — likely the single highest-value, lowest-effort fix in the repo, because it's blocking the project's *strongest* bet (#2).

### 2. No trained weights exist for any of the project's own models

Tree-wide `find` for `*.pt / *.pth / *.safetensors / *.ckpt / *.bin` returns only:
- `resources/cad_operations/cad-feature-detection/files/TITAN-1M.bin`
- `resources/cad_operations/cad-feature-detection/feature_detector/model/best.ckpt`

Both are external third-party CAD-feature-detection assets — **nothing for ll_stepnet / ll_gen / ll_ocadr / geotoken**. Every neural model in the project is randomly initialized; outputs are noise until the (real) training loops are run on real data. The one exception is ll_ocadr, which loads a *pretrained HF LLM* via `from_pretrained`, but its 3D encoders are untrained.

### 3. ll_ocadr "vLLM integration" is aspirational

`ll_ocadr/vllm/latticelabs_ocadr.py` defines a genuine HF multimodal `nn.Module`, but the class is **not** registered via vLLM's `ModelRegistry` and does **not** inherit `SupportsMultiModal`, so it cannot actually run inside vLLM as written. Only `ll_ocadr/run_ll_ocadr.py:21` touches the real vLLM API. ll_ocadr also has **no pytest suite** — `ll_ocadr/test_ll_ocadr.py` is a manual `__main__` CLI script.

### 4. ll_clouds is an empty scaffold

`ll_clouds/` contains only `pyproject.toml` — zero source files. Cannot be installed, imported, or tested.

---

## Per-package status

| Package | LOC | Verdict | Notes |
|---|---|---|---|
| **geotoken** | ~15K | ✅ **REAL & COMPLETE** | Strongest package. Real adaptive quantization (`geotoken/quantization/adaptive.py`), verified round-trip (`tests/unit/test_adaptive.py:18-26`, `assert np.max(errors) < 0.05`). 357 tests pass, 82.7% coverage. |
| **ll_stepnet** | ~28K | ✅ **REAL BUT UNTRAINED** | Correct symmetric-normalized sparse GCN (`encoder.py:397-451`, forward+backward verified), real `nn.TransformerEncoder` (`encoder.py:38-73`), real CE losses in pretrain (`pretrain.py:62-330`), genuine WGAN-GP (`training/gan_trainer.py:166-200`) and DDPM. No checkpoints. |
| **cadling** | ~121K | ✅ **PARTIAL → REAL** | Real OCC `STEPControl_Reader` path + text fallback (`backend/step/step_backend.py:336-389`); flagged placeholders now genuinely implemented — mate detection via OCC `BRepExtrema` (`models/assembly_analysis.py:416-593`), 1682-line graph builder, 1719-line geometry extractors. SDG is real LLM calls (`sdg/qa/generate.py:277,289,357,391`), not templated filler. |
| **ll_gen** | ~34K | ✅ **REAL, WIRED, UNTRAINED** (M1+M2) | Deterministic dispose + LLM-code propose real; **neural-latent propose + RL track now wired** (M1) and the RL loop runs (M2): live-graph log-probs (VAE/VQ-VAE), real `train_step` update, `python -m ll_gen.training.run` CLI, checkpoint round-trip. Diffusion RL is a documented decoupled exception. Untrained (M3). ruff/black clean. |
| **ll_ocadr** | ~5K | ✅ **REAL, HF-NATIVE, TESTED** (M4) | Real PointNet++ + ViT encoders; **HF-native inference path** (`run_ll_ocadr_hf.py`) runs mesh→text; **real pytest suite** (21 passed / 2 skipped). vLLM honestly documented as experimental/future. 3D encoders untrained. |
| **ll_clouds** | ~1.5K | ✅ **REAL & COMPLETE** (M5) | Built from empty: PointCloud models, PLY/PCD/XYZ I/O + mesh sampling, normalize/voxel/FPS/outlier preprocessing, k-NN PCA normals + curvature, ICP, RANSAC plane + DBSCAN clustering, lazy cadling/ll_ocadr bridges. 74 tests, 93% coverage, ruff/black/mypy clean. |

> Note: a subagent estimated "~85–90% of the codebase is genuine implementation." That is a hand-wave, not a measured figure — treat it as directional. What is *measured*: geotoken's 357 passing tests, the verified forward/backward passes, and the verified-by-hand defects above.

---

## Technical bets vs. real-world state of the art

### Bet 1 — Neural nets on B-Rep / STEP (ll_stepnet): *grounded research, thin in production*
Face segmentation / feature recognition on B-Rep genuinely works — BRepNet ~90.2–92.5%, UV-Net ~89% on Fusion 360 segmentation. Hard limit: **representation instability** (the same solid has many valid B-Reps; an [Autodesk patent](https://image-ppubs.uspto.gov/dirsearch-public/print/downloadPdf/12288013) calls consistent output across B-Reps "difficult, if not impossible"). **Autodesk is essentially the only production user.** Published IP, not a market.
Refs: [BRepNet](https://arxiv.org/pdf/2104.00706) · [UV-Net](https://arxiv.org/pdf/2006.10211) · [GDL-for-CAD survey](https://arxiv.org/pdf/2402.17695)

### Bet 2 — Generative CAD "neural propose / deterministic dispose" (ll_gen): *strongest bet*
Propose-with-neural, validate-with-deterministic-execution is the **convergent 2025–26 SOTA architecture**. Text2CAD-Bench (2026) confirms your representation choice: **CadQuery code beats command-sequence** targets (DeepSeek invalidity 13% → 67% when switching away from code). Honest reliability numbers: validity is real but degrades sharply — invalidity 11–24% on easy prompts, **68–93% on hard prompts**; domain-tuned models often produce *executable but geometrically wrong* output (which is exactly why the deterministic-validate stage matters). Live market: **Zoo.dev / KittyCAD** (Sequoia-backed) ships a commercial propose-then-validate loop. The irony: this is your best idea *and* your most broken wiring (defect #1).
Refs: [Text2CAD-Bench](https://arxiv.org/html/2605.18430) · [CADCodeVerify (ICLR'25)](https://arxiv.org/abs/2410.05340) · [CAD-Coder](https://arxiv.org/pdf/2505.19713) · [SkexGen](https://arxiv.org/pdf/2207.04632) · [Zoo](https://zoo.dev/zookeeper)

### Bet 3 — Geometric/mesh tokenization with adaptive quantization (geotoken): *sound and SOTA-aligned*
Coordinate quantization underpins the entire autoregressive-mesh field (MeshGPT residual VQ, MeshAnything V2, Meshtron at 1024 levels / 64k faces). "Adaptive" non-uniform bit allocation is a legitimate refinement, not a gimmick. Core tension it can't escape: coarse = lost geometry, fine = exploding sequence length.
Refs: [MeshGPT](https://nihalsid.github.io/mesh-gpt/) · [MeshAnything V2](https://arxiv.org/html/2408.02555v1) · [Compressive tokenization](https://arxiv.org/html/2411.07025v1)

### Bet 4 — "Docling for CAD" (cadling): *least-validated as a named niche, but closer to a real edge than it looks*
**STEP-LLM (Jan 2026)** independently published nearly your exact DFS-reserialization + RAG-over-STEP techniques on ~40K STEP-caption pairs — which both validates the method *and* proves a serious team is already in the lane. But "STEP/STL/BRep → structured doc → RAG → synthetic Q&A" is **not yet a recognized commercial niche**; real demand sits in *engineering-knowledge search* (Leo AI, Onshape AI Advisor), which is text-adjacent, not geometry-RAG. Position accordingly.
Refs: [STEP-LLM](https://arxiv.org/abs/2601.12641) · [Onshape AI Advisor](https://www.onshape.com/en/blog/ai-advisor-artificial-intelligence-cad-engineering-software) · [Leo AI](https://www.getleo.ai/blog/ai-cad-design-2026-whats-real)

### Bet 5 — "DeepSeek-OCR-inspired 3D for LLMs" (ll_ocadr): *weakest; the analogy breaks*
[DeepSeek-OCR](https://arxiv.org/abs/2510.18234) works because rendered text has an **exact reconstruction target** to grade against; a point cloud has no canonical token string, so the optical-compression → exact-decode trick has nothing to anchor to. The field already feeds 3D to LLMs via point-encoder → projector ([PointLLM](https://arxiv.org/abs/2308.16911), ShapeLLM) **without** the OCR framing — which is, in fact, what `latticelabs_ocadr.py` actually implements. The OCR analogy is marketing wrapped around a sound-but-different mechanism, and it's the single most criticizable claim in the README.

---

## Recommended next steps (highest-leverage first)

1. **Fix the broken wiring (defect #1).** Repoint ll_gen's generator imports to the real modules (`stepnet.vae` / `stepnet.diffusion` / `stepnet.vqvae`, `stepnet.generation_pipeline`) and drop the wrong `ll_stepnet.` prefix. Add a smoke test that imports and runs one generation step so this can't regress silently. This unblocks your strongest bet for near-zero effort.
2. **Train *something*, even small.** One trained checkpoint on one path turns "scaffolding" into "demo." Until then, every neural output is noise and the project can't be evaluated on results.
3. **Lead with #2, reframe #4, cut/rebuild #5.** Make propose-then-validate the headline (convergent SOTA + live market). Position cadling as engineering-knowledge Q&A (proven demand), not geometry-RAG (unproven). Remove the DeepSeek-OCR claim — keep the PointLLM/ShapeLLM-style encoder→projector that the code already is.
4. **Add pytest coverage to ll_ocadr** and either register the model into vLLM properly (`ModelRegistry` + `SupportsMultiModal`) or stop describing it as a vLLM integration.
5. **Resolve ll_clouds** — implement it or remove the empty scaffold.
6. **Don't try to out-build Autodesk/Zoo on generation.** Solo, the open lane is the unglamorous plumbing (structured extraction, tokenization, validation harnesses) that funded teams under-invest in — not the model that needs their datasets and kernel.

---

## Bottom line

Not vaporware, not naïve — a genuinely-built, research-grounded toolkit that is roughly **one integration pass and one training run away from demoing**, and a very long way (data, distribution, trust, funding) from **winning** the markets it targets. The realism gap is not in code quality; it's in the distance between "the algorithms are real" and "the system produces correct CAD that someone will pay for."
