# SPEC-2 Documentation Site — Accuracy Review (M6 / T6.3)

**Date:** 2026-06-10
**Reviewer:** Maintainer + Claude
**Scope:** Every published page of the docs site (`site/`) checked against the
actual code. Public-grade accuracy bar (SPEC-2 NFR-ACC, FR-16/17/18).

## Method

Each capability claim and code example was verified against the source tree by
reading the relevant module (`file:line` where it mattered). Where a claim came
from a package README that itself disagreed with the code, the **code won** and
the page was corrected.

## Findings and corrections

| # | Page | Claim (as written) | Reality (code) | Fix |
|---|---|---|---|---|
| 1 | `ll_gen/usage`, `tutorials/generate-cad` | `GenerationOrchestrator.generate(...)` returns `result.valid` / `result.metrics` | `DisposalResult` has `is_valid` and `geometry_report` (`ll_gen/ll_gen/proposals/disposal_result.py:170`) | Corrected to `is_valid` / `geometry_report` (+ `step_path`) |
| 2 | `ll_gen/usage`, `tutorials/generate-cad` | `force_route=GenerationRoute.CODE` | Enum members are `CODE_CADQUERY`, `CODE_OPENSCAD`, `CODE_PYTHONOCC`, `NEURAL_VAE/DIFFUSION/VQVAE` (`ll_gen/ll_gen/config.py:22`) | Corrected to `CODE_CADQUERY` (+ listed all members) |
| 3 | `get-started/quickstart` | `from stepnet.encoder import StepNetEncoder; encoder.encode(step_data)` (from root README) | Class is `STEPEncoder` (`ll_stepnet/stepnet/encoder.py:454`); no `.encode()` — called via `__call__(token_ids, topology_data=...)` | Replaced with the real tokenizer→features→topology→`STEPEncoder(...)` pipeline |
| 4 | `tutorials/parse-a-step-file` | `Path("part.json").write_text(doc.export_to_json())` | `export_to_json()` returns a **dict** (`cadling/.../base_models.py:486`), not a string | Wrapped in `json.dumps(...)` |
| 5 | `geotoken/overview` | "best-tested package (460+ test cases)" | `grep -rc "def test_"` → **435** test functions | Softened to "400+ tests" (defensible lower bound) |

## Verified accurate (checked against code)

Because the **root README was shown to be unreliable** (findings 3 and 5 above),
all README-derived code examples on public pages were re-verified against the
source, not trusted on faith:

- **`ll_ocadr` examples** — `build_model_and_tokenizer(...)` returns the 4-tuple
  `(model, tokenizer, config, processor)` (`run_ll_ocadr_hf.py:91`); and
  `run_inference(model, processor, tokenizer, mesh_file, prompt, ...)` matches the
  documented call order (`run_ll_ocadr_hf.py:94`). ✓
- **`geotoken` examples** — `analyze_impact(...)` returns an object with
  `.mean_error` and `.hausdorff_distance` (`impact/analyzer.py:29-30`);
  `CommandSequenceTokenizer().tokenize(...)` returns a `TokenSequence` with
  `.command_tokens` (`tokenizer/token_types.py:272`). ✓
- **`ll_ocadr` vLLM framing** — every ll_ocadr page presents vLLM as
  experimental / not-functional (`ll_ocadr/README.md`; SPEC-1 §FR-O4). No page
  claims working vLLM. ✓ (FR-17)
- **`ll_gen` "models ship untrained"** — grounded in `proof_of_life.py` and the
  root README; reward gated on a closed solid. ✓
- **`ll_brepnet` = empty scaffold** — all files 0 bytes; documented only as a
  Roadmap stub, never a shipping package. ✓ (FR-7b, R10)
- **`ll_clouds` API** — `PointCloud`, `icp`, `ransac_plane`, `euclidean_cluster`,
  preprocessing/feature signatures verified directly from source.
- **cadling chunk API** — `chunk.text`, `chunk.meta`, `chunk.chunk_id`,
  `chunk.meta.entity_ids` verified (`cadling/.../chunker/base_chunker.py:73`).
- **cadling `doc.topology` / `TopologyGraph.adjacency_list`** verified
  (`base_models.py:420,272`).
- **Generated API reference** — produced *from* the code by `gen_api.py` (static
  griffe parse), so it cannot drift from signatures; each symbol links to its
  GitHub source line.

## Maturity badges (per-package, verified honest)

| Package | Badge | Basis |
|---|---|---|
| cadling | Beta | broad + complete, some methods still hardening (RequiredToBeCorrected) |
| ll_stepnet | Untrained | architectures + trainer present, no checkpoints |
| geotoken | Stable | pure NumPy, 435 tests, best-tested |
| ll_ocadr | HF-native | HF path real/tested; vLLM experimental |
| ll_gen | Untrained | propose→dispose + RL real; generators untrained |
| ll_clouds | Core | installable core library, full suite |

## Verdict

After corrections 1–5, and re-verification of the README-derived `ll_ocadr` and
`geotoken` examples against source, **no published page asserts a capability the
code does not back**. Coverage: every neural/generation code example and every
example sourced from the (unreliable) root README was checked line-by-line
against the code; package-README-derived examples were spot-checked at their
return/parameter contracts; the generated API reference is produced *from* the
code and cannot drift. The accuracy bar (NFR-ACC) is met for launch.
