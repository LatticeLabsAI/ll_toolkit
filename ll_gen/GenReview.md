# Code Review: ll_gen (Round 3)

## Summary

ll_gen's "Neural Propose, Deterministic Dispose" architecture is structurally sound after the prior fix rounds. The subprocess sandbox, gradient flow restoration, OCC API corrections, and shared tokenization are all in place. The remaining findings cluster around three themes: (1) RL training correctness issues (`zero_grad` ordering, EMA baseline bias, generate-without-no_grad memory leak), (2) OCC API misuses in secondary paths (line edges, shell repair, surface executor imports), and (3) test quality gaps (vacuous assertions, zero coverage of `train_step`/`generate`/`generate_batch`). No new security vulnerabilities were found.

## Findings

### Critical

- **`zero_grad()` called after `backward()`, risking stale gradient accumulation** (`training/rl_trainer.py:214`) — The optimizer zeroes gradients after the loss backward pass, not before the teacher-forcing forward. If `generate()` (called at line 164 without `torch.no_grad()`) builds a computation graph, stale gradients from that inference pass pollute the update. Correct order: `zero_grad()` → forward → `backward()` → clip → `step()`.

- **`generate()` runs without `torch.no_grad()` during `train_step`** (`generators/neural_vae.py:78`, `neural_vqvae.py:80`) — During RL training the model is in `.train()` mode, so the inference-time generation step accumulates a full computation graph in memory. This is both a correctness problem (gradients flow through the wrong trajectory) and a memory leak that will OOM on long training runs.

- **`_get_log_probs` teacher-forcing signature may not match model's real forward API** (`training/rl_trainer.py:344`) — Calls `self.generator._model(token_ids_t.unsqueeze(0))` assuming a simple `(input_ids) → logits` signature. VAE/VQ-VAE models have encoder/decoder paths with conditioning inputs. If the output shape doesn't match `(1, seq_len, vocab_size)`, log-probs are silently computed on garbage dimensions.

- **Subprocess stdout polluted by user `print()` calls breaks JSON IPC** (`disposal/code_executor.py:386-391`) — The wrapper scripts execute user code via `exec()` in the subprocess and emit JSON via `print(json.dumps(meta))`. Any user code that also calls `print()` (common for debugging) prepends text to stdout, causing `json.loads(stdout)` to fail with `JSONDecodeError`. Fix: write JSON to a separate file, or redirect user stdout to stderr in the wrapper.

- **`quantize_parameter` and dequantizer have incompatible normalization** (`datasets/_tokenization.py:59-87`, `proposals/command_proposal.py:240`) — Tokenizer computes `value / normalization_range` and clamps to `[0,1]`, discarding negative values. Dequantizer expects symmetric `[-range/2, +range/2]`. DeepCAD coordinates in `[-1,1]` lose their sign, producing systematically wrong geometry on round-trip.

### High

- **EMA baseline initialization biases early training** (`training/rl_trainer.py:177-183`) — Baseline initialized at `0.0` with decay `0.99`. For rewards in `[0,1]`, the first ~100 steps have inflated advantages, producing destabilizing policy updates. Should initialize with the first observed reward.

- **`_logits_to_token_ids` batch-dimension ambiguity** (`generators/base.py:243-244`) — `command_logits[0]` strips the first dimension regardless of whether it's batch or sequence. For 2D input `(seq_len, vocab)`, this peels the first timestep, silently decoding only `seq_len - 1` tokens.

- **`generate_candidates` reuses `self._model.last_latent` for all candidates** (`generators/neural_vae.py:181-193`) — The model stores only the last-generated latent. In a batch of N candidates, every candidate except the last gets the wrong latent vector.

- **`_apply_shell_fixes` does not write back repaired shells** (`disposal/repairer.py:348-389`) — Unlike wire/face fixers which use `ShapeBuild_ReShape`, shell fixer returns the original unmodified shape. Shell repair is a silent no-op.

- **`_apply_boolean_operation` reads wrong key (`"parameters"` vs `"params"`)** (`disposal/command_executor.py:544`) — Key mismatch means boolean type always defaults to union. Cut and intersection operations never execute.

- **`_create_line_edge` passes `gp_Pnt2d` to `BRepBuilderAPI_MakeEdge`** (`disposal/command_executor.py:346`) — No valid pythonocc overload for 2D points. All LINE sketch commands produce SWIG `TypeError` or `None` edges. Should use `gp_Pnt(x, y, 0.0)`.

- **`brepbndlib_Add` and `topexp_MapShapes` imported incorrectly** (`disposal/surface_executor.py:30,33`) — These are module-level instances in pythonocc (`brepbndlib.Add`, `topexp.MapShapes`), not standalone functions. The imports fail silently inside the broad `try/except ImportError`, setting `_OCC_AVAILABLE = False` even when pythonocc is installed. The entire surface executor is disabled.

- **`_run_in_subprocess` returns inconsistent types** (`disposal/code_executor.py:399-417`) — Returns either `TopoDS_Shape` or `dict`. No type guard in `DisposalEngine._execute` before passing to `validate_shape`, which will crash on a dict.

- **Export only triggers on last retry attempt** (`pipeline/orchestrator.py:194`) — `export and (attempt == retries)` means shapes that succeed on attempt 1 of 3 are never exported. The loop breaks on success, so the export flag is always `False` for early successes.

- **`HybridShapeEncoder.state_dict()`/`load_state_dict()` incompatible with PyTorch conventions** (`embeddings/hybrid_encoder.py:388-440`) — Custom nested-dict format breaks `torch.save`/`torch.load(..., weights_only=True)` and standard PyTorch tooling.

- **`decode_latents` passes numpy array where tensor expected** (`generators/latent_sampler.py:280`) — After converting to tensor for `set_latent()`, the original numpy `latent` is passed to `decode()`. The tensor conversion is wasted.

- **Reward config docstrings diverge from actual defaults** (`config.py:207-212`, `feedback/reward_signal.py:9-15`) — Docstrings say `+1.0/+0.3/+0.2/+0.1` but actual defaults are `0.8/0.16/0.0/0.16`. Max achievable reward before clamp is ~1.8, compressing gradient signal.

- **`train_step` has zero test coverage** (`tests/test_training.py`) — Core RL path (advantage computation, gradient clipping, baseline update) is entirely untested.

- **Sandbox security tests are vacuous** (`tests/test_code_executor.py:516-537`) — Assert only that a locally-constructed list has length > 0. Never exercise the real sandbox.

### Medium

- **`_create_placeholder_face_grids` computes `num_faces` but returns `[]`** (`generators/neural_diffusion.py:437-448`) — Dead variable; method name promises placeholders but returns nothing.

- **`compute_mmd` reduces point clouds to centroids** (`training/metrics.py:144-145`) — Collapses all shape information. Same-centroid, different-geometry shapes score MMD=0.

- **`generate_from_error_context` uses string comparison for error categories** (`generators/neural_vae.py:253-258`) — Compares against `"topology_error"` etc. If caller passes `ErrorCategory` enum instead of string value, all comparisons fail silently.

- **`train_epoch` mutates caller's dataset list in place** (`training/rl_trainer.py:413`) — `np.random.shuffle(dataset)` is a side-effect callers don't expect.

- **`HybridShapeEncoder.forward()` calls `.detach()` unconditionally** (`embeddings/hybrid_encoder.py:267`) — Breaks gradient flow for end-to-end fine-tuning.

- **`_try_initialize_gnn` initial import probe is dead code** (`embeddings/hybrid_encoder.py:169-179`) — The result is never used; actual initialization always happens in the second block.

- **EOS token may be appended twice** (`generators/base.py:259-299`) — Natural termination via `EOS_CMD_TOKEN_ID=11` appends EOS, then the final guard appends `EOS_TOKEN_ID=2` again.

- **`save_checkpoint` incompatible with `weights_only=True` load** (`training/rl_trainer.py:548`) — Stores `train_history` (list of dicts) via `torch.save`, which fails with `weights_only=True`.

- **Massive duplication: lazy-import helpers in all 4 dataset loaders** — Identical `_get_torch()`, `_get_datasets()`, `_get_numpy()`, `_get_geotoken()` patterns.

- **`SketchGraphsDataset.__getitem__` duplicates `_tokenize_sketchgraphs_sample`** (~170 lines copy-pasted).

- **`_check_euler` hardcodes `== 2` for all geometries** (`feedback/feedback_builder.py:404-408`) — Fails for tori (χ=0), through-holes (χ=0). Penalizes valid non-genus-0 shapes.

- **`BOPAlgo_*` keys in `OCC_ERROR_MAP` silently missing from `BRepCheck` module** (`feedback/error_mapper.py:459-463`) — `getattr(brep_mod, "BOPAlgo_AlertTooFewArguments")` returns `None`; BOPAlgo errors never surface.

- **`validate_shape` assumes valid on any exception** (`disposal/engine.py:155-162`) — Broad `except Exception` marks `is_valid = True`. OCC crashes produce false-positive validation.

- **`_sample_edge_points` divides by zero when `n_samples=1`** (`disposal/surface_executor.py:403`)

- **CLIP threshold 0.25 on softmax over 4 images is effectively "above average"** (`pipeline/verification.py:374`) — Nearly always passes regardless of semantic match.

- **Dead `_timeout_handler` and `signal` import** (`disposal/code_executor.py:17,54-64`) — Never registered; creates false impression SIGALRM is active.

- **Conditional-export assertions in disposal tests** (`tests/test_disposal.py:741-755`) — `if result.step_path:` guard makes assertions vacuous.

- **`generate()` and `generate_batch()` never behaviorally tested** (`tests/test_pipeline.py`) — Only `hasattr`/`callable` checks.

- **`test_verify_multi_dimension_pattern` asserts `>= 0`** (`tests/test_verification.py:149`) — Always true for any list.

- **Conditional confidence assertions** (`tests/test_verification.py:338-344`, `tests/test_pipeline.py:458`) — `if result.matches_intent:` guard skips assertion when verification fails.

### Low

- **`seed` parameter accepted but unused in `interpolate()`** (`generators/latent_sampler.py:57`)
- **f-string logging throughout** vs `%s` convention in other modules
- **`import copy` deferred inside `with_error_context`** (`proposals/base.py:92`) — stdlib import; no benefit to deferring
- **`LatentProposal` imports numpy unconditionally at module level** (`proposals/latent_proposal.py:20`)
- **`DisposalResult.to_dict()` omits `render_paths`** (`proposals/disposal_result.py:267-327`)
- **`HybridShapeEncoder.to()` only accepts `str` device** (`embeddings/hybrid_encoder.py:374`) — Breaks `encoder.to(torch.float16)`
- **`PARAM_OFFSET` redefined locally in `rl_trainer.py:275`** — Duplicates `generators/base.py`
- **`_geotoken` lazy accessor declared but never called** in deepcad/text2cad/sketchgraphs loaders
- **`TimeoutError` shadows built-in** (`disposal/code_executor.py:48`)
- **`step_failed` initialized inside `except` branch** (`training/rl_trainer.py:233`) — Should initialize before `try`
- **Duplicate `VerificationResult` test classes** across `test_pipeline.py` and `test_verification.py`
- **Non-deterministic `command_proposal` fixture** (`tests/conftest.py:337`) — Uses unseeded `np.random.randn(256)`
- **`test_summary_with_nan_values` avoids actual NaN** (`tests/test_training.py:863`) — `float("nan") if False else 0.5`

## Strengths

- **Subprocess-isolated code execution** — All LLM-generated CAD code runs in a child process via `subprocess.run()` with timeout. No `exec()` in the parent. Wrapper script pattern is clean and auditable.

- **Correct REINFORCE teacher-forcing formulation** — `_get_log_probs` correctly shifts inputs/targets by one position, applies `log_softmax`, gathers per-token probabilities, and computes entropy for exploration bonus.

- **Numerically stable SLERP interpolation** — Near-parallel fallback, proper epsilon guard, normalization before dot product, `arccos` domain clipping. A subtle algorithm implemented correctly.

- **Exhaustive OCC error catalog** — 37 `BRepCheck_Status` codes mapped to 6 neural-interpretable categories with severity, descriptions, and correction suggestions. Production-quality error mapping.

- **Dense tiered reward signal** — Provides gradient signal at every validation stage rather than sparse binary valid/invalid. The tier design (shape exists → manifold → watertight → Euler → no self-intersection) is architecturally sound.

- **Deterministic hash-based fallback embeddings** — SHA-256 seeded `RandomState` produces reproducible pseudo-embeddings when transformers unavailable. Same prompt always produces same embedding with stored metadata for traceability.

- **Shared tokenization kernel** — `_tokenization.py` centralizes constants and functions. `validate_token_space` proactively catches misconfiguration before tokenizing.

- **Layered OCC repair pipeline** — `ShapeFix_Shape` → wire → face → fuzzy tolerance escalation with `ShapeBuild_ReShape` write-back and re-validation between passes (wire and face correctly implemented).

- **Consistent lazy-import pattern** — Every heavy dependency guarded behind `try/except ImportError` with `_AVAILABLE` flags and graceful degradation. Full package importable without torch/OCC/cadquery.

- **Comprehensive test dependency isolation** — Skip markers consistently applied. Full unit suite (1222 tests) runs in any environment.

- **High-quality test fixtures** — Realistic domain-accurate values (correct surface areas, real BRepCheck codes, proper Euler characteristics). Subprocess IPC boundary tests cover all 5 outcome branches.

## Recommendations

**Priority 1 — RL Training Correctness:**
1. Move `zero_grad()` before the teacher-forcing forward pass, not after `backward()`
2. Wrap `generate()` in `torch.no_grad()` during `train_step` to prevent graph accumulation
3. Initialize EMA baseline with first observed reward instead of `0.0`
4. Validate that `_get_log_probs` model call matches the actual model's forward signature

**Priority 2 — Data Pipeline Correctness:**
5. Fix `quantize_parameter` to handle signed coordinates: `(value + range/2) / range`
6. Fix subprocess JSON IPC to use a separate output file instead of stdout
7. Add type guard for `_run_in_subprocess` mixed return types

**Priority 3 — OCC API Fixes:**
8. Fix `_create_line_edge` to use `gp_Pnt(x, y, 0.0)` instead of `gp_Pnt2d`
9. Fix `_apply_shell_fixes` to use `ShapeBuild_ReShape` (matching wire/face pattern)
10. Fix `_apply_boolean_operation` key: `"params"` not `"parameters"`
11. Fix `surface_executor.py` imports: `brepbndlib` and `topexp` are module instances
12. Fix export condition: trigger on success, not just last attempt

**Priority 4 — Test Coverage:**
13. Add behavioral tests for `train_step`, `generate()`, `generate_batch()`
14. Replace vacuous sandbox tests with real execution-through-sandbox tests
15. Replace all conditional assertions with unconditional ones
16. Seed all random state in test fixtures
