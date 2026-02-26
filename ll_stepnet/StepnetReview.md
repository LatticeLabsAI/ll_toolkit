# Code Review: ll_stepnet

## Summary

ll_stepnet is an ambitious neural network package for STEP/B-Rep CAD files with strong architectural separation (tokenizer → encoder → generative models → trainers). However, the codebase has several **critical training-correctness bugs** (gradient zeroing order, unregistered projection layers, broken caption embedding paths), **security vulnerabilities** (unsafe `torch.load` across 7+ files), and significant **code duplication** between streaming and non-streaming trainer variants. The generation pipeline's fallback decode paths create random projection layers on every call, making VQ-VAE and Diffusion generation non-functional.

## Findings

### Critical

1. **`STEPTrainer.train_epoch` zeros gradients after backward — model never learns** (`trainer.py:131`) — The order is `loss.backward()` → `optimizer.zero_grad()` → `optimizer.step()`. Gradients are wiped before the step, so every parameter update uses zero gradients. The model will never converge. Correct order: `zero_grad()` → `forward` → `backward` → `step`.

2. **`torch.load` without `weights_only=True` — arbitrary code execution** (`trainer.py:262`, `vae_trainer.py:599`, `diffusion_trainer.py:586`, `gan_trainer.py:513`, `streaming_vae_trainer.py:558`, `streaming_diffusion_trainer.py:643`, `streaming_gan_trainer.py:623`, `examples/inference_example.py:61,97`) — All checkpoint loading uses `torch.load()` without `weights_only=True`. Malicious checkpoints can execute arbitrary Python via pickle. Fix: add `weights_only=True` (PyTorch ≥ 2.0).

3. **Unregistered projection layers created fresh on every forward/generate call** (`encoder.py:398-403`, `generation_pipeline.py:674,701,730-745`) — New `nn.Linear` instances with random Xavier weights are instantiated per call, never trained, never stored. Results are non-deterministic garbage. These must be created once as `nn.Module` attributes.

4. **`STEPForCaptioning.forward()` feeds raw integer token IDs to TransformerDecoder** (`tasks.py:65`) — `caption_ids` is unsqueezed to `[B, T, 1]` and cast to float without embedding. The decoder expects `[B, T, d_model]`. The model will fail to converge. Token IDs must pass through `nn.Embedding` first.

5. **Unbounded `log_var` causes KL divergence overflow in STEPVAE** (`vae.py:326-328`) — `log_var.exp()` is unclamped. Large positive `log_var` values overflow to Inf, NaN-poisoning KL loss and all gradients. Standard fix: `log_var = torch.clamp(log_var, -30, 20)`.

6. **PNDM buffer not reset between diffusion stages** (`diffusion.py:75,157-158,509`) — `_pndm_ets` accumulates noise predictions across all 4 stages. Stages 2-4 use stale predictions from prior stages in the Adams-Bashforth formula. `reset_pndm()` must be called before each stage's timestep loop.

7. **`UnifiedTrainer` passes wrong arguments to all inner trainers** (`unified_trainer.py:194-206,344-348,362,375,387`) — `GANTrainer`/`DiffusionTrainer` constructors receive wrong keyword arguments. `train_epoch()`, `validate()`, `save_checkpoint()`, `load_checkpoint()` are all called with incompatible signatures. The unified trainer will raise `TypeError` on first use.

8. **VAE decoder uses `memory=hidden` (self-attention instead of cross-attention on z)** (`vae.py:248`) — Passing `hidden` as both `tgt` and `memory` means the decoder cross-attends to itself, not to the latent z. The z conditioning is not properly injected.

### High

9. **`STEPForCausalLM.generate()` permanently sets model to eval mode** (`pretrain.py:161`) — `self.eval()` inside `generate()` persists after the call. If called mid-training, all subsequent training runs without dropout.

10. **`STEPTokenizer.encode()` uses Python's non-deterministic `hash()` for unknown tokens** (`tokenizer.py:119`) — `hash()` varies across processes/restarts (PYTHONHASHSEED). Embeddings learned in one process won't transfer. Use `hashlib` for deterministic hashing.

11. **`StreamingVAETrainer.train_step` assumes tuple model output** (`streaming_vae_trainer.py:328`) — `reconstructed, mu, log_var = self.model(input_ids)` crashes if the model returns a dict (STEPVAE convention). The non-streaming `VAETrainer` has `_unpack_model_output()` to handle this; the streaming variant lacks it.

12. **`_beam_search_decode` only masks beam indices 1 and N+1** (`beam_search.py:325`) — For `num_beams=4`, beams at indices 2,3 are left at score 0.0 instead of `-inf`. Correct: `beam_scores.view(batch_size, num_beams)[:, 1:] = float("-inf")`.

13. **`GANTrainer.validate` uses training data as reference distribution** (`gan_trainer.py:346`) — Validation metrics are computed against `self.train_dataloader`, contaminating evaluation.

14. **`PrefetchingIterator` sentinel may be dropped on full queue** (`streaming_processor.py:193-201`) — If the queue is full when the worker thread raises an exception, `put(sentinel, timeout=1.0)` raises `queue.Full`, which is caught with `pass`. The main thread deadlocks waiting.

15. **`STEPForCausalLM`/`STEPForMaskedLM` use `torch.randn` for missing features** (`pretrain.py:106,285`) — Random noise as default features introduces non-determinism. `STEPEncoder` uses `torch.zeros` for the same case. Use zeros consistently.

16. **`mask_tokens` mutates caller's `input_ids` in-place** (`pretrain.py:440,445`) — Without cloning `input_ids` first, shared/pinned DataLoader tensors are silently corrupted.

17. **`data.py` STEP parsing crashes on malformed DATA sections** (`data.py:89-93`) — `content.index('ENDSEC;', data_start)` raises `ValueError` if `ENDSEC;` is absent.

18. **Hardcoded personal path in tool** (`tools/deduplicate_methods.py:302`) — `gatgpt_core = "/Users/ryanoboyle/gatgpt/core"` fails for any other developer.

19. **`DiffusionTrainer` `noise_mse` metric is identical to `loss`** (`diffusion_trainer.py:284-285`) — `total_noise_mse += loss.item()` computes the same value as `total_loss`. The metric provides no additional information.

### Medium

20. **Massive code duplication across streaming trainers** — `_build_dataset_from_config` is copy-pasted identically 3 times (`streaming_vae_trainer.py:187-230`, `streaming_diffusion_trainer.py:210-252`, `streaming_gan_trainer.py:222-264`). `_create_scheduler` duplicated twice. Extract to shared utilities.

21. **`StreamingDiffusionTrainer._add_noise` uses linear schedule vs cosine in non-streaming** (`streaming_diffusion_trainer.py:299-303`) — The two variants use different noise schedules, producing different training dynamics.

22. **`mask_tokens` ignores caller-supplied `replace_prob`/`random_prob` parameters** (`pretrain.py:407-408,439,443`) — The function hardcodes 80/10/10 ratios. The parameters are dead code.

23. **GNN message passing lacks self-loops** (`encoder.py:279-308`) — Standard GCN includes self-loops (`adj + I`). The current implementation only aggregates neighbor features.

24. **`STEPQAConfig` uses same embedding for questions and answers** (`tasks.py:398,475`) — Despite config exposing separate `step_vocab_size` and `text_vocab_size`, a single `question_embedding` encodes both.

25. **`STEPAnnotatedOutput.format()` mutates instance state** (`annotations.py:134`) — Violates Command-Query Separation. Calling `format()` has side effects.

26. **`visualize_latent_space` loads entire validation set into RAM** (`vae_trainer.py:413-431`) — No subsampling before t-SNE. Will OOM on large datasets.

27. **EMA codebook device mismatch risk** (`vqvae.py:241-243`) — In-place buffer operations assume same device as `flat_inputs`. Multi-GPU scenarios may fail silently.

28. **`DisentangledCodebooks.decode_quantized()` depends on state set by `encode()`** (`vqvae.py:505-519`) — Instance attributes are not in `state_dict`, not moved by `.to()`, and race in multi-threaded contexts.

29. **`_reconstruct` catches all `Exception`, swallowing bugs** (`generation_pipeline.py:902-905`) — Programming errors (AttributeError, TypeError) are silently converted to result dicts.

30. **`test_with_step_files.py` uses hardcoded relative paths** (`tests/test_with_step_files.py:41-47`) — Fails when pytest runs from any directory other than `ll_stepnet/`.

31. **`test_streaming_integration.py` lambda captures loop variable** (`tests/test_streaming_integration.py:109-113`) — All lambdas capture final value of `i`. Fix: `lambda i=i: {"id": i}`.

32. **`test_temperature_effects` tests nothing about temperature** (`tests/test_beam_search.py:610-641`) — Only asserts outputs are `not None`. A no-op temperature would pass.

33. **`pretrain_unsupervised.py` processes batch element-by-element** (`examples/pretrain_unsupervised.py:195-207`) — Python loop inside training loop defeats batching, running at ~1/batch_size GPU utilization. Users will copy this pattern.

34. **`LatentGenerator` uses `BatchNorm1d` which fails at batch_size=1** (`latent_gan.py:51`) — Use `LayerNorm` or `InstanceNorm1d` for inference compatibility.

35. **`VertexRefinementHead` produces unbounded positions** (`vertex_prediction.py:319-320`) — Coarse positions are bounded by `tanh` but refinement steps can push beyond `[-1, 1]`.

36. **`CADGenerationError` dataclass + Exception is pickle-incompatible** (`generation/errors.py:22-23`) — Breaks in multiprocessing DataLoader contexts with `num_workers > 0`.

### Low

37. **`STEPTokenizerConfig` defined but never consumed by `STEPTokenizer`** (`config.py:10-23`) — The config and tokenizer are disconnected.

38. **`max_length` parameter accepted but unused in `STEPForCausalLM`** (`pretrain.py:37`) — Stored nowhere, positional buffer hardcoded to 5000.

39. **Missing `from __future__ import annotations`** in 6+ core modules — Inconsistent with CLAUDE.md convention.

40. **`math.pi` approximated as `3.14159` in LR scheduler lambdas** (`streaming_vae_trainer.py:183`, `streaming_diffusion_trainer.py:206`) — Use `math.pi` for precision.

41. **`GANTrainer.train` never updates `best_wasserstein_dist`** (`gan_trainer.py:427-474`) — Best-model checkpointing is broken for epoch-based GAN training.

42. **History list lengths inconsistent when validation absent** (`vae_trainer.py:521-533`) — Train/val metric lists have different lengths, breaking downstream analysis.

43. **`DecodingStrategy.DIVERSE_BEAM` exported but not implemented** (`beam_search.py:107`) — Falls through to standard beam search silently.

44. **`output_heads.py` uses absolute import `from stepnet.vertex_prediction`** (`output_heads.py:174`) — Should be relative `from .vertex_prediction` for portability.

45. **Legacy trainer imports torch unconditionally** (`trainer.py:6-13`) — Unlike `training/` trainers which guard imports, this crashes if torch is missing.

## Strengths

- **Clean separation of concerns** — Tokenizer, feature extractor, topology builder, and encoder are independent, single-responsibility components that compose well.

- **Excellent lazy import discipline** — Heavy optional deps (matplotlib, scipy, transformers, geotoken, cadling, trimesh) are consistently lazy-imported with actionable error messages.

- **Robust DFS traversal** (`reserialization.py:287-336`) — Iterative DFS with explicit stack, visited-set, configurable max-depth avoids Python recursion limits.

- **`_unpack_model_output` in VAETrainer** — Auto-detection of dict vs. tuple model output with one-time logging is backward-compatible and clean.

- **EMA model implementation** — Correct in-place `mul_` + `add_` without gradient tracking in diffusion trainers, included in checkpoints.

- **Structured error hierarchy** (`generation/errors.py`) — Rich `CADGenerationError` with `recoverable`, `context`, `original_exception` fields and factory methods.

- **Fallback handler design** (`generation/fallbacks.py`) — Per-error-type strategy dispatch, bounded retry counts, fixed-size cache for substitution, configurable exhaustion strategy.

- **Cross-package integration tests** — `test_geotoken_bridge.py` and `test_native_integration.py` verify `CommandType` values, `PARAMETER_MASKS`, and feature dimensionality stay in sync across packages.

- **OpenMP guard in conftest.py** — Torch-first import, `OMP_NUM_THREADS=1`, and linked upstream issue documentation is defense-in-depth for macOS.

- **`STEPEncoder.prepare_topology_data`** — Accepts both dict and cadling `TopologyGraph` with defensive edge-index clipping.

## Recommendations

**Priority 1 — Training-breaking bugs:**
1. Fix gradient zeroing order in `trainer.py` (finding #1)
2. Add `weights_only=True` to all `torch.load` calls (finding #2)
3. Register projection layers as `nn.Module` attributes (finding #3)
4. Add `nn.Embedding` for caption token IDs (finding #4)
5. Clamp `log_var` in STEPVAE (finding #5)
6. Reset PNDM buffer between diffusion stages (finding #6)
7. Fix `UnifiedTrainer` argument passing (finding #7)

**Priority 2 — Correctness:**
8. Replace `hash()` with deterministic hash in tokenizer (finding #10)
9. Add `_unpack_model_output` to streaming VAE trainer (finding #11)
10. Fix beam score initialization (finding #12)
11. Clone `input_ids` in `mask_tokens` (finding #16)

**Priority 3 — Architecture:**
12. Extract shared trainer utilities to eliminate 3x code duplication (finding #20)
13. Unify noise schedule between streaming/non-streaming diffusion trainers (finding #21)
14. Remove `.eval()` side effects from `generate()` methods (findings #9, related)

**Priority 4 — Testing:**
15. Fix lambda variable capture in streaming tests (finding #31)
16. Add meaningful assertions to temperature and callback tests (findings #32, related)
17. Use `Path(__file__)` for test file paths (finding #30)
