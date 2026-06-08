# Plan M6 — Cross-Cutting Quality Gate + Docs Truth-Up

| | |
|---|---|
| **Spec** | SPEC-1 §5 M6, §3.1 FR-G5, §9.7/9.8 |
| **Goal** | All three packages pass lint/type/test gates with zero stubs; status docs reflect the achieved state with no over-claiming. |
| **Depends on** | M1–M5 (this is the closing gate) |
| **Owner** | Maintainer · **Mode** Inline/sequential · **Tests** run all suites · **Commits** per task |
| **Status** | Not started |

## Pre-flight
- Confirm M1–M5 done checklists are all ticked.
- Have the active conda env with torch/cadquery/pythonocc available, or explicitly record which gates can't run locally (don't fake green).

## Tasks

### T6.1 — Lint + format + types
- Run and fix to clean:
  ```bash
  ruff check ll_gen/ll_gen ll_ocadr ll_clouds/ll_clouds
  black --check ll_gen ll_ocadr ll_clouds
  mypy ll_gen/ll_gen ll_ocadr/vllm ll_clouds/ll_clouds
  ```
- Fix real issues; do not blanket-`# noqa`/`# type: ignore` to silence — justify any suppression inline.
- **Commit:** `style(ll_gen,ll_ocadr,ll_clouds): ruff/black/mypy clean`

### T6.2 — Stub & fabrication scan (must be zero)
- Across `ll_gen/`, `ll_ocadr/`, `ll_clouds/`:
  ```bash
  grep -rnE "TODO|FIXME|XXX|raise NotImplementedError" --include=*.py
  grep -rnE "^\s*pass\s*$" --include=*.py        # inspect each: only legit (abstract/protocol) allowed
  grep -rniE "placeholder|not yet implemented|using defaults|for now|in a real" --include=*.py
  ```
- Any incomplete logic found → implement it (no leaving stubs). Any hardcoded/fake return where real logic belongs → replace with real computation.
- **Commit:** `fix(ll_gen,ll_ocadr,ll_clouds): eliminate residual stubs/placeholders`

### T6.3 — Full test run, per package
- ```bash
  cd ll_gen   && pytest -q
  cd ll_ocadr && pytest -q -m "not slow"
  cd ll_clouds&& pytest -q --cov=ll_clouds
  ```
- Zero failures; record skip reasons (env-gated) explicitly. Fix any regression introduced across milestones.
- **Commit (if fixes):** `test: fix regressions surfaced by full-suite run`

### T6.4 — Truth-up `STATUS.md`
- Update the per-package table and the four-axis verdict in `STATUS.md`: `ll_gen` neural track wired/runnable (+checkpoints if M3 done), `ll_ocadr` HF-native+tested, `ll_clouds` real. Keep the honest framing — only claim what's verified; cite the new tests.
- **Commit:** `docs: update STATUS.md to reflect ll_gen/ll_ocadr/ll_clouds completion`

### T6.5 — Truth-up root `README.md`
- Update the package table (`ll-clouds` now real; `ll_gen` neural path working; `ll_ocadr` HF-native). Remove any "DeepSeek-OCR works" / working-vLLM implications (NG5 framing stays a separate concern, but don't assert falsehoods).
- Ensure documented commands actually run (the README examples).
- **Commit:** `docs: update README package table + runnable examples`

### T6.6 — Spec closeout
- In `docs/specs/SPEC-1-...md`, fill any Open Questions that got resolved during execution (OQ1–OQ5) with the actual decisions; flip Status to `Done` (or note remaining deferred items per NG1/NG2).
- **Commit:** `docs(spec): close out SPEC-1 open questions + status`

## Verification (the gate)
```bash
# all must pass / be clean:
ruff check ll_gen/ll_gen ll_ocadr ll_clouds/ll_clouds
black --check ll_gen ll_ocadr ll_clouds
mypy ll_gen/ll_gen ll_ocadr/vllm ll_clouds/ll_clouds
( cd ll_gen && pytest -q ) && ( cd ll_ocadr && pytest -q -m "not slow" ) && ( cd ll_clouds && pytest -q )
grep -rnE "TODO|FIXME|raise NotImplementedError" ll_gen/ll_gen ll_ocadr ll_clouds/ll_clouds --include=*.py   # expect empty
```

## Milestone risks
- **Env can't run a gate locally** — record exactly which gate is unverified and why (e.g., no CadQuery); do not report it green. Per repo rule: no premature "100% passing" claims without evidence.
- **A late fix breaks an earlier milestone's test** — full-suite run in T6.3 is the safety net; treat any failure as a regression to fix, not skip.

## Done checklist
- [ ] ruff/black/mypy clean across the 3 packages.
- [ ] Stub/fabrication scan returns zero (or only justified abstract `pass`).
- [ ] All three suites pass (skips documented).
- [ ] `STATUS.md` + `README.md` updated, no over-claiming, examples runnable.
- [ ] SPEC-1 open questions closed; status flipped.
