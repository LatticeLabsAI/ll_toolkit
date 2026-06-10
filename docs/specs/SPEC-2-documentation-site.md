# SPEC-2: LatticeLabs Toolkit Documentation Site

> A unified Astro + Starlight documentation site for the six implemented LatticeLabs packages (plus an honest roadmap entry for the empty `ll_brepnet` scaffold), built by curating existing prose, authoring the gaps, and auto-generating API reference from docstrings — held to a public-grade accuracy bar.

| Field | Value |
|---|---|
| **Spec ID** | SPEC-2 |
| **Title** | LatticeLabs Toolkit Documentation Site (`site/`, Astro + Starlight) |
| **Status** | Draft |
| **Version** | 1.0 |
| **Author** | LatticeLabs (LayerDynamics) + Claude |
| **Owner** | Maintainer (solo) |
| **Created** | 2026-06-10 |
| **Source material** | `site/` scaffold (Starlight starter), `README.md` (root package table), per-package READMEs + `docs/` (cadling, geotoken, ll_stepnet, ll_ocadr, ll_clouds), root `docs/` conceptual essays, `docs/specs/SPEC-1-*.md` (house style), `docs/2026-06-09-partial-deceptive-code-audit.md` (accuracy-risk evidence) |
| **Related** | `docs/specs/SPEC-1-ll-gen-ocadr-clouds-completion.md`, `CLAUDE.md` (repo conventions), `docs/HowNNsGenerateCAD.md`, `docs/HowCADGenerationModelsActuallyWorkInside.md`, `docs/AICADGenerationDoesNotWorkHowYouThink.md`, `docs/GenerationImplementation.md` |

---

## 1. Background

### 1.1 Problem Statement

The LatticeLabs Toolkit is a monorepo of **six implemented packages** — `cadling`, `ll_stepnet`, `geotoken`, `ll_ocadr`, `ll_gen`, `ll_clouds` (the exact set in the canonical root `README.md` table) — plus an **empty, untracked `ll_brepnet` scaffold** (every file 0 bytes; not in the README). Its documentation is **scattered, uneven, and partly stale**. Today a reader who wants to understand or use the toolkit faces three problems:

1. **No single entry point.** Documentation lives in per-package `README.md` files of wildly varying depth (cadling 20 KB, ll_clouds 1 KB, `ll_gen` has *no* README; `ll_brepnet` has no README **and no code**), plus a `docs/` folder per a couple of packages, plus ~10 conceptual essays in the root `docs/`. There is no navigable, searchable, cross-linked surface.

2. **The `site/` scaffold is empty.** `site/` is a freshly-generated Starlight starter (`@astrojs/starlight ^0.40.0`, `astro ^6.4.5`). It still carries the stock title "My Docs" (`site/astro.config.mjs:9`), a placeholder `withastro/starlight` GitHub social link (`:10`), a starter sidebar (`:11-23`), and the stock starter `README.md`. Seven content directories exist (`site/src/content/docs/{cadling,geotoken,ll_brepnet,ll_clouds,ll_gen,ll_ocadr,ll_stepnet}/`) — note one per *implemented* package **plus a `ll_brepnet/` dir for the not-yet-built package** — but every one is **empty**; the only real content is `index.mdx` (default splash), `guides/example.md`, and `reference/example.md`.

3. **Some existing docs over-claim.** This monorepo has a *documented, recurring* gap between prose and code: `docs/2026-06-09-partial-deceptive-code-audit.md`, the stale `cadling/docs/RequiredToBeCorrected.md`, `cadling/docs/UNIMPLEMENTED.md`, and SPEC-1's own commit trail (`docs(ll_gen): correct M3 overclaim …`, `docs(m6): truth-up STATUS.md, README, SPEC-1 …`). Publishing the existing prose verbatim would broadcast claims the code does not back (e.g. "vLLM integration works", "trained models ship").

### 1.2 Current State

| Surface | What exists today | Gap |
|---|---|---|
| `site/` | Starlight starter: `astro.config.mjs` (title "My Docs", placeholder sidebar), `index.mdx` (default splash), `guides/example.md`, `reference/example.md`, `src/content.config.ts`, default `README.md` | De-scaffold; build real IA |
| `cadling` | `README.md` (20 KB) + `docs/`: `Overview.md`, `Purpose.md`, `Development.md`, `Plan.md`, `Adjustments.md`, `visualization_plan.md` (+ internal: `RequiredToBeCorrected.md`, `UNIMPLEMENTED.md`, `Todo.md`, `ReviewFindings.md`, `plans/`) | Curate the user-facing subset; exclude internal status docs |
| `geotoken` | `README.md` (5 KB) + `docs/`: `architecture.md`, `integration.md`, `data-requirements.md`, `api/README.md`, `examples/README.md` | Best-documented package; migrate mostly as-is |
| `ll_stepnet` | `README.md` (7 KB), no `docs/` | Migrate README; author concepts + API |
| `ll_ocadr` | `README.md` (3 KB), no `docs/` | Migrate README **with vLLM-claim correction**; author API |
| `ll_gen` | **No README**, no `docs/` (described only in root `README.md`); **real code** (`ll_gen/ll_gen/…`) | Author from scratch (from code) |
| `ll_clouds` | `README.md` (1 KB), no `docs/`; **real code** | Author concepts + API |
| `ll_brepnet` | **No README, empty `docs/`, and every source file is 0 bytes** (untracked; `pyproject.toml`/`requirements.txt`/all 9 `.py` empty); not in root README | **Roadmap page only** — honestly mark "planned / not implemented"; no Usage/API |
| Root `docs/` | ~10 conceptual essays: `HowNNsGenerateCAD.md`, `HowCADGenerationModelsActuallyWorkInside.md`, `AICADGenerationDoesNotWorkHowYouThink.md`, `GenerationImplementation.md`, `HuggingfaceLatticeLabsIntegrationPlan.md`, `ResearchVSCodebase.md`, `training_data_requirements.md` (+ `plans/`, `specs/`, the audit) | Reframe essays as public explainers; exclude plans/specs/audit |
| Hosting / CI | None. No `firebase.json`, no `.firebaserc`, no `.github/workflows/`. (`firebase-debug.log` at root is from an unrelated MCP call.) | Build the deploy pipeline |

### 1.3 Target Users

**Both internal and external** (public-grade accuracy bar — confirmed in discovery).

- **External / open-source users** — engineers evaluating or adopting one or more packages. Need: what each package is, how to install it, runnable usage, conceptual explainers ("how does neural CAD generation actually work?"), and an honest maturity signal so they don't build on something that "ships untrained".
- **Internal contributors / the maintainer** — need: architecture orientation, development setup, the cross-package integration story (geotoken ↔ ll_stepnet ↔ cadling), and a reference that stays in sync with code.

### 1.4 Motivation

- **Adoption & legibility.** The toolkit spans CAD parsing, tokenization, three neural sub-projects, point clouds, and generation. Without a unified, searchable site, even the maintainer cannot quickly answer "where does X live and how do I call it?"
- **Accuracy as a feature.** SPEC-1 spent six milestones *truthing-up* over-claims. A docsite is the natural place to make the honest, verified state of each package the default public message — turning a liability (scattered, occasionally inflated docs) into a strength (one accurate surface).
- **The scaffold is already here.** `site/` exists; the cost to go from empty scaffold to real site is bounded and high-leverage.

### 1.5 Assumptions

- **A1** The repo stays on GitHub at `LatticeLabsAI/ll_toolkit` (confirmed: `git remote -v`). GitHub Pages project-site URL is `https://latticelabsai.github.io/ll_toolkit/`, base path `/ll_toolkit/`.
- **A2** Toolchain: Node ≥ 22.12.0 (Astro 6 engine floor; local is v24.8.0), Astro 6.4.5, Starlight 0.40.0. Static output only — no SSR/server runtime.
- **A3** Python packages use **Google-style docstrings** (`CLAUDE.md` → "Google-style docstrings: Required for public APIs"), which the API generator targets.
- **A4** Content authority order: **code > existing docs**. Where a doc and the code disagree, the code wins and the doc is corrected, not copied.
- **A5** English-only for v1; no localization.
- **A6** The site is documentation only — no interactive playground, no live model inference embedded in pages.

---

## 2. Goals and Non-Goals

### 2.1 Goals

- **G1 — One navigable, searchable site for all six implemented packages.** Every implemented package has, at minimum, Overview, Installation, Usage/How-to, and an API reference, reachable from a coherent sidebar and full-text search. The empty `ll_brepnet` scaffold gets a single honest roadmap page, not a fabricated package section.
- **G2 — Full Diátaxis depth** (confirmed in discovery): Tutorials (learning-oriented), How-to Guides (task-oriented), Reference (information-oriented, incl. generated API), and Explanation/Concepts (understanding-oriented).
- **G3 — Three-pronged content strategy** (confirmed: "migrate & curate … plus api from docstrings"):
  - **(a) Curate** — consolidate existing accurate prose into Starlight MDX.
  - **(b) Author** — write the gaps fresh (`ll_gen` from its code, tutorials, missing concepts; and a roadmap stub for `ll_brepnet`).
  - **(c) Generate** — auto-produce API reference from Python docstrings at build time.
- **G4 — Public-grade accuracy.** No published page claims behavior the code does not back. Maturity/status is explicit per package; capability claims are traceable to code (file/entry-point citations, SPEC-1 discipline).
- **G5 — Automated build & deploy.** A GitHub Actions workflow runs the API generator, `astro check`, internal link validation, and deploys to GitHub Pages on merge to `main`; the same checks gate PRs without deploying.
- **G6 — Low-drift by construction.** Generated API reference is regenerated every build; internal links are CI-validated; published content excludes internal status/planning docs that go stale.

### 2.2 Non-Goals

- **NG1 — No code changes to the six implemented packages** beyond, at most, docstring touch-ups required for the generator to read a public symbol. No refactors, no new features. **Writing `ll_brepnet`'s implementation is explicitly out of scope** — this spec documents what exists, it does not build the missing package. (Backfilling an `ll_gen` package README is OQ6, not in scope by default.)
- **NG2 — No publishing of internal status/planning docs.** `RequiredToBeCorrected.md`, `UNIMPLEMENTED.md`, `Todo.md`, `ReviewFindings.md`, `docs/plans/`, `docs/specs/`, and the deceptive-code audit are **not** user-facing pages. Contributor docs may link to them in-repo.
- **NG3 — No doc versioning in v1.** Single "latest" site tracking `main`. (Versioning = OQ5.)
- **NG4 — No localization / i18n in v1.**
- **NG5 — No SSR, no server, no auth.** Pure static site; no gated content.
- **NG6 — No interactive features** (live model demos, in-browser CAD viewers, runnable code sandboxes). Static prose, code blocks, and images only.
- **NG7 — No custom design system.** Use Starlight's theme with light branding (title, logo, colors); no bespoke component library.

---

## 3. Requirements

### 3.1 Functional Requirements

#### Site foundation & information architecture

| ID | Priority | Requirement |
|---|---|---|
| FR-1 | MUST | De-scaffold the starter: replace `title: 'My Docs'` (`site/astro.config.mjs:9`) and the placeholder GitHub social link `withastro/starlight` (`:10`); replace the stock `site/README.md`; replace the default splash `index.mdx` with a real landing page. |
| FR-2 | MUST | Configure GitHub-Pages project-site routing in `astro.config.mjs`: `site: 'https://latticelabsai.github.io'`, `base: '/ll_toolkit/'`, so all internal links and asset URLs resolve under the base path. |
| FR-3 | MUST | Top-level IA: a landing page plus a sidebar with one group per **implemented** package (6), a **Roadmap** entry for `ll_brepnet`, and the Diátaxis cross-cuts — **Get Started / Tutorials**, **How-to Guides**, **Concepts (Explanation)**, **Reference**, and **Contributing**. |
| FR-4 | MUST | The sidebar replaces the starter's `Guides`/`Reference` example groups and points at real content; the `guides/example.md` and `reference/example.md` placeholders are removed or repurposed. |

#### Content coverage (per package)

| ID | Priority | Requirement |
|---|---|---|
| FR-5 | MUST | Each of the **6 implemented** packages (cadling, ll_stepnet, geotoken, ll_ocadr, ll_gen, ll_clouds) has at least: **Overview** (what it is, one marquee capability), **Installation**, **Usage / How-to** (≥1 runnable example), and an **API Reference**. |
| FR-6 | MUST | Curate existing accurate prose into Starlight pages: `cadling` (README + `docs/Overview.md`, `Purpose.md`, `Development.md`), `geotoken` (`docs/architecture.md`, `integration.md`, `data-requirements.md`, `api/`, `examples/`), `ll_stepnet`/`ll_ocadr`/`ll_clouds` READMEs — correcting any claim the code does not back. |
| FR-7 | MUST | Author fresh docs for `ll_gen` (Overview, Install, Usage, Concepts), grounded in its **actual code** (`ll_gen/pipeline/orchestrator.py`, `python -m ll_gen.training.run`, public classes) — it has no README but is a real package. |
| FR-7b | MUST | `ll_brepnet` is represented by **exactly one honest roadmap page** stating it is a planned, not-yet-implemented B-Rep face-graph network (the directory contains only 0-byte scaffold files). It MUST NOT receive Installation, Usage, or API-reference pages, and MUST NOT appear in the package list as if shippable. (Accuracy: FR-16/17.) |
| FR-8 | MUST | A **Concepts/Explanation** section built by reframing the root `docs/` essays — `HowNNsGenerateCAD.md`, `HowCADGenerationModelsActuallyWorkInside.md`, `AICADGenerationDoesNotWorkHowYouThink.md`, `GenerationImplementation.md` — as public explainers (internal framing/status removed). |
| FR-9 | SHOULD | ≥1 end-to-end **Tutorial** per major workflow: (a) parse a STEP/STL file with `cadling`; (b) tokenize a mesh with `geotoken`; (c) run the `ll_gen` propose→dispose generation loop; (d) HF-native inference with `ll_ocadr` (`run_ll_ocadr_hf.py`). |
| FR-10 | MUST | Internal status/planning docs are **excluded** from the published site (see NG2). A build-time guard (allowlist or ignore-glob) prevents accidental inclusion. |

#### Auto-generated API reference

| ID | Priority | Requirement |
|---|---|---|
| FR-11 | MUST | A build-time generator parses the **public** Python modules of each package and emits API-reference MDX into `site/src/content/docs/<pkg>/reference/`. Parsing is **static** (AST-based, e.g. `griffe`) — it must not *import* the packages (avoids pulling heavy deps: torch, pythonocc, trimesh; avoids the OpenMP crash class, `CLAUDE.md` "OpenMP / PyTorch Import Order"). |
| FR-12 | MUST | The generator renders Google-style docstrings (summary, Args, Returns, Raises, Examples) into readable MDX; classes list public methods with signatures; modules list public functions/classes. |
| FR-13 | MUST | The generator runs as a **prebuild** step (before `astro build`/`astro check`) wired into `package.json` and CI. Generation failure fails the build (no silent partial output). |
| FR-14 | SHOULD | Each generated symbol page links its source location (`<pkg>/path/file.py:line`) for traceability (SPEC-1 discipline). |
| FR-15 | COULD | Generated API pages render a deprecation/"experimental" marker when the docstring carries one. |

#### Accuracy guardrails

| ID | Priority | Requirement |
|---|---|---|
| FR-16 | MUST | Every package landing page carries a **maturity/status badge** reflecting the real code state, e.g. `ll_gen` → "models ship untrained" (root `README.md`), `ll_ocadr` → "HF-native; vLLM experimental/future" (root `README.md`, SPEC-1 §FR-O4), `ll_clouds` → "core library" (SPEC-1 G4). |
| FR-17 | MUST | No published page asserts a capability the code does not back. Specifically: the `ll_ocadr` pages MUST present vLLM as experimental/future, never as working (corrects the historical over-claim; SPEC-1 §FR-O4, audit doc). |
| FR-18 | MUST | Capability claims on Overview/Concepts pages are traceable to code — each marquee claim cites a module, class, or entry point (e.g. `ll_gen` propose→dispose → `ll_gen/pipeline/orchestrator.py`; `python -m ll_gen.training.run`). |

#### Build quality, search, SEO

| ID | Priority | Requirement |
|---|---|---|
| FR-19 | MUST | `astro check` passes with zero errors (content-collection schema + types valid). |
| FR-20 | MUST | Internal link integrity is validated in CI; any broken internal link **fails the build** (link-checker over the built `dist/`, e.g. Starlight link-validation or `lychee` scoped to internal links). |
| FR-21 | MUST | Full-text search is enabled (Starlight's built-in Pagefind) and indexes all published pages. |
| FR-22 | SHOULD | Every page sets `title` + `description` frontmatter; the site emits a sitemap and Open Graph/social metadata for the landing + package pages. |
| FR-23 | COULD | A site logo/favicon replaces the starter `public/favicon.svg`; light brand colors set in Starlight config. |

#### CI/CD & deployment

| ID | Priority | Requirement |
|---|---|---|
| FR-24 | MUST | A GitHub Actions workflow (`.github/workflows/docs.yml`) on push to `main`: checkout → setup Node ≥22.12 (pinned) + Python (for the generator) → install → run API generator → `astro check` → `astro build` → internal link check → deploy to GitHub Pages (official `actions/deploy-pages`). |
| FR-25 | MUST | The same build + check steps run on pull requests **without** deploying, gating merges (PR build must pass). |
| FR-26 | SHOULD | The workflow is path-filtered to trigger on changes under `site/`, the six implemented package source trees (for regenerated API), and the workflow file itself. |
| FR-27 | SHOULD | Node and Python dependency caches are used to keep CI build < 3 min (NFR-1). |

### 3.2 Non-Functional Requirements

#### Performance

| Metric | Target | Measurement |
|---|---|---|
| CI build (generator + `astro build` + Pagefind + link check) | < 3 min | GitHub Actions job duration, warm cache |
| Local production build (`npm run build`) | < 60 s | wall-clock on the dev machine |
| Lighthouse Performance (landing + a package page) | ≥ 90 | Lighthouse CI / manual run on built output |
| Largest page transfer (gzipped HTML+CSS, excl. images) | < 250 KB | browser devtools on built output |

#### Reliability / Integrity

| Metric | Target |
|---|---|
| Broken internal links | 0 (hard CI gate, FR-20) |
| `astro check` errors | 0 (hard CI gate, FR-19) |
| API generator failures masked | 0 — generation failure fails the build (FR-13) |
| Deploys to `main` that skip checks | 0 — deploy is downstream of all gates (FR-24) |

#### Accessibility & SEO

| Metric | Target | Measurement |
|---|---|---|
| Lighthouse Accessibility | ≥ 95 | Lighthouse on landing + package page |
| WCAG | AA (Starlight default conformance retained) | manual + axe (SHOULD) |
| Lighthouse SEO | ≥ 90 | Lighthouse; sitemap + per-page meta present |

#### Accuracy (the load-bearing NFR for *this* repo)

- **NFR-ACC** Public-grade: zero unverified capability claims at launch (FR-17/18). An **accuracy review pass** (M6) checks every package page against code before launch and is recorded.

#### Maintainability & Portability

- **NFR-MNT** Docs source co-located in `site/src/content/docs/`; generated API has a single deterministic source (the package code) and is reproducible from a documented command. No hand-edited generated files.
- **NFR-PORT** Output is pure static HTML/CSS/JS (`dist/`), servable by any static host; no server runtime, no DB, no secrets at runtime. (Deploy secret is the GitHub Pages token only.)
- **NFR-CONV** Markdown/MDX follows repo doc conventions (`CLAUDE.md`): every code block tagged with its language; correct heading levels and emphasis.

---

## 4. Architecture

### 4.1 System Overview

A **static documentation site**: content authored/curated as MDX plus an API-reference layer generated from Python source at build time, compiled by Astro+Starlight into static HTML, and deployed to GitHub Pages by CI.

```
                        LatticeLabs monorepo (source of truth)
   ┌───────────────────────────────────────────────────────────────────────┐
   │  Existing prose                Python source (6 pkgs)   Root docs/      │
   │  cadling/README+docs           cadling/ … ll_clouds/    essays (concepts)│
   │  geotoken/docs/*               (public modules)         (reframed)      │
   │  *_/README.md                                                            │
   └──────────┬───────────────────────────┬───────────────────────┬─────────┘
      (a) curate (manual)        (c) generate (build-time)   (b) author (manual)
              │                            │                       │
              ▼                            ▼                       ▼
   ┌───────────────────────────────────────────────────────────────────────┐
   │              site/src/content/docs/  (Starlight content collection)    │
   │   index.mdx · <pkg>/{overview,install,usage}.md · concepts/*.md ·      │
   │   guides/*.md · tutorials/*.md · <pkg>/reference/*.mdx (GENERATED)      │
   └───────────────────────────────┬───────────────────────────────────────┘
                                    │  prebuild: api-gen  →  astro check → astro build → Pagefind
                                    ▼
                         dist/ (static HTML/CSS/JS + search index)
                                    │  internal link check (gate)
                                    ▼
                    GitHub Actions → actions/deploy-pages
                                    ▼
                 https://latticelabsai.github.io/ll_toolkit/
```

### 4.2 Component Design

#### Component: Astro + Starlight site (`site/`)
- **Responsibility:** Compile the content collection into a themed, searchable static site.
- **Technology:** Astro 6.4.5, `@astrojs/starlight` 0.40.0, `sharp` (image opt), Node ≥22.12.
- **Interfaces:** `npm run build` → `dist/`; `npm run dev` → localhost:4321; content collection defined in `site/src/content.config.ts` (`docsLoader` + `docsSchema`).
- **Dependencies:** the content collection (4.3), the API generator output (4.4).

#### Component: Content collection (`site/src/content/docs/`)
- **Responsibility:** Hold all published pages as Markdown/MDX with Starlight frontmatter (`title`, `description`, optional `sidebar`/`badge`).
- **Technology:** Markdown/MDX validated by `docsSchema()`.
- **Interfaces:** file path → route (Starlight routing). Sidebar wired in `astro.config.mjs`.
- **Dependencies:** none at runtime; populated by curation (b), authoring (a), and the generator (c).

#### Component: API-reference generator (`site/scripts/gen-api.*`)
- **Responsibility:** Statically parse each package's public Python API and emit reference MDX into `<pkg>/reference/`.
- **Technology:** Python + `griffe` (static AST parsing of Google-style docstrings) → a small renderer that writes MDX. Invoked from a `package.json` `prebuild` script.
- **Interfaces:** input = list of package roots + public-module allowlist; output = MDX files; exit non-zero on any parse/render error (FR-13).
- **Dependencies:** the package source trees. **Does not import** the packages (FR-11) — no torch/pythonocc/OpenMP exposure.

#### Component: CI/CD workflow (`.github/workflows/docs.yml`)
- **Responsibility:** Build, validate, and deploy on `main`; gate PRs.
- **Technology:** GitHub Actions; `actions/setup-node`, `actions/setup-python`, `actions/configure-pages`, `actions/upload-pages-artifact`, `actions/deploy-pages`.
- **Interfaces:** triggers on `push: main` and `pull_request` (path-filtered, FR-26); GitHub Pages environment.
- **Dependencies:** repo Pages settings (Pages source = GitHub Actions).

### 4.3 Content Model & Information Architecture

```
src/content/docs/
  index.mdx                         # Landing: what the toolkit is, the 6 packages, quick links
  get-started/
    installation.md                 # monorepo install (conda-forge torch caveat), per-pkg pip -e
    quickstart.md                   # smallest end-to-end win
  concepts/                         # EXPLANATION (Diátaxis) — from root docs/ essays
    how-neural-cad-generation-works.md
    inside-cad-generation-models.md
    what-generation-can-and-cannot-do.md
    tokenization-overview.md
  tutorials/                        # LEARNING — FR-9 end-to-end walkthroughs
    parse-a-step-file.md            # cadling
    tokenize-a-mesh.md              # geotoken
    generate-cad.md                 # ll_gen propose→dispose
    ocadr-hf-inference.md           # ll_ocadr run_ll_ocadr_hf.py
  guides/                           # HOW-TO — task recipes (cross-package)
    ...
  <pkg>/                            # one per IMPLEMENTED package: cadling, ll_stepnet,
    overview.md                     #   geotoken, ll_ocadr, ll_gen, ll_clouds
    installation.md
    usage.md
    concepts.md                     # pkg-specific explanation (optional)
    reference/                      # REFERENCE — GENERATED (FR-11)
      <module>.mdx ...
  roadmap/
    ll_brepnet.md                   # honest "planned, not implemented" stub (FR-7b)
  contributing/
    development.md                  # from cadling/docs/Development.md, generalized
    docs-site.md                    # how to build/preview/add docs (this site)
```

Sidebar (in `astro.config.mjs`) is organized **Diátaxis-first** at the top (Get Started, Tutorials, How-to, Concepts) then **package reference** groups, then Contributing — so newcomers learn and adopters look up.

### 4.4 API Generation Pipeline (FR-11–FR-15)

```
gen-api  (prebuild)
  for pkg in [cadling, ll_stepnet, geotoken, ll_ocadr, ll_gen, ll_clouds]:   # ll_brepnet excluded: 0-byte files, nothing to parse
    modules = griffe.load(pkg, public-only, STATIC)        # AST parse — no import()
    for module in modules:
       render summary · classes(methods, signatures) · functions · Args/Returns/Raises/Examples
       emit  site/src/content/docs/<pkg>/reference/<module>.mdx   (with source file:line link)
    if any error: exit 1                                    # fail the build (FR-13)
```

- **Why static (`griffe`):** importing these packages would execute `import torch` / `pythonocc` and risk the documented OpenMP crash (`CLAUDE.md`), and would require the full conda env in CI. Static parsing needs only the source tree.
- **Determinism:** output is a pure function of source; regenerated each build (G6). Generated files are git-ignored and reproduced by `npm run gen-api` (default; OQ3).

### 4.5 Data Flow — "publish a doc change"

```
1. Author edits/adds MDX under site/src/content/docs/ (or edits a Python docstring)
2. Open PR → Actions PR job: setup → gen-api → astro check → astro build → link-check
3. Any gate red → PR blocked (FR-25)
4. Merge to main → Actions main job repeats build + gates → upload-pages-artifact → deploy-pages
5. Live at https://latticelabsai.github.io/ll_toolkit/  (Pagefind search index included)
```

### 4.6 Deployment & Infrastructure

- **Host:** GitHub Pages (project site). **Source:** "GitHub Actions" (not branch-based).
- **URL/base:** `https://latticelabsai.github.io/ll_toolkit/` · `base: '/ll_toolkit/'` (FR-2). Custom domain deferred (OQ4).
- **Environments:** one (production = `main`). PRs produce build artifacts but do not deploy (preview deploys = optional future, NG-aligned).
- **Secrets:** none beyond the Actions-provided Pages token (`permissions: pages: write, id-token: write`). No runtime secrets (NFR-PORT).

### 4.7 Observability

- **Build observability:** Actions logs per step; the link-checker and `astro check` print actionable failures; the generator logs per-package symbol counts and exits non-zero on error.
- **Content health:** internal-link report on every build (FR-20); optional scheduled link check for *external* links (COULD) so rot is caught without blocking deploys.
- **Usage (optional, OQ):** privacy-light analytics (e.g. self-host Plausible) — not in v1 by default (NG5-adjacent: no trackers unless chosen).

---

## 5. Implementation Plan & Milestones

Owner for all milestones: **Maintainer**. Each milestone exits with its quality gate green.

### M1 — Scaffold cleanup & IA foundation
- T1.1 De-scaffold: real `title`, logo/social config, remove starter `README.md` boilerplate, replace `index.mdx` splash with a real landing page (FR-1).
- T1.2 Set `site` + `base` for GitHub Pages (FR-2); verify built links resolve under `/ll_toolkit/`.
- T1.3 Build the sidebar/IA skeleton (FR-3/FR-4): Get Started, Tutorials, How-to, Concepts, 6 implemented-package groups, a Roadmap entry, Contributing; remove `guides/example.md`, `reference/example.md`.
- T1.4 Landing page lists the 6 implemented packages with one-line descriptions + status badges (FR-16, sourced from root `README.md`), and notes `ll_brepnet` as planned under Roadmap.
- **Exit:** `npm run build` + `astro check` clean; nav shows every section and all 6 implemented packages + the Roadmap entry; a placeholder page exists per implemented package.

### M2 — CI/CD pipeline
- T2.1 `.github/workflows/docs.yml`: PR job (build + `astro check` + link check, no deploy) and `main` job (same + deploy via `actions/deploy-pages`) (FR-24/25).
- T2.2 Pin Node ≥22.12; enable npm cache; configure Pages env + permissions (FR-27).
- T2.3 Add the internal link-check step gating the build (FR-20).
- **Exit:** a merged PR auto-deploys to `https://latticelabsai.github.io/ll_toolkit/`; a PR with a broken internal link is blocked.

### M3 — Curate existing prose + Concepts
- T3.1 `cadling`: Overview/Install/Usage from `README.md` + `docs/{Overview,Purpose,Development}.md` (FR-6); status badge.
- T3.2 `geotoken`: migrate `docs/{architecture,integration,data-requirements}.md` + `api/`,`examples/` (FR-6).
- T3.3 `ll_stepnet`, `ll_ocadr`, `ll_clouds`: Overview/Install/Usage from READMEs (FR-6); **`ll_ocadr` vLLM corrected to experimental/future** (FR-17).
- T3.4 Concepts section from root essays, reframed as public explainers (FR-8); internal framing/status stripped (NG2).
- T3.5 Wire the exclusion guard so internal status docs can never be published (FR-10).
- **Exit:** five packages have Overview/Install/Usage; Concepts live; zero internal-status pages published; over-claims corrected.

### M4 — Author the gaps + Tutorials
- T4.1 `ll_gen` docs authored from code: propose→dispose loop, neural generators (untrained), `python -m ll_gen.training.run` (FR-7/FR-18).
- T4.2 `ll_brepnet` **roadmap stub** authored (FR-7b): one honest page — "planned B-Rep face-graph network, not yet implemented (scaffold only)"; no Install/Usage/API.
- T4.3 Tutorials (FR-9): parse-a-step (cadling), tokenize-a-mesh (geotoken), generate-cad (ll_gen), ocadr-hf-inference (ll_ocadr).
- T4.4 Contributing pages: `development.md` (generalized from `cadling/docs/Development.md`) + `docs-site.md` (how to add/build docs).
- **Exit:** all 6 implemented packages have the four-quadrant set; the `ll_brepnet` roadmap stub is published; ≥1 tutorial per major workflow; Contributing live.

### M5 — Docstring API reference generator
- T5.1 Implement `gen-api` (griffe static parse → MDX) with per-package public-module allowlist (FR-11/FR-12).
- T5.2 Wire as `prebuild` in `package.json` and into both CI jobs; fail-loud on error (FR-13); source `file:line` links (FR-14).
- T5.3 Generate `reference/` for all 6 implemented packages; verify pages render and are searchable. (`ll_brepnet` is skipped — 0-byte source.)
- T5.4 Git-ignore generated output; document `npm run gen-api` reproduction (OQ3 default).
- **Exit:** CI regenerates API MDX for all 6 implemented packages every build; build gated on successful generation; reference is searchable.

### M6 — Polish & launch gate
- T6.1 Search (Pagefind) verified across all pages (FR-21); per-page `title`/`description`, sitemap, OG meta (FR-22); logo/favicon/colors (FR-23).
- T6.2 NFR targets met: Lighthouse Perf ≥90, A11y ≥95, SEO ≥90; CI build < 3 min; zero broken internal links.
- T6.3 **Accuracy review pass** (NFR-ACC): every package page checked against code; no unverified capability claim; status badges correct (FR-16/17/18). Result recorded.
- T6.4 Update root `README.md` to link the live docs site.
- **Exit:** all NFR targets met; accuracy review signed off and recorded; public launch.

### Dependency order

```
M1 ──► M2 ─────────────► (deploy live)
  │                        ▲
  ├──► M3 ──┐              │
  ├──► M4 ──┼──► M6 (launch gate)
  └──► M5 ──┘              │
                           │
  M3, M4, M5 depend on M1 (IA); integrate through M2's CI from the start.
  M3/M4/M5 run in parallel.  M6 is last (needs all content + generator).
```

---

## 6. Content Plan & Source Mapping

The authority rule (A4): **code > existing docs**. Each row says where a page comes from and how it's produced.

| Target area | Source | Method | Accuracy note |
|---|---|---|---|
| Landing + Get Started | root `README.md` (package table, install, conda-forge caveat) | curate | Keep the conda-forge/OpenMP caveat verbatim — it's load-bearing |
| `cadling/*` | `cadling/README.md` + `docs/{Overview,Purpose,Development}.md` | curate | Exclude `RequiredToBeCorrected`/`UNIMPLEMENTED`/`Todo`/`ReviewFindings`/`plans` |
| `geotoken/*` | `geotoken/docs/{architecture,integration,data-requirements}.md`, `api/`, `examples/`, README | curate | Best-documented; migrate near-as-is |
| `ll_stepnet/*` | `ll_stepnet/README.md` | curate + author concepts | — |
| `ll_ocadr/*` | `ll_ocadr/README.md` | curate **with correction** | vLLM = experimental/future, never "works" (FR-17; SPEC-1 §FR-O4; audit) |
| `ll_clouds/*` | `ll_clouds/README.md` (1 KB) | author (expand) | "core library" maturity (SPEC-1 G4) |
| `ll_gen/*` | **code** (`pipeline/orchestrator.py`, `training/run`) + root `README.md` line | author | "models ship untrained" (root README); cite entry points |
| `roadmap/ll_brepnet.md` | directory listing only (all files 0 bytes; file *names* suggest a UV-Net/BRepNet-style B-Rep face-graph net) | author (roadmap stub) | **Not implemented** — state "planned/scaffold only"; no Install/Usage/API (FR-7b) |
| Concepts | root `docs/{HowNNsGenerateCAD, HowCADGenerationModelsActuallyWorkInside, AICADGenerationDoesNotWorkHowYouThink, GenerationImplementation}.md` | curate/reframe | Strip internal status; keep the honest "what it can't do" framing |
| `<pkg>/reference/*` | package Python source (public modules) | **generate** (griffe, static) | Regenerated every build; no hand edits |
| Contributing | `cadling/docs/Development.md` + new | curate + author | Generalize cadling-specific steps to the monorepo |

**Excluded from publication (NG2):** `cadling/docs/RequiredToBeCorrected.md`, `UNIMPLEMENTED.md`, `Todo.md`, `ReviewFindings.md`, `Plan.md`, `Adjustments.md`, `visualization_plan.md`, `docs/plans/**`, `docs/specs/**`, `docs/2026-06-09-partial-deceptive-code-audit.md`, `docs/Huggingface*Plan.md`, `docs/ResearchVSCodebase.md`.

---

## 7. Testing & Quality Strategy

- **Build gates (hard, CI):** `astro check` zero errors (FR-19); API generation succeeds (FR-13); zero broken **internal** links over `dist/` (FR-20). Any red → no merge / no deploy.
- **Generator tests:** a unit check that `gen-api` produces a non-empty `reference/` for a known package and a known public class (e.g. `geotoken.GeoTokenizer`) with its documented methods; and that an intentionally malformed input makes the generator exit non-zero (proves fail-loud, FR-13).
- **Accuracy review (manual, M6, recorded):** per-package checklist — does every capability claim trace to code? Is every maturity badge correct? Is `ll_ocadr` vLLM framed as future? (NFR-ACC, FR-16/17/18.)
- **External-link check (soft):** scheduled (non-blocking) job reports rotted external links (FR/observability) — does not gate deploys.
- **Lighthouse (M6):** Perf ≥90, A11y ≥95, SEO ≥90 on landing + one package + one generated reference page.
- **Manual smoke:** `npm run dev` renders every top-level route; search returns hits for each package name and a key class; base-path links work on the deployed Pages URL (catches FR-2 regressions).

---

## 8. Risks & Mitigations

| # | Risk | Severity | Mitigation |
|---|---|---|---|
| **R1** | **Publishing claims the code doesn't back** (the repo's documented, recurring failure mode — audit doc, stale `RequiredToBeCorrected`, SPEC-1's "truth-up" commits). Highest risk for *this* site. | **High** | Status badges per package (FR-16); capability claims cite code (FR-18); explicit `ll_ocadr` vLLM correction (FR-17); **exclude** internal status docs (FR-10/NG2); mandatory accuracy-review gate before launch (M6/T6.3); authority rule code>docs (A4). |
| R2 | Python docstring→MDX under a JS toolchain has no turnkey TypeDoc equivalent; generator is bespoke and could slip. | Med-High | Use mature `griffe` for static parsing; keep the renderer small and tested (§7); **reference can ship hand-authored** if the generator slips (M5 is parallel, not on the launch critical path for *coverage* — generated API is an upgrade over hand-authored, FR-5 is still met). |
| R3 | Doc drift as code evolves (concepts/guides especially). | Med | API reference regenerated every build (G6); CI internal-link gate; per-package status badges; concepts framed to age well ("what it can/can't do"). |
| R4 | GitHub Pages **base-path** breakage — relative links/assets 404 under `/ll_toolkit/`. | Med | Set `base` early (FR-2, M1); use Starlight link helpers/base-aware URLs; internal link check over built output catches it; deployed-URL smoke test (§7). |
| R5 | Importing packages to introspect triggers torch/pythonocc + OpenMP crash and forces a heavy CI env. | Med | **Static** parsing only (FR-11) — `griffe` never imports; CI needs no conda/torch for the generator. |
| R6 | CI build time creep (Pagefind + generation across 6 packages + link check). | Low-Med | npm/pip caches (FR-27); path-filtered triggers (FR-26); budget < 3 min (NFR-1); parallelize generation if needed. |
| R7 | Scope creep into exhaustive tutorials for every API. | Med | Tutorials bounded to major workflows (FR-9, four total); Reference (generated) covers the long tail. |
| R8 | Node/toolchain mismatch (Astro 6 needs Node ≥22.12). | Low | Pin Node 22 in the workflow (FR-24/T2.2); document the floor (A2). |
| R9 | `ll_gen` has no README — authoring may misread intent from code alone. | Med | Author strictly from code (entry points, tests, examples) and cite it (FR-18); flag uncertain maturity rather than guess; maintainer reviews in the accuracy pass (M6). |
| R10 | **`ll_brepnet` documented as if real** — the empty sidebar dir / suggestive file names tempt a fabricated package section (the precise R1 failure mode). | High | Roadmap-stub-only treatment (FR-7b); excluded from FR-5/generator; AC#3 names six packages and explicitly excludes it; accuracy pass (M6/T6.3) confirms no Install/Usage/API page exists for it. |

---

## 9. Acceptance Criteria (Definition of Done)

Launch-ready when **all** hold:

1. `site/` builds clean: `npm run build` + `astro check` zero errors; output is static `dist/` (FR-19, NFR-PORT).
2. The site is live at `https://latticelabsai.github.io/ll_toolkit/`, deployed by Actions on merge to `main`; PRs run the same gates without deploying (FR-24/25).
3. All **6 implemented packages** (cadling, ll_stepnet, geotoken, ll_ocadr, ll_gen, ll_clouds) have Overview, Installation, Usage, and a **generated** API reference; `ll_brepnet` is represented **only** by a roadmap stub (no Install/Usage/API — FR-7b); the four Diátaxis quadrants are represented (Get Started/Tutorials, How-to, Concepts, Reference) (FR-3/5/8/9/11, G2).
4. The starter is fully de-scaffolded — no "My Docs", no `withastro/starlight` placeholder link, no `example.md` pages (FR-1/4).
5. `gen-api` runs in CI, statically (no package import), regenerates API MDX for all 6 implemented packages, and **fails the build on error** (FR-11/13).
6. **Zero broken internal links** and search (Pagefind) indexes all pages — both verified in CI/build (FR-20/21).
7. **Accuracy review recorded:** every package page checked against code; zero unverified capability claims; `ll_ocadr` vLLM shown as experimental/future; per-package status badges correct (FR-16/17/18, NFR-ACC).
8. No internal status/planning doc is published (FR-10/NG2).
9. NFR targets met: CI build < 3 min; Lighthouse Perf ≥90 / A11y ≥95 / SEO ≥90 (NFR-1, A11y/SEO).
10. Root `README.md` links the live docs site (T6.4).

---

## 10. Open Questions (each with an owner — no TBD)

| # | Question | Owner | Default if unanswered |
|---|---|---|---|
| OQ1 | Confirm the content strategy is the three-pronged **curate + author + generate** (interpreting the discovery note "plus api from docstrings" as *additive* to the recommended curate-and-author approach). | Maintainer | Proceed as three-pronged (G3). If the intent was *generate-only*, M3/M4 curation collapses into authored stubs + generated reference. |
| OQ2 | Docstring→MDX tooling: `griffe`-based custom generator (default) vs. `pdoc`/`sphinx-autodoc2` adapted vs. hand-authored reference. | Maintainer | Custom `griffe` static generator → MDX (FR-11), chosen for Google-style support + no-import safety. |
| OQ3 | Commit generated API MDX to git, or git-ignore and regenerate in CI? | Maintainer | Git-ignore + regenerate (mirrors SPEC-1's reproduce-don't-commit stance for large/derived artifacts). |
| OQ4 | Custom domain vs. the `github.io/ll_toolkit/` project path? | Maintainer | Project path `/ll_toolkit/` for v1; custom domain later. |
| OQ5 | Doc versioning (single "latest" vs. versioned releases)? | Maintainer | Single "latest" tracking `main` (NG3); revisit when the toolkit cuts releases. |
| OQ6 | For `ll_gen`, also backfill a package `README.md` so repo and site agree, or site-only for now? | Maintainer | Site-only now; backfill the README as a follow-up (NG1). |
| OQ7 | Privacy-light analytics (e.g. self-hosted Plausible) or none? | Maintainer | None in v1 (no trackers). |
| OQ8 | `ll_brepnet`: publish a roadmap stub now (default), or omit it from the site entirely until it has real code? | Maintainer | Publish a single honest **roadmap stub** (FR-7b); never a full package section. If the maintainer prefers, omit it until code lands — but do **not** document it as functional either way. |

---

## 11. Verification Log (evidence this spec is grounded, not assumed)

- **Scaffold state:** `site/` is a Starlight starter — `astro.config.mjs` title `'My Docs'` (`site/astro.config.mjs:9`), social link `withastro/starlight` (`:10`), starter sidebar (`:11-23`); only `index.mdx` (splash), `guides/example.md`, `reference/example.md` exist; the 7 content dirs under `src/content/docs/` (6 implemented + a `ll_brepnet/` placeholder) are **empty** (confirmed via `find src/content -type f` → 3 files, and per-dir `ls -A`).
- **Toolchain:** `@astrojs/starlight@0.40.0`, `astro@6.4.5`, `sharp@0.34.5` (`site/package.json`); Astro engine floor Node ≥22.12.0 (`node_modules/astro/package.json` engines); local Node v24.8.0, npm 11.6.0.
- **Remote/host:** `origin = https://github.com/LatticeLabsAI/ll_toolkit.git` (`git remote -v`) → Pages URL `https://latticelabsai.github.io/ll_toolkit/`, base `/ll_toolkit/`.
- **No deploy/CI today:** no `firebase.json`, `.firebaserc`, or `.github/workflows/` (confirmed via `ls`); `firebase-debug.log` is from an unrelated MCP call (`developerknowledge.googleapis.com`).
- **Per-package doc inventory (drives §6):** cadling = 20 KB README + 11 `docs/` files; geotoken = README + `docs/{architecture,integration,data-requirements}.md`,`api/`,`examples/`; ll_stepnet/ll_ocadr/ll_clouds = README only (7 KB / 3 KB / 1 KB); **ll_gen = no README, no docs** (but real code); **ll_brepnet = no README, empty docs/** (confirmed via per-package `find … -name '*.md'`).
- **Package count is SIX, not seven (drives the whole ll_brepnet treatment):** the canonical root `README.md` package table lists exactly six — cadling, ll-stepnet, geotoken, ll-ocadr, ll-gen, ll-clouds. `ll_brepnet` is **absent from it** and **untracked in git** (`git status` → `?? ll_brepnet/`).
- **`ll_brepnet` is an empty scaffold (drives FR-7b/R10/OQ8):** every file is **0 bytes** — `pyproject.toml`, `requirements.txt`, `environment.yaml`, and all 9 `.py` files under `ll_brepnet/ll_brepnet/` (dataloaders, models incl. `ll_brepnet.py`/`uvnet_encoders.py`, pipelines incl. `extract_brepnet_data_from_step.py`, eval) — confirmed via `wc -c` (`0 total`) and `find ll_brepnet -type f ! -empty` (no output). Nothing to install, import, parse, or document as functional.
- **Conceptual essays exist:** root `docs/` holds `HowNNsGenerateCAD.md`, `HowCADGenerationModelsActuallyWorkInside.md`, `AICADGenerationDoesNotWorkHowYouThink.md`, `GenerationImplementation.md` (`ls docs/`).
- **Accuracy-risk evidence (drives R1/FR-16/17):** `docs/2026-06-09-partial-deceptive-code-audit.md`; stale `cadling/docs/RequiredToBeCorrected.md` + `UNIMPLEMENTED.md`; SPEC-1 status line and commit trail ("correct M3 overclaim", "truth-up STATUS.md, README, SPEC-1"); root `README.md` states `ll_gen` "models ship untrained" and `ll_ocadr` "vLLM serving is experimental/future".
- **Docstring convention (drives FR-11/12):** `CLAUDE.md` → "Google-style docstrings: Required for public APIs".
- **OpenMP hazard (drives static-parse decision, FR-11/R5):** `CLAUDE.md` "OpenMP / PyTorch Import Order (macOS Critical)" — importing torch + conda mix crashes; static parsing avoids importing the packages.
- **House style match:** structure mirrors `docs/specs/SPEC-1-ll-gen-ocadr-clouds-completion.md` (metadata table, MoSCoW FRs, ASCII architecture, milestones with exit criteria, risk register, open questions with owners, verification log).
