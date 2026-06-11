---
title: Working on the docs site
description: How this documentation site is built — Astro + Starlight, content layout, the link-validation and API-generation steps, and how to add a page.
sidebar:
  label: Docs site
  order: 2
---

This documentation site lives in [`site/`](https://github.com/LatticeLabsAI/ll_toolkit/tree/main/site)
and is built with [Astro](https://astro.build/) +
[Starlight](https://starlight.astro.build/). It deploys to GitHub Pages.

## Local development

```bash
cd site
npm install
npm run dev        # live preview at http://localhost:4321/ll_toolkit/
npm run check      # astro check (content + types)
npm run build      # production build → dist/ (validates internal links)
npm run preview    # serve the production build locally
```

## Content layout

```text
site/src/content/docs/
  index.mdx                # landing page
  get-started/             # installation, quickstart
  concepts/                # explanation (Diátaxis)
  tutorials/               # learning-oriented walkthroughs
  guides/                  # task-oriented how-tos
  <pkg>/                   # one per package: overview, installation, usage
    reference/             # GENERATED API reference (do not hand-edit)
  roadmap/                 # roadmap + shipped milestones (e.g. ll_brepnet → done)
  contributing/            # this section
```

The sidebar is configured in `site/astro.config.mjs`; each group autogenerates
from its directory, ordered by each page's `sidebar.order` frontmatter.

## Internal-link validation

Builds **fail on broken internal links** via the `starlight-links-validator`
plugin. Write internal links with the full base path, e.g.
`/ll_toolkit/cadling/usage/`. Run `npm run build` before opening a PR.

## Generated API reference

The per-package `reference/` pages are **generated from Python docstrings** by
`site/scripts/gen_api.py` (static parsing with [griffe](https://mkdocstrings.github.io/griffe/)
— it never imports the packages, so no torch/pythonocc is needed):

```bash
cd site
npm run gen:api    # regenerate src/content/docs/<pkg>/reference/*
```

Generation also runs automatically before `npm run build` (a `prebuild` hook).
Generated files are git-ignored — never hand-edit them; change the Python
docstring instead.

## The accuracy bar

This site holds a **public-grade accuracy bar**: no page may claim behavior the
code does not back. Concretely:

- Every package page carries an honest maturity badge.
- Capability claims trace to code (cite a module, class, or entry point).
- Internal status/planning docs (e.g. `RequiredToBeCorrected.md`) are **not**
  published.
- Where a package is not implemented (e.g. `ll_brepnet`), it gets a roadmap
  entry, not a fabricated package section.

## Adding a page

1. Create a `.md`/`.mdx` file under the right directory with `title` +
   `description` frontmatter (and `sidebar.order` to place it).
2. Link to it with a full-base path; run `npm run build` to confirm links pass.
3. Open a PR — CI runs `gen:api`, `astro check`, the build, and link validation,
   and deploys on merge to `main`.

## Deployment

A GitHub Actions workflow
([`.github/workflows/docs.yml`](https://github.com/LatticeLabsAI/ll_toolkit/blob/main/.github/workflows/docs.yml))
builds and validates on every PR and deploys to GitHub Pages on push to `main`.
The published site is at `https://latticelabsai.github.io/ll_toolkit/`.
