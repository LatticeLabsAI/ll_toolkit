# Plan M5 — `ll_clouds` Core Point-Cloud Library

| | |
|---|---|
| **Spec** | SPEC-1 §5 M5, §3.1 FR-C1…C6, §4.3 |
| **Goal** | Build `ll_clouds` from an empty scaffold into a real, installable point-cloud library: I/O, preprocessing, features, registration/segmentation, Pydantic models, full pytest suite (≥80% coverage). |
| **Depends on** | — (independent) |
| **Owner** | Maintainer · **Mode** Inline/sequential · **Tests** TDD · **Commits** per task |
| **Status** | Not started · **Decision baked in:** core standalone library; backend numpy/scipy/trimesh (open3d optional, OQ2) |

## Context (verified)
- `ll_clouds/` contains only `pyproject.toml` (name `ll_clouds`, desc "Pointcloud Processing and Analyzing for the LatticeLabs CAD toolkit", deps numpy+pydantic; optional `cad`/`ml`/`data`/`vision`/`conditioning`; full marker taxonomy + ruff/black/mypy config already present).
- `[tool.setuptools.packages.find] include = ["./*"]` — package code must live so it's discoverable; create `ll_clouds/ll_clouds/`.
- Target layout (SPEC §4.3): `datamodel.py, io.py, preprocess.py, features.py, registration.py, segmentation.py, bridges.py`.

## Pre-flight
- Resolve **OQ2** (numpy/scipy/trimesh required; open3d optional accelerator) and **OQ5** (bridges live in `ll_clouds`, lazy imports).
- Skim `ll_ocadr/vllm/lattice_encoder/geometry_net.py` FPS (so `ll_clouds` FPS is interface-compatible) and `geotoken/geotoken/curvature.py` (reuse the vectorized cotangent approach where applicable, don't reinvent).
- Confirm `[tool.setuptools.packages.find]` discovers `ll_clouds/ll_clouds/` (adjust to `include=["ll_clouds*"]` if needed) — verify with an editable install.

## Tasks

### T5.0 — Package skeleton + installability
- Create `ll_clouds/ll_clouds/__init__.py`, `ll_clouds/README.md`, `ll_clouds/tests/conftest.py` (OpenMP-safe if torch ever used; numpy-only otherwise).
- Verify `pip install -e ./ll_clouds` succeeds and `import ll_clouds` works. Fix `packages.find` if discovery fails.
- **Commit:** `chore(ll_clouds): package skeleton + editable install`

### T5.1 — Data models (FR-C5) [TDD]
- **Red:** `tests/unit/test_datamodel.py` — construct `PointCloud(points=(N,3))`; optional `normals/colors/labels/metadata`; validators reject mismatched lengths; `__len__`/`num_points`.
- **Green:** `datamodel.py` — Pydantic v2 `PointCloud` with `arbitrary_types_allowed=True` (ndarray), plus result models (`RegistrationResult`, `SegmentationResult`).
- **Commit:** `feat(ll_clouds): PointCloud + result Pydantic models`

### T5.2 — I/O (FR-C1) [TDD]
- **Red:** `tests/unit/test_io.py` — write→read round-trip for PLY/PCD/XYZ preserves points (and normals where supported); mesh→point-cloud sampling yields requested count on a unit cube/sphere (trimesh, lazy).
- **Green:** `io.py` — `read_ply/read_pcd/read_xyz/write_*`, `sample_from_mesh(mesh, n, method="poisson"|"uniform")`.
- **Commit:** `feat(ll_clouds): point-cloud I/O (PLY/PCD/XYZ) + mesh sampling`

### T5.3 — Preprocessing (FR-C2) [TDD]
- **Red:** `tests/unit/test_preprocess.py` — `normalize` centers to ~0 and scales to unit (idempotent on already-normalized); `voxel_downsample(size)` reduces count and snaps to grid; `farthest_point_downsample(k)` returns exactly k well-spread points; `remove_statistical_outliers` drops planted outliers, keeps inliers.
- **Green:** `preprocess.py`. FPS interface-compatible with `ll_ocadr` geometry_net.
- **Commit:** `feat(ll_clouds): normalize, voxel/FPS downsample, outlier removal`

### T5.4 — Features (FR-C3) [TDD]
- **Red:** `tests/unit/test_features.py` — normals on a planar cloud are constant ±sign and ⟂ plane; curvature on a sphere of radius r ≈ 1/r within tolerance; `bbox/centroid/extent` correct on known shapes.
- **Green:** `features.py` — k-NN PCA normals (scipy `cKDTree`), curvature estimate, stats.
- **Commit:** `feat(ll_clouds): per-point normals, curvature, geometry stats`

### T5.5 — Registration (FR-C4 part 1) [TDD]
- **Red:** `tests/unit/test_registration.py` — apply a known rotation+translation to a cloud; `icp(source, target)` recovers the inverse transform within tolerance; converges in a bounded iteration count.
- **Green:** `registration.py` — point-to-point ICP (kd-tree correspondences, SVD pose, iteration/tolerance controls), returns `RegistrationResult`.
- **Commit:** `feat(ll_clouds): point-to-point ICP registration`

### T5.6 — Segmentation (FR-C4 part 2) [TDD]
- **Red:** `tests/unit/test_segmentation.py` — RANSAC plane finds a planted plane (inlier ratio ≥ threshold, normal within tolerance); Euclidean clustering separates two well-separated blobs into 2 clusters.
- **Green:** `segmentation.py` — `ransac_plane`, `euclidean_cluster` (kd-tree region growing), return `SegmentationResult`.
- **Commit:** `feat(ll_clouds): RANSAC plane segmentation + Euclidean clustering`

### T5.7 — Bridges (FR-C6) [TDD, lazy imports]
- **Red:** `tests/integration/test_bridges.py` (mark `requires_cadling`/`requires_torch` as appropriate) — `from_cadling_document(doc)`/`from_mesh` produce a valid `PointCloud`; `from_ll_ocadr_input(...)` converts ll_ocadr's mesh/point representation; conversions are lazy (importing `ll_clouds` without cadling/torch must not fail).
- **Green:** `bridges.py` — lazy `try/except ImportError` guarded converters; no hard deps.
- **Commit:** `feat(ll_clouds): lazy bridges to cadling + ll_ocadr`

### T5.8 — Public API + coverage gate
- Export the public surface from `ll_clouds/__init__.py`; write `README.md` usage examples that match real signatures.
- Run coverage; ensure ≥80% on `ll_clouds/ll_clouds`; backfill tests for gaps.
- **Commit:** `docs(ll_clouds): public API exports + README; coverage ≥80%`

## Verification
```bash
pip install -e ./ll_clouds
cd ll_clouds && pytest tests/ -v --cov=ll_clouds --cov-report=term-missing
# coverage on ll_clouds/ll_clouds must be ≥ 80%
ruff check ll_clouds && black --check ll_clouds && mypy ll_clouds/ll_clouds
```

## Milestone risks
- **R4 (backend choice)** — default numpy/scipy/trimesh; open3d only as optional accelerator behind a flag, never required.
- **Discovery config** — `include=["./*"]` may not pick up `ll_clouds/ll_clouds`; fix `packages.find` in T5.0 and verify import before building modules.
- **Don't reinvent** — reuse `geotoken` curvature math and `ll_ocadr` FPS conventions where they fit.

## Done checklist
- [ ] `pip install -e ./ll_clouds` works; `import ll_clouds` works.
- [ ] All FR-C1…C6 modules implemented with passing TDD tests.
- [ ] Coverage ≥80% on `ll_clouds/ll_clouds`.
- [ ] ruff/black/mypy clean.
- [ ] Bridges are lazy — `import ll_clouds` works without cadling/torch installed.
