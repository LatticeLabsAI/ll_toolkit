---
license: other
configs:
- config_name: 3d
  data_files:
  - split: train
    path: "3d/metadata.parquet"
- config_name: 3d-public-domain
  data_files:
  - split: train
    path: "3d/metadata-public-domain.parquet"
- config_name: cad
  data_files:
  - split: train
    path: "cad/metadata.parquet"
- config_name: cad-permissive
  data_files:
  - split: train
    path: "cad/metadata-permissive.parquet"
- config_name: cad-public-domain
  data_files:
  - split: train
    path: "cad/metadata-public-domain.parquet"
- config_name: geo
  data_files:
  - split: train
    path: "geo/metadata.parquet"
- config_name: geo-public-domain
  data_files:
  - split: train
    path: "geo/metadata-public-domain.parquet"
---

# LayerDynamics/three-indexer

Published by the LatticeLabs Three Indexer. Per-asset license + attribution live in each
config's `metadata.parquet`. See `docs/specs/SPEC-1-latticelabs-three-indexer.md`.
