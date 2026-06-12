---
license: other
configs:
- config_name: 3d
  data_files:
  - split: train
    path: "3d/*.parquet"
- config_name: cad
  data_files:
  - split: train
    path: "cad/*.parquet"
- config_name: geo
  data_files:
  - split: train
    path: "geo/*.parquet"
---

# LayerDynamics/three-indexer

Published by the LatticeLabs Three Indexer. Per-asset license + attribution live in each
config's `metadata.parquet`. See `docs/specs/SPEC-1-latticelabs-three-indexer.md`.
