"""Build a BRepNet-style ``.npz`` record from a precomputed JSON description.

This is the non-STEP front-end: when a caller already has the B-Rep coedge
topology and per-entity features (e.g. exported from another CAD tool or a
cached intermediate), they can supply it as JSON and obtain a ``.npz`` record
identical in schema to :mod:`ll_brepnet.pipelines.extract_brepnet_data_from_step`,
ready for the dataset loader.

Expected JSON schema (UV-grids are optional and default to zeros)::

    {
      "coedge_to_next": [int, ...],   # length C
      "coedge_to_prev": [int, ...],   # length C
      "coedge_to_mate": [int, ...],   # length C
      "coedge_to_face": [int, ...],   # length C, values in [0, F)
      "coedge_to_edge": [int, ...],   # length C, values in [0, E)
      "coedge_reversed": [0|1, ...],  # optional, length C, default zeros
      "face_features": [[float, ...], ...],   # [F, Df]
      "edge_features": [[float, ...], ...],    # [E, De]
      "face_point_grids": [...],   # optional [F, 7, U, V]
      "edge_point_grids": [...]    # optional [E, 6, U]
    }
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np

from .extract_brepnet_data_from_step import NUM_U, NUM_V

_log = logging.getLogger(__name__)

_COEDGE_INT_KEYS = (
    "coedge_to_next",
    "coedge_to_prev",
    "coedge_to_mate",
    "coedge_to_face",
    "coedge_to_edge",
)


class BRepJsonExtractor:
    """Convert a JSON topology/feature description into BRepNet ``.npz`` arrays.

    Args:
        topology: The parsed JSON dictionary (see module docstring for schema).
        num_u, num_v: UV-grid resolution used when grids are absent.
    """

    def __init__(self, topology: dict, num_u: int = NUM_U, num_v: int = NUM_V):
        self.topology = topology
        self.num_u = num_u
        self.num_v = num_v

    def extract_arrays(self) -> dict[str, np.ndarray]:
        """Validate the description and return the named ``.npz`` arrays."""
        t = self.topology

        for key in (*_COEDGE_INT_KEYS, "face_features", "edge_features"):
            if key not in t:
                raise ValueError(f"JSON topology missing required key: {key!r}")

        face_features = np.asarray(t["face_features"], dtype=np.float32)
        edge_features = np.asarray(t["edge_features"], dtype=np.float32)
        if face_features.ndim != 2 or edge_features.ndim != 2:
            raise ValueError("face_features and edge_features must be 2-D [N, D]")
        num_faces, num_edges = face_features.shape[0], edge_features.shape[0]

        coedge_arrays: dict[str, np.ndarray] = {}
        lengths = set()
        for key in _COEDGE_INT_KEYS:
            arr = np.asarray(t[key], dtype=np.int64).ravel()
            coedge_arrays[key] = arr
            lengths.add(arr.shape[0])
        if len(lengths) != 1:
            raise ValueError(f"coedge_to_* arrays have inconsistent lengths: {lengths}")
        num_coedges = lengths.pop()

        # Bounds checks: incidence must reference valid entities.
        if num_coedges:
            if int(coedge_arrays["coedge_to_face"].max(initial=0)) >= num_faces:
                raise ValueError("coedge_to_face references a face index >= num_faces")
            if int(coedge_arrays["coedge_to_edge"].max(initial=0)) >= num_edges:
                raise ValueError("coedge_to_edge references an edge index >= num_edges")
            for key in ("coedge_to_next", "coedge_to_prev", "coedge_to_mate"):
                if int(coedge_arrays[key].max(initial=0)) >= num_coedges:
                    raise ValueError(f"{key} references a coedge index >= num_coedges")

        if "coedge_reversed" in t:
            coedge_reversed = np.asarray(t["coedge_reversed"], dtype=np.float32).ravel()
            if coedge_reversed.shape[0] != num_coedges:
                raise ValueError("coedge_reversed length must equal num_coedges")
        else:
            coedge_reversed = np.zeros(num_coedges, dtype=np.float32)

        if "face_point_grids" in t:
            face_point_grids = np.asarray(t["face_point_grids"], dtype=np.float32)
            if face_point_grids.shape != (num_faces, 7, self.num_u, self.num_v):
                raise ValueError(
                    "face_point_grids must have shape "
                    f"[{num_faces}, 7, {self.num_u}, {self.num_v}]"
                )
        else:
            face_point_grids = np.zeros((num_faces, 7, self.num_u, self.num_v), dtype=np.float32)

        if "edge_point_grids" in t:
            edge_point_grids = np.asarray(t["edge_point_grids"], dtype=np.float32)
            if edge_point_grids.shape != (num_edges, 6, self.num_u):
                raise ValueError(f"edge_point_grids must have shape [{num_edges}, 6, {self.num_u}]")
        else:
            edge_point_grids = np.zeros((num_edges, 6, self.num_u), dtype=np.float32)

        return {
            **coedge_arrays,
            "coedge_reversed": coedge_reversed,
            "face_features": face_features,
            "edge_features": edge_features,
            "face_point_grids": face_point_grids,
            "edge_point_grids": edge_point_grids,
            "num_faces": np.int64(num_faces),
            "num_edges": np.int64(num_edges),
            "num_coedges": np.int64(num_coedges),
        }

    def process(self, output_path: Path) -> Path:
        """Extract arrays and write them to ``output_path`` (a ``.npz`` file)."""
        arrays = self.extract_arrays()
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(output_path, **arrays)
        _log.info(
            "Wrote %s (F=%d E=%d C=%d)",
            output_path.name,
            int(arrays["num_faces"]),
            int(arrays["num_edges"]),
            int(arrays["num_coedges"]),
        )
        return output_path


def extract_brepnet_data_from_json(json_path: Path, output_path: Path) -> Path:
    """Read a JSON topology file and write the corresponding ``.npz`` record."""
    json_path = Path(json_path)
    topology = json.loads(json_path.read_text())
    return BRepJsonExtractor(topology).process(output_path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build a BRepNet-style .npz from a JSON topology description."
    )
    parser.add_argument("--json", required=True, help="Input JSON topology file")
    parser.add_argument("--output", required=True, help="Output .npz path")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    out = extract_brepnet_data_from_json(Path(args.json), Path(args.output))
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
