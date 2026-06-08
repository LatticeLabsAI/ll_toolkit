"""Point-cloud I/O for ll_clouds.

Supports ASCII PLY, PCD, and XYZ read/write (points and, where the format
allows, per-point normals). Mesh -> point-cloud sampling uses trimesh, imported
lazily so the core I/O has no hard trimesh dependency.
"""

from __future__ import annotations

import os
from typing import Any

import numpy as np

from .datamodel import PointCloud

_FLOAT_FMT = "%.9g"


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------


def write_point_cloud(pc: PointCloud, path: str) -> None:
    """Write a PointCloud to ``path`` (format chosen by extension)."""
    ext = os.path.splitext(path)[1].lower()
    if ext == ".xyz":
        _write_xyz(pc, path)
    elif ext == ".pcd":
        _write_pcd(pc, path)
    elif ext == ".ply":
        _write_ply(pc, path)
    else:
        raise ValueError(f"Unsupported point-cloud format for writing: {ext!r}")


def read_point_cloud(path: str) -> PointCloud:
    """Read a PointCloud from ``path`` (format chosen by extension)."""
    ext = os.path.splitext(path)[1].lower()
    if ext == ".xyz":
        return _read_xyz(path)
    if ext == ".pcd":
        return _read_pcd(path)
    if ext == ".ply":
        return _read_ply(path)
    raise ValueError(f"Unsupported point-cloud format for reading: {ext!r}")


# ---------------------------------------------------------------------------
# XYZ (plain text: x y z [nx ny nz])
# ---------------------------------------------------------------------------


def _stack(pc: PointCloud) -> np.ndarray:
    if pc.normals is not None:
        return np.concatenate([pc.points, pc.normals], axis=1)
    return pc.points


def _write_xyz(pc: PointCloud, path: str) -> None:
    np.savetxt(path, _stack(pc), fmt=_FLOAT_FMT)


def _read_xyz(path: str) -> PointCloud:
    data = np.loadtxt(path, ndmin=2)
    if data.shape[1] not in (3, 6):
        raise ValueError(f"XYZ file must have 3 or 6 columns, got {data.shape[1]}")
    points = data[:, :3]
    normals = data[:, 3:6] if data.shape[1] == 6 else None
    return PointCloud(points=points, normals=normals)


# ---------------------------------------------------------------------------
# PCD (ASCII subset)
# ---------------------------------------------------------------------------


def _write_pcd(pc: PointCloud, path: str) -> None:
    has_normals = pc.normals is not None
    fields = ["x", "y", "z"] + (
        ["normal_x", "normal_y", "normal_z"] if has_normals else []
    )
    n = pc.num_points
    arr = _stack(pc)
    header = (
        "# .PCD v0.7 - Point Cloud Data file format\n"
        "VERSION 0.7\n"
        f"FIELDS {' '.join(fields)}\n"
        f"SIZE {' '.join(['4'] * len(fields))}\n"
        f"TYPE {' '.join(['F'] * len(fields))}\n"
        f"COUNT {' '.join(['1'] * len(fields))}\n"
        f"WIDTH {n}\n"
        "HEIGHT 1\n"
        "VIEWPOINT 0 0 0 1 0 0 0\n"
        f"POINTS {n}\n"
        "DATA ascii\n"
    )
    with open(path, "w") as fh:
        fh.write(header)
        for row in arr:
            fh.write(" ".join(_FLOAT_FMT % v for v in row) + "\n")


def _read_pcd(path: str) -> PointCloud:
    fields: list[str] = []
    data_start = None
    lines = []
    with open(path) as fh:
        lines = fh.read().splitlines()
    for i, line in enumerate(lines):
        if line.startswith("FIELDS"):
            fields = line.split()[1:]
        elif line.startswith("DATA"):
            if "ascii" not in line:
                raise ValueError("Only ASCII PCD is supported")
            data_start = i + 1
            break
    if data_start is None or not fields:
        raise ValueError("Malformed PCD: missing FIELDS or DATA section")

    rows = [[float(x) for x in ln.split()] for ln in lines[data_start:] if ln.strip()]
    data = np.asarray(rows, dtype=np.float64).reshape(-1, len(fields))
    idx = {name: k for k, name in enumerate(fields)}
    points = data[:, [idx["x"], idx["y"], idx["z"]]]
    normals = None
    if {"normal_x", "normal_y", "normal_z"} <= set(idx):
        normals = data[:, [idx["normal_x"], idx["normal_y"], idx["normal_z"]]]
    return PointCloud(points=points, normals=normals)


# ---------------------------------------------------------------------------
# PLY (ASCII subset)
# ---------------------------------------------------------------------------


def _write_ply(pc: PointCloud, path: str) -> None:
    has_normals = pc.normals is not None
    n = pc.num_points
    props = ["x", "y", "z"] + (["nx", "ny", "nz"] if has_normals else [])
    arr = _stack(pc)
    header_lines = ["ply", "format ascii 1.0", f"element vertex {n}"]
    header_lines += [f"property float {p}" for p in props]
    header_lines.append("end_header")
    with open(path, "w") as fh:
        fh.write("\n".join(header_lines) + "\n")
        for row in arr:
            fh.write(" ".join(_FLOAT_FMT % v for v in row) + "\n")


def _read_ply(path: str) -> PointCloud:
    with open(path) as fh:
        lines = fh.read().splitlines()
    if not lines or lines[0].strip() != "ply":
        raise ValueError("Not a PLY file")
    props: list[str] = []
    n_vertices = 0
    header_end = None
    in_vertex_element = False
    for i, line in enumerate(lines):
        s = line.strip()
        if s.startswith("element vertex"):
            n_vertices = int(s.split()[-1])
            in_vertex_element = True
        elif s.startswith("element"):
            # A different element (e.g. "element face N") ends the vertex block;
            # only vertex properties belong in `props`.
            in_vertex_element = False
        elif s.startswith("property") and in_vertex_element:
            props.append(s.split()[-1])
        elif s == "end_header":
            header_end = i + 1
            break
    if header_end is None:
        raise ValueError("Malformed PLY: missing end_header")

    rows = [
        [float(x) for x in ln.split()]
        for ln in lines[header_end : header_end + n_vertices]
    ]
    data = np.asarray(rows, dtype=np.float64).reshape(n_vertices, len(props))
    idx = {name: k for k, name in enumerate(props)}
    points = data[:, [idx["x"], idx["y"], idx["z"]]]
    normals = None
    if {"nx", "ny", "nz"} <= set(idx):
        normals = data[:, [idx["nx"], idx["ny"], idx["nz"]]]
    return PointCloud(points=points, normals=normals)


# ---------------------------------------------------------------------------
# Mesh sampling (trimesh, lazy)
# ---------------------------------------------------------------------------


def sample_from_mesh(
    mesh: Any,
    n: int,
    with_normals: bool = False,
    method: str = "surface",
) -> PointCloud:
    """Sample ``n`` points from a mesh (a trimesh.Trimesh or a file path).

    Args:
        mesh: a ``trimesh.Trimesh`` object or a path to a mesh file.
        n: number of points to sample.
        with_normals: also return the face normal at each sampled point.
        method: ``"surface"`` for uniform area-weighted surface sampling.

    Returns:
        A PointCloud with ``n`` points (and normals when requested).
    """
    try:
        import trimesh
    except ImportError as exc:  # pragma: no cover - exercised only without trimesh
        raise ImportError("sample_from_mesh requires trimesh") from exc

    if isinstance(mesh, str):
        mesh = trimesh.load(mesh, force="mesh")

    if method != "surface":
        raise ValueError(f"Unsupported sampling method: {method!r}")

    points, face_idx = trimesh.sample.sample_surface(mesh, n)
    normals: np.ndarray | None = None
    if with_normals:
        normals = np.asarray(mesh.face_normals[face_idx], dtype=np.float64)
    return PointCloud(points=np.asarray(points, dtype=np.float64), normals=normals)
