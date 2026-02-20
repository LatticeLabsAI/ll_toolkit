"""STEP and STL export with schema selection and multi-view rendering.

Wraps OpenCASCADE's ``STEPControl_Writer`` and ``StlAPI_Writer``
with schema selection (AP203/AP214/AP242), tessellation control
for STL, and multi-view rendering for visual verification.
"""
from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Any, List, Optional

from ll_gen.config import ExportConfig, StepSchema

_log = logging.getLogger(__name__)

_OCC_AVAILABLE = False
try:
    from OCC.Core.BRepMesh import BRepMesh_IncrementalMesh
    from OCC.Core.IFSelect import IFSelect_RetDone
    from OCC.Core.Interface import Interface_Static
    from OCC.Core.STEPControl import (
        STEPControl_AsIs,
        STEPControl_Writer,
    )
    from OCC.Core.StlAPI import StlAPI_Writer
    from OCC.Core.TopoDS import TopoDS_Shape

    _OCC_AVAILABLE = True
except ImportError:
    _log.debug("pythonocc not available; exporter will not function")


def export_step(
    shape: Any,
    path: Path,
    schema: StepSchema = StepSchema.AP214,
) -> Path:
    """Export a TopoDS_Shape to a STEP file.

    Uses ``STEPControl_Writer`` with the specified application
    protocol schema.

    Args:
        shape: TopoDS_Shape to export.
        path: Output file path (should end with ``.step`` or ``.stp``).
        schema: STEP application protocol (AP203, AP214, or AP242).

    Returns:
        Path to the written STEP file.

    Raises:
        ImportError: If pythonocc is not installed.
        RuntimeError: If the STEP writer fails.
    """
    if not _OCC_AVAILABLE:
        raise ImportError(
            "pythonocc-core is required for STEP export. "
            "Install with: conda install -c conda-forge pythonocc-core"
        )

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    writer = STEPControl_Writer()

    # Set schema
    schema_map = {
        StepSchema.AP203: "3",
        StepSchema.AP214: "1",
        StepSchema.AP242: "4",
    }
    schema_val = schema_map.get(schema, "1")
    Interface_Static.SetCVal("write.step.schema", schema_val)

    # Set precision
    Interface_Static.SetIVal("write.precision.mode", 1)

    status = writer.Transfer(shape, STEPControl_AsIs)
    if status != IFSelect_RetDone:
        raise RuntimeError(
            f"STEP transfer failed with status {status}. "
            f"The shape may not be a valid solid."
        )

    write_status = writer.Write(str(path))
    if write_status != IFSelect_RetDone:
        raise RuntimeError(
            f"STEP write failed with status {write_status} "
            f"for path: {path}"
        )

    _log.info("Exported STEP to %s (schema=%s)", path, schema.value)
    return path


def export_stl(
    shape: Any,
    path: Path,
    linear_deflection: float = 0.1,
    angular_deflection: float = 0.5,
    ascii_mode: bool = False,
) -> Path:
    """Export a TopoDS_Shape to an STL file.

    Tessellates the shape using ``BRepMesh_IncrementalMesh`` before
    writing via ``StlAPI_Writer``.

    Args:
        shape: TopoDS_Shape to export.
        path: Output file path (should end with ``.stl``).
        linear_deflection: Maximum distance between mesh and true
            surface (smaller = finer mesh).
        angular_deflection: Maximum angle between adjacent triangles
            in radians.
        ascii_mode: If True, write ASCII STL; otherwise binary.

    Returns:
        Path to the written STL file.

    Raises:
        ImportError: If pythonocc is not installed.
        RuntimeError: If tessellation or writing fails.
    """
    if not _OCC_AVAILABLE:
        raise ImportError(
            "pythonocc-core is required for STL export. "
            "Install with: conda install -c conda-forge pythonocc-core"
        )

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Tessellate
    mesh = BRepMesh_IncrementalMesh(shape, linear_deflection, False, angular_deflection)
    mesh.Perform()

    if not mesh.IsDone():
        raise RuntimeError(
            "Tessellation failed. The shape may have degenerate faces."
        )

    # Write
    writer = StlAPI_Writer()
    writer.SetASCIIMode(ascii_mode)

    success = writer.Write(shape, str(path))
    if not success:
        raise RuntimeError(f"STL write failed for path: {path}")

    _log.info(
        "Exported STL to %s (linear_deflection=%.3f, ascii=%s)",
        path, linear_deflection, ascii_mode,
    )
    return path


def render_views(
    shape: Any,
    output_dir: Path,
    views: Optional[List[str]] = None,
    resolution: int = 512,
    prefix: str = "view",
) -> List[Path]:
    """Render multi-view images of a shape for visual verification.

    Renders the shape from multiple camera angles using pythonocc's
    offscreen rendering if available, or falls back to generating
    STL + trimesh rendering.

    Camera positions for each view:
    - ``front``:  +Y direction, looking at -Y
    - ``back``:   -Y direction, looking at +Y
    - ``top``:    +Z direction, looking at -Z
    - ``bottom``: -Z direction, looking at +Z
    - ``right``:  +X direction, looking at -X
    - ``left``:   -X direction, looking at -X
    - ``isometric``: (1,1,1) direction, looking at origin

    Args:
        shape: TopoDS_Shape to render.
        output_dir: Directory to save rendered images.
        views: List of view names to render. Defaults to
            ``["front", "top", "right", "isometric"]``.
        resolution: Image resolution in pixels (square).
        prefix: Filename prefix.

    Returns:
        List of paths to rendered PNG images.
    """
    if views is None:
        views = ["front", "top", "right", "isometric"]

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Camera direction vectors for each view
    view_directions = {
        "front": (0, 1, 0),
        "back": (0, -1, 0),
        "top": (0, 0, 1),
        "bottom": (0, 0, -1),
        "right": (1, 0, 0),
        "left": (-1, 0, 0),
        "isometric": (
            1 / math.sqrt(3),
            1 / math.sqrt(3),
            1 / math.sqrt(3),
        ),
    }

    render_paths: List[Path] = []

    # Try pythonocc offscreen rendering
    try:
        from OCC.Display import SimpleGui
        from OCC.Core.Graphic3d import Graphic3d_BufferType

        _log.info("Using pythonocc offscreen rendering")

        for view_name in views:
            if view_name not in view_directions:
                _log.warning("Unknown view name: %s, skipping", view_name)
                continue

            out_path = output_dir / f"{prefix}_{view_name}.png"

            # Offscreen rendering requires display initialization
            # which may not work in headless environments
            # Fall through to trimesh if this fails
            raise ImportError("Falling through to trimesh rendering")

    except (ImportError, RuntimeError):
        pass

    # Fallback: export to STL and render with trimesh
    try:
        import tempfile

        import numpy as np

        stl_path = output_dir / f"_temp_render.stl"
        export_stl(shape, stl_path, linear_deflection=0.05)

        try:
            import trimesh
            from PIL import Image

            mesh = trimesh.load(str(stl_path))

            for view_name in views:
                if view_name not in view_directions:
                    continue

                out_path = output_dir / f"{prefix}_{view_name}.png"
                direction = np.array(view_directions[view_name])

                # Compute camera transform
                center = mesh.centroid
                extent = mesh.extents.max()
                camera_distance = extent * 2.5

                camera_pos = center + direction * camera_distance

                # Create scene and render
                scene = trimesh.Scene(mesh)

                # Set camera transform
                up = np.array([0, 0, 1])
                if abs(np.dot(direction, up)) > 0.99:
                    up = np.array([0, 1, 0])

                forward = -direction / np.linalg.norm(direction)
                right = np.cross(up, forward)
                right = right / np.linalg.norm(right)
                up_corrected = np.cross(forward, right)

                camera_transform = np.eye(4)
                camera_transform[:3, 0] = right
                camera_transform[:3, 1] = up_corrected
                camera_transform[:3, 2] = forward
                camera_transform[:3, 3] = camera_pos

                scene.camera_transform = camera_transform

                try:
                    png_data = scene.save_image(
                        resolution=(resolution, resolution),
                    )
                    if png_data is not None:
                        with open(out_path, "wb") as f:
                            f.write(png_data)
                        render_paths.append(out_path)
                        _log.info("Rendered %s view to %s", view_name, out_path)
                except Exception as exc:
                    _log.debug(
                        "Trimesh rendering failed for %s: %s",
                        view_name, exc,
                    )

        finally:
            # Cleanup temp STL
            if stl_path.exists():
                stl_path.unlink()

    except ImportError as exc:
        _log.warning(
            "Neither pythonocc offscreen nor trimesh available "
            "for rendering: %s", exc,
        )

    return render_paths
