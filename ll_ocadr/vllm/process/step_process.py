"""
STEP file processing for LL-OCADR.
Handles CAD-specific preprocessing for STEP files using pythonocc-core.
"""

import numpy as np
from typing import Tuple, Optional
from pathlib import Path

try:
    from OCC.Core.STEPControl import STEPControl_Reader
    from OCC.Core.BRepMesh import BRepMesh_IncrementalMesh
    from OCC.Core.BRep import BRep_Tool
    from OCC.Core.TopExp import TopExp_Explorer
    from OCC.Core.TopAbs import TopAbs_FACE, TopAbs_VERTEX
    from OCC.Core.TopoDS import topods
    from OCC.Core.gp import gp_Pnt
    from OCC.Core.TopLoc import TopLoc_Location
    from OCC.Core.Bnd import Bnd_Box
    from OCC.Core.BRepBndLib import brepbndlib
    PYTHONOCC_AVAILABLE = True
except ImportError:
    PYTHONOCC_AVAILABLE = False
    print("Warning: pythonocc-core not available. STEP file support disabled.")
    print("Install with: conda install -c conda-forge pythonocc-core")


class STEPProcessor:
    """
    CAD-specific preprocessing for STEP files.
    Handles B-Rep geometry conversion to triangle mesh.
    """

    def __init__(self, tessellation_tolerance: float = 0.1):
        """
        Initialize STEP processor.

        Args:
            tessellation_tolerance: Mesh tessellation tolerance in mm
                                   Lower = finer mesh, higher = coarser mesh
        """
        if not PYTHONOCC_AVAILABLE:
            raise ImportError(
                "pythonocc-core is required for STEP file processing. "
                "Install with: conda install -c conda-forge pythonocc-core"
            )

        self.tessellation_tolerance = tessellation_tolerance

    def load_step_file(
        self,
        step_file: str,
        compute_normals: bool = True
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Load STEP file and convert to triangle mesh.

        Args:
            step_file: Path to STEP file
            compute_normals: Whether to compute vertex normals

        Returns:
            Tuple of (vertices, faces, normals, bbox):
                - vertices: [N, 3] vertex coordinates
                - faces: [F, 3] triangle face indices
                - normals: [N, 3] vertex normals
                - bbox: (min_xyz, max_xyz) bounding box
        """
        step_path = Path(step_file)
        if not step_path.exists():
            raise FileNotFoundError(f"STEP file not found: {step_file}")

        print(f"Loading STEP file: {step_path.name}")

        # Read STEP file
        reader = STEPControl_Reader()
        status = reader.ReadFile(str(step_file))

        if status != 1:  # IFSelect_RetDone
            raise RuntimeError(f"Failed to read STEP file: {step_file}")

        # Transfer roots to shape
        reader.TransferRoots()
        shape = reader.Shape()

        if shape.IsNull():
            raise RuntimeError(f"Failed to extract shape from STEP file: {step_file}")

        print(f"✓ Loaded STEP shape")

        # Tessellate B-Rep to triangle mesh
        vertices, faces = self._tessellate_shape(shape)

        print(f"✓ Tessellated to {len(vertices)} vertices, {len(faces)} faces")

        # Compute bounding box
        bbox = self._compute_bbox(shape)

        # Compute normals if requested
        if compute_normals:
            normals = self._compute_vertex_normals(vertices, faces)
        else:
            normals = np.zeros_like(vertices)

        return vertices, faces, normals, bbox

    def _tessellate_shape(
        self,
        shape
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Tessellate B-Rep shape to triangle mesh.

        Args:
            shape: TopoDS_Shape from STEP file

        Returns:
            Tuple of (vertices, faces)
        """
        # Perform incremental mesh tessellation
        mesh = BRepMesh_IncrementalMesh(shape, self.tessellation_tolerance)
        mesh.Perform()

        if not mesh.IsDone():
            raise RuntimeError("Mesh tessellation failed")

        # Extract triangles from all faces
        vertices_list = []
        faces_list = []
        vertex_offset = 0

        # Explore all faces in the shape
        explorer = TopExp_Explorer(shape, TopAbs_FACE)

        while explorer.More():
            face = topods.Face(explorer.Current())
            location = TopLoc_Location()
            facing = BRep_Tool.Triangulation(face, location)

            if facing is not None:
                # Get transformation
                transform = location.Transformation()

                # Extract vertices
                num_vertices = facing.NbNodes()
                face_vertices = np.zeros((num_vertices, 3))

                for i in range(1, num_vertices + 1):
                    vertex = facing.Node(i)
                    # Apply transformation
                    vertex.Transform(transform)
                    face_vertices[i - 1] = [vertex.X(), vertex.Y(), vertex.Z()]

                # Extract triangles
                num_triangles = facing.NbTriangles()
                face_faces = np.zeros((num_triangles, 3), dtype=np.int32)

                for i in range(1, num_triangles + 1):
                    triangle = facing.Triangle(i)
                    v1, v2, v3 = triangle.Get()
                    # Adjust indices to be 0-based and offset by accumulated vertices
                    face_faces[i - 1] = [
                        v1 - 1 + vertex_offset,
                        v2 - 1 + vertex_offset,
                        v3 - 1 + vertex_offset
                    ]

                vertices_list.append(face_vertices)
                faces_list.append(face_faces)
                vertex_offset += num_vertices

            explorer.Next()

        # Concatenate all faces
        if not vertices_list:
            raise RuntimeError("No triangulation data found in STEP file")

        vertices = np.vstack(vertices_list)
        faces = np.vstack(faces_list)

        return vertices, faces

    def _compute_bbox(self, shape) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute bounding box of shape.

        Args:
            shape: TopoDS_Shape

        Returns:
            Tuple of (min_xyz, max_xyz)
        """
        bbox = Bnd_Box()
        brepbndlib.Add(shape, bbox)

        xmin, ymin, zmin, xmax, ymax, zmax = bbox.Get()

        min_xyz = np.array([xmin, ymin, zmin])
        max_xyz = np.array([xmax, ymax, zmax])

        return min_xyz, max_xyz

    def _compute_vertex_normals(
        self,
        vertices: np.ndarray,
        faces: np.ndarray
    ) -> np.ndarray:
        """
        Compute vertex normals from face normals.

        Args:
            vertices: [N, 3] vertex coordinates
            faces: [F, 3] triangle face indices

        Returns:
            normals: [N, 3] vertex normals
        """
        # Initialize normals
        normals = np.zeros_like(vertices)

        # Compute face normals and accumulate to vertices
        for face in faces:
            v0, v1, v2 = vertices[face]

            # Compute face normal via cross product
            edge1 = v1 - v0
            edge2 = v2 - v0
            face_normal = np.cross(edge1, edge2)

            # Accumulate to vertices
            normals[face[0]] += face_normal
            normals[face[1]] += face_normal
            normals[face[2]] += face_normal

        # Normalize
        norms = np.linalg.norm(normals, axis=1, keepdims=True)
        norms[norms == 0] = 1  # Avoid division by zero
        normals = normals / norms

        return normals

    def validate_step_file(self, step_file: str) -> bool:
        """
        Validate STEP file can be loaded.

        Args:
            step_file: Path to STEP file

        Returns:
            True if valid, False otherwise
        """
        if not PYTHONOCC_AVAILABLE:
            return False

        step_path = Path(step_file)

        if not step_path.exists():
            print(f"✗ File not found: {step_file}")
            return False

        if step_path.suffix.lower() not in ['.step', '.stp']:
            print(f"✗ Not a STEP file: {step_file}")
            return False

        try:
            reader = STEPControl_Reader()
            status = reader.ReadFile(str(step_file))
            return status == 1
        except Exception as e:
            print(f"✗ Error validating STEP file: {e}")
            return False


def extract_step_metadata(step_file: str) -> dict:
    """
    Extract metadata from STEP file without full tessellation.

    Args:
        step_file: Path to STEP file

    Returns:
        Dictionary with metadata
    """
    if not PYTHONOCC_AVAILABLE:
        return {"error": "pythonocc-core not available"}

    try:
        reader = STEPControl_Reader()
        status = reader.ReadFile(str(step_file))

        if status != 1:
            return {"error": "Failed to read STEP file"}

        reader.TransferRoots()
        shape = reader.Shape()

        if shape.IsNull():
            return {"error": "Failed to extract shape"}

        # Compute bounding box
        bbox = Bnd_Box()
        brepbndlib.Add(shape, bbox)
        xmin, ymin, zmin, xmax, ymax, zmax = bbox.Get()

        # Count faces
        explorer = TopExp_Explorer(shape, TopAbs_FACE)
        num_faces = 0
        while explorer.More():
            num_faces += 1
            explorer.Next()

        # Count vertices
        explorer = TopExp_Explorer(shape, TopAbs_VERTEX)
        num_vertices = 0
        while explorer.More():
            num_vertices += 1
            explorer.Next()

        return {
            "num_faces": num_faces,
            "num_vertices": num_vertices,
            "bbox_min": [xmin, ymin, zmin],
            "bbox_max": [xmax, ymax, zmax],
            "bbox_volume": (xmax - xmin) * (ymax - ymin) * (zmax - zmin)
        }

    except Exception as e:
        return {"error": str(e)}


# Convenience function for quick loading
def load_step(
    step_file: str,
    tessellation_tolerance: float = 0.1
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """
    Quick load STEP file to mesh.

    Args:
        step_file: Path to STEP file
        tessellation_tolerance: Tessellation tolerance in mm

    Returns:
        Tuple of (vertices, faces, normals, bbox)
    """
    processor = STEPProcessor(tessellation_tolerance=tessellation_tolerance)
    return processor.load_step_file(step_file, compute_normals=True)
