"""
Mesh preprocessing pipeline for LL-OCADR.
Handles loading, chunking, and tokenization of 3D meshes.
Mirrors DeepSeek-OCR's image_process.py for 3D geometry.
"""

import math
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
import numpy as np
import torch
import trimesh
from transformers import BatchFeature, LlamaTokenizerFast
from transformers.processing_utils import ProcessorMixin

from file_content_chunker import UnifiedCADContentChunker


@dataclass
class BRepData:
    """B-Rep (Boundary Representation) data for STEP/CAD files."""
    surfaces: List[Dict]  # Analytical surfaces (cylinders, planes, cones, etc.)
    curves: List[Dict]    # Parametric curves (circles, ellipses, lines, splines)
    faces: List[Dict]     # Topological faces with surface references
    edges: List[Dict]     # Topological edges with curve references
    vertices: List[Dict]  # Topological vertices (not tessellation vertices!)
    bbox: Tuple[np.ndarray, np.ndarray]
    metadata: Dict = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

    @property
    def num_surfaces(self) -> int:
        return len(self.surfaces)

    @property
    def num_faces(self) -> int:
        return len(self.faces)

    @property
    def num_edges(self) -> int:
        return len(self.edges)

    @property
    def num_vertices(self) -> int:
        return len(self.vertices)


@dataclass
class MeshData:
    """Triangle mesh representation for STL/OBJ files."""
    vertices: np.ndarray  # [N, 3] - vertex coordinates
    faces: np.ndarray     # [F, 3] - face indices
    normals: np.ndarray   # [N, 3] - vertex normals
    bbox: Tuple[np.ndarray, np.ndarray]  # (min_xyz, max_xyz)
    metadata: Dict = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

        # Calculate bbox if not provided
        if self.bbox is None and self.vertices is not None:
            self.bbox = (
                np.min(self.vertices, axis=0),
                np.max(self.vertices, axis=0)
            )

    @property
    def num_vertices(self) -> int:
        return len(self.vertices) if self.vertices is not None else 0

    @property
    def num_faces(self) -> int:
        return len(self.faces) if self.faces is not None else 0

    @property
    def bbox_volume(self) -> float:
        """Calculate bounding box volume."""
        if self.bbox is None:
            return 0.0
        min_xyz, max_xyz = self.bbox
        dimensions = max_xyz - min_xyz
        return np.prod(dimensions)


@dataclass
class MeshChunk:
    """Spatial subdivision of a mesh (like image tiles)."""
    vertices: np.ndarray  # [M, 3] - local vertex coordinates
    faces: np.ndarray     # [K, 3] - face indices (reindexed to local vertices)
    normals: np.ndarray   # [M, 3] - vertex normals
    bbox: Tuple[np.ndarray, np.ndarray]  # Spatial bounding box
    chunk_id: Tuple[int, int, int]  # (x, y, z) position in grid

    @property
    def num_vertices(self) -> int:
        return len(self.vertices) if self.vertices is not None else 0

    @property
    def num_faces(self) -> int:
        return len(self.faces) if self.faces is not None else 0


def compute_bbox_volume(bbox: Tuple[np.ndarray, np.ndarray]) -> float:
    """Calculate volume of a bounding box."""
    min_xyz, max_xyz = bbox
    dimensions = max_xyz - min_xyz
    return np.prod(dimensions)


def vertices_in_bbox(vertices: np.ndarray,
                     bbox_min: np.ndarray,
                     bbox_max: np.ndarray) -> np.ndarray:
    """
    Find vertices within a bounding box.

    Args:
        vertices: [N, 3] vertex coordinates
        bbox_min: [3] minimum corner
        bbox_max: [3] maximum corner

    Returns:
        mask: [N] boolean mask of vertices inside bbox
    """
    mask = np.all((vertices >= bbox_min) & (vertices <= bbox_max), axis=1)
    return mask


def extract_faces_in_region(faces: np.ndarray, vertex_mask: np.ndarray) -> np.ndarray:
    """
    Extract faces that have all vertices in the masked region.

    Args:
        faces: [F, 3] face indices
        vertex_mask: [N] boolean mask of included vertices

    Returns:
        local_faces: [K, 3] reindexed faces for local vertices
    """
    # Find faces where all 3 vertices are in the region
    face_mask = np.all(vertex_mask[faces], axis=1)
    selected_faces = faces[face_mask]

    # Reindex faces to local vertex indices
    vertex_indices = np.where(vertex_mask)[0]
    index_map = {old_idx: new_idx for new_idx, old_idx in enumerate(vertex_indices)}

    local_faces = np.array([
        [index_map[selected_faces[i, 0]], index_map[selected_faces[i, 1]], index_map[selected_faces[i, 2]]]
        for i in range(len(selected_faces))
    ], dtype=np.int32)

    return local_faces


def dynamic_mesh_partition(mesh: MeshData,
                          min_chunk_size: Optional[int] = None,
                          max_chunks: int = 27) -> List[MeshChunk]:
    """
    Octree-based spatial subdivision similar to image tiling.
    Automatically determines optimal chunk size if not specified.

    Args:
        mesh: Full mesh data
        min_chunk_size: Minimum faces per chunk. If None, determined dynamically.
        max_chunks: Maximum subdivisions (2x2x2=8 or 3x3x3=27)

    Returns:
        List of MeshChunk objects with spatial subdivisions
    """
    total_faces = mesh.num_faces

    # Determine optimal min_chunk_size based on total faces
    # Similar to DeepSeek-OCR's dynamic tile sizing
    if min_chunk_size is None:
        # Target ~384 tokens per chunk (96 faces × 4 tokens)
        # Scales with complexity: more faces = larger chunks to limit total chunks
        if total_faces < 500:
            min_chunk_size = total_faces  # Single chunk for small meshes
        elif total_faces < 2000:
            min_chunk_size = 200  # Small chunks for moderate meshes
        elif total_faces < 10000:
            min_chunk_size = 400  # Medium chunks
        else:
            min_chunk_size = 800  # Larger chunks for complex meshes

    # Determine optimal subdivision based on mesh complexity
    if total_faces <= min_chunk_size:
        subdivision = (1, 1, 1)  # No chunking
    elif total_faces <= min_chunk_size * 8:
        subdivision = (2, 2, 2)  # 8 chunks
    else:
        subdivision = (3, 3, 3)  # 27 chunks

    # Create spatial grid
    chunks = []
    bbox_min, bbox_max = mesh.bbox
    grid_size = (bbox_max - bbox_min) / np.array(subdivision)

    for ix in range(subdivision[0]):
        for iy in range(subdivision[1]):
            for iz in range(subdivision[2]):
                # Define chunk bounding box
                chunk_min = bbox_min + np.array([ix, iy, iz]) * grid_size
                chunk_max = chunk_min + grid_size

                # Extract vertices within bbox
                mask = vertices_in_bbox(mesh.vertices, chunk_min, chunk_max)

                if not np.any(mask):
                    # Empty chunk, skip
                    continue

                chunk_vertices = mesh.vertices[mask]
                chunk_normals = mesh.normals[mask]

                # Extract and reindex faces
                chunk_faces = extract_faces_in_region(mesh.faces, mask)

                if len(chunk_faces) == 0:
                    # No faces in this chunk
                    continue

                chunks.append(MeshChunk(
                    vertices=chunk_vertices,
                    faces=chunk_faces,
                    normals=chunk_normals,
                    bbox=(chunk_min, chunk_max),
                    chunk_id=(ix, iy, iz)
                ))

    return chunks


def create_global_view(mesh: MeshData, target_faces: int = 4096) -> MeshData:
    """
    Downsample full mesh for global context.
    Analogous to resizing image to 1024x1024.

    Args:
        mesh: Full resolution mesh
        target_faces: Target face count for global view

    Returns:
        Downsampled mesh maintaining overall shape
    """
    if mesh.num_faces <= target_faces:
        return mesh  # Already small enough

    # Create trimesh object
    tmesh = trimesh.Trimesh(
        vertices=mesh.vertices,
        faces=mesh.faces,
        vertex_normals=mesh.normals
    )

    # Try quadric decimation first, fall back to simple face sampling
    try:
        simplified_mesh = tmesh.simplify_quadric_decimation(target_faces)
    except (ImportError, AttributeError, ModuleNotFoundError):
        # Fall back to simple uniform face sampling
        # This is fast and doesn't require extra dependencies
        face_indices = np.random.choice(
            len(tmesh.faces),
            size=min(target_faces, len(tmesh.faces)),
            replace=False
        )
        # Keep only selected faces and their vertices
        selected_faces = tmesh.faces[face_indices]
        # Get unique vertices used by selected faces
        used_vertices = np.unique(selected_faces.flatten())
        vertex_map = {old: new for new, old in enumerate(used_vertices)}
        # Reindex faces
        new_faces = np.array([[vertex_map[f[0]], vertex_map[f[1]], vertex_map[f[2]]]
                              for f in selected_faces], dtype=np.int32)
        simplified_mesh = trimesh.Trimesh(
            vertices=tmesh.vertices[used_vertices],
            faces=new_faces,
            process=False
        )

    # Ensure normals are computed
    if simplified_mesh.vertex_normals is None or len(simplified_mesh.vertex_normals) == 0:
        simplified_mesh.vertex_normals = simplified_mesh.vertex_normals

    return MeshData(
        vertices=simplified_mesh.vertices,
        faces=simplified_mesh.faces,
        normals=simplified_mesh.vertex_normals,
        bbox=(
            np.min(simplified_mesh.vertices, axis=0),
            np.max(simplified_mesh.vertices, axis=0)
        ),
        metadata={'downsampled': True, 'original_faces': mesh.num_faces}
    )


class CADLoader:
    """Unified loader for STEP (B-Rep) and STL/OBJ/PLY (mesh) formats."""

    def __init__(self):
        self.loaders = {
            '.step': self._load_step,
            '.stp': self._load_step,
            '.stl': self._load_stl_obj,
            '.obj': self._load_stl_obj,
            '.ply': self._load_stl_obj,
        }

    def load(self, file_path: str):
        """
        Load CAD file - returns BRepData for STEP or MeshData for STL/OBJ.

        Args:
            file_path: Path to CAD/mesh file

        Returns:
            BRepData for STEP files, MeshData for STL/OBJ/PLY files
        """
        import os
        _, ext = os.path.splitext(file_path.lower())

        if ext not in self.loaders:
            raise ValueError(f"Unsupported file format: {ext}")

        return self.loaders[ext](file_path)

    def _load_stl_obj(self, mesh_file: str) -> MeshData:
        """
        Load STL, OBJ, or PLY using trimesh.
        """
        mesh = trimesh.load_mesh(mesh_file)

        # Ensure watertight (close holes)
        if not mesh.is_watertight:
            mesh.fill_holes()

        # Compute vertex normals if missing
        if mesh.vertex_normals is None or len(mesh.vertex_normals) == 0:
            mesh.vertex_normals = mesh.vertex_normals

        return MeshData(
            vertices=mesh.vertices,
            faces=mesh.faces,
            normals=mesh.vertex_normals,
            bbox=tuple(mesh.bounds),
            metadata={'file_type': 'mesh', 'watertight': mesh.is_watertight}
        )

    def _load_step(self, step_file: str) -> BRepData:
        """
        Load STEP file as B-Rep (preserves parametric surfaces, not tessellated).
        """
        try:
            from OCC.Core.STEPControl import STEPControl_Reader
            from OCC.Core.TopExp import TopExp_Explorer
            from OCC.Core.TopAbs import TopAbs_FACE, TopAbs_EDGE, TopAbs_VERTEX
            from OCC.Core.BRep import BRep_Tool
            from OCC.Core.GeomAbs import GeomAbs_Plane, GeomAbs_Cylinder, GeomAbs_Cone, GeomAbs_Sphere, GeomAbs_Torus
            from OCC.Core.BRepAdaptor import BRepAdaptor_Surface, BRepAdaptor_Curve
            from OCC.Extend.TopologyUtils import TopologyExplorer
            from OCC.Core.gp import gp_Pnt
        except ImportError:
            raise ImportError(
                "pythonocc-core is required for STEP file support. "
                "Install with: conda install -c conda-forge pythonocc-core"
            )

        # Read STEP file
        reader = STEPControl_Reader()
        status = reader.ReadFile(step_file)

        if status != 1:
            raise ValueError(f"Failed to read STEP file: {step_file}")

        reader.TransferRoots()
        shape = reader.Shape()

        # Extract B-Rep data (NO tessellation!)
        explorer = TopologyExplorer(shape)

        surfaces = []
        curves = []
        faces_topo = []
        edges_topo = []
        vertices_topo = []

        all_points = []

        # Extract topological faces with their analytical surfaces
        for idx, face in enumerate(explorer.faces()):
            adaptor = BRepAdaptor_Surface(face)
            surface_type = adaptor.GetType()

            surface_info = {'face_id': idx}

            if surface_type == GeomAbs_Plane:
                pln = adaptor.Plane()
                surface_info.update({
                    'type': 'PLANE',
                    'location': [pln.Location().X(), pln.Location().Y(), pln.Location().Z()],
                    'normal': [pln.Axis().Direction().X(), pln.Axis().Direction().Y(), pln.Axis().Direction().Z()]
                })
            elif surface_type == GeomAbs_Cylinder:
                cyl = adaptor.Cylinder()
                surface_info.update({
                    'type': 'CYLINDER',
                    'radius': cyl.Radius(),
                    'axis_location': [cyl.Location().X(), cyl.Location().Y(), cyl.Location().Z()],
                    'axis_direction': [cyl.Axis().Direction().X(), cyl.Axis().Direction().Y(), cyl.Axis().Direction().Z()]
                })
            elif surface_type == GeomAbs_Cone:
                cone = adaptor.Cone()
                surface_info.update({
                    'type': 'CONE',
                    'radius': cone.RefRadius(),
                    'semi_angle': cone.SemiAngle()
                })
            elif surface_type == GeomAbs_Sphere:
                sphere = adaptor.Sphere()
                surface_info.update({
                    'type': 'SPHERE',
                    'radius': sphere.Radius(),
                    'center': [sphere.Location().X(), sphere.Location().Y(), sphere.Location().Z()]
                })
            elif surface_type == GeomAbs_Torus:
                torus = adaptor.Torus()
                surface_info.update({
                    'type': 'TORUS',
                    'major_radius': torus.MajorRadius(),
                    'minor_radius': torus.MinorRadius()
                })
            else:
                surface_info.update({'type': 'NURBS'})

            surfaces.append(surface_info)
            faces_topo.append({'surface_id': len(surfaces) - 1, 'face_id': idx})

        # Extract edges with curves
        for idx, edge in enumerate(explorer.edges()):
            curve_adaptor = BRepAdaptor_Curve(edge)
            curve_type = curve_adaptor.GetType()

            # Get start/end points
            first_param = curve_adaptor.FirstParameter()
            last_param = curve_adaptor.LastParameter()
            start_pnt = gp_Pnt()
            end_pnt = gp_Pnt()
            curve_adaptor.D0(first_param, start_pnt)
            curve_adaptor.D0(last_param, end_pnt)

            edge_info = {
                'edge_id': idx,
                'start': [start_pnt.X(), start_pnt.Y(), start_pnt.Z()],
                'end': [end_pnt.X(), end_pnt.Y(), end_pnt.Z()]
            }
            edges_topo.append(edge_info)

            all_points.extend([[start_pnt.X(), start_pnt.Y(), start_pnt.Z()],
                             [end_pnt.X(), end_pnt.Y(), end_pnt.Z()]])

        # Extract vertices
        for idx, vertex in enumerate(explorer.vertices()):
            pnt = BRep_Tool.Pnt(vertex)
            vertices_topo.append({
                'vertex_id': idx,
                'point': [pnt.X(), pnt.Y(), pnt.Z()]
            })
            all_points.append([pnt.X(), pnt.Y(), pnt.Z()])

        # Calculate bounding box
        if all_points:
            points_array = np.array(all_points)
            bbox = (np.min(points_array, axis=0), np.max(points_array, axis=0))
        else:
            bbox = (np.zeros(3), np.zeros(3))

        return BRepData(
            surfaces=surfaces,
            curves=curves,
            faces=faces_topo,
            edges=edges_topo,
            vertices=vertices_topo,
            bbox=bbox,
            metadata={'file_type': 'step', 'num_topo_faces': len(faces_topo)}
        )


class LLOCADRProcessor:
    """
    Main preprocessing pipeline. Mirrors DeepseekOCRProcessor.
    Processes actual file format content (like OCR processes document text).
    """

    def __init__(self, tokenizer, mesh_token_id: int, chunk_size: Optional[int] = None):
        self.tokenizer = tokenizer
        self.mesh_token_id = mesh_token_id  # From vocab: "<mesh>"
        self.loader = CADLoader()
        # chunk_size=None enables dynamic chunking based on file analysis
        self.content_chunker = UnifiedCADContentChunker(chunk_size=chunk_size)

    def _chunk_brep(self, brep: BRepData, max_surfaces_per_chunk: int = 10) -> List[Dict]:
        """
        Chunk BRepData by grouping surfaces.

        Args:
            brep: BRepData object
            max_surfaces_per_chunk: Max surfaces per chunk (analogous to max faces in mesh chunking)

        Returns:
            List of BRep chunks
        """
        chunks = []
        num_surfaces = len(brep.surfaces)

        for i in range(0, num_surfaces, max_surfaces_per_chunk):
            end_idx = min(i + max_surfaces_per_chunk, num_surfaces)

            chunk = {
                'surfaces': brep.surfaces[i:end_idx],
                'faces': brep.faces[i:end_idx],  # Assume faces align with surfaces
                'chunk_id': len(chunks),
                'start_surface': i,
                'end_surface': end_idx
            }
            chunks.append(chunk)

        return chunks

    def get_chunks(self, file_path: str, chunk_type: str = 'both') -> Dict:
        """
        Generate and return all chunks for a CAD file.

        Args:
            file_path: Path to CAD/mesh file
            chunk_type: 'file' for file content chunks, 'spatial' for geometry chunks, 'both' for all

        Returns:
            Dictionary with chunk information
        """
        result = {
            'file_path': file_path,
            'file_content_chunks': None,
            'spatial_chunks': None,
            'data_type': None
        }

        # Load CAD file
        data = self.loader.load(file_path)
        result['data_type'] = type(data).__name__

        # Get file content chunks
        if chunk_type in ['file', 'both']:
            file_chunks = self.content_chunker.chunk_file(file_path)
            result['file_content_chunks'] = {
                'num_chunks': len(file_chunks),
                'chunks': file_chunks,
                'stats': self.content_chunker.get_chunk_statistics(file_chunks)
            }

        # Get spatial/topological chunks
        if chunk_type in ['spatial', 'both']:
            if isinstance(data, BRepData):
                spatial_chunks = self._chunk_brep(data)
                result['spatial_chunks'] = {
                    'type': 'brep_topology',
                    'num_chunks': len(spatial_chunks),
                    'chunks': spatial_chunks,
                    'total_surfaces': data.num_surfaces,
                    'total_faces': data.num_faces,
                    'total_edges': data.num_edges
                }
            elif isinstance(data, MeshData):
                spatial_chunks = dynamic_mesh_partition(data)
                result['spatial_chunks'] = {
                    'type': 'mesh_octree',
                    'num_chunks': len(spatial_chunks),
                    'chunks': [
                        {
                            'chunk_id': i,
                            'num_vertices': chunk.num_vertices,
                            'num_faces': chunk.num_faces,
                            'bbox': chunk.bbox,
                            'grid_position': chunk.chunk_id
                        }
                        for i, chunk in enumerate(spatial_chunks)
                    ],
                    'total_vertices': data.num_vertices,
                    'total_faces': data.num_faces
                }

        return result

    def tokenize_with_meshes(self,
                            mesh_files: List[str],
                            conversation: str,
                            cropping: bool = True) -> Dict[str, torch.Tensor]:
        """
        Full preprocessing pipeline:
        1. Load CAD files (BRep or Mesh)
        2. Chunk based on data type
        3. Convert to torch tensors
        4. Replace <mesh> tokens with appropriate count

        Args:
            mesh_files: List of paths to CAD/mesh files
            conversation: Text with <mesh> placeholders
            cropping: Whether to use chunking

        Returns:
            Dictionary with preprocessed data
        """
        cad_data = []
        chunks_list = []
        spatial_partitions = []

        for cad_file in mesh_files:
            # Load CAD file (BRepData or MeshData)
            data = self.loader.load(cad_file)

            # Handle based on type
            if isinstance(data, BRepData):
                # STEP file - chunk by topology
                if cropping:
                    chunks = self._chunk_brep(data)
                    chunks_list.append(chunks)
                    subdivision = (1, 1, 1)  # BRep doesn't use spatial subdivision
                    spatial_partitions.append(subdivision)
                else:
                    chunks_list.append([])
                    spatial_partitions.append((1, 1, 1))

                cad_data.append(data)

            elif isinstance(data, MeshData):
                # STL/OBJ file - spatial chunking
                if cropping:
                    chunks = dynamic_mesh_partition(data)
                    chunks_list.append(chunks)

                    # Determine subdivision from chunks
                    if len(chunks) == 0:
                        subdivision = (1, 1, 1)
                    elif len(chunks) <= 8:
                        subdivision = (2, 2, 2)
                    else:
                        subdivision = (3, 3, 3)
                    spatial_partitions.append(subdivision)
                else:
                    chunks_list.append([])
                    spatial_partitions.append((1, 1, 1))

                # Create global view for meshes
                global_mesh = create_global_view(data, target_faces=4096)
                cad_data.append(global_mesh)
            else:
                raise ValueError(f"Unknown data type: {type(data)}")

        # Convert to tensors (handles both BRepData and MeshData)
        vertex_coords, vertex_normals = self._cad_to_tensors(cad_data)
        chunks_coords, chunks_normals = self._chunks_to_tensors(chunks_list)

        # Tokenize conversation and replace <mesh> placeholders
        tokenized = self._create_token_sequence(conversation, cad_data, chunks_list)

        return {
            'input_ids': tokenized['input_ids'],
            'vertex_coords': vertex_coords,
            'vertex_normals': vertex_normals,
            'chunks_coords': chunks_coords,
            'chunks_normals': chunks_normals,
            'mesh_spatial_partition': torch.tensor(spatial_partitions, dtype=torch.long),
            'num_mesh_tokens': tokenized['num_mesh_tokens']
        }

    def _cad_to_tensors(self, cad_data: List) -> Tuple[torch.Tensor, torch.Tensor]:
        """Convert list of CAD data (BRep or Mesh) to batched tensors."""
        coords_batch = []
        normals_batch = []

        for data in cad_data:
            if isinstance(data, BRepData):
                # For BRep, extract representative points from topology
                points = []
                for vertex in data.vertices:
                    points.append(vertex['point'])

                # Convert to arrays
                if points:
                    coords = np.array(points, dtype=np.float32)
                    # For BRep, use zero normals (surfaces have analytical normals, not vertex normals)
                    normals = np.zeros_like(coords)
                else:
                    coords = np.zeros((1, 3), dtype=np.float32)
                    normals = np.zeros((1, 3), dtype=np.float32)

                coords_batch.append(coords)
                normals_batch.append(normals)

            elif isinstance(data, MeshData):
                # For meshes, use vertices and normals directly
                coords_batch.append(data.vertices)
                normals_batch.append(data.normals)

        # Pad to same size
        max_verts = max(c.shape[0] for c in coords_batch)

        padded_coords_batch = []
        padded_normals_batch = []

        for coords, normals in zip(coords_batch, normals_batch):
            padded_coords = np.zeros((max_verts, 3), dtype=np.float32)
            padded_normals = np.zeros((max_verts, 3), dtype=np.float32)

            n = coords.shape[0]
            padded_coords[:n] = coords
            padded_normals[:n] = normals

            padded_coords_batch.append(padded_coords)
            padded_normals_batch.append(padded_normals)

        return (
            torch.from_numpy(np.stack(padded_coords_batch)),
            torch.from_numpy(np.stack(padded_normals_batch))
        )

    def _chunks_to_tensors(self, chunks_list: List[List]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Convert list of chunk lists (BRep or Mesh) to batched tensors."""
        if not chunks_list or not chunks_list[0]:
            # No chunks, return empty tensors
            return torch.zeros(1, 0, 0, 3), torch.zeros(1, 0, 0, 3)

        coords_batch = []
        normals_batch = []

        for chunks in chunks_list:
            if not chunks:
                # Empty chunk list
                chunk_coords = np.zeros((1, 1, 3), dtype=np.float32)
                chunk_normals = np.zeros((1, 1, 3), dtype=np.float32)
                coords_batch.append(chunk_coords)
                normals_batch.append(chunk_normals)
                continue

            # Check chunk type
            first_chunk = chunks[0]

            if isinstance(first_chunk, dict):
                # BRep chunks - extract surface centers/features
                chunk_features = []
                for chunk in chunks:
                    # Use first surface location as representative point
                    if chunk['surfaces']:
                        surf = chunk['surfaces'][0]
                        if 'location' in surf:
                            chunk_features.append(surf['location'])
                        elif 'center' in surf:
                            chunk_features.append(surf['center'])
                        elif 'axis_location' in surf:
                            chunk_features.append(surf['axis_location'])
                        else:
                            chunk_features.append([0.0, 0.0, 0.0])
                    else:
                        chunk_features.append([0.0, 0.0, 0.0])

                chunk_coords = np.array(chunk_features, dtype=np.float32).reshape(len(chunks), 1, 3)
                chunk_normals = np.zeros_like(chunk_coords)

            else:
                # MeshChunk objects
                max_verts = max(c.num_vertices for c in chunks)
                chunk_coords = np.zeros((len(chunks), max_verts, 3), dtype=np.float32)
                chunk_normals = np.zeros((len(chunks), max_verts, 3), dtype=np.float32)

                for i, chunk in enumerate(chunks):
                    n = chunk.num_vertices
                    chunk_coords[i, :n] = chunk.vertices
                    chunk_normals[i, :n] = chunk.normals

            coords_batch.append(chunk_coords)
            normals_batch.append(chunk_normals)

        # Pad to same dimensions
        max_chunks = max(c.shape[0] for c in coords_batch)
        max_verts = max(c.shape[1] for c in coords_batch)

        padded_coords = []
        padded_normals = []

        for coords, normals in zip(coords_batch, normals_batch):
            pad_coords = np.zeros((max_chunks, max_verts, 3), dtype=np.float32)
            pad_normals = np.zeros((max_chunks, max_verts, 3), dtype=np.float32)

            nc, nv = coords.shape[:2]
            pad_coords[:nc, :nv] = coords
            pad_normals[:nc, :nv] = normals

            padded_coords.append(pad_coords)
            padded_normals.append(pad_normals)

        return (
            torch.from_numpy(np.stack(padded_coords)),
            torch.from_numpy(np.stack(padded_normals))
        )

    def _create_token_sequence(self, conversation: str,
                              cad_data: List,
                              chunks_list: List[List]) -> Dict:
        """Create token sequence with mesh placeholders replaced."""
        # Split conversation by <mesh> markers
        parts = conversation.split('<mesh>')

        # Calculate tokens per CAD file
        num_mesh_tokens = []
        for i, (data, chunks) in enumerate(zip(cad_data, chunks_list)):
            # Global tokens: fixed ~384
            global_tokens = 384

            # Local tokens: depends on chunks
            if len(chunks) > 1:
                local_tokens = len(chunks) * 128
                boundary_tokens = int(np.cbrt(len(chunks)))  # Layers
                total = global_tokens + local_tokens + boundary_tokens + 1
            else:
                total = global_tokens + 1

            num_mesh_tokens.append(total)

        # Build token sequence
        token_ids = []

        for i, part in enumerate(parts):
            # Add text tokens
            if part:
                text_tokens = self.tokenizer.encode(part, add_special_tokens=(i == 0))
                token_ids.extend(text_tokens)

            # Add mesh placeholder tokens
            if i < len(num_mesh_tokens):
                token_ids.extend([self.mesh_token_id] * num_mesh_tokens[i])

        return {
            'input_ids': torch.tensor([token_ids], dtype=torch.long),
            'num_mesh_tokens': num_mesh_tokens
        }
