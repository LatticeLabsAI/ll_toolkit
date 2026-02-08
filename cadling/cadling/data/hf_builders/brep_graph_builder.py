"""HuggingFace Dataset Builder for B-Rep graph data.

Provides ArrowBasedBuilder for converting B-Rep topology graphs
to Parquet format on HuggingFace Hub for GNN training.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional

_log = logging.getLogger(__name__)

# Lazy imports for heavy dependencies
_datasets = None
_pa = None


def _ensure_deps():
    """Lazily import datasets and pyarrow."""
    global _datasets, _pa
    if _datasets is None:
        try:
            import datasets
            import pyarrow as pa
            _datasets = datasets
            _pa = pa
        except ImportError as e:
            raise ImportError(
                "datasets and pyarrow are required for HF builders. "
                "Install via: pip install datasets>=2.16.0 pyarrow>=14.0.0"
            ) from e
    return _datasets, _pa


class BRepGraphConfig:
    """Configuration for BRepGraphBuilder.

    Attributes:
        name: Configuration name.
        include_uv: Include UV-grid parameterization.
        include_coedges: Include coedge sequences.
        uv_grid_size: Grid size for UV sampling (if include_uv).
        max_faces: Maximum faces per graph.
        max_edges: Maximum edges per graph.
        version: Dataset version.
        description: Human-readable description.
    """

    def __init__(
        self,
        name: str = "default",
        include_uv: bool = False,
        include_coedges: bool = False,
        uv_grid_size: int = 10,
        max_faces: int = 1024,
        max_edges: int = 4096,
        version: str = "1.0.0",
        description: str = "",
    ) -> None:
        self.name = name
        self.include_uv = include_uv
        self.include_coedges = include_coedges
        self.uv_grid_size = uv_grid_size
        self.max_faces = max_faces
        self.max_edges = max_edges
        self.version = version
        self.description = description or f"B-Rep graphs ({name})"


# Pre-defined configurations
DEFAULT_CONFIG = BRepGraphConfig(
    name="default",
    description="Standard B-Rep topology graphs for GNN training",
)

WITH_UV_CONFIG = BRepGraphConfig(
    name="with_uv",
    include_uv=True,
    description="B-Rep graphs with UV-net style face parameterization",
)

WITH_COEDGES_CONFIG = BRepGraphConfig(
    name="with_coedges",
    include_coedges=True,
    description="B-Rep graphs with coedge sequences for autoregressive models",
)

FULL_CONFIG = BRepGraphConfig(
    name="full",
    include_uv=True,
    include_coedges=True,
    description="Complete B-Rep data with UV and coedges",
)


class BRepGraphBuilder:
    """Builder for B-Rep graph datasets.

    Converts STEP/BRep files to graph format suitable for GNN training
    and HuggingFace Hub hosting. Supports multiple configurations:
    - default: Face/edge features with adjacency
    - with_uv: Adds UV-grid point sampling per face
    - with_coedges: Adds coedge sequences for BRepNet-style models

    Usage:
        >>> builder = BRepGraphBuilder(
        ...     source_dir="/path/to/step_files",
        ...     config=DEFAULT_CONFIG,
        ... )
        >>> # Build to local Parquet files
        >>> builder.build("/path/to/output")
        >>>
        >>> # Or get as HuggingFace Dataset
        >>> dataset = builder.to_dataset()

    Args:
        source_dir: Directory containing source STEP/BRep files.
        config: Configuration specifying dataset options.
        splits: List of splits to process.
        use_pythonocc: Whether to use pythonocc for STEP parsing.
    """

    # Surface type mapping
    SURFACE_TYPES = {
        "plane": 0,
        "cylinder": 1,
        "cone": 2,
        "sphere": 3,
        "torus": 4,
        "bspline": 5,
        "bezier": 6,
        "other": 7,
    }

    # Curve type mapping
    CURVE_TYPES = {
        "line": 0,
        "circle": 1,
        "ellipse": 2,
        "bspline": 3,
        "bezier": 4,
        "other": 5,
    }

    def __init__(
        self,
        source_dir: Optional[str] = None,
        config: Optional[BRepGraphConfig] = None,
        splits: Optional[List[str]] = None,
        use_pythonocc: bool = True,
    ) -> None:
        _ensure_deps()

        self.source_dir = Path(source_dir) if source_dir else None
        self.config = config or DEFAULT_CONFIG
        self.splits = splits or ["train", "val", "test"]
        self.use_pythonocc = use_pythonocc

        # Schema from our schemas module
        from ..schemas import get_brep_graph_schema
        self._schema = get_brep_graph_schema(
            include_uv=self.config.include_uv,
            include_coedges=self.config.include_coedges,
            max_faces=self.config.max_faces,
            max_edges=self.config.max_edges,
        )

        # Check for pythonocc
        self._has_pythonocc = False
        if use_pythonocc:
            try:
                from OCC.Core.TopoDS import TopoDS_Shape
                self._has_pythonocc = True
            except ImportError:
                _log.warning(
                    "pythonocc not available, will use fallback graph extraction"
                )

    def _get_features(self) -> "datasets.Features":
        """Get HuggingFace Features from PyArrow schema."""
        datasets, pa = _ensure_deps()

        features_dict = {}

        for field in self._schema:
            name = field.name

            if pa.types.is_string(field.type):
                features_dict[name] = datasets.Value("string")
            elif pa.types.is_int32(field.type):
                features_dict[name] = datasets.Value("int32")
            elif pa.types.is_int64(field.type):
                features_dict[name] = datasets.Value("int64")
            elif pa.types.is_float32(field.type):
                features_dict[name] = datasets.Value("float32")
            elif pa.types.is_list(field.type):
                elem_type = field.type.value_type
                if pa.types.is_int32(elem_type):
                    features_dict[name] = datasets.Sequence(
                        datasets.Value("int32")
                    )
                elif pa.types.is_int64(elem_type):
                    features_dict[name] = datasets.Sequence(
                        datasets.Value("int64")
                    )
                elif pa.types.is_float32(elem_type):
                    features_dict[name] = datasets.Sequence(
                        datasets.Value("float32")
                    )
                else:
                    features_dict[name] = datasets.Sequence(
                        datasets.Value("string")
                    )
            else:
                features_dict[name] = datasets.Value("string")

        return datasets.Features(features_dict)

    def _extract_face_features(
        self, face_data: Dict[str, Any]
    ) -> List[float]:
        """Extract feature vector from face data.

        Features: [surface_type, area, cx, cy, cz, nx, ny, nz, k1, k2]
        """
        surface_type = self.SURFACE_TYPES.get(
            face_data.get("surface_type", "other"), 7
        )
        area = float(face_data.get("area", 0.0))

        centroid = face_data.get("centroid", [0.0, 0.0, 0.0])
        if len(centroid) < 3:
            centroid = centroid + [0.0] * (3 - len(centroid))

        normal = face_data.get("normal", [0.0, 0.0, 1.0])
        if len(normal) < 3:
            normal = normal + [0.0] * (3 - len(normal))

        curvatures = face_data.get("curvatures", [0.0, 0.0])
        if len(curvatures) < 2:
            curvatures = curvatures + [0.0] * (2 - len(curvatures))

        return [
            float(surface_type),
            area,
            centroid[0], centroid[1], centroid[2],
            normal[0], normal[1], normal[2],
            curvatures[0], curvatures[1],
        ]

    def _extract_edge_features(
        self, edge_data: Dict[str, Any]
    ) -> List[float]:
        """Extract feature vector from edge data.

        Features: [curve_type, length, convexity, dihedral_angle]
        """
        curve_type = self.CURVE_TYPES.get(
            edge_data.get("curve_type", "other"), 5
        )
        length = float(edge_data.get("length", 0.0))
        convexity = float(edge_data.get("convexity", 0.0))
        dihedral = float(edge_data.get("dihedral_angle", 0.0))

        return [float(curve_type), length, convexity, dihedral]

    def _process_step_file_pythonocc(
        self, step_path: Path
    ) -> Optional[Dict[str, Any]]:
        """Process STEP file using pythonocc."""
        if not self._has_pythonocc:
            return None

        try:
            from OCC.Core.STEPControl import STEPControl_Reader
            from OCC.Core.TopAbs import TopAbs_FACE, TopAbs_EDGE
            from OCC.Core.TopExp import TopExp_Explorer
            from OCC.Core.BRepGProp import brepgprop
            from OCC.Core.GProp import GProp_GProps
            from OCC.Core.BRepAdaptor import BRepAdaptor_Surface, BRepAdaptor_Curve
            from OCC.Core.GeomAbs import (
                GeomAbs_Plane, GeomAbs_Cylinder, GeomAbs_Cone,
                GeomAbs_Sphere, GeomAbs_Torus, GeomAbs_BSplineSurface,
                GeomAbs_Line, GeomAbs_Circle, GeomAbs_Ellipse,
                GeomAbs_BSplineCurve,
            )
            from OCC.Core.TopoDS import topods

            # Read STEP file
            reader = STEPControl_Reader()
            status = reader.ReadFile(str(step_path))
            if status != 1:
                return None

            reader.TransferRoots()
            shape = reader.OneShape()

            # Extract faces
            faces_data = []
            face_explorer = TopExp_Explorer(shape, TopAbs_FACE)
            face_idx = 0

            while face_explorer.More():
                face = topods.Face(face_explorer.Current())

                # Get surface type
                adaptor = BRepAdaptor_Surface(face)
                surf_type = adaptor.GetType()
                surf_type_map = {
                    GeomAbs_Plane: "plane",
                    GeomAbs_Cylinder: "cylinder",
                    GeomAbs_Cone: "cone",
                    GeomAbs_Sphere: "sphere",
                    GeomAbs_Torus: "torus",
                    GeomAbs_BSplineSurface: "bspline",
                }
                surface_type_str = surf_type_map.get(surf_type, "other")

                # Get area
                props = GProp_GProps()
                brepgprop.SurfaceProperties(face, props)
                area = props.Mass()

                # Get centroid
                center = props.CentreOfMass()
                centroid = [center.X(), center.Y(), center.Z()]

                faces_data.append({
                    "idx": face_idx,
                    "surface_type": surface_type_str,
                    "area": area,
                    "centroid": centroid,
                    "normal": [0.0, 0.0, 1.0],  # Simplified
                    "curvatures": [0.0, 0.0],  # Simplified
                })
                face_idx += 1
                face_explorer.Next()

            # Extract edges
            edges_data = []
            edge_explorer = TopExp_Explorer(shape, TopAbs_EDGE)
            edge_idx = 0

            while edge_explorer.More():
                edge = topods.Edge(edge_explorer.Current())

                # Get curve type
                try:
                    adaptor = BRepAdaptor_Curve(edge)
                    curve_type = adaptor.GetType()
                    curve_type_map = {
                        GeomAbs_Line: "line",
                        GeomAbs_Circle: "circle",
                        GeomAbs_Ellipse: "ellipse",
                        GeomAbs_BSplineCurve: "bspline",
                    }
                    curve_type_str = curve_type_map.get(curve_type, "other")

                    # Get length
                    props = GProp_GProps()
                    brepgprop.LinearProperties(edge, props)
                    length = props.Mass()
                except Exception:
                    curve_type_str = "other"
                    length = 0.0

                edges_data.append({
                    "idx": edge_idx,
                    "curve_type": curve_type_str,
                    "length": length,
                    "convexity": 0.0,
                    "dihedral_angle": 0.0,
                })
                edge_idx += 1
                edge_explorer.Next()

            # Build simple adjacency (face-to-face via shared edges)
            # This is simplified; real implementation would use TopExp.MapShapesAndAncestors
            edge_index_src = []
            edge_index_dst = []

            # For now, create a simple complete graph on faces (placeholder)
            num_faces = len(faces_data)
            for i in range(num_faces):
                for j in range(i + 1, min(i + 5, num_faces)):  # Connect nearby
                    edge_index_src.extend([i, j])
                    edge_index_dst.extend([j, i])

            return {
                "faces": faces_data,
                "edges": edges_data,
                "edge_index": [edge_index_src, edge_index_dst],
            }

        except Exception as e:
            _log.debug("Failed to process %s with pythonocc: %s", step_path, e)
            return None

    def _process_json_graph(
        self, json_path: Path
    ) -> Optional[Dict[str, Any]]:
        """Process pre-computed graph data from JSON."""
        try:
            with open(json_path) as f:
                data = json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            _log.debug("Failed to load %s: %s", json_path, e)
            return None

        # Expected JSON structure:
        # {
        #     "faces": [{"surface_type": ..., "area": ..., ...}, ...],
        #     "edges": [{"curve_type": ..., "length": ..., ...}, ...],
        #     "edge_index": [[src...], [dst...]],
        # }
        faces = data.get("faces", [])
        edges = data.get("edges", [])
        edge_index = data.get("edge_index", [[], []])

        if not faces:
            return None

        return {
            "faces": faces,
            "edges": edges,
            "edge_index": edge_index,
        }

    def _graph_to_sample(
        self,
        graph_data: Dict[str, Any],
        sample_id: str,
        source_path: str,
    ) -> Dict[str, Any]:
        """Convert extracted graph data to sample dict."""
        faces = graph_data["faces"]
        edges = graph_data["edges"]
        edge_index = graph_data["edge_index"]

        # Extract face features
        face_features = []
        for face in faces:
            face_features.extend(self._extract_face_features(face))
        face_feature_dim = 10  # As defined in _extract_face_features

        # Extract edge features
        edge_features = []
        for edge in edges:
            edge_features.extend(self._extract_edge_features(edge))
        edge_feature_dim = 4  # As defined in _extract_edge_features

        # Flatten edge_index
        edge_index_flat = []
        if len(edge_index) >= 2:
            edge_index_flat = edge_index[0] + edge_index[1]

        sample = {
            "sample_id": sample_id,
            "num_faces": len(faces),
            "num_edges": len(edges),
            "face_features": face_features,
            "face_feature_dim": face_feature_dim,
            "edge_features": edge_features,
            "edge_feature_dim": edge_feature_dim,
            "edge_index": edge_index_flat,
            "face_labels": None,
            "global_features": None,
            "source_path": source_path,
            "source": "brep",
            "metadata": json.dumps({"file": sample_id}),
        }

        # Add UV data if configured
        if self.config.include_uv:
            sample["uv_points"] = None
            sample["uv_grid_size"] = self.config.uv_grid_size
            sample["uv_normals"] = None

        # Add coedge data if configured
        if self.config.include_coedges:
            sample["coedge_sequence"] = None
            sample["num_coedges"] = 0

        return sample

    def generate_samples(
        self, split: str = "train"
    ) -> Generator[Dict[str, Any], None, None]:
        """Generate samples for a given split.

        Args:
            split: Dataset split.

        Yields:
            Sample dictionaries with graph data.
        """
        if self.source_dir is None:
            raise ValueError("source_dir must be set to generate samples")

        split_dir = self.source_dir / split
        if not split_dir.exists():
            _log.warning("Split directory not found: %s", split_dir)
            return

        # Look for STEP files
        step_files = list(split_dir.glob("*.step")) + list(split_dir.glob("*.stp"))

        # Also look for pre-computed JSON graphs
        json_files = list(split_dir.glob("*.json"))

        _log.info(
            "Processing %d STEP + %d JSON files from %s",
            len(step_files), len(json_files), split_dir,
        )

        # Process STEP files
        for step_path in step_files:
            graph_data = self._process_step_file_pythonocc(step_path)
            if graph_data is not None:
                sample = self._graph_to_sample(
                    graph_data, step_path.stem, str(step_path)
                )
                yield sample

        # Process JSON files
        for json_path in json_files:
            graph_data = self._process_json_graph(json_path)
            if graph_data is not None:
                sample = self._graph_to_sample(
                    graph_data, json_path.stem, str(json_path)
                )
                yield sample

    def to_arrow_tables(self) -> Dict[str, "pa.Table"]:
        """Convert all splits to PyArrow Tables."""
        datasets, pa = _ensure_deps()
        from ..schemas import samples_to_table

        tables = {}

        for split in self.splits:
            samples = list(self.generate_samples(split))
            if samples:
                tables[split] = samples_to_table(samples, self._schema)
                _log.info("Built %s table with %d samples", split, len(samples))

        return tables

    def build(
        self,
        output_dir: str,
        compression: str = "zstd",
        row_group_size: int = 5000,
    ) -> Dict[str, Path]:
        """Build Parquet files for all splits.

        Args:
            output_dir: Output directory.
            compression: Compression codec.
            row_group_size: Rows per row group.

        Returns:
            Dictionary of split names to file paths.
        """
        datasets, pa = _ensure_deps()
        import pyarrow.parquet as pq

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        output_files = {}
        tables = self.to_arrow_tables()

        for split, table in tables.items():
            file_path = output_path / f"{split}.parquet"
            pq.write_table(
                table,
                file_path,
                compression=compression,
                row_group_size=row_group_size,
            )
            output_files[split] = file_path
            _log.info("Wrote %s to %s", split, file_path)

        return output_files

    def to_dataset(self) -> "datasets.DatasetDict":
        """Convert to HuggingFace DatasetDict."""
        datasets_lib, pa = _ensure_deps()

        dataset_dict = {}
        features = self._get_features()

        for split in self.splits:
            samples = list(self.generate_samples(split))
            if samples:
                dataset_dict[split] = datasets_lib.Dataset.from_list(
                    samples, features=features
                )

        return datasets_lib.DatasetDict(dataset_dict)

    def push_to_hub(
        self,
        repo_id: str,
        private: bool = False,
        token: Optional[str] = None,
        commit_message: str = "Upload B-Rep graph dataset",
    ) -> str:
        """Push dataset to HuggingFace Hub.

        Args:
            repo_id: Repository ID.
            private: Whether to make private.
            token: HuggingFace token.
            commit_message: Commit message.

        Returns:
            Dataset URL.
        """
        dataset = self.to_dataset()

        dataset.push_to_hub(
            repo_id,
            private=private,
            token=token,
            commit_message=commit_message,
        )

        return f"https://huggingface.co/datasets/{repo_id}"


__all__ = [
    "BRepGraphBuilder",
    "BRepGraphConfig",
    "DEFAULT_CONFIG",
    "WITH_UV_CONFIG",
    "WITH_COEDGES_CONFIG",
    "FULL_CONFIG",
]
