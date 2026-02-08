# Complete Placeholder Implementations Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Implement all remaining ~170 placeholder methods across the cadling codebase to achieve production-ready functionality.

**Architecture:** Follow existing patterns in each module. Use three-strategy approach (STEP text → OCC geometry → graph/heuristic fallback) for feature extraction. Return structured error dicts instead of `None`. Add debug logging to all exception handlers.

**Tech Stack:** Python 3.9+, pythonocc-core (optional), trimesh (optional), numpy, torch, torch_geometric

---

## Overview

This plan addresses ~170 remaining placeholder implementations organized by priority and dependency order:

| Phase | Area | Methods | Priority |
|-------|------|---------|----------|
| A | Geometry Extractors | ~25 | HIGH |
| B | Graph Building | ~10 | HIGH |
| C | Interference & Normalization | ~15 | MEDIUM |
| D | Assembly Analysis | ~10 | MEDIUM |
| E | Backend Methods | ~15 | MEDIUM |
| F | Chunkers & Serializers | ~15 | LOW |
| G | Remaining Models | ~30 | LOW |

**Already Fixed (Previous Session):** topology_validation.py, mesh_quality.py, surface_analysis.py, pattern_detection.py, feature_extractor.py, stepnet_integration.py, hybrid_pipeline.py, vision_pipeline.py, vlm_pipeline.py, multi_view_fusion_pipeline.py, threaded_geometry_vlm_pipeline.py, design_intent_inference_model.py, feature_recognition_vlm_model.py, pmi_extraction_model.py, feature_recognition.py, mesh_segmentation.py, mesh_chunker.py, stl_chunker.py

---

## Phase A: Geometry Extractors (HIGH PRIORITY)

**File:** `cadling/models/segmentation/geometry_extractors.py`

These extractors return hardcoded defaults. Need real implementations.

### Task A1: PocketGeometryExtractor._extract_from_step_text

**Files:**
- Modify: `cadling/models/segmentation/geometry_extractors.py:446-500`

**Current:** Returns `None` without parsing STEP text

**Implementation:**
```python
def _extract_from_step_text(
    self, face_entities: List[Dict]
) -> Optional[Dict[str, Any]]:
    """Extract pocket parameters from STEP entity text.

    Looks for:
    - PLANE entities for bottom and side faces
    - B_SPLINE_SURFACE_WITH_KNOTS for complex pockets
    - CARTESIAN_POINT coordinates for dimensions
    """
    if not face_entities:
        return None

    try:
        # Collect all plane faces
        plane_faces = []
        all_points = []

        for entity in face_entities:
            entity_type = entity.get("type", "")
            raw_text = entity.get("raw_text", "")

            if "PLANE" in entity_type:
                plane_faces.append(entity)

            # Extract CARTESIAN_POINT coordinates
            point_matches = re.findall(
                r"CARTESIAN_POINT\s*\([^)]*,\s*\(\s*([-\d.E+]+)\s*,\s*([-\d.E+]+)\s*,\s*([-\d.E+]+)\s*\)",
                raw_text,
                re.IGNORECASE
            )
            for match in point_matches:
                try:
                    all_points.append([float(match[0]), float(match[1]), float(match[2])])
                except ValueError:
                    continue

        if len(all_points) < 4:
            return None

        # Compute bounding box from points
        points_array = np.array(all_points)
        min_coords = points_array.min(axis=0)
        max_coords = points_array.max(axis=0)
        dimensions = max_coords - min_coords

        # Sort dimensions to get width, length, depth
        sorted_dims = sorted(enumerate(dimensions), key=lambda x: x[1], reverse=True)

        # Largest two are width/length, smallest is depth
        width = sorted_dims[0][1]
        length = sorted_dims[1][1]
        depth = sorted_dims[2][1]

        # Compute center location
        center = ((min_coords + max_coords) / 2).tolist()

        # Determine pocket type
        aspect_ratio = min(width, length) / max(width, length) if max(width, length) > 0 else 0
        pocket_type = "circular" if aspect_ratio > 0.9 else "rectangular"

        return {
            "width": float(width),
            "length": float(length),
            "depth": float(depth),
            "location": center,
            "pocket_type": pocket_type,
            "confidence": 0.75,
        }

    except Exception as e:
        _log.debug(f"STEP text pocket extraction failed: {e}")
        return None
```

### Task A2: PocketGeometryExtractor._extract_from_occ_faces

**Files:**
- Modify: `cadling/models/segmentation/geometry_extractors.py:502-560`

**Implementation:**
```python
def _extract_from_occ_faces(
    self, face_ids: List[int], graph: Any
) -> Optional[Dict[str, Any]]:
    """Extract pocket parameters from OCC face geometry."""
    if not self.has_pythonocc:
        return None

    if not hasattr(graph, "faces") or not graph.faces:
        return None

    try:
        from OCC.Core.BRepAdaptor import BRepAdaptor_Surface
        from OCC.Core.BRepGProp import brepgprop
        from OCC.Core.GProp import GProp_GProps
        from OCC.Core.GeomAbs import GeomAbs_Plane
        from OCC.Core.Bnd import Bnd_Box
        from OCC.Core.BRepBndLib import brepbndlib

        # Collect faces
        occ_faces = [graph.faces[i] for i in face_ids if i < len(graph.faces)]

        if not occ_faces:
            return None

        # Compute combined bounding box
        combined_box = Bnd_Box()
        plane_normals = []

        for face in occ_faces:
            # Add to bounding box
            brepbndlib.Add(face, combined_box)

            # Get surface type
            adaptor = BRepAdaptor_Surface(face)
            if adaptor.GetType() == GeomAbs_Plane:
                plane = adaptor.Plane()
                normal = plane.Axis().Direction()
                plane_normals.append([normal.X(), normal.Y(), normal.Z()])

        # Extract dimensions from bounding box
        xmin, ymin, zmin, xmax, ymax, zmax = combined_box.Get()

        width = xmax - xmin
        length = ymax - ymin
        depth = zmax - zmin

        # Sort to identify depth (smallest dimension for typical pocket)
        dims = sorted([width, length, depth])

        center = [(xmin + xmax) / 2, (ymin + ymax) / 2, (zmin + zmax) / 2]

        # Determine pocket type from plane normals
        pocket_type = "rectangular"
        if plane_normals:
            # Check if normals suggest circular pocket
            normals_array = np.array(plane_normals)
            normal_variance = np.var(normals_array, axis=0).sum()
            if normal_variance > 0.5:
                pocket_type = "circular"

        return {
            "width": float(dims[2]),  # Largest
            "length": float(dims[1]),  # Middle
            "depth": float(dims[0]),   # Smallest
            "location": center,
            "pocket_type": pocket_type,
            "confidence": 0.8,
        }

    except Exception as e:
        _log.debug(f"OCC pocket extraction failed: {e}")
        return None
```

### Task A3: PocketGeometryExtractor._extract_from_graph_features

**Files:**
- Modify: `cadling/models/segmentation/geometry_extractors.py:562-610`

**Implementation:**
```python
def _extract_from_graph_features(
    self, face_ids: List[int], graph: Any
) -> Optional[Dict[str, Any]]:
    """Extract pocket parameters from graph node features."""
    if not hasattr(graph, "x") or graph.x is None:
        return None

    try:
        # Get features for pocket faces
        features = graph.x[face_ids].numpy() if hasattr(graph.x, "numpy") else np.array(graph.x[face_ids])

        if len(features) == 0:
            return None

        # Extract area features (typically in feature vector)
        # Assume feature layout: [surface_type_onehot(8), area(1), curvature(2), normal(3), centroid(3), ...] = 24 dims
        areas = features[:, 8] if features.shape[1] > 8 else np.ones(len(features))
        total_area = areas.sum()

        # Estimate dimensions from total area (assuming rectangular pocket)
        # area = 2*(w*l + w*d + l*d) for box, approximate with sqrt
        estimated_dim = np.sqrt(total_area / 6) if total_area > 0 else 10.0

        # Get centroids from features
        if features.shape[1] >= 14:
            centroids = features[:, 11:14]  # centroid x, y, z
            center = centroids.mean(axis=0).tolist()

            # Estimate dimensions from centroid spread
            spreads = centroids.max(axis=0) - centroids.min(axis=0)
            width, length, depth = sorted(spreads, reverse=True)
        else:
            center = [0.0, 0.0, 0.0]
            width = length = depth = estimated_dim

        return {
            "width": float(max(width, 1.0)),
            "length": float(max(length, 1.0)),
            "depth": float(max(depth, 1.0)),
            "location": center,
            "pocket_type": "unknown",
            "confidence": 0.4,
        }

    except Exception as e:
        _log.debug(f"Graph-based pocket extraction failed: {e}")
        return None
```

### Task A4: BossGeometryExtractor (Complete Implementation)

**Files:**
- Modify: `cadling/models/segmentation/geometry_extractors.py:620-750`

**Implementation:** Follow same three-strategy pattern as PocketGeometryExtractor:
- `_extract_from_step_text`: Parse CYLINDRICAL_SURFACE for circular boss, PLANE for rectangular
- `_extract_from_occ_faces`: Use BRepAdaptor to get cylinder radius, compute height from bbox
- `_extract_from_graph_features`: Estimate from node feature areas and centroids

### Task A5: FilletGeometryExtractor (Complete Implementation)

**Files:**
- Modify: `cadling/models/segmentation/geometry_extractors.py:760-880`

**Implementation:**
- `_extract_from_step_text`: Parse TOROIDAL_SURFACE minor_radius parameter
- `_extract_from_occ_faces`: Use BRepAdaptor_Surface for GeomAbs_Torus, extract minor radius
- `_extract_from_graph_features`: Estimate radius from curvature features

### Task A6: ChamferGeometryExtractor (Complete Implementation)

**Files:**
- Modify: `cadling/models/segmentation/geometry_extractors.py:890-1000`

**Implementation:**
- `_extract_from_step_text`: Parse CONICAL_SURFACE semi_angle, compute distance
- `_extract_from_occ_faces`: Use BRepAdaptor for GeomAbs_Cone
- `_extract_from_graph_features`: Estimate from edge angle features

---

## Phase B: Graph Building (HIGH PRIORITY)

**File:** `cadling/models/segmentation/brep_graph_builder.py`

### Task B1: _extract_face_entities

**Files:**
- Modify: `cadling/models/segmentation/brep_graph_builder.py:110-170`

**Current:** Returns empty list "for now"

**Implementation:**
```python
def _extract_face_entities(self, doc: "CADlingDocument") -> List[Dict]:
    """Extract ADVANCED_FACE and FACE_SURFACE entities from document."""
    face_entities = []

    # Get all items from document
    for item in doc.items:
        entity_type = getattr(item, "entity_type", "") or item.properties.get("entity_type", "")

        # Check for face entity types
        if any(ft in entity_type.upper() for ft in ["ADVANCED_FACE", "FACE_SURFACE", "FACE_OUTER_BOUND"]):
            face_entity = {
                "entity_id": getattr(item, "entity_id", None) or item.properties.get("entity_id"),
                "type": entity_type,
                "raw_text": getattr(item, "raw_text", "") or item.properties.get("raw_text", ""),
                "params": getattr(item, "params", []) or item.properties.get("params", []),
                "item": item,
            }

            # Extract geometric properties if available
            if "geometry_analysis" in item.properties:
                face_entity["geometry"] = item.properties["geometry_analysis"]
            if "surface_analysis" in item.properties:
                face_entity["surface"] = item.properties["surface_analysis"]

            face_entities.append(face_entity)

    _log.debug(f"Extracted {len(face_entities)} face entities from document")
    return face_entities
```

### Task B2: _build_face_adjacency

**Files:**
- Modify: `cadling/models/segmentation/brep_graph_builder.py:175-250`

**Current:** Uses "simplified approach"

**Implementation:**
```python
def _build_face_adjacency(
    self, doc: "CADlingDocument", face_entities: List[Dict]
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Build face-to-face adjacency from shared edges/vertices."""
    num_faces = len(face_entities)

    if num_faces == 0:
        return np.zeros((2, 0), dtype=np.int64), None

    # Build entity ID to index mapping
    id_to_idx = {}
    for idx, entity in enumerate(face_entities):
        entity_id = entity.get("entity_id")
        if entity_id is not None:
            id_to_idx[entity_id] = idx

    # Extract edge references from each face
    face_edges = []
    for entity in face_entities:
        edges = self._extract_edge_references(entity)
        face_edges.append(set(edges))

    # Build adjacency from shared edges
    edges_src = []
    edges_dst = []
    edge_features = []

    for i in range(num_faces):
        for j in range(i + 1, num_faces):
            shared = face_edges[i] & face_edges[j]
            if shared:
                # Faces share at least one edge - they are adjacent
                edges_src.extend([i, j])
                edges_dst.extend([j, i])

                # Compute edge features
                feat = self._compute_edge_features(
                    face_entities[i], face_entities[j], shared
                )
                edge_features.extend([feat, feat])  # Undirected

    edge_index = np.array([edges_src, edges_dst], dtype=np.int64)
    edge_attr = np.array(edge_features) if edge_features else None

    _log.debug(f"Built adjacency graph: {num_faces} nodes, {len(edges_src)} edges")
    return edge_index, edge_attr

def _extract_edge_references(self, face_entity: Dict) -> List[int]:
    """Extract edge entity references from face."""
    edges = []
    raw_text = face_entity.get("raw_text", "")

    # Look for EDGE_LOOP, EDGE_CURVE references
    edge_matches = re.findall(r"#(\d+)", raw_text)
    for match in edge_matches:
        edges.append(int(match))

    return edges

def _compute_edge_features(
    self, face1: Dict, face2: Dict, shared_edges: set
) -> np.ndarray:
    """Compute features for edge between two faces."""
    # Feature vector: [edge_type(2), dihedral_angle(1), edge_length(1), convexity(1), ...]
    features = np.zeros(8)

    # Get surface normals if available
    n1 = face1.get("geometry", {}).get("normal", [0, 0, 1])
    n2 = face2.get("geometry", {}).get("normal", [0, 0, 1])

    # Compute dihedral angle
    n1 = np.array(n1)
    n2 = np.array(n2)
    cos_angle = np.clip(np.dot(n1, n2), -1, 1)
    dihedral_angle = np.arccos(cos_angle)

    features[0] = 1.0  # Edge exists
    features[1] = dihedral_angle
    features[2] = 1.0 if dihedral_angle < np.pi / 2 else 0.0  # Convex
    features[3] = len(shared_edges)  # Number of shared edges

    return features
```

### Task B3: _extract_face_features

**Files:**
- Modify: `cadling/models/segmentation/brep_graph_builder.py:260-350`

**Implementation:**
```python
def _extract_face_features(
    self, doc: "CADlingDocument", face_entities: List[Dict]
) -> np.ndarray:
    """Extract geometric features for each face node."""
    # Feature vector per face: 24 dimensions
    # [surface_type_onehot(8), area(1), gaussian_curv(1), mean_curv(1),
    #  normal(3), centroid(3), bbox_size(3), aspect_ratio(1), ...]

    num_faces = len(face_entities)
    features = np.zeros((num_faces, 24))

    surface_types = ["PLANE", "CYLINDER", "CONE", "SPHERE", "TORUS", "BSPLINE", "BEZIER", "OTHER"]

    for idx, entity in enumerate(face_entities):
        entity_type = entity.get("type", "").upper()

        # One-hot encode surface type
        for i, stype in enumerate(surface_types):
            if stype in entity_type:
                features[idx, i] = 1.0
                break
        else:
            features[idx, 7] = 1.0  # OTHER

        # Get geometry analysis if available
        geometry = entity.get("geometry", {})
        surface = entity.get("surface", {})

        # Area (feature 8)
        features[idx, 8] = geometry.get("area", 1.0)

        # Curvatures (features 9-10)
        features[idx, 9] = surface.get("gaussian_curvature", 0.0)
        features[idx, 10] = surface.get("mean_curvature", 0.0)

        # Normal (features 11-13)
        normal = geometry.get("normal", [0, 0, 1])
        features[idx, 11:14] = normal[:3] if len(normal) >= 3 else [0, 0, 1]

        # Centroid (features 14-16)
        centroid = geometry.get("centroid", [0, 0, 0])
        features[idx, 14:17] = centroid[:3] if len(centroid) >= 3 else [0, 0, 0]

        # Bounding box size (features 17-19)
        bbox = geometry.get("bounding_box", {})
        features[idx, 17] = bbox.get("size_x", 1.0)
        features[idx, 18] = bbox.get("size_y", 1.0)
        features[idx, 19] = bbox.get("size_z", 1.0)

        # Aspect ratios and other features
        sizes = features[idx, 17:20]
        if sizes.max() > 0:
            features[idx, 20] = sizes.min() / sizes.max()  # Aspect ratio

    return features
```

---

## Phase C: Interference & Normalization (MEDIUM PRIORITY)

### Task C1: InterferenceCheckModel._check_pair_interference

**File:** `cadling/models/interference_check.py:340-400`

**Implementation:** Use OCC BRepAlgoAPI_Common for boolean intersection, compute interference volume

### Task C2: InterferenceCheckModel._compute_clearance

**File:** `cadling/models/interference_check.py:410-470`

**Implementation:** Use OCC BRepExtrema_DistShapeShape for minimum distance computation

### Task C3: GeometryNormalizationModel._normalize_occ_shape

**File:** `cadling/models/geometry_normalization.py:150-200`

**Implementation:** Use GProp_GProps for centroid, apply gp_Trsf transformations

### Task C4: GeometryNormalizationModel._normalize_trimesh

**File:** `cadling/models/geometry_normalization.py:210-260`

**Implementation:** Use trimesh centroid and apply_transform methods

### Task C5: GeometryNormalizationModel._compute_pca_rotation

**File:** `cadling/models/geometry_normalization.py:270-320`

**Implementation:** Use numpy SVD on vertex coordinates for principal axis alignment

---

## Phase D: Assembly Analysis (MEDIUM PRIORITY)

### Task D1: AssemblyHierarchyPipeline._detect_concentric_mate

**File:** `cadling/experimental/pipeline/assembly_hierarchy_pipeline.py:400-450`

**Implementation:**
```python
def _detect_concentric_mate(
    self, shape1, shape2
) -> Optional[Dict[str, Any]]:
    """Detect concentric mate between cylindrical surfaces."""
    if not self.has_pythonocc:
        return None

    try:
        from OCC.Core.TopExp import TopExp_Explorer
        from OCC.Core.TopAbs import TopAbs_FACE
        from OCC.Core.BRepAdaptor import BRepAdaptor_Surface
        from OCC.Core.GeomAbs import GeomAbs_Cylinder
        from OCC.Core.gp import gp_Lin

        # Find cylindrical faces in both shapes
        cylinders1 = self._extract_cylindrical_faces(shape1)
        cylinders2 = self._extract_cylindrical_faces(shape2)

        if not cylinders1 or not cylinders2:
            return None

        # Check for coaxial cylinders
        for cyl1 in cylinders1:
            axis1 = cyl1["axis"]
            radius1 = cyl1["radius"]

            for cyl2 in cylinders2:
                axis2 = cyl2["axis"]
                radius2 = cyl2["radius"]

                # Check if axes are parallel (dot product ~1)
                dot = abs(np.dot(axis1, axis2))
                if dot > 0.99:
                    # Check if axes are colinear (same line)
                    # ... distance between axis lines < tolerance

                    return {
                        "mate_type": "concentric",
                        "axis": axis1.tolist(),
                        "radius1": radius1,
                        "radius2": radius2,
                        "confidence": 0.85,
                    }

        return None

    except Exception as e:
        _log.debug(f"Concentric mate detection failed: {e}")
        return None
```

### Task D2: AssemblyHierarchyPipeline._detect_planar_contact

**File:** `cadling/experimental/pipeline/assembly_hierarchy_pipeline.py:460-520`

**Implementation:** Find opposing planar faces within tolerance, check for overlap

### Task D3: AssemblyHierarchyPipeline._build_hierarchical_tree

**File:** `cadling/experimental/pipeline/assembly_hierarchy_pipeline.py:530-600`

**Current:** Creates flat structure "for now"

**Implementation:** Parse NEXT_ASSEMBLY_USAGE_OCCURRENCE entities to build parent-child relationships

---

## Phase E: Backend Methods (MEDIUM PRIORITY)

### Task E1: TopologyBuilder (Return Proper Structures)

**File:** `cadling/backend/step/topology_builder.py:300-350`

**Fix:** Replace `return None` with proper error dict or empty topology structure

### Task E2: IGESBackend Error Handling

**File:** `cadling/backend/iges_backend.py:290-330`

**Fix:** Return structured error results instead of `None`

### Task E3: BRepBackend Exception Handler

**File:** `cadling/backend/brep/brep_backend.py:560-580`

**Fix:** Add logging and return error dict instead of `pass`

### Task E4: STEPBackend Methods

**File:** `cadling/backend/step/step_backend.py:340-390`

**Fix:** Implement proper fallbacks or error handling

---

## Phase F: Chunkers & Serializers (LOW PRIORITY)

### Task F1: BaseChunker Abstract Methods

**File:** `cadling/chunker/base_chunker.py:145-180`

**Fix:** These are ABCs - ensure all subclasses implement properly

### Task F2: HybridChunker._compute_chunk_embedding

**File:** `cadling/chunker/hybrid_chunker.py:230-260`

**Fix:** Already returns `None` appropriately, but add fallback embedding computation

### Task F3: Tokenizer Abstract Methods

**File:** `cadling/chunker/tokenizer/tokenizer.py:35-80`

**Fix:** These are ABCs - ensure all subclasses implement

### Task F4: Serializer Abstract Methods

**File:** `cadling/chunker/serializer/serializer.py:38-55`

**Fix:** These are ABCs - ensure all subclasses implement

---

## Phase G: Remaining Models (LOW PRIORITY)

### Task G1: ManufacturabilityAssessmentModel

**File:** `cadling/experimental/models/manufacturability_assessment_model.py:130-150, 600-620`

**Fix:** Add debug logging to `pass` statements in exception handlers

### Task G2: GeometricConstraintModel

**File:** `cadling/experimental/models/geometric_constraint_model.py:400-420`

**Fix:** Implement proper constraint detection instead of "simplified heuristic"

### Task G3: DataLoaders Skipped Samples

**File:** `cadling/models/segmentation/training/data_loaders.py:190-290`

**Note:** These `return None` are INTENTIONAL to avoid fake data - add better logging only

---

## Verification Commands

After each phase, run:

```bash
cd /Users/ryanoboyle/LatticeLabs_toolkit/cadling

# Syntax check
python -c "import ast; ast.parse(open('cadling/models/segmentation/geometry_extractors.py').read())"

# Import check (with conda env)
source ~/.zshrc && eval "$(conda shell.zsh hook)" && conda activate cadling
python -c "from cadling.models.segmentation.geometry_extractors import HoleGeometryExtractor; print('OK')"

# Run tests if available
pytest tests/ -v -x --tb=short
```

---

## Estimated Effort

| Phase | Methods | Est. Time |
|-------|---------|-----------|
| A | 25 | 4-5 hours |
| B | 10 | 2-3 hours |
| C | 10 | 2 hours |
| D | 8 | 2 hours |
| E | 15 | 1.5 hours |
| F | 15 | 1 hour |
| G | 10 | 1 hour |
| **Total** | **~95** | **~14-16 hours** |

Note: Some methods in RequiredToBeCorrected.md are:
- Abstract base class methods (intentional `pass`)
- Already fixed in previous session
- Intentional graceful degradation (skip when deps unavailable)

This plan covers the ~95 methods that need actual implementation.
