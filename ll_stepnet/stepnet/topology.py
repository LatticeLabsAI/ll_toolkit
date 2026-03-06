"""
STEP Topology Builder Module
Constructs topological graph from STEP entity references.
"""

from __future__ import annotations

import hashlib
import logging

import torch
from typing import List, Dict, Tuple, Set, Optional
from collections import defaultdict

_log = logging.getLogger(__name__)


class STEPTopologyBuilder:
    """
    Builds topological graphs from STEP entity relationships.
    Separate from tokenization and feature extraction.
    """

    def __init__(self):
        """Initialize topology builder."""
        pass

    def build_reference_graph(self, features_list: List[Dict]) -> Dict:
        """
        Build entity reference graph from extracted features.

        Args:
            features_list: List of feature dicts from STEPFeatureExtractor

        Returns:
            Dictionary with:
                - adjacency_dict: Dict[int, List[int]] - entity_id → referenced_ids
                - edge_list: List[Tuple[int, int]] - list of (from, to) edges
                - num_nodes: int
        """
        adjacency_dict = defaultdict(list)
        all_entity_ids = set()

        for features in features_list:
            entity_id = features['entity_id']
            references = features.get('references', [])

            if entity_id is not None:
                all_entity_ids.add(entity_id)
                adjacency_dict[entity_id] = references

                # Track referenced entities too
                for ref_id in references:
                    all_entity_ids.add(ref_id)

        # Create edge list
        edge_list = []
        for from_id, to_ids in adjacency_dict.items():
            for to_id in to_ids:
                edge_list.append((from_id, to_id))

        # Create mapping from entity_id to index
        node_ids = sorted(all_entity_ids)
        id_to_idx = {entity_id: idx for idx, entity_id in enumerate(node_ids)}

        return {
            'adjacency_dict': dict(adjacency_dict),
            'edge_list': edge_list,
            'num_nodes': len(all_entity_ids),
            'node_ids': node_ids,
            'id_to_idx': id_to_idx
        }

    def build_adjacency_matrix(self, reference_graph: Dict) -> torch.Tensor:
        """
        Convert reference graph to sparse adjacency matrix.

        Returns a sparse COO tensor to avoid O(N^2) memory on large
        B-Rep graphs.

        Args:
            reference_graph: Output from build_reference_graph

        Returns:
            Sparse COO adjacency matrix [N, N] where N = num_nodes
        """
        node_ids = reference_graph['node_ids']
        num_nodes = len(node_ids)

        # Create ID to index mapping
        id_to_idx = {node_id: idx for idx, node_id in enumerate(node_ids)}

        # Collect edge indices
        rows: List[int] = []
        cols: List[int] = []
        for from_id, to_id in reference_graph['edge_list']:
            if from_id in id_to_idx and to_id in id_to_idx:
                rows.append(id_to_idx[from_id])
                cols.append(id_to_idx[to_id])

        if rows:
            indices = torch.tensor([rows, cols], dtype=torch.long)
            values = torch.ones(len(rows), dtype=torch.float32)
            return torch.sparse_coo_tensor(
                indices, values, (num_nodes, num_nodes),
            ).coalesce()
        else:
            return torch.sparse_coo_tensor(
                torch.zeros(2, 0, dtype=torch.long),
                torch.zeros(0, dtype=torch.float32),
                (num_nodes, num_nodes),
            )

    def build_edge_index(self, reference_graph: Dict) -> torch.Tensor:
        """
        Build edge index in PyTorch Geometric format.

        Args:
            reference_graph: Output from build_reference_graph

        Returns:
            Edge index tensor [2, num_edges] for PyG
        """
        node_ids = reference_graph['node_ids']
        id_to_idx = {node_id: idx for idx, node_id in enumerate(node_ids)}

        edge_index = []

        for from_id, to_id in reference_graph['edge_list']:
            if from_id in id_to_idx and to_id in id_to_idx:
                from_idx = id_to_idx[from_id]
                to_idx = id_to_idx[to_id]
                edge_index.append([from_idx, to_idx])

        if edge_index:
            return torch.tensor(edge_index, dtype=torch.long).t()
        else:
            return torch.zeros(2, 0, dtype=torch.long)

    def compute_node_degrees(self, reference_graph: Dict) -> Dict[int, Dict[str, int]]:
        """
        Compute in-degree and out-degree for each node.

        Args:
            reference_graph: Output from build_reference_graph

        Returns:
            Dict mapping node_id → {'in_degree': int, 'out_degree': int}
        """
        degrees = defaultdict(lambda: {'in_degree': 0, 'out_degree': 0})

        for from_id, to_id in reference_graph['edge_list']:
            degrees[from_id]['out_degree'] += 1
            degrees[to_id]['in_degree'] += 1

        return dict(degrees)

    def identify_topology_types(self, features_list: List[Dict]) -> Dict[str, List[int]]:
        """
        Categorize entities by topological role.

        Args:
            features_list: List of feature dicts

        Returns:
            Dict mapping category → list of entity IDs
        """
        categories = {
            'vertices': [],
            'edges': [],
            'faces': [],
            'shells': [],
            'solids': [],
            'geometry': [],
            'other': []
        }

        for features in features_list:
            entity_id = features['entity_id']
            entity_type = features.get('entity_type', '')

            if entity_type in ['VERTEX_POINT']:
                categories['vertices'].append(entity_id)
            elif entity_type in ['EDGE_CURVE', 'ORIENTED_EDGE', 'EDGE_LOOP']:
                categories['edges'].append(entity_id)
            elif entity_type in ['ADVANCED_FACE', 'FACE_BOUND', 'FACE_OUTER_BOUND']:
                categories['faces'].append(entity_id)
            elif entity_type in ['CLOSED_SHELL', 'OPEN_SHELL']:
                categories['shells'].append(entity_id)
            elif entity_type in ['MANIFOLD_SOLID_BREP']:
                categories['solids'].append(entity_id)
            elif entity_type in ['CYLINDRICAL_SURFACE', 'PLANE', 'CONICAL_SURFACE',
                               'CIRCLE', 'LINE', 'B_SPLINE_CURVE']:
                categories['geometry'].append(entity_id)
            else:
                categories['other'].append(entity_id)

        return categories

    def build_node_features(self, features_list: List[Dict], reference_graph: Dict) -> torch.Tensor:
        """
        Build node feature matrix from extracted features.

        Args:
            features_list: List of feature dicts from STEPFeatureExtractor
            reference_graph: Output from build_reference_graph

        Returns:
            Node features tensor [num_nodes, feature_dim]
        """
        # Create mapping from entity_id to index
        id_to_idx = reference_graph['id_to_idx']
        num_nodes = reference_graph['num_nodes']

        # Feature dimension: 128 (numeric params) + 1 (entity type hash)
        feature_dim = 129
        node_features = torch.zeros(num_nodes, feature_dim, dtype=torch.float32)

        for features in features_list:
            entity_id = features['entity_id']
            if entity_id not in id_to_idx:
                continue

            idx = id_to_idx[entity_id]

            # Numeric parameters (pad/truncate to 128 dims)
            numeric_params = features.get('numeric_params', [])
            if len(numeric_params) > 0:
                # Take first 128 params
                params_to_use = numeric_params[:128]
                node_features[idx, :len(params_to_use)] = torch.tensor(params_to_use, dtype=torch.float32)

            # Entity type as hashed feature (last dimension)
            entity_type = features.get('entity_type', '')
            if entity_type:
                # Deterministic hash to [0, 1] range (hashlib is process-independent)
                type_hash = (int(hashlib.sha256(entity_type.encode()).hexdigest(), 16) % 10000) / 10000.0
                node_features[idx, -1] = type_hash

        return node_features

    def build_complete_topology(self, features_list: List[Dict], compact: bool = True) -> Dict:
        """
        Build complete topology representation.

        Args:
            features_list: List of feature dicts from STEPFeatureExtractor.
            compact: If True (default), use ``build_compact_node_features()``
                to produce 48-dim features in cadling's native layout.  This
                matches the default ``input_dim=48`` of
                :class:`STEPGraphEncoder`.  Pass ``compact=False`` to use the
                legacy ``build_node_features()`` (129-dim: 128 numeric + 1
                type hash).

        Returns:
            Complete topology dictionary with:
                - reference_graph
                - adjacency_matrix
                - edge_index
                - node_degrees
                - topology_types
                - node_features
                - num_nodes
                - num_edges
        """
        reference_graph = self.build_reference_graph(features_list)
        adjacency_matrix = self.build_adjacency_matrix(reference_graph)
        edge_index = self.build_edge_index(reference_graph)
        node_degrees = self.compute_node_degrees(reference_graph)
        topology_types = self.identify_topology_types(features_list)

        if compact:
            node_features = self.build_compact_node_features(features_list, reference_graph)
        else:
            node_features = self.build_node_features(features_list, reference_graph)

        return {
            'reference_graph': reference_graph,
            'adjacency_matrix': adjacency_matrix,
            'edge_index': edge_index,
            'node_degrees': node_degrees,
            'topology_types': topology_types,
            'node_features': node_features,
            'num_nodes': reference_graph['num_nodes'],
            'num_edges': len(reference_graph['edge_list'])
        }

    def build_coedge_structure(self, features_list: List[Dict]) -> Dict:
        """Build coedge adjacency structure from STEP topology.

        In B-Rep topology, each topological edge is shared by (at most) two
        adjacent faces. Each such sharing creates two oriented coedges -- one
        per face. This method reconstructs the coedge-level graph with
        next/prev/mate pointers from the STEP entity hierarchy.

        The coedge structure is the primary input format for the BRepNet
        architecture (see cadling.models.segmentation.architectures.brep_net).

        Args:
            features_list: List of feature dicts from STEPFeatureExtractor.
                Each dict should have 'entity_id', 'entity_type', 'references',
                and optionally 'numeric_params'.

        Returns:
            Dictionary with:
                - coedge_features: torch.Tensor [num_coedges, feature_dim]
                - next_indices: torch.Tensor [num_coedges]
                - prev_indices: torch.Tensor [num_coedges]
                - mate_indices: torch.Tensor [num_coedges]
                - face_indices: torch.Tensor [num_coedges]
                - num_coedges: int
                - num_faces: int
                - face_entity_ids: List[int] - entity IDs of faces
                - edge_entity_ids: List[int] - entity IDs of edges
        """
        # Categorize entities by topological role
        topology_types = self.identify_topology_types(features_list)

        face_ids = topology_types.get('faces', [])
        edge_ids = topology_types.get('edges', [])

        # Build entity ID -> features lookup
        id_to_features: Dict[int, Dict] = {}
        for features in features_list:
            eid = features.get('entity_id')
            if eid is not None:
                id_to_features[eid] = features

        # Precompute set of EDGE_CURVE IDs for O(1) membership tests
        edge_curve_ids: set = {
            eid for eid, feat in id_to_features.items()
            if feat.get('entity_type', '') == 'EDGE_CURVE'
        }
        edge_ids_set: set = set(edge_ids)

        # Build face -> ordered edge references
        # Each face entity (ADVANCED_FACE) references bounds, which reference edge loops,
        # which reference oriented edges, which reference edge curves.
        face_to_edges: Dict[int, List[int]] = {}
        face_id_to_idx: Dict[int, int] = {}

        for face_idx, face_id in enumerate(face_ids):
            face_id_to_idx[face_id] = face_idx
            face_features = id_to_features.get(face_id, {})
            refs = face_features.get('references', [])

            # Collect all edge-related entity IDs reachable from this face.
            # Walk the reference chain: ADVANCED_FACE -> FACE_BOUND ->
            # EDGE_LOOP -> ORIENTED_EDGE -> EDGE_CURVE
            ordered_edges: List[int] = []

            for ref_id in refs:
                ref_features = id_to_features.get(ref_id, {})
                ref_type = ref_features.get('entity_type', '')

                if ref_type in ('FACE_BOUND', 'FACE_OUTER_BOUND'):
                    # FACE_BOUND references an EDGE_LOOP
                    bound_refs = ref_features.get('references', [])
                    for loop_id in bound_refs:
                        loop_features = id_to_features.get(loop_id, {})
                        loop_type = loop_features.get('entity_type', '')

                        if loop_type == 'EDGE_LOOP':
                            # EDGE_LOOP references ORIENTED_EDGEs
                            loop_refs = loop_features.get('references', [])
                            for oe_id in loop_refs:
                                oe_features = id_to_features.get(oe_id, {})
                                oe_type = oe_features.get('entity_type', '')

                                if oe_type == 'ORIENTED_EDGE':
                                    # ORIENTED_EDGE references EDGE_CURVE
                                    oe_refs = oe_features.get('references', [])
                                    for ec_id in oe_refs:
                                        ec_features = id_to_features.get(ec_id, {})
                                        ec_type = ec_features.get('entity_type', '')
                                        if ec_type == 'EDGE_CURVE':
                                            ordered_edges.append(ec_id)
                                            break
                                    else:
                                        # Use the oriented edge itself
                                        ordered_edges.append(oe_id)
                                elif oe_type == 'EDGE_CURVE':
                                    ordered_edges.append(oe_id)
                        elif loop_id in edge_ids_set:
                            ordered_edges.append(loop_id)

                elif ref_type in ('EDGE_LOOP', 'ORIENTED_EDGE', 'EDGE_CURVE'):
                    ordered_edges.append(ref_id)

            # If no edges found via hierarchy, try direct edge references
            if not ordered_edges:
                for ref_id in refs:
                    if ref_id in edge_ids_set or ref_id in edge_curve_ids:
                        ordered_edges.append(ref_id)

            face_to_edges[face_id] = ordered_edges

        # Build edge -> list of face IDs (for mate detection)
        edge_to_faces: Dict[int, List[int]] = defaultdict(list)
        for face_id, edge_list in face_to_edges.items():
            for eid in edge_list:
                edge_to_faces[eid].append(face_id)

        # Create coedges: each (face_id, edge_id, pos) occurrence
        coedge_list: List[Dict] = []
        # Position-based index for next/prev navigation (always unique)
        coedge_pos_to_idx: Dict[Tuple[int, int], int] = {}
        # Edge-based index for mate detection across faces
        coedge_edge_to_idx: Dict[Tuple[int, int], List[int]] = defaultdict(list)

        for face_id, edge_list in face_to_edges.items():
            for pos, edge_id in enumerate(edge_list):
                ci = len(coedge_list)
                coedge_pos_to_idx[(face_id, pos)] = ci
                coedge_edge_to_idx[(face_id, edge_id)].append(ci)
                coedge_list.append({
                    'face_id': face_id,
                    'edge_id': edge_id,
                    'pos': pos,
                    'loop_size': len(edge_list),
                })

        num_coedges = len(coedge_list)

        if num_coedges == 0:
            _log.warning("No coedges found in topology")
            return {
                'coedge_features': torch.zeros((0, 12), dtype=torch.float32),
                'next_indices': torch.zeros(0, dtype=torch.long),
                'prev_indices': torch.zeros(0, dtype=torch.long),
                'mate_indices': torch.zeros(0, dtype=torch.long),
                'face_indices': torch.zeros(0, dtype=torch.long),
                'num_coedges': 0,
                'num_faces': len(face_ids),
                'face_entity_ids': face_ids,
                'edge_entity_ids': edge_ids,
            }

        # Build next / prev / mate index tensors
        next_indices = torch.zeros(num_coedges, dtype=torch.long)
        prev_indices = torch.zeros(num_coedges, dtype=torch.long)
        mate_indices = torch.zeros(num_coedges, dtype=torch.long)
        face_indices_t = torch.zeros(num_coedges, dtype=torch.long)

        for ci, coedge in enumerate(coedge_list):
            face_id = coedge['face_id']
            edge_id = coedge['edge_id']
            pos = coedge['pos']
            loop_size = coedge['loop_size']
            edges_in_face = face_to_edges[face_id]

            face_indices_t[ci] = face_id_to_idx.get(face_id, 0)

            # Next in loop (cyclic) — use position-based key
            next_pos = (pos + 1) % loop_size
            next_key = (face_id, next_pos)
            next_indices[ci] = coedge_pos_to_idx.get(next_key, ci)

            # Prev in loop (cyclic) — use position-based key
            prev_pos = (pos - 1) % loop_size
            prev_key = (face_id, prev_pos)
            prev_indices[ci] = coedge_pos_to_idx.get(prev_key, ci)

            # Mate: same edge, different face
            mate_idx = ci  # default self (boundary)
            for other_face_id in edge_to_faces.get(edge_id, []):
                if other_face_id != face_id:
                    other_coedges = coedge_edge_to_idx.get((other_face_id, edge_id), [])
                    if other_coedges:
                        mate_idx = other_coedges[0]
                        break
            mate_indices[ci] = mate_idx

        # Build coedge features (12 dims)
        # [curve_type_onehot(6), length(1), tangent(3), curvature(1), convexity(1)]
        curve_type_names = ['LINE', 'CIRCLE', 'ELLIPSE', 'B_SPLINE', 'PARABOLA', 'OTHER']

        coedge_features = torch.zeros(num_coedges, 12, dtype=torch.float32)

        for ci, coedge in enumerate(coedge_list):
            edge_id = coedge['edge_id']
            edge_features = id_to_features.get(edge_id, {})

            # Determine curve type
            edge_type = edge_features.get('entity_type', 'OTHER').upper()
            curve_idx = 5  # OTHER
            for ct_idx, ct_name in enumerate(curve_type_names):
                if ct_name in edge_type:
                    curve_idx = ct_idx
                    break

            # One-hot curve type (dims 0-5)
            coedge_features[ci, curve_idx] = 1.0

            # Edge length from numeric params (dim 6)
            numeric_params = edge_features.get('numeric_params', [])
            if numeric_params:
                # Use first numeric param as length estimate
                coedge_features[ci, 6] = abs(numeric_params[0]) if numeric_params[0] != 0 else 1.0
            else:
                coedge_features[ci, 6] = 1.0

            # Tangent vector (dims 7-9) - extract from numeric params if available
            if len(numeric_params) >= 4:
                coedge_features[ci, 7] = numeric_params[1]
                coedge_features[ci, 8] = numeric_params[2]
                coedge_features[ci, 9] = numeric_params[3]
            else:
                coedge_features[ci, 7:10] = torch.tensor([0.0, 0.0, 1.0])

            # Curvature (dim 10)
            if curve_idx == 0:  # LINE
                coedge_features[ci, 10] = 0.0
            elif curve_idx == 1 and len(numeric_params) > 0:  # CIRCLE
                radius = abs(numeric_params[0])
                coedge_features[ci, 10] = 1.0 / radius if radius > 1e-10 else 0.0
            else:
                coedge_features[ci, 10] = 0.0

            # Convexity (dim 11): 1=convex, 0=concave, 0.5=unknown
            coedge_features[ci, 11] = 0.5

        _log.info(
            f"Built coedge structure: {num_coedges} coedges, "
            f"{len(face_ids)} faces, {len(edge_ids)} edges"
        )

        return {
            'coedge_features': coedge_features,
            'next_indices': next_indices,
            'prev_indices': prev_indices,
            'mate_indices': mate_indices,
            'face_indices': face_indices_t,
            'num_coedges': num_coedges,
            'num_faces': len(face_ids),
            'face_entity_ids': face_ids,
            'edge_entity_ids': edge_ids,
        }

    def build_compact_node_features(
        self,
        features_list: List[Dict],
        reference_graph: Optional[Dict] = None,
        feature_dim: int = 48,
    ) -> torch.Tensor:
        """Build node features in cadling-compatible compact format.

        Produces node features in the same 48-dim layout used by
        cadling's TopologyGraph, so both ll_stepnet and cadling feed
        the same native representation into STEPGraphEncoder (default
        input_dim=48) and geotoken's GraphTokenizer.

        Feature layout (48 dims):
            [0:32]  — first 32 numeric parameters (zero-padded)
            [32:48] — entity type one-hot (16 common B-Rep types)

        Args:
            features_list: Feature dicts from STEPFeatureExtractor.
            reference_graph: Optional reference graph from build_reference_graph().
                If provided, uses its num_nodes and id_to_idx to ensure
                node_features shape matches adjacency_matrix. If None,
                uses len(features_list) as num_nodes (legacy behavior).
            feature_dim: Output feature dimension (default 48).

        Returns:
            torch.Tensor of shape [num_nodes, feature_dim].
        """
        import numpy as np

        # 16 common B-Rep entity types for one-hot encoding
        _BREP_TYPES = [
            'ADVANCED_FACE', 'FACE_BOUND', 'FACE_OUTER_BOUND',
            'EDGE_LOOP', 'ORIENTED_EDGE', 'EDGE_CURVE',
            'VERTEX_POINT', 'CARTESIAN_POINT', 'DIRECTION',
            'LINE', 'CIRCLE', 'B_SPLINE_CURVE_WITH_KNOTS',
            'CYLINDRICAL_SURFACE', 'PLANE', 'CONICAL_SURFACE',
            'CLOSED_SHELL',
        ]
        _type_to_idx = {t: i for i, t in enumerate(_BREP_TYPES)}
        num_type_slots = 16
        num_numeric_slots = feature_dim - num_type_slots  # 32

        # Use reference_graph for consistent sizing with adjacency_matrix
        if reference_graph is not None:
            num_nodes = reference_graph['num_nodes']
            id_to_idx = reference_graph['id_to_idx']
        else:
            # Legacy behavior: use features_list length
            num_nodes = max(len(features_list), 1)
            id_to_idx = None

        node_feats = np.zeros((num_nodes, feature_dim), dtype=np.float32)

        for list_idx, feat in enumerate(features_list):
            entity_id = feat.get('entity_id')

            # Determine the node index
            if id_to_idx is not None and entity_id is not None:
                if entity_id not in id_to_idx:
                    continue
                idx = id_to_idx[entity_id]
            else:
                # Legacy: features_list index ordering
                idx = list_idx
                if idx >= num_nodes:
                    continue

            # Numeric parameters
            params = feat.get('numeric_params', [])
            n_fill = min(len(params), num_numeric_slots)
            for j in range(n_fill):
                node_feats[idx, j] = float(params[j])

            # Entity type one-hot
            etype = feat.get('entity_type', '')
            tidx = _type_to_idx.get(etype, -1)
            if 0 <= tidx < num_type_slots:
                node_feats[idx, num_numeric_slots + tidx] = 1.0

        return torch.tensor(node_feats, dtype=torch.float32)

    @staticmethod
    def to_cadling_topology_graph(topo_dict: Dict):
        """Convert a ``build_complete_topology()`` output dict to a cadling TopologyGraph.

        This closes the round-trip: cadling → ll_stepnet → cadling.
        The cadling ``TopologyGraph`` is constructed from the adjacency
        matrix and node features stored in *topo_dict*.

        Args:
            topo_dict: Dictionary returned by :meth:`build_complete_topology`.
                Must contain ``adjacency_matrix`` (tensor or array) and
                ``node_features`` (tensor or array).

        Returns:
            A ``cadling.datamodel.base_models.TopologyGraph`` instance.

        Raises:
            ImportError: If cadling is not installed.
        """
        # Lazy import to avoid hard dependency on cadling
        from cadling.datamodel.base_models import TopologyGraph, CADItemLabel

        adj = topo_dict['adjacency_matrix']
        node_feats = topo_dict['node_features']

        # Convert to numpy if tensors (handle sparse adjacency gracefully)
        if hasattr(adj, 'is_sparse') and adj.is_sparse:
            adj = adj.coalesce()
            rows_t, cols_t = adj.indices()
            rows_list = rows_t.tolist()
            cols_list = cols_t.tolist()
        elif hasattr(adj, 'layout') and adj.layout == torch.sparse_csr:
            adj = adj.to_sparse_coo().coalesce()
            rows_t, cols_t = adj.indices()
            rows_list = rows_t.tolist()
            cols_list = cols_t.tolist()
        else:
            if hasattr(adj, 'cpu'):
                adj = adj.detach().cpu().numpy()
            nonzero = adj.nonzero()
            # numpy nonzero returns tuple of arrays; torch nonzero returns (K,2)
            if isinstance(nonzero, tuple):
                rows_np, cols_np = nonzero
            else:
                rows_np, cols_np = nonzero[:, 0], nonzero[:, 1]
            rows_list = rows_np.tolist() if hasattr(rows_np, 'tolist') else list(rows_np)
            cols_list = cols_np.tolist() if hasattr(cols_np, 'tolist') else list(cols_np)

        if hasattr(node_feats, 'cpu'):
            node_feats = node_feats.detach().cpu().numpy()

        num_nodes = int(node_feats.shape[0])

        # Build adjacency_list: Dict[int, List[int]]
        adjacency_list: Dict[int, List[int]] = {}
        for src, dst in zip(rows_list, cols_list):
            adjacency_list.setdefault(src, []).append(dst)

        num_edges = len(rows_list)

        # Convert node features to list of lists
        node_features_list = node_feats.tolist()

        # Edge features if present
        edge_feats_raw = topo_dict.get('edge_features')
        edge_features_list = None
        if edge_feats_raw is not None:
            if hasattr(edge_feats_raw, 'cpu'):
                edge_feats_raw = edge_feats_raw.detach().cpu().numpy()
            edge_features_list = edge_feats_raw.tolist()

        return TopologyGraph(
            num_nodes=num_nodes,
            num_edges=num_edges,
            adjacency_list=adjacency_list,
            node_features=node_features_list,
            edge_features=edge_features_list,
        )
