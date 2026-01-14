"""
STEP Topology Builder Module
Constructs topological graph from STEP entity references.
"""

import torch
from typing import List, Dict, Tuple, Set
from collections import defaultdict


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
        Convert reference graph to adjacency matrix.

        Args:
            reference_graph: Output from build_reference_graph

        Returns:
            Adjacency matrix [N, N] where N = num_nodes
        """
        node_ids = reference_graph['node_ids']
        num_nodes = len(node_ids)

        # Create ID to index mapping
        id_to_idx = {node_id: idx for idx, node_id in enumerate(node_ids)}

        # Build adjacency matrix
        adj_matrix = torch.zeros(num_nodes, num_nodes, dtype=torch.float32)

        for from_id, to_id in reference_graph['edge_list']:
            if from_id in id_to_idx and to_id in id_to_idx:
                from_idx = id_to_idx[from_id]
                to_idx = id_to_idx[to_id]
                adj_matrix[from_idx, to_idx] = 1.0

        return adj_matrix

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
                # Simple hash to [0, 1] range
                type_hash = (hash(entity_type) % 10000) / 10000.0
                node_features[idx, -1] = type_hash

        return node_features

    def build_complete_topology(self, features_list: List[Dict]) -> Dict:
        """
        Build complete topology representation.

        Args:
            features_list: List of feature dicts from STEPFeatureExtractor

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
