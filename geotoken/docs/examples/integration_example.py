#!/usr/bin/env python3
"""Example: Full integration with cadling and ll_stepnet data formats."""
from __future__ import annotations

import numpy as np

from geotoken import (
    GeoTokenizer,
    GraphTokenizer,
    CommandSequenceTokenizer,
    CADVocabulary,
    QuantizationConfig,
    PrecisionTier,
)
from geotoken.vertex import VertexValidator, VertexClusterer, VertexMerger


def simulate_cadling_data():
    """Simulate data that would come from cadling."""
    # Mesh data (from MeshItem.to_numpy())
    vertices = np.random.randn(50, 3).astype(np.float32) * 10
    faces = np.random.randint(0, 50, (100, 3)).astype(np.int64)

    # Topology graph data (from TopologyGraph)
    node_features = np.random.randn(10, 48).astype(np.float32)
    edge_index = np.array([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                           [1, 2, 3, 4, 5, 6, 7, 8, 9, 0]], dtype=np.int64)
    edge_features = np.random.randn(10, 16).astype(np.float32)

    # Command sequence (from Sketch2DItem)
    commands = [
        {"type": "SOL", "params": [0.5, 0.5] + [0]*14},
        {"type": "LINE", "params": [0.0, 0.0, 0, 1.0, 0.0] + [0]*11},
        {"type": "EXTRUDE", "params": [0]*15 + [5.0]},
        {"type": "EOS", "params": [0]*16},
    ]

    return {
        "vertices": vertices,
        "faces": faces,
        "node_features": node_features,
        "edge_index": edge_index,
        "edge_features": edge_features,
        "commands": commands,
    }


def main():
    print("=== GeoToken Integration Example ===\n")

    # Simulate cadling data
    data = simulate_cadling_data()
    print("Simulated cadling data:")
    print(f"  Vertices: {data['vertices'].shape}")
    print(f"  Faces: {data['faces'].shape}")
    print(f"  Node features: {data['node_features'].shape}")
    print(f"  Edge features: {data['edge_features'].shape}")
    print(f"  Commands: {len(data['commands'])}")

    # Initialize tokenizers
    mesh_tokenizer = GeoTokenizer(QuantizationConfig(tier=PrecisionTier.STANDARD))
    graph_tokenizer = GraphTokenizer()
    command_tokenizer = CommandSequenceTokenizer()
    vocab = CADVocabulary()

    # Tokenize all components
    print("\n--- Mesh Tokenization ---")
    mesh_tokens = mesh_tokenizer.tokenize(data["vertices"], data["faces"])
    print(f"  Coordinate tokens: {len(mesh_tokens.coordinate_tokens)}")

    print("\n--- Graph Tokenization ---")
    graph_tokens = graph_tokenizer.tokenize(
        data["node_features"],
        data["edge_index"],
        data["edge_features"]
    )
    print(f"  Node tokens: {len(graph_tokens.graph_node_tokens)}")
    print(f"  Edge tokens: {len(graph_tokens.graph_edge_tokens)}")

    print("\n--- Command Tokenization ---")
    cmd_tokens = command_tokenizer.tokenize(data["commands"])
    print(f"  Command tokens: {len(cmd_tokens.command_tokens)}")

    # Encode to IDs (for ll_stepnet model input)
    mesh_ids = vocab.encode_full_sequence(mesh_tokens)
    graph_ids = vocab.encode_full_sequence(graph_tokens)
    cmd_ids = vocab.encode(cmd_tokens.command_tokens)

    print("\n--- Vocabulary Encoding (for ll_stepnet) ---")
    print(f"  Mesh token IDs: {len(mesh_ids)}")
    print(f"  Graph token IDs: {len(graph_ids)}")
    print(f"  Command token IDs: {len(cmd_ids)}")

    # Post-processing example
    print("\n--- Vertex Post-Processing ---")
    validator = VertexValidator()
    report = validator.validate(data["vertices"], data["faces"])
    print(f"  Bounds check: {report.bounds.all_in_bounds if report.bounds else 'N/A'}")
    print(f"  Manifold check: {report.manifold.is_manifold if report.manifold else 'N/A'}")

    # Cluster duplicates
    clusterer = VertexClusterer(merge_distance=0.01)
    clustering = clusterer.cluster(data["vertices"])
    print(f"  Clusters found: {clustering.num_clusters}")

    # Reconstruct and verify
    print("\n--- Roundtrip Verification ---")
    reconstructed = mesh_tokenizer.detokenize(mesh_tokens)
    error = np.linalg.norm(data["vertices"] - reconstructed, axis=1)
    print(f"  Mean reconstruction error: {np.mean(error):.6f}")
    print(f"  Max reconstruction error: {np.max(error):.6f}")

    print("\n=== Integration Complete ===")


if __name__ == "__main__":
    main()
