#!/usr/bin/env python3
"""Example: Mesh tokenization with adaptive quantization."""
from __future__ import annotations

import numpy as np

from geotoken import (
    GeoTokenizer,
    QuantizationConfig,
    PrecisionTier,
    AdaptiveBitAllocationConfig,
)


def create_cube_mesh() -> tuple[np.ndarray, np.ndarray]:
    """Create a unit cube mesh."""
    vertices = np.array([
        [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],  # Bottom face
        [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1],  # Top face
    ], dtype=np.float32)

    faces = np.array([
        [0, 1, 2], [0, 2, 3],  # Bottom
        [4, 6, 5], [4, 7, 6],  # Top
        [0, 4, 5], [0, 5, 1],  # Front
        [2, 6, 7], [2, 7, 3],  # Back
        [0, 3, 7], [0, 7, 4],  # Left
        [1, 5, 6], [1, 6, 2],  # Right
    ], dtype=np.int64)

    return vertices, faces


def main():
    print("=== GeoToken Mesh Tokenization Example ===\n")

    # Create sample mesh
    vertices, faces = create_cube_mesh()
    print(f"Input mesh: {len(vertices)} vertices, {len(faces)} faces")

    # Configure adaptive quantization
    config = QuantizationConfig(
        tier=PrecisionTier.STANDARD,
        adaptive=True,
        bit_allocation=AdaptiveBitAllocationConfig(
            base_bits=8,
            max_additional_bits=4,
            curvature_weight=0.7,
            density_weight=0.3,
        )
    )

    # Tokenize
    tokenizer = GeoTokenizer(config)
    tokens = tokenizer.tokenize(vertices, faces)

    print(f"\nTokenization results:")
    print(f"  Coordinate tokens: {len(tokens.coordinate_tokens)}")
    print(f"  Geometry tokens: {len(tokens.geometry_tokens)}")

    # Show bits per vertex
    bits = [t.bits for t in tokens.coordinate_tokens]
    print(f"  Bits per vertex: min={min(bits)}, max={max(bits)}, mean={np.mean(bits):.1f}")

    # Reconstruct
    reconstructed = tokenizer.detokenize(tokens)

    # Calculate error
    error = np.linalg.norm(vertices - reconstructed, axis=1)
    print(f"\nReconstruction error:")
    print(f"  Mean: {np.mean(error):.6f}")
    print(f"  Max: {np.max(error):.6f}")

    # Analyze impact
    impact = tokenizer.analyze_impact(vertices, faces)
    print(f"\nImpact analysis:")
    print(f"  Hausdorff distance: {impact.hausdorff_distance:.6f}")
    print(f"  Total bits: {impact.total_bits_used}")
    print(f"  Mean bits/vertex: {impact.mean_bits_per_vertex:.1f}")

    # Compare precision tiers
    print("\n=== Precision Tier Comparison ===")
    for tier in [PrecisionTier.DRAFT, PrecisionTier.STANDARD, PrecisionTier.PRECISION]:
        cfg = QuantizationConfig(tier=tier, adaptive=True)
        tok = GeoTokenizer(cfg)
        tokens = tok.tokenize(vertices, faces)
        recon = tok.detokenize(tokens)
        err = np.max(np.linalg.norm(vertices - recon, axis=1))
        print(f"  {tier.name}: max_error={err:.6f}")


if __name__ == "__main__":
    main()
