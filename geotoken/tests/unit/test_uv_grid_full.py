"""Tests for full UV-grid quantization (face and edge grids).

Tests the FaceUVGridTokens and EdgeUVGridTokens dataclasses and
their quantize/dequantize methods in UVGridQuantizer.
"""

from __future__ import annotations

import numpy as np
import pytest

from geotoken.quantization.uv_grid_quantizer import (
    UVGridQuantizer,
    UVGridTokens,
    FaceUVGridTokens,
    EdgeUVGridTokens,
)


class TestFaceUVGridTokens:
    """Test [10, 10, 7] face UV-grid quantization."""

    def test_quantize_face_grid_shape(self):
        """Test basic face grid quantization produces correct shapes."""
        quantizer = UVGridQuantizer(grid_resolution=(10, 10), bits=8)

        # Create [10, 10, 7] face grid
        face_grid = np.random.rand(10, 10, 7).astype(np.float32)
        # Set trim mask to binary values
        face_grid[..., 6] = (face_grid[..., 6] > 0.5).astype(np.float32)

        result = quantizer.quantize_face_uv_grid(face_grid, face_index=0)

        assert isinstance(result, FaceUVGridTokens)
        assert result.quantized_xyz.shape == (100, 3)
        assert result.quantized_normals.shape == (100, 3)
        assert result.trim_mask.shape == (100,)
        assert result.face_index == 0
        assert result.grid_resolution == (10, 10)
        assert result.bits == 8

    def test_quantize_face_grid_different_resolution(self):
        """Test face grid quantization with non-standard resolution."""
        quantizer = UVGridQuantizer(grid_resolution=(5, 8), bits=8)

        face_grid = np.random.rand(5, 8, 7).astype(np.float32)
        face_grid[..., 6] = (face_grid[..., 6] > 0.5).astype(np.float32)

        result = quantizer.quantize_face_uv_grid(face_grid, face_index=3)

        assert result.quantized_xyz.shape == (40, 3)  # 5 * 8 = 40
        assert result.quantized_normals.shape == (40, 3)
        assert result.trim_mask.shape == (40,)
        assert result.grid_resolution == (5, 8)

    def test_quantize_face_grid_preserves_params(self):
        """Test that quantization parameters are stored."""
        quantizer = UVGridQuantizer(bits=8)

        face_grid = np.random.rand(10, 10, 7).astype(np.float32)
        face_grid[..., 6] = (face_grid[..., 6] > 0.5).astype(np.float32)

        result = quantizer.quantize_face_uv_grid(face_grid)

        assert result.params_xyz is not None
        assert result.params_normals is not None
        # XYZ and normals should have separate params
        assert result.params_xyz is not result.params_normals

    def test_face_grid_roundtrip(self):
        """Test quantize-dequantize roundtrip preserves values."""
        quantizer = UVGridQuantizer(bits=8)

        face_grid = np.random.rand(10, 10, 7).astype(np.float32)
        face_grid[..., 6] = (face_grid[..., 6] > 0.5).astype(np.float32)

        tokens = quantizer.quantize_face_uv_grid(face_grid)
        reconstructed = quantizer.dequantize_face_grid(tokens)

        # Check shape
        assert reconstructed.shape == face_grid.shape

        # XYZ should be close (quantization error for 8-bit)
        xyz_error = np.abs(reconstructed[..., :3] - face_grid[..., :3]).max()
        assert xyz_error < 0.05, f"XYZ max error {xyz_error} too high"

        # Normals should be close
        normal_error = np.abs(reconstructed[..., 3:6] - face_grid[..., 3:6]).max()
        assert normal_error < 0.05, f"Normal max error {normal_error} too high"

        # Trim mask should be preserved exactly
        np.testing.assert_array_equal(
            reconstructed[..., 6].astype(bool),
            face_grid[..., 6].astype(bool),
        )

    def test_face_grid_trim_mask_preserved(self):
        """Test that trim mask is preserved as boolean without quantization loss."""
        quantizer = UVGridQuantizer(bits=8)

        face_grid = np.zeros((10, 10, 7), dtype=np.float32)
        # Create a specific pattern in trim mask
        face_grid[2:5, 3:7, 6] = 1.0
        face_grid[7:9, 1:3, 6] = 1.0

        tokens = quantizer.quantize_face_uv_grid(face_grid)
        reconstructed = quantizer.dequantize_face_grid(tokens)

        # Check exact trim mask preservation
        original_mask = face_grid[..., 6].astype(bool)
        reconstructed_mask = reconstructed[..., 6].astype(bool)
        np.testing.assert_array_equal(original_mask, reconstructed_mask)

    def test_face_grid_invalid_shape_raises(self):
        """Test that invalid shapes raise ValueError."""
        quantizer = UVGridQuantizer(bits=8)

        # Wrong number of channels
        with pytest.raises(ValueError, match="must be"):
            quantizer.quantize_face_uv_grid(np.random.rand(10, 10, 3))

        # Wrong number of dimensions
        with pytest.raises(ValueError, match="must be"):
            quantizer.quantize_face_uv_grid(np.random.rand(100, 7))

    def test_dequantize_face_grid_missing_params_raises(self):
        """Test that dequantizing without params raises ValueError."""
        quantizer = UVGridQuantizer(bits=8)

        tokens = FaceUVGridTokens(
            quantized_xyz=np.zeros((100, 3), dtype=np.int32),
            quantized_normals=np.zeros((100, 3), dtype=np.int32),
            trim_mask=np.zeros((100,), dtype=bool),
            params_xyz=None,  # Missing params
            params_normals=None,
        )

        with pytest.raises(ValueError, match="missing params"):
            quantizer.dequantize_face_grid(tokens)


class TestEdgeUVGridTokens:
    """Test [10, 6] edge UV-grid quantization."""

    def test_quantize_edge_grid_shape(self):
        """Test basic edge grid quantization produces correct shapes."""
        quantizer = UVGridQuantizer(bits=8)

        edge_grid = np.random.rand(10, 6).astype(np.float32)

        result = quantizer.quantize_edge_uv_grid(edge_grid, edge_index=0)

        assert isinstance(result, EdgeUVGridTokens)
        assert result.quantized_xyz.shape == (10, 3)
        assert result.quantized_tangents.shape == (10, 3)
        assert result.edge_index == 0
        assert result.num_samples == 10
        assert result.bits == 8

    def test_quantize_edge_grid_different_samples(self):
        """Test edge grid quantization with different sample counts."""
        quantizer = UVGridQuantizer(bits=8)

        edge_grid = np.random.rand(25, 6).astype(np.float32)

        result = quantizer.quantize_edge_uv_grid(edge_grid, edge_index=5)

        assert result.quantized_xyz.shape == (25, 3)
        assert result.quantized_tangents.shape == (25, 3)
        assert result.num_samples == 25
        assert result.edge_index == 5

    def test_quantize_edge_grid_preserves_params(self):
        """Test that quantization parameters are stored."""
        quantizer = UVGridQuantizer(bits=8)

        edge_grid = np.random.rand(10, 6).astype(np.float32)

        result = quantizer.quantize_edge_uv_grid(edge_grid)

        assert result.params_xyz is not None
        assert result.params_tangents is not None
        # XYZ and tangents should have separate params
        assert result.params_xyz is not result.params_tangents

    def test_edge_grid_roundtrip(self):
        """Test quantize-dequantize roundtrip preserves values."""
        quantizer = UVGridQuantizer(bits=8)

        edge_grid = np.random.rand(10, 6).astype(np.float32)

        tokens = quantizer.quantize_edge_uv_grid(edge_grid)
        reconstructed = quantizer.dequantize_edge_grid(tokens)

        # Check shape
        assert reconstructed.shape == edge_grid.shape

        # Values should be close (quantization error for 8-bit)
        max_error = np.abs(reconstructed - edge_grid).max()
        assert max_error < 0.05, f"Max error {max_error} too high"

    def test_edge_grid_roundtrip_higher_bits(self):
        """Test that higher bit depth reduces quantization error."""
        # 8-bit
        quantizer_8 = UVGridQuantizer(bits=8)
        edge_grid = np.random.rand(10, 6).astype(np.float32)

        tokens_8 = quantizer_8.quantize_edge_uv_grid(edge_grid)
        recon_8 = quantizer_8.dequantize_edge_grid(tokens_8)
        error_8 = np.abs(recon_8 - edge_grid).max()

        # 12-bit (if supported)
        quantizer_12 = UVGridQuantizer(bits=12)
        tokens_12 = quantizer_12.quantize_edge_uv_grid(edge_grid)
        recon_12 = quantizer_12.dequantize_edge_grid(tokens_12)
        error_12 = np.abs(recon_12 - edge_grid).max()

        # Higher bits should have lower error
        assert error_12 < error_8, "12-bit should have lower error than 8-bit"

    def test_edge_grid_invalid_shape_raises(self):
        """Test that invalid shapes raise ValueError."""
        quantizer = UVGridQuantizer(bits=8)

        # Wrong number of columns
        with pytest.raises(ValueError, match="must be"):
            quantizer.quantize_edge_uv_grid(np.random.rand(10, 3))

        # Wrong number of dimensions
        with pytest.raises(ValueError, match="must be"):
            quantizer.quantize_edge_uv_grid(np.random.rand(10, 2, 6))

    def test_dequantize_edge_grid_missing_params_raises(self):
        """Test that dequantizing without params raises ValueError."""
        quantizer = UVGridQuantizer(bits=8)

        tokens = EdgeUVGridTokens(
            quantized_xyz=np.zeros((10, 3), dtype=np.int32),
            quantized_tangents=np.zeros((10, 3), dtype=np.int32),
            params_xyz=None,  # Missing params
            params_tangents=None,
        )

        with pytest.raises(ValueError, match="missing params"):
            quantizer.dequantize_edge_grid(tokens)


class TestUVGridQuantizerCompatibility:
    """Test backward compatibility with existing UVGridTokens."""

    def test_existing_surface_samples_still_works(self):
        """Ensure existing quantize_surface_samples API unchanged."""
        quantizer = UVGridQuantizer(grid_resolution=(5, 5), bits=8)

        uv = np.random.rand(25, 2).astype(np.float32)
        xyz = np.random.rand(25, 3).astype(np.float32)

        result = quantizer.quantize_surface_samples(uv, xyz, face_index=0)

        assert isinstance(result, UVGridTokens)
        assert result.quantized_grid.shape == (25, 3)
        assert result.face_index == 0

    def test_existing_dequantize_still_works(self):
        """Ensure existing dequantize API unchanged."""
        quantizer = UVGridQuantizer(bits=8)

        uv = np.random.rand(25, 2).astype(np.float32)
        xyz = np.random.rand(25, 3).astype(np.float32)

        tokens = quantizer.quantize_surface_samples(uv, xyz)
        reconstructed = quantizer.dequantize(tokens)

        assert reconstructed.shape == (25, 3)

    def test_to_flat_tokens_still_works(self):
        """Ensure existing to_flat_tokens API unchanged."""
        quantizer = UVGridQuantizer(bits=8)

        uv = np.random.rand(25, 2).astype(np.float32)
        xyz = np.random.rand(25, 3).astype(np.float32)

        tokens = quantizer.quantize_surface_samples(uv, xyz)
        flat = quantizer.to_flat_tokens(tokens)

        assert isinstance(flat, list)
        assert len(flat) == 75  # 25 samples * 3 coords


class TestDataclassDefaults:
    """Test dataclass default values and initialization."""

    def test_face_uv_grid_tokens_defaults(self):
        """Test FaceUVGridTokens has sensible defaults."""
        tokens = FaceUVGridTokens()

        assert tokens.face_index == -1
        assert tokens.grid_resolution == (10, 10)
        assert tokens.quantized_xyz.shape == (0, 3)
        assert tokens.quantized_normals.shape == (0, 3)
        assert tokens.trim_mask.shape == (0,)
        assert tokens.params_xyz is None
        assert tokens.params_normals is None
        assert tokens.bits == 8

    def test_edge_uv_grid_tokens_defaults(self):
        """Test EdgeUVGridTokens has sensible defaults."""
        tokens = EdgeUVGridTokens()

        assert tokens.edge_index == -1
        assert tokens.num_samples == 10
        assert tokens.quantized_xyz.shape == (0, 3)
        assert tokens.quantized_tangents.shape == (0, 3)
        assert tokens.params_xyz is None
        assert tokens.params_tangents is None
        assert tokens.bits == 8
