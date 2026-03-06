"""UV-grid surface quantizer for B-Rep face parameterization.

Tokenizes UV-parameter surface samples from B-Rep faces into a parallel
token channel.  Each B-Rep face is sampled on a regular ``(U, V)`` grid;
the corresponding 3-D surface points ``(x, y, z)`` are quantized into
discrete tokens using :class:`FeatureVectorQuantizer` under the hood.

This gives the model an explicit *surface shape* signal in addition to
the topology-graph and command-sequence token channels already provided
by :class:`GraphTokenizer` and :class:`CommandSequenceTokenizer`.

Classes:
    UVGridTokens: Dataclass holding quantized grid tokens for a single face.
    UVGridQuantizer: Quantize UV-parameter grid samples from B-Rep faces.

Example::

    import numpy as np
    from geotoken.quantization.uv_grid_quantizer import UVGridQuantizer

    quantizer = UVGridQuantizer(grid_resolution=(5, 5), bits=8)

    # Single surface
    uv = np.random.rand(25, 2).astype(np.float32)
    xyz = np.random.rand(25, 3).astype(np.float32)
    result = quantizer.quantize_surface_samples(uv, xyz)
    print(result.quantized_grid.shape)  # (25, 3) int32
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Optional

import numpy as np

from geotoken.quantization.feature_quantizer import (
    FeatureVectorQuantizer,
    FeatureQuantizationParams,
)

if TYPE_CHECKING:
    from cadling.datamodel.base_models import TopologyGraph

_log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class UVGridTokens:
    """Quantized UV-grid tokens for a single B-Rep face.

    Attributes:
        face_index: Index of the face in the topology graph.
        grid_resolution: ``(num_u, num_v)`` grid dimensions.
        uv_samples: Original UV parameter values — shape ``(U*V, 2)``.
        xyz_samples: Original 3-D surface points — shape ``(U*V, 3)``.
        quantized_grid: Quantized XYZ values — shape ``(U*V, 3)``
            with integer values in ``[0, 2^bits - 1]``.
        params: Normalization parameters used for quantization.
        bits: Bit-width per coordinate dimension.
    """

    face_index: Optional[int] = None
    grid_resolution: tuple[int, int] = (5, 5)
    uv_samples: np.ndarray = field(
        default_factory=lambda: np.empty((0, 2), dtype=np.float32)
    )
    xyz_samples: np.ndarray = field(
        default_factory=lambda: np.empty((0, 3), dtype=np.float32)
    )
    quantized_grid: np.ndarray = field(
        default_factory=lambda: np.empty((0, 3), dtype=np.int32)
    )
    params: FeatureQuantizationParams | None = None
    bits: int = 8
    is_approximated: bool = False


@dataclass
class FaceUVGridTokens:
    """Quantized tokens for [10, 10, 7] face UV-grid.

    Represents a full face UV-grid with xyz points, normals, and trim mask.
    The grid is sampled on a regular ``(U, V)`` parameter grid.

    Channels in the input [num_u, num_v, 7] grid:
        - 0-2: XYZ surface points (quantized)
        - 3-5: Surface normals (quantized separately)
        - 6: Trim mask (preserved as boolean, NOT quantized)

    Attributes:
        face_index: Index of the face in the topology graph.
        grid_resolution: ``(num_u, num_v)`` grid dimensions.
        quantized_xyz: Quantized XYZ values — shape ``(U*V, 3)`` int32.
        quantized_normals: Quantized normal values — shape ``(U*V, 3)`` int32.
        trim_mask: Boolean trim mask — shape ``(U*V,)`` bool.
        params_xyz: Normalization parameters for XYZ quantization.
        params_normals: Normalization parameters for normals quantization.
        bits: Bit-width per coordinate dimension.
    """

    face_index: Optional[int] = None
    grid_resolution: tuple[int, int] = (10, 10)
    quantized_xyz: np.ndarray = field(
        default_factory=lambda: np.empty((0, 3), dtype=np.int32)
    )
    quantized_normals: np.ndarray = field(
        default_factory=lambda: np.empty((0, 3), dtype=np.int32)
    )
    trim_mask: np.ndarray = field(
        default_factory=lambda: np.empty((0,), dtype=bool)
    )
    params_xyz: FeatureQuantizationParams | None = None
    params_normals: FeatureQuantizationParams | None = None
    bits: int = 8
    is_approximated: bool = False


@dataclass
class EdgeUVGridTokens:
    """Quantized tokens for [10, 6] edge UV-grid.

    Represents an edge sampled at regular parameter intervals with
    xyz points and tangent vectors.

    Channels in the input [num_samples, 6] grid:
        - 0-2: XYZ edge points (quantized)
        - 3-5: Tangent vectors (quantized separately)

    Attributes:
        edge_index: Index of the edge in the topology graph.
        num_samples: Number of samples along the edge.
        quantized_xyz: Quantized XYZ values — shape ``(N, 3)`` int32.
        quantized_tangents: Quantized tangent values — shape ``(N, 3)`` int32.
        params_xyz: Normalization parameters for XYZ quantization.
        params_tangents: Normalization parameters for tangent quantization.
        bits: Bit-width per coordinate dimension.
    """

    edge_index: int = -1
    num_samples: int = 10
    quantized_xyz: np.ndarray = field(
        default_factory=lambda: np.empty((0, 3), dtype=np.int32)
    )
    quantized_tangents: np.ndarray = field(
        default_factory=lambda: np.empty((0, 3), dtype=np.int32)
    )
    params_xyz: FeatureQuantizationParams | None = None
    params_tangents: FeatureQuantizationParams | None = None
    bits: int = 8
    is_approximated: bool = False


# ---------------------------------------------------------------------------
# Main quantizer
# ---------------------------------------------------------------------------


class UVGridQuantizer:
    """Quantize UV-parameter grid samples from B-Rep surface patches.

    Samples each B-Rep face on a regular ``(U, V)`` parameter grid,
    evaluates the surface mapping ``(u, v) → (x, y, z)``, and
    quantizes the resulting 3-D points using
    :class:`FeatureVectorQuantizer`.

    The quantized grid can be linearised into a flat token sequence
    (row-major order) and prepended/appended to the topology-graph
    token stream to give the model an explicit *surface shape* channel.

    Args:
        grid_resolution: ``(num_u, num_v)`` sample count along each
            parameter direction.  Total samples per face =
            ``num_u × num_v``.  Default ``(5, 5)`` → 25 samples.
        bits: Quantization bit-width per coordinate axis.
            Default ``8`` → 256 levels.

    Example::

        quantizer = UVGridQuantizer(grid_resolution=(5, 5), bits=8)
        tokens = quantizer.quantize_surface_samples(uv, xyz)
    """

    def __init__(
        self,
        grid_resolution: tuple[int, int] = (5, 5),
        bits: int = 8,
    ) -> None:
        self.grid_resolution = grid_resolution
        self.bits = bits

        # Separate per-channel quantizers to avoid shared state corruption.
        # Face and edge XYZ get independent instances so interleaved
        # fit()/quantize() calls never corrupt each other's cached params.
        self._face_xyz_quantizer = FeatureVectorQuantizer(bits=bits)
        self._edge_xyz_quantizer = FeatureVectorQuantizer(bits=bits)
        self._normals_quantizer = FeatureVectorQuantizer(bits=bits)
        self._tangents_quantizer = FeatureVectorQuantizer(bits=bits)

        # Backward-compatible aliases
        self._xyz_quantizer = self._face_xyz_quantizer
        self._quantizer = self._face_xyz_quantizer

        # Global normalization params (set via fit_global)
        self._global_xyz_params: FeatureQuantizationParams | None = None
        self._global_normals_params: FeatureQuantizationParams | None = None
        self._global_tangents_params: FeatureQuantizationParams | None = None

        _log.debug(
            "UVGridQuantizer init: resolution=%s, bits=%d",
            grid_resolution,
            bits,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit_global(
        self,
        all_xyz_samples: np.ndarray,
        all_normals_samples: np.ndarray | None = None,
        all_tangents_samples: np.ndarray | None = None,
    ) -> None:
        """Fit global normalization params on combined data from all faces/edges.

        When global params are set, ``quantize_surface_samples``,
        ``quantize_face_uv_grid``, and ``quantize_edge_uv_grid`` will use
        them instead of per-face fitting.  This ensures cross-face token
        comparability — the same XYZ coordinate always maps to the same
        token regardless of which face it belongs to.

        Args:
            all_xyz_samples: Combined XYZ points from all faces/edges,
                shape ``(N, 3)`` float32.
            all_normals_samples: Optional combined normals from all faces,
                shape ``(M, 3)`` float32.
            all_tangents_samples: Optional combined tangents from all edges,
                shape ``(K, 3)`` float32.
        """
        all_xyz_samples = np.asarray(all_xyz_samples, dtype=np.float32)
        if all_xyz_samples.ndim != 2 or all_xyz_samples.shape[1] != 3:
            raise ValueError(
                f"all_xyz_samples must be (N, 3), got {all_xyz_samples.shape}"
            )

        self._global_xyz_params = self._face_xyz_quantizer.fit(all_xyz_samples)
        self._edge_xyz_quantizer.fit(all_xyz_samples)
        _log.info(
            "Fit global XYZ params on %d samples", len(all_xyz_samples),
        )

        if all_normals_samples is not None:
            all_normals_samples = np.asarray(all_normals_samples, dtype=np.float32)
            self._global_normals_params = self._normals_quantizer.fit(all_normals_samples)
            _log.info(
                "Fit global normals params on %d samples",
                len(all_normals_samples),
            )

        if all_tangents_samples is not None:
            all_tangents_samples = np.asarray(all_tangents_samples, dtype=np.float32)
            self._global_tangents_params = self._tangents_quantizer.fit(all_tangents_samples)
            _log.info(
                "Fit global tangents params on %d samples",
                len(all_tangents_samples),
            )

    def quantize_surface_samples(
        self,
        uv_samples: np.ndarray,
        xyz_samples: np.ndarray,
        face_index: Optional[int] = None,
    ) -> UVGridTokens:
        """Quantize ``(u, v) → (x, y, z)`` mappings from a single surface.

        The caller is responsible for evaluating the surface at the
        desired UV points.  This method quantizes the *xyz* side and
        packages the result into :class:`UVGridTokens`.

        Args:
            uv_samples: UV parameters — shape ``(N, 2)`` where
                ``N = num_u * num_v``.
            xyz_samples: Corresponding 3-D points — shape ``(N, 3)``.
            face_index: Optional face index for bookkeeping.

        Returns:
            :class:`UVGridTokens` with quantized grid.

        Raises:
            ValueError: If shapes are inconsistent.
        """
        uv_samples = np.asarray(uv_samples, dtype=np.float32)
        xyz_samples = np.asarray(xyz_samples, dtype=np.float32)

        if uv_samples.ndim != 2 or uv_samples.shape[1] != 2:
            raise ValueError(
                f"uv_samples must be (N, 2), got {uv_samples.shape}"
            )
        if xyz_samples.ndim != 2 or xyz_samples.shape[1] != 3:
            raise ValueError(
                f"xyz_samples must be (N, 3), got {xyz_samples.shape}"
            )
        if uv_samples.shape[0] != xyz_samples.shape[0]:
            raise ValueError(
                f"Sample count mismatch: uv={uv_samples.shape[0]} "
                f"vs xyz={xyz_samples.shape[0]}"
            )

        # Use global params if available; otherwise fit per-surface
        if self._global_xyz_params is not None:
            params = self._global_xyz_params
        else:
            params = self._xyz_quantizer.fit(xyz_samples)
        quantized = self._xyz_quantizer.quantize(xyz_samples, params).astype(np.int32)

        return UVGridTokens(
            face_index=face_index,
            grid_resolution=self.grid_resolution,
            uv_samples=uv_samples,
            xyz_samples=xyz_samples,
            quantized_grid=quantized,
            params=params,
            bits=self.bits,
        )

    def quantize_from_topology(
        self,
        topology: TopologyGraph,
    ) -> dict[int, UVGridTokens]:
        """Extract and quantize UV grids for all faces in a topology graph.

        The topology graph's 48-dim node features encode UV statistics
        at known indices (see ``cadling.backend.step.enhanced_features``).
        When raw UV samples are not directly available, this method
        synthesises a regular grid from the per-face UV bounds stored
        in the feature vector.

        Args:
            topology: A :class:`TopologyGraph` from a CADlingDocument.

        Returns:
            Mapping of ``{face_index → UVGridTokens}``.
            Faces without UV data are silently skipped.
        """
        node_features = topology.to_numpy_node_features()
        if node_features is None or len(node_features) == 0:
            _log.debug("No node features in topology; skipping UV grid")
            return {}

        results: dict[int, UVGridTokens] = {}
        num_u, num_v = self.grid_resolution

        for face_idx in range(len(node_features)):
            feats = node_features[face_idx]

            # Extract UV bounds from the 48-dim feature vector.
            # By convention (see enhanced_features.py):
            #   indices [30:32] → u_min, u_max
            #   indices [32:34] → v_min, v_max
            #   indices [34:40] → xyz statistics (mean_x,y,z, std_x,y,z)
            if len(feats) < 40:
                continue  # Not an enhanced feature vector

            u_min, u_max = float(feats[30]), float(feats[31])
            v_min, v_max = float(feats[32]), float(feats[33])

            # Skip degenerate parameter spaces
            if abs(u_max - u_min) < 1e-12 or abs(v_max - v_min) < 1e-12:
                continue

            # Generate regular grid in UV space
            u_vals = np.linspace(u_min, u_max, num_u, dtype=np.float32)
            v_vals = np.linspace(v_min, v_max, num_v, dtype=np.float32)
            uu, vv = np.meshgrid(u_vals, v_vals, indexing="ij")
            uv_grid = np.column_stack([uu.ravel(), vv.ravel()])

            # Synthesise approximate xyz from the UV grid + feature stats.
            # This is a linear approximation; actual evaluation requires
            # the B-Rep surface, which may not be available at this stage.
            xyz_mean = feats[34:37].astype(np.float32)  # (3,)
            xyz_std = feats[37:40].astype(np.float32)   # (3,)

            # Map UV → approximate XYZ via linear interpolation
            # u_frac / v_frac ∈ [0, 1]
            u_frac = (uv_grid[:, 0] - u_min) / (u_max - u_min)
            v_frac = (uv_grid[:, 1] - v_min) / (v_max - v_min)

            # Simple bilinear model: xyz ≈ mean + std * (u_frac + v_frac - 1)
            t = (u_frac + v_frac)[:, np.newaxis] / 2.0  # (N, 1)
            xyz_approx = xyz_mean[np.newaxis, :] + xyz_std[np.newaxis, :] * (
                2.0 * t - 1.0
            )
            xyz_approx = xyz_approx.astype(np.float32)

            try:
                tokens = self.quantize_surface_samples(
                    uv_grid, xyz_approx, face_index=face_idx,
                )
                tokens.is_approximated = True
                results[face_idx] = tokens
            except Exception as exc:
                _log.debug(
                    "UV grid quantization failed for face %d: %s",
                    face_idx,
                    exc,
                )

        _log.info(
            "UV grid quantization: %d/%d faces quantized",
            len(results),
            len(node_features),
        )
        return results

    def dequantize(self, tokens: UVGridTokens) -> np.ndarray:
        """Dequantize grid tokens back to approximate 3-D coordinates.

        Args:
            tokens: :class:`UVGridTokens` with quantized grid.

        Returns:
            Reconstructed ``(N, 3)`` float32 array.

        Raises:
            ValueError: If params are missing from the token object.
        """
        if tokens.params is None:
            raise ValueError("Cannot dequantize: missing params")

        return self._xyz_quantizer.dequantize(
            tokens.quantized_grid, tokens.params,
        )

    def to_flat_tokens(self, tokens: UVGridTokens) -> list[int]:
        """Linearise quantized grid to a flat integer token list.

        The grid is serialised in row-major order (U-major, then V).
        Each sample produces ``3`` tokens (quantized x, y, z),
        so the total length is ``num_u * num_v * 3``.

        Args:
            tokens: :class:`UVGridTokens` with quantized grid.

        Returns:
            Flat list of integer tokens.
        """
        return tokens.quantized_grid.ravel().tolist()

    # ------------------------------------------------------------------
    # Full face/edge UV-grid quantization (cadling integration)
    # ------------------------------------------------------------------

    def quantize_face_uv_grid(
        self,
        uv_grid: np.ndarray,
        face_index: Optional[int] = None,
    ) -> FaceUVGridTokens:
        """Quantize full [num_u, num_v, 7] face UV-grid.

        Processes a face UV-grid with xyz points, normals, and trim mask.
        XYZ and normals are quantized independently; the trim mask is
        preserved as a boolean array without quantization.

        Channels:
            - 0-2: XYZ points (quantized)
            - 3-5: normals (quantized separately)
            - 6: trim mask (preserved as bool)

        Args:
            uv_grid: Face UV-grid — shape ``(num_u, num_v, 7)`` float32.
            face_index: Optional face index for bookkeeping.

        Returns:
            :class:`FaceUVGridTokens` with quantized xyz, normals, and
            preserved trim mask.

        Raises:
            ValueError: If shape is not ``(num_u, num_v, 7)``.
        """
        uv_grid = np.asarray(uv_grid, dtype=np.float32)

        if uv_grid.ndim != 3 or uv_grid.shape[2] != 7:
            raise ValueError(
                f"uv_grid must be (num_u, num_v, 7), got {uv_grid.shape}"
            )

        num_u, num_v = uv_grid.shape[:2]
        n_samples = num_u * num_v

        # Extract channels
        xyz = uv_grid[..., :3].reshape(n_samples, 3)
        normals = uv_grid[..., 3:6].reshape(n_samples, 3)
        trim_mask = uv_grid[..., 6].reshape(n_samples).astype(bool)

        # Quantize XYZ (use global params if available)
        if self._global_xyz_params is not None:
            params_xyz = self._global_xyz_params
        else:
            params_xyz = self._xyz_quantizer.fit(xyz)
        quantized_xyz = self._xyz_quantizer.quantize(xyz, params_xyz).astype(np.int32)

        # Quantize normals independently (use global params if available)
        if self._global_normals_params is not None:
            params_normals = self._global_normals_params
        else:
            params_normals = self._normals_quantizer.fit(normals)
        quantized_normals = self._normals_quantizer.quantize(normals, params_normals).astype(np.int32)

        _log.debug(
            "Quantized face %d UV-grid: %dx%d samples",
            face_index,
            num_u,
            num_v,
        )

        return FaceUVGridTokens(
            face_index=face_index,
            grid_resolution=(num_u, num_v),
            quantized_xyz=quantized_xyz,
            quantized_normals=quantized_normals,
            trim_mask=trim_mask,
            params_xyz=params_xyz,
            params_normals=params_normals,
            bits=self.bits,
        )

    def quantize_edge_uv_grid(
        self,
        uv_grid: np.ndarray,
        edge_index: int = -1,
    ) -> EdgeUVGridTokens:
        """Quantize full [num_samples, 6] edge UV-grid.

        Processes an edge UV-grid with xyz points and tangent vectors.
        Both are quantized independently.

        Channels:
            - 0-2: XYZ points (quantized)
            - 3-5: tangent vectors (quantized)

        Args:
            uv_grid: Edge UV-grid — shape ``(num_samples, 6)`` float32.
            edge_index: Optional edge index for bookkeeping.

        Returns:
            :class:`EdgeUVGridTokens` with quantized xyz and tangents.

        Raises:
            ValueError: If shape is not ``(num_samples, 6)``.
        """
        uv_grid = np.asarray(uv_grid, dtype=np.float32)

        if uv_grid.ndim != 2 or uv_grid.shape[1] != 6:
            raise ValueError(
                f"uv_grid must be (num_samples, 6), got {uv_grid.shape}"
            )

        num_samples = uv_grid.shape[0]

        # Extract channels
        xyz = uv_grid[:, :3]
        tangents = uv_grid[:, 3:6]

        # Quantize XYZ with edge-dedicated quantizer (avoids shared state
        # corruption when interleaved with face quantization calls)
        if self._global_xyz_params is not None:
            params_xyz = self._global_xyz_params
        else:
            params_xyz = self._edge_xyz_quantizer.fit(xyz)
        quantized_xyz = self._edge_xyz_quantizer.quantize(xyz, params_xyz).astype(np.int32)

        # Quantize tangents independently (use global params if available)
        if self._global_tangents_params is not None:
            params_tangents = self._global_tangents_params
        else:
            params_tangents = self._tangents_quantizer.fit(tangents)
        quantized_tangents = self._tangents_quantizer.quantize(tangents, params_tangents).astype(np.int32)

        _log.debug(
            "Quantized edge %d UV-grid: %d samples",
            edge_index,
            num_samples,
        )

        return EdgeUVGridTokens(
            edge_index=edge_index,
            num_samples=num_samples,
            quantized_xyz=quantized_xyz,
            quantized_tangents=quantized_tangents,
            params_xyz=params_xyz,
            params_tangents=params_tangents,
            bits=self.bits,
        )

    def dequantize_face_grid(self, tokens: FaceUVGridTokens) -> np.ndarray:
        """Dequantize face grid tokens back to [num_u, num_v, 7] array.

        Reconstructs the full face UV-grid from quantized tokens.
        XYZ and normals are dequantized; the trim mask is restored
        from the boolean array.

        Args:
            tokens: :class:`FaceUVGridTokens` with quantized data.

        Returns:
            Reconstructed ``(num_u, num_v, 7)`` float32 array.

        Raises:
            ValueError: If params are missing from the token object.
        """
        if tokens.params_xyz is None or tokens.params_normals is None:
            raise ValueError("Cannot dequantize: missing params")

        num_u, num_v = tokens.grid_resolution
        n_samples = num_u * num_v

        # Dequantize XYZ and normals
        xyz = self._xyz_quantizer.dequantize(tokens.quantized_xyz, tokens.params_xyz)
        normals = self._normals_quantizer.dequantize(
            tokens.quantized_normals, tokens.params_normals
        )

        # Reconstruct grid
        result = np.zeros((num_u, num_v, 7), dtype=np.float32)
        result[..., :3] = xyz.reshape(num_u, num_v, 3)
        result[..., 3:6] = normals.reshape(num_u, num_v, 3)
        result[..., 6] = tokens.trim_mask.reshape(num_u, num_v).astype(np.float32)

        return result

    def dequantize_edge_grid(self, tokens: EdgeUVGridTokens) -> np.ndarray:
        """Dequantize edge grid tokens back to [num_samples, 6] array.

        Reconstructs the full edge UV-grid from quantized tokens.

        Args:
            tokens: :class:`EdgeUVGridTokens` with quantized data.

        Returns:
            Reconstructed ``(num_samples, 6)`` float32 array.

        Raises:
            ValueError: If params are missing from the token object.
        """
        if tokens.params_xyz is None or tokens.params_tangents is None:
            raise ValueError("Cannot dequantize: missing params")

        # Dequantize XYZ and tangents (edge-dedicated quantizer)
        xyz = self._edge_xyz_quantizer.dequantize(tokens.quantized_xyz, tokens.params_xyz)
        tangents = self._tangents_quantizer.dequantize(
            tokens.quantized_tangents, tokens.params_tangents
        )

        # Reconstruct grid
        result = np.concatenate([xyz, tangents], axis=1)
        return result
