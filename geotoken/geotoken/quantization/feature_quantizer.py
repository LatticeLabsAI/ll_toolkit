"""Feature vector quantization for B-Rep topology graph data.

Quantizes N-dimensional dense feature vectors (e.g., 48-dim face features
or 16-dim edge features from cadling's enhanced_features module) into
discrete integer tokens suitable for transformer vocabulary encoding.

Supports per-dimension normalization to handle features with different
scales and distributions.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

_log = logging.getLogger(__name__)


@dataclass
class FeatureQuantizationParams:
    """Fitted normalization parameters for feature quantization.

    Stores per-dimension min/max values learned from training data,
    used to map feature values to the [0, levels-1] quantization range.

    Args:
        dim: Feature dimensionality.
        min_vals: Per-dimension minimum values (shape: (D,)).
        max_vals: Per-dimension maximum values (shape: (D,)).
        scale: Pre-computed per-dimension scale factor.
    """
    dim: int
    min_vals: np.ndarray = field(repr=False)
    max_vals: np.ndarray = field(repr=False)
    scale: np.ndarray = field(repr=False)  # (max_vals - min_vals) per dimension


class FeatureVectorQuantizer:
    """Quantize dense feature vectors to discrete token sequences.

    Maps N-dimensional float feature vectors to integer values in
    [0, 2^bits - 1] using per-dimension linear quantization. Supports
    fit/quantize workflow for learning normalization from data.

    Args:
        bits: Quantization bit width per dimension. Default 8 → 256 levels.
        strategy: Normalization strategy:
            "per_dimension" - Normalize each feature dim independently (default)
            "global" - Use a single min/max across all dimensions

    Example:
        quantizer = FeatureVectorQuantizer(bits=8)
        params = quantizer.fit(training_features)  # (N, 48) float32
        quantized = quantizer.quantize(features, params)  # (N, 48) int
        reconstructed = quantizer.dequantize(quantized, params)  # (N, 48) float32
    """

    def __init__(self, bits: int = 8, strategy: str = "per_dimension"):
        self.bits = bits
        self.levels = 2 ** bits
        self.strategy = strategy
        self._params: Optional[FeatureQuantizationParams] = None

    def fit(self, features: np.ndarray) -> FeatureQuantizationParams:
        """Compute normalization parameters from feature data.

        Learns per-dimension (or global) min/max values that will be
        used to map features to the quantization range.

        Note:
            This method caches the returned params in ``self._params``
            for convenience (so that subsequent ``quantize()`` calls can
            omit the *params* argument).  However, because the cached
            state is overwritten on every call, callers that need
            thread-safety or interleaved fits should use the **returned**
            ``FeatureQuantizationParams`` object directly instead of
            relying on the cached instance state.

        Args:
            features: Feature array of shape (N, D) where N is number
                of samples and D is feature dimensionality.

        Returns:
            FeatureQuantizationParams with learned normalization values.

        Raises:
            ValueError: If features array is not 2D.
        """
        if self._params is not None:
            _log.warning(
                "FeatureVectorQuantizer.fit() is overwriting previously "
                "fitted params (dim=%d). Use the returned params object "
                "for thread-safe / multi-fit workflows.",
                self._params.dim,
            )

        if features.ndim != 2:
            raise ValueError(f"Expected 2D features, got shape {features.shape}")

        dim = features.shape[1]

        if self.strategy == "per_dimension":
            min_vals = features.min(axis=0)
            max_vals = features.max(axis=0)
        elif self.strategy == "global":
            global_min = features.min()
            global_max = features.max()
            min_vals = np.full(dim, global_min)
            max_vals = np.full(dim, global_max)
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")

        # Prevent division by zero for constant features
        range_vals = max_vals - min_vals
        range_vals = np.where(range_vals < 1e-10, 1.0, range_vals)

        params = FeatureQuantizationParams(
            dim=dim,
            min_vals=min_vals.astype(np.float32),
            max_vals=max_vals.astype(np.float32),
            scale=range_vals.astype(np.float32),
        )
        self._params = params

        _log.debug(
            "Fit FeatureVectorQuantizer: dim=%d, %d levels, strategy=%s",
            dim, self.levels, self.strategy,
        )
        return params

    def quantize(
        self,
        features: np.ndarray,
        params: Optional[FeatureQuantizationParams] = None,
    ) -> np.ndarray:
        """Quantize float features to integer tokens.

        Maps each feature dimension from [min_d, max_d] to [0, levels-1].
        Values outside the fitted range are clamped.

        Args:
            features: Feature array of shape (N, D) float32.
            params: Normalization params (uses fitted params if None).

        Returns:
            Quantized array of shape (N, D) int64 with values in [0, levels-1].

        Raises:
            ValueError: If params not provided and fit() hasn't been called.
            ValueError: If feature dimensionality doesn't match params.
        """
        p = params or self._params
        if p is None:
            raise ValueError("Must call fit() or provide params before quantize()")

        if features.ndim == 1:
            features = features.reshape(1, -1)

        if features.shape[1] != p.dim:
            raise ValueError(
                f"Feature dim {features.shape[1]} != fitted dim {p.dim}"
            )

        # Normalize to [0, 1]
        normalized = (features - p.min_vals) / p.scale
        normalized = np.clip(normalized, 0.0, 1.0)

        # Map to [0, levels-1]
        quantized = np.round(normalized * (self.levels - 1)).astype(np.int64)
        quantized = np.clip(quantized, 0, self.levels - 1)

        return quantized

    def dequantize(
        self,
        quantized: np.ndarray,
        params: Optional[FeatureQuantizationParams] = None,
    ) -> np.ndarray:
        """Reconstruct approximate float features from quantized tokens.

        Inverse of quantize(): maps integer tokens back to approximate
        continuous feature values.

        Args:
            quantized: Quantized array of shape (N, D) int.
            params: Normalization params (uses fitted params if None).

        Returns:
            Reconstructed feature array of shape (N, D) float32.
        """
        p = params or self._params
        if p is None:
            raise ValueError("Must call fit() or provide params before dequantize()")

        # Map [0, levels-1] → [0, 1]
        normalized = quantized.astype(np.float32) / (self.levels - 1)

        # Denormalize to original range
        reconstructed = normalized * p.scale + p.min_vals

        return reconstructed

    def quantize_single(self, feature_vector: np.ndarray, params: Optional[FeatureQuantizationParams] = None) -> list[int]:
        """Quantize a single feature vector and return as list of ints.

        Convenience method for quantizing individual vectors.

        Args:
            feature_vector: 1D feature array of shape (D,).
            params: Normalization params.

        Returns:
            List of quantized integer values.
        """
        q = self.quantize(feature_vector.reshape(1, -1), params)
        return q[0].tolist()

    @property
    def fitted_params(self) -> Optional[FeatureQuantizationParams]:
        """Return the fitted quantization parameters, or None."""
        return self._params
