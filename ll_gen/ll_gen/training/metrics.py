"""Generation quality metrics for evaluating neural CAD generation.

Provides metrics for validity, compilability, coverage, and statistical
distance measures (MMD, JSD) between generated and reference point clouds.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np

from ll_gen.proposals.disposal_result import DisposalResult

_log = logging.getLogger(__name__)


@dataclass
class GenerationMetrics:
    """Aggregated generation quality metrics.

    Attributes:
        validity_rate: Fraction of proposals passing validation (0–1).
        compile_rate: Fraction of proposals that execute without error (0–1).
        coverage: Coverage of reference shape set via COV metric (0–1).
        mmd: Minimum Matching Distance between generated and reference sets.
        jsd: Jensen-Shannon Divergence between point distributions (0–1).
        mean_reward: Mean disposal reward signal across all samples.
        reward_std: Standard deviation of reward signal.
        num_samples: Total number of samples evaluated.
        num_valid: Count of valid samples.
        num_compiled: Count of successfully compiled samples.
    """

    validity_rate: float = 0.0
    compile_rate: float = 0.0
    coverage: float = 0.0
    mmd: float = 0.0
    jsd: float = 0.0
    mean_reward: float = 0.0
    reward_std: float = 0.0
    num_samples: int = 0
    num_valid: int = 0
    num_compiled: int = 0

    def summary(self) -> dict[str, Any]:
        """Generate a summary dict suitable for logging or JSON export.

        Returns:
            Dictionary with all metric values and counts.
        """
        return {
            "validity_rate": self.validity_rate,
            "compile_rate": self.compile_rate,
            "coverage": self.coverage,
            "mmd": self.mmd,
            "jsd": self.jsd,
            "mean_reward": self.mean_reward,
            "reward_std": self.reward_std,
            "num_samples": self.num_samples,
            "num_valid": self.num_valid,
            "num_compiled": self.num_compiled,
        }


class MetricsComputer:
    """Computes generation quality metrics from disposal results.

    Supports MMD (minimum matching distance), JSD (Jensen-Shannon divergence),
    coverage analysis, and reward aggregation.

    Attributes:
        num_bins: Number of histogram bins for JSD computation.
        kernel_bandwidth: Bandwidth for RBF kernel in MMD.
    """

    def __init__(self, num_bins: int = 64, kernel_bandwidth: float = 0.1) -> None:
        """Initialize the metrics computer.

        Args:
            num_bins: Number of histogram bins for JSD computation.
            kernel_bandwidth: Bandwidth parameter for RBF kernel in MMD.
        """
        self.num_bins = num_bins
        self.kernel_bandwidth = kernel_bandwidth

    def compute_validity_rate(self, results: list[DisposalResult]) -> float:
        """Compute fraction of valid samples.

        Args:
            results: List of disposal results.

        Returns:
            Validity rate in [0, 1]. Returns 0.0 if empty.
        """
        if not results:
            return 0.0
        valid_count = sum(1 for r in results if r.is_valid)
        return valid_count / len(results)

    def compute_compile_rate(self, results: list[DisposalResult]) -> float:
        """Compute fraction of samples that compiled (produced a shape).

        A shape is "compiled" if result.has_shape is True
        (i.e., result.shape is not None).

        Args:
            results: List of disposal results.

        Returns:
            Compile rate in [0, 1]. Returns 0.0 if empty.
        """
        if not results:
            return 0.0
        compiled_count = sum(1 for r in results if r.has_shape)
        return compiled_count / len(results)

    def compute_mmd(
        self,
        set1: list[np.ndarray],
        set2: list[np.ndarray],
        kernel: str = "rbf",
    ) -> float:
        """Compute Maximum Mean Discrepancy between two point cloud sets.

        MMD = E[k(x,x')] + E[k(y,y')] - 2*E[k(x,y)]

        Each element in set1/set2 is a point cloud (N, 3). Points are
        flattened to centroids for efficient pairwise comparison.

        Args:
            set1: List of point clouds (each (N, 3) array).
            set2: List of point clouds (each (M, 3) array).
            kernel: Kernel type ("rbf" or other).

        Returns:
            MMD value. Returns 0.0 if either set is empty.
        """
        if not set1 or not set2:
            return 0.0

        # Compute centroids for each point cloud
        centroids1 = np.array([pc.mean(axis=0) for pc in set1])
        centroids2 = np.array([pc.mean(axis=0) for pc in set2])

        # Sample up to 1000 pairs for efficiency
        sample_size = min(1000, len(centroids1) * len(centroids2))
        if sample_size == 0:
            return 0.0

        # Randomly sample pairs if needed
        if len(centroids1) * len(centroids2) > sample_size:
            indices1 = np.random.choice(
                len(centroids1), size=sample_size, replace=True
            )
            indices2 = np.random.choice(
                len(centroids2), size=sample_size, replace=True
            )
            pair1 = centroids1[indices1]
            pair2 = centroids2[indices2]
        else:
            pair1 = np.repeat(
                centroids1, len(centroids2), axis=0
            )
            pair2 = np.tile(centroids2, (len(centroids1), 1))

        if kernel == "rbf":
            if len(centroids1) * len(centroids2) > sample_size:
                # Sampled estimation — use sampled pairs for ALL three kernels
                k_11 = self._rbf_kernel(
                    centroids1[np.random.choice(len(centroids1), size=min(sample_size, len(centroids1)), replace=True)],
                    centroids1[np.random.choice(len(centroids1), size=min(sample_size, len(centroids1)), replace=True)]
                )
                k_22 = self._rbf_kernel(
                    centroids2[np.random.choice(len(centroids2), size=min(sample_size, len(centroids2)), replace=True)],
                    centroids2[np.random.choice(len(centroids2), size=min(sample_size, len(centroids2)), replace=True)]
                )
                k_12 = self._rbf_kernel(pair1, pair2)
            else:
                # Full pairwise estimation for all three kernels
                k_11 = self._rbf_kernel(centroids1, centroids1)
                k_22 = self._rbf_kernel(centroids2, centroids2)
                # Full cross-kernel
                pair1_full = np.repeat(centroids1, len(centroids2), axis=0)
                pair2_full = np.tile(centroids2, (len(centroids1), 1))
                k_12 = self._rbf_kernel(pair1_full, pair2_full)

            e_k11 = np.mean(k_11)
            e_k22 = np.mean(k_22)
            e_k12 = np.mean(k_12)

            mmd = e_k11 + e_k22 - 2.0 * e_k12
            return max(0.0, float(mmd))
        else:
            return 0.0

    def _rbf_kernel(self, x_data: np.ndarray, y_data: np.ndarray) -> np.ndarray:
        """Compute RBF kernel matrix.

        k(x,y) = exp(-||x-y||^2 / (2 * bandwidth^2))

        Args:
            x_data: (N, D) array.
            y_data: (M, D) array.

        Returns:
            (N, M) kernel matrix.
        """
        # Pairwise squared distances
        sq_dists = (
            np.sum(x_data ** 2, axis=1, keepdims=True)
            - 2.0 * np.dot(x_data, y_data.T)
            + np.sum(y_data ** 2, axis=1, keepdims=True).T
        )
        sq_dists = np.maximum(sq_dists, 0.0)
        sigma_sq = 2.0 * (self.kernel_bandwidth ** 2)
        return np.exp(-sq_dists / sigma_sq)

    def compute_jsd(
        self,
        dist1: np.ndarray,
        dist2: np.ndarray,
        num_bins: int | None = None,
    ) -> float:
        """Compute Jensen-Shannon Divergence between two point distributions.

        Computes 1D histograms per axis, averages JSD across axes.
        JSD = 0.5 * KL(P||M) + 0.5 * KL(Q||M) where M = 0.5*(P+Q)

        Args:
            dist1: (N, 3) point array.
            dist2: (M, 3) point array.
            num_bins: Number of histogram bins. Uses self.num_bins if None.

        Returns:
            JSD in [0, 1]. Returns 0.0 if either distribution is empty.
        """
        if dist1.size == 0 or dist2.size == 0:
            return 0.0

        if num_bins is None:
            num_bins = self.num_bins

        jsd_values = []

        # Compute JSD for each axis
        for axis in range(3):
            # Determine range
            min_val = min(dist1[:, axis].min(), dist2[:, axis].min())
            max_val = max(dist1[:, axis].max(), dist2[:, axis].max())

            if min_val == max_val:
                # Degenerate case — both distributions are constant
                jsd_values.append(0.0)
                continue

            # Histogram both distributions
            hist1, _ = np.histogram(
                dist1[:, axis], bins=num_bins, range=(min_val, max_val)
            )
            hist2, _ = np.histogram(
                dist2[:, axis], bins=num_bins, range=(min_val, max_val)
            )

            # Normalize to probabilities
            p = hist1.astype(np.float64) / hist1.sum()
            q = hist2.astype(np.float64) / hist2.sum()

            # Compute JSD with smoothing
            eps = 1e-10
            p = np.clip(p, eps, 1.0)
            q = np.clip(q, eps, 1.0)

            # M = 0.5 * (P + Q)
            m = 0.5 * (p + q)

            # KL divergence with log smoothing
            kl_pm = np.sum(p * (np.log(p) - np.log(m)))
            kl_qm = np.sum(q * (np.log(q) - np.log(m)))

            jsd = 0.5 * (kl_pm + kl_qm)
            jsd = min(1.0, max(0.0, jsd))  # Numerical safety
            jsd_values.append(jsd)

        # Average across axes and return
        avg_jsd = float(np.mean(jsd_values))
        return avg_jsd

    def compute_coverage(
        self,
        generated: list[np.ndarray],
        reference: list[np.ndarray],
        threshold: float = 0.05,
    ) -> float:
        """Compute coverage: fraction of reference shapes covered by generated.

        For each reference point cloud, check if any generated point cloud
        is within Chamfer distance threshold (computed on centroids).

        Args:
            generated: List of generated point clouds (each (N, 3)).
            reference: List of reference point clouds (each (M, 3)).
            threshold: Chamfer distance threshold for coverage.

        Returns:
            Coverage in [0, 1]. Returns 0.0 if either set is empty.
        """
        if not generated or not reference:
            return 0.0

        # Compute centroids
        gen_centroids = np.array([pc.mean(axis=0) for pc in generated])
        ref_centroids = np.array([pc.mean(axis=0) for pc in reference])

        # For each reference, check if covered by any generated
        covered = 0
        for ref_c in ref_centroids:
            # Minimum distance to any generated centroid
            distances = np.linalg.norm(gen_centroids - ref_c, axis=1)
            if np.min(distances) <= threshold:
                covered += 1

        return covered / len(reference)

    def compute_all(
        self,
        results: list[DisposalResult],
        reference_points: list[np.ndarray] | None = None,
    ) -> GenerationMetrics:
        """Compute all metrics at once.

        Extracts point clouds from results that have geometry_report with
        bounding_box, then calls individual compute_* methods.

        Args:
            results: List of disposal results.
            reference_points: Optional reference point clouds for coverage/MMD/JSD.

        Returns:
            Populated GenerationMetrics dataclass.
        """
        if not results:
            return GenerationMetrics()

        # Extract point clouds from results
        generated_points = []
        for r in results:
            if r.geometry_report and r.geometry_report.bounding_box:
                # Create a simple point cloud from bounding box
                # (in practice, you'd use actual geometry)
                bbox = r.geometry_report.bounding_box
                if len(bbox) >= 6:
                    # Create 8 corners of the bounding box
                    x_min, y_min, z_min, x_max, y_max, z_max = (
                        bbox[0], bbox[1], bbox[2],
                        bbox[3], bbox[4], bbox[5]
                    )
                    corners = np.array([
                        [x_min, y_min, z_min],
                        [x_min, y_min, z_max],
                        [x_min, y_max, z_min],
                        [x_min, y_max, z_max],
                        [x_max, y_min, z_min],
                        [x_max, y_min, z_max],
                        [x_max, y_max, z_min],
                        [x_max, y_max, z_max],
                    ])
                    generated_points.append(corners)

        # Compute scalar metrics
        validity_rate = self.compute_validity_rate(results)
        compile_rate = self.compute_compile_rate(results)

        # Compute reward statistics
        rewards = [r.reward_signal for r in results]
        mean_reward = float(np.mean(rewards)) if rewards else 0.0
        reward_std = float(np.std(rewards)) if rewards else 0.0

        # Compute distance metrics if reference is provided
        mmd = 0.0
        jsd = 0.0
        coverage = 0.0

        if reference_points and generated_points:
            mmd = self.compute_mmd(generated_points, reference_points)
            if reference_points[0].ndim == 2 and reference_points[0].shape[1] == 3:
                if generated_points:
                    ref_combined = np.vstack(reference_points)
                    gen_combined = np.vstack(generated_points)
                    jsd = self.compute_jsd(gen_combined, ref_combined)
            coverage = self.compute_coverage(generated_points, reference_points)

        return GenerationMetrics(
            validity_rate=validity_rate,
            compile_rate=compile_rate,
            coverage=coverage,
            mmd=mmd,
            jsd=jsd,
            mean_reward=mean_reward,
            reward_std=reward_std,
            num_samples=len(results),
            num_valid=sum(1 for r in results if r.is_valid),
            num_compiled=sum(1 for r in results if r.has_shape),
        )
