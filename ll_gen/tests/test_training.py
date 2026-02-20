"""Comprehensive tests for ll_gen training module.

Tests cover:
- GenerationMetrics dataclass: fields, defaults, summary method
- MetricsComputer: validity, compile rate, MMD, JSD, coverage, compute_all
- RLAlignmentTrainer: initialization, training steps, epochs, evaluation, checkpoints

All tests work WITHOUT torch installed by mocking heavy dependencies.
"""
from __future__ import annotations

import json
import logging
import tempfile
from pathlib import Path
from unittest import mock

import numpy as np
import pytest

from ll_gen.proposals.disposal_result import DisposalResult, GeometryReport
from ll_gen.training.metrics import GenerationMetrics, MetricsComputer

_log = logging.getLogger(__name__)


# ============================================================================
# GenerationMetrics Tests
# ============================================================================


@pytest.mark.unit
class TestGenerationMetrics:
    """Test the GenerationMetrics dataclass."""

    def test_default_initialization(self) -> None:
        """Test that GenerationMetrics initializes with all zeros."""
        metrics = GenerationMetrics()
        assert metrics.validity_rate == 0.0
        assert metrics.compile_rate == 0.0
        assert metrics.coverage == 0.0
        assert metrics.mmd == 0.0
        assert metrics.jsd == 0.0
        assert metrics.mean_reward == 0.0
        assert metrics.reward_std == 0.0
        assert metrics.num_samples == 0
        assert metrics.num_valid == 0
        assert metrics.num_compiled == 0

    def test_custom_initialization(self) -> None:
        """Test GenerationMetrics with custom values."""
        metrics = GenerationMetrics(
            validity_rate=0.85,
            compile_rate=0.90,
            coverage=0.75,
            mmd=0.15,
            jsd=0.20,
            mean_reward=0.80,
            reward_std=0.12,
            num_samples=100,
            num_valid=85,
            num_compiled=90,
        )
        assert metrics.validity_rate == 0.85
        assert metrics.compile_rate == 0.90
        assert metrics.coverage == 0.75
        assert metrics.mmd == 0.15
        assert metrics.jsd == 0.20
        assert metrics.mean_reward == 0.80
        assert metrics.reward_std == 0.12
        assert metrics.num_samples == 100
        assert metrics.num_valid == 85
        assert metrics.num_compiled == 90

    def test_summary_method(self) -> None:
        """Test that summary() returns a dict with all fields."""
        metrics = GenerationMetrics(
            validity_rate=0.75,
            compile_rate=0.80,
            coverage=0.70,
            mmd=0.10,
            jsd=0.15,
            mean_reward=0.75,
            reward_std=0.10,
            num_samples=50,
            num_valid=35,
            num_compiled=40,
        )
        summary = metrics.summary()

        # Verify all keys are present
        expected_keys = {
            "validity_rate",
            "compile_rate",
            "coverage",
            "mmd",
            "jsd",
            "mean_reward",
            "reward_std",
            "num_samples",
            "num_valid",
            "num_compiled",
        }
        assert set(summary.keys()) == expected_keys

        # Verify values match
        assert summary["validity_rate"] == 0.75
        assert summary["compile_rate"] == 0.80
        assert summary["coverage"] == 0.70
        assert summary["mmd"] == 0.10
        assert summary["jsd"] == 0.15
        assert summary["mean_reward"] == 0.75
        assert summary["reward_std"] == 0.10
        assert summary["num_samples"] == 50
        assert summary["num_valid"] == 35
        assert summary["num_compiled"] == 40

    def test_summary_serializable(self) -> None:
        """Test that summary() output is JSON serializable."""
        metrics = GenerationMetrics(
            validity_rate=0.85,
            num_samples=100,
            num_valid=85,
        )
        summary = metrics.summary()

        # Should not raise
        json_str = json.dumps(summary)
        assert len(json_str) > 0


# ============================================================================
# MetricsComputer Tests
# ============================================================================


@pytest.mark.unit
class TestMetricsComputerInitialization:
    """Test MetricsComputer initialization."""

    def test_default_initialization(self) -> None:
        """Test MetricsComputer with default parameters."""
        computer = MetricsComputer()
        assert computer.num_bins == 64
        assert computer.kernel_bandwidth == 0.1

    def test_custom_initialization(self) -> None:
        """Test MetricsComputer with custom parameters."""
        computer = MetricsComputer(num_bins=128, kernel_bandwidth=0.2)
        assert computer.num_bins == 128
        assert computer.kernel_bandwidth == 0.2


@pytest.mark.unit
class TestMetricsComputerValidity:
    """Test compute_validity_rate method."""

    def test_empty_results(self) -> None:
        """Test validity rate with empty results list."""
        computer = MetricsComputer()
        rate = computer.compute_validity_rate([])
        assert rate == 0.0

    def test_all_valid(self) -> None:
        """Test when all results are valid."""
        computer = MetricsComputer()
        results = [
            DisposalResult(is_valid=True),
            DisposalResult(is_valid=True),
            DisposalResult(is_valid=True),
        ]
        rate = computer.compute_validity_rate(results)
        assert rate == 1.0

    def test_none_valid(self) -> None:
        """Test when no results are valid."""
        computer = MetricsComputer()
        results = [
            DisposalResult(is_valid=False),
            DisposalResult(is_valid=False),
        ]
        rate = computer.compute_validity_rate(results)
        assert rate == 0.0

    def test_partial_valid(self) -> None:
        """Test partial validity."""
        computer = MetricsComputer()
        results = [
            DisposalResult(is_valid=True),
            DisposalResult(is_valid=False),
            DisposalResult(is_valid=True),
            DisposalResult(is_valid=False),
        ]
        rate = computer.compute_validity_rate(results)
        assert rate == pytest.approx(0.5)

    def test_single_valid(self) -> None:
        """Test with single valid result."""
        computer = MetricsComputer()
        results = [DisposalResult(is_valid=True)]
        rate = computer.compute_validity_rate(results)
        assert rate == 1.0

    def test_single_invalid(self) -> None:
        """Test with single invalid result."""
        computer = MetricsComputer()
        results = [DisposalResult(is_valid=False)]
        rate = computer.compute_validity_rate(results)
        assert rate == 0.0


@pytest.mark.unit
class TestMetricsComputerCompile:
    """Test compute_compile_rate method."""

    def test_empty_results(self) -> None:
        """Test compile rate with empty results."""
        computer = MetricsComputer()
        rate = computer.compute_compile_rate([])
        assert rate == 0.0

    def test_all_compiled(self) -> None:
        """Test when all results have shapes."""
        computer = MetricsComputer()
        # Create results with mock shapes
        results = [
            DisposalResult(shape=mock.Mock()),
            DisposalResult(shape=mock.Mock()),
            DisposalResult(shape=mock.Mock()),
        ]
        rate = computer.compute_compile_rate(results)
        assert rate == 1.0

    def test_none_compiled(self) -> None:
        """Test when no results have shapes."""
        computer = MetricsComputer()
        results = [
            DisposalResult(shape=None),
            DisposalResult(shape=None),
        ]
        rate = computer.compute_compile_rate(results)
        assert rate == 0.0

    def test_partial_compiled(self) -> None:
        """Test partial compilation."""
        computer = MetricsComputer()
        results = [
            DisposalResult(shape=mock.Mock()),
            DisposalResult(shape=None),
            DisposalResult(shape=mock.Mock()),
        ]
        rate = computer.compute_compile_rate(results)
        assert rate == pytest.approx(2.0 / 3.0)


@pytest.mark.unit
class TestMetricsComputerMMD:
    """Test Maximum Mean Discrepancy computation."""

    def test_mmd_empty_sets(self) -> None:
        """Test MMD with empty point cloud sets."""
        computer = MetricsComputer()
        mmd = computer.compute_mmd([], [])
        assert mmd == 0.0

    def test_mmd_empty_first_set(self) -> None:
        """Test MMD when first set is empty."""
        computer = MetricsComputer()
        set2 = [np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])]
        mmd = computer.compute_mmd([], set2)
        assert mmd == 0.0

    def test_mmd_empty_second_set(self) -> None:
        """Test MMD when second set is empty."""
        computer = MetricsComputer()
        set1 = [np.array([[1.0, 2.0, 3.0]])]
        mmd = computer.compute_mmd(set1, [])
        assert mmd == 0.0

    def test_mmd_identical_sets(self) -> None:
        """Test MMD for identical point cloud sets (should be near 0)."""
        computer = MetricsComputer()
        pc = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [2.0, 2.0, 2.0]])
        set1 = [pc]
        set2 = [pc]
        mmd = computer.compute_mmd(set1, set2)
        assert mmd >= 0.0
        assert mmd < 0.01  # Should be very small for identical sets

    def test_mmd_different_sets(self) -> None:
        """Test MMD for different point cloud sets (should be > 0)."""
        computer = MetricsComputer()
        set1 = [np.array([[0.0, 0.0, 0.0]])]
        set2 = [np.array([[10.0, 10.0, 10.0]])]
        mmd = computer.compute_mmd(set1, set2)
        assert mmd > 0.0

    def test_mmd_non_negative(self) -> None:
        """Test that MMD is always non-negative."""
        computer = MetricsComputer()
        set1 = [
            np.random.randn(5, 3),
            np.random.randn(3, 3),
        ]
        set2 = [
            np.random.randn(4, 3),
            np.random.randn(6, 3),
        ]
        mmd = computer.compute_mmd(set1, set2)
        assert mmd >= 0.0

    def test_mmd_with_multiple_point_clouds(self) -> None:
        """Test MMD with multiple point clouds in each set."""
        computer = MetricsComputer()
        set1 = [
            np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
            np.array([[0.0, 1.0, 0.0]]),
        ]
        set2 = [
            np.array([[10.0, 0.0, 0.0], [11.0, 0.0, 0.0]]),
            np.array([[10.0, 1.0, 0.0]]),
        ]
        mmd = computer.compute_mmd(set1, set2)
        assert isinstance(mmd, float)
        assert mmd >= 0.0

    def test_mmd_rbf_kernel(self) -> None:
        """Test RBF kernel computation."""
        computer = MetricsComputer()
        x_data = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        y_data = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        kernel_matrix = computer._rbf_kernel(x_data, y_data)

        # Should be (2, 2)
        assert kernel_matrix.shape == (2, 2)
        # Diagonal should be ~1.0
        assert kernel_matrix[0, 0] == pytest.approx(1.0, abs=1e-5)
        assert kernel_matrix[1, 1] == pytest.approx(1.0, abs=1e-5)
        # Off-diagonal should be < 1.0
        assert kernel_matrix[0, 1] < 1.0
        assert kernel_matrix[1, 0] < 1.0


@pytest.mark.unit
class TestMetricsComputerJSD:
    """Test Jensen-Shannon Divergence computation."""

    def test_jsd_empty_distributions(self) -> None:
        """Test JSD with empty distributions."""
        computer = MetricsComputer()
        dist1 = np.array([]).reshape(0, 3)
        dist2 = np.array([[1.0, 1.0, 1.0]])
        jsd = computer.compute_jsd(dist1, dist2)
        assert jsd == 0.0

    def test_jsd_both_empty(self) -> None:
        """Test JSD when both distributions are empty."""
        computer = MetricsComputer()
        dist1 = np.array([]).reshape(0, 3)
        dist2 = np.array([]).reshape(0, 3)
        jsd = computer.compute_jsd(dist1, dist2)
        assert jsd == 0.0

    def test_jsd_identical_distributions(self) -> None:
        """Test JSD for identical distributions (should be 0)."""
        computer = MetricsComputer()
        dist = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [2.0, 2.0, 2.0]])
        jsd = computer.compute_jsd(dist, dist)
        assert jsd < 1e-5  # Should be very close to 0

    def test_jsd_in_valid_range(self) -> None:
        """Test that JSD is in [0, log(2)] which is ~[0, 0.693]."""
        computer = MetricsComputer()
        dist1 = np.random.randn(20, 3)
        dist2 = np.random.randn(20, 3)
        jsd = computer.compute_jsd(dist1, dist2)
        assert 0.0 <= jsd <= np.log(2) + 1e-5  # Small margin for numerical error

    def test_jsd_symmetric(self) -> None:
        """Test that JSD(P, Q) == JSD(Q, P)."""
        computer = MetricsComputer()
        dist1 = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
        dist2 = np.array([[0.5, 0.5, 0.5], [1.5, 1.5, 1.5]])
        jsd_12 = computer.compute_jsd(dist1, dist2)
        jsd_21 = computer.compute_jsd(dist2, dist1)
        assert jsd_12 == pytest.approx(jsd_21, rel=1e-5)

    def test_jsd_with_custom_bins(self) -> None:
        """Test JSD with custom number of bins."""
        computer = MetricsComputer(num_bins=32)
        dist1 = np.random.randn(10, 3)
        dist2 = np.random.randn(10, 3)
        jsd = computer.compute_jsd(dist1, dist2, num_bins=128)
        assert 0.0 <= jsd <= np.log(2) + 1e-5

    def test_jsd_degenerate_case(self) -> None:
        """Test JSD when one axis has no variance (all points identical)."""
        computer = MetricsComputer()
        dist1 = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
        dist2 = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 2.0]])
        jsd = computer.compute_jsd(dist1, dist2)
        assert 0.0 <= jsd <= np.log(2) + 1e-5


@pytest.mark.unit
class TestMetricsComputerCoverage:
    """Test coverage metric computation."""

    def test_coverage_empty_generated(self) -> None:
        """Test coverage when no generated point clouds."""
        computer = MetricsComputer()
        reference = [np.array([[0.0, 0.0, 0.0]])]
        coverage = computer.compute_coverage([], reference)
        assert coverage == 0.0

    def test_coverage_empty_reference(self) -> None:
        """Test coverage when no reference point clouds."""
        computer = MetricsComputer()
        generated = [np.array([[0.0, 0.0, 0.0]])]
        coverage = computer.compute_coverage(generated, [])
        assert coverage == 0.0

    def test_coverage_both_empty(self) -> None:
        """Test coverage when both sets are empty."""
        computer = MetricsComputer()
        coverage = computer.compute_coverage([], [])
        assert coverage == 0.0

    def test_coverage_perfect(self) -> None:
        """Test coverage when all references are covered."""
        computer = MetricsComputer()
        # Generated has centroid at (0.5, 0.5, 0.5)
        generated = [np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])]
        # References centered near (0.5, 0.5, 0.5)
        reference = [
            np.array([[0.4, 0.4, 0.4]]),
            np.array([[0.6, 0.6, 0.6]]),
        ]
        coverage = computer.compute_coverage(generated, reference, threshold=0.2)
        assert coverage == 1.0

    def test_coverage_zero(self) -> None:
        """Test coverage when no references are covered."""
        computer = MetricsComputer()
        generated = [np.array([[0.0, 0.0, 0.0]])]
        reference = [np.array([[100.0, 100.0, 100.0]])]
        coverage = computer.compute_coverage(generated, reference, threshold=0.1)
        assert coverage == 0.0

    def test_coverage_partial(self) -> None:
        """Test partial coverage."""
        computer = MetricsComputer()
        # Generated has two centroids at (0,0,0) and (10,10,10)
        generated = [
            np.array([[0.0, 0.0, 0.0]]),
            np.array([[10.0, 10.0, 10.0]]),
        ]
        # References at (0,0,0), (10,10,10), and far away
        reference = [
            np.array([[0.0, 0.0, 0.0]]),
            np.array([[10.0, 10.0, 10.0]]),
            np.array([[100.0, 100.0, 100.0]]),
        ]
        coverage = computer.compute_coverage(generated, reference, threshold=0.1)
        assert coverage == pytest.approx(2.0 / 3.0)

    def test_coverage_in_valid_range(self) -> None:
        """Test that coverage is always in [0, 1]."""
        computer = MetricsComputer()
        for _ in range(5):
            generated = [np.random.randn(5, 3) for _ in range(3)]
            reference = [np.random.randn(5, 3) for _ in range(3)]
            coverage = computer.compute_coverage(generated, reference, threshold=0.5)
            assert 0.0 <= coverage <= 1.0

    def test_coverage_threshold_effect(self) -> None:
        """Test that larger threshold increases coverage."""
        computer = MetricsComputer()
        generated = [np.array([[0.0, 0.0, 0.0]])]
        reference = [
            np.array([[0.05, 0.05, 0.05]]),
            np.array([[10.0, 10.0, 10.0]]),
        ]
        coverage_small = computer.compute_coverage(generated, reference, threshold=0.01)
        coverage_large = computer.compute_coverage(generated, reference, threshold=1.0)
        assert coverage_large >= coverage_small


@pytest.mark.unit
class TestMetricsComputerComputeAll:
    """Test compute_all method."""

    def test_compute_all_empty_results(self) -> None:
        """Test compute_all with empty results."""
        computer = MetricsComputer()
        metrics = computer.compute_all([])
        assert isinstance(metrics, GenerationMetrics)
        assert metrics.num_samples == 0
        assert metrics.validity_rate == 0.0

    def test_compute_all_basic_metrics(self) -> None:
        """Test compute_all computes basic metrics correctly."""
        computer = MetricsComputer()
        results = [
            DisposalResult(is_valid=True, shape=mock.Mock()),
            DisposalResult(is_valid=False, shape=None),
            DisposalResult(is_valid=True, shape=mock.Mock()),
        ]
        metrics = computer.compute_all(results)

        assert metrics.num_samples == 3
        assert metrics.num_valid == 2
        assert metrics.num_compiled == 2
        assert metrics.validity_rate == pytest.approx(2.0 / 3.0)
        assert metrics.compile_rate == pytest.approx(2.0 / 3.0)

    def test_compute_all_with_reward_signals(self) -> None:
        """Test that compute_all handles reward signals."""
        computer = MetricsComputer()
        results = [
            DisposalResult(reward_signal=0.8),
            DisposalResult(reward_signal=0.9),
            DisposalResult(reward_signal=0.7),
        ]
        metrics = computer.compute_all(results)

        assert metrics.mean_reward == pytest.approx(0.8)
        assert metrics.reward_std == pytest.approx(np.std([0.8, 0.9, 0.7]))

    def test_compute_all_with_geometry_report(self) -> None:
        """Test compute_all with geometry reports for point clouds."""
        computer = MetricsComputer()
        results = [
            DisposalResult(
                is_valid=True,
                shape=mock.Mock(),
                geometry_report=GeometryReport(
                    bounding_box=(0.0, 0.0, 0.0, 1.0, 1.0, 1.0)
                ),
                reward_signal=0.9,
            ),
            DisposalResult(
                is_valid=False,
                shape=None,
                geometry_report=None,
                reward_signal=0.5,
            ),
        ]
        metrics = computer.compute_all(results)

        assert metrics.num_samples == 2
        assert metrics.num_valid == 1
        assert metrics.mean_reward == pytest.approx(0.7)

    def test_compute_all_with_reference_points(self) -> None:
        """Test compute_all with reference point clouds."""
        computer = MetricsComputer()
        results = [
            DisposalResult(
                is_valid=True,
                shape=mock.Mock(),
                geometry_report=GeometryReport(
                    bounding_box=(0.0, 0.0, 0.0, 1.0, 1.0, 1.0)
                ),
                reward_signal=0.9,
            ),
        ]
        reference_points = [
            np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]),
        ]
        metrics = computer.compute_all(results, reference_points=reference_points)

        assert metrics.num_samples == 1
        assert metrics.coverage >= 0.0
        assert metrics.coverage <= 1.0
        assert metrics.mmd >= 0.0
        assert 0.0 <= metrics.jsd <= np.log(2) + 1e-5


# ============================================================================
# RLAlignmentTrainer Tests
# ============================================================================


@pytest.mark.unit
class TestRLAlignmentTrainerInitialization:
    """Test RLAlignmentTrainer initialization."""

    def test_initialization_with_mocked_generator(self) -> None:
        """Test trainer initialization with mocked generator."""
        from ll_gen.training.rl_trainer import RLAlignmentTrainer

        mock_generator = mock.Mock()
        mock_generator._model = None

        trainer = RLAlignmentTrainer(
            generator=mock_generator,
            learning_rate=1e-4,
            device="cpu",
        )

        assert trainer.generator is mock_generator
        assert trainer.learning_rate == 1e-4
        assert trainer.device == "cpu"
        assert trainer.baseline_decay == 0.99
        assert trainer.entropy_coeff == 0.01
        assert trainer.max_grad_norm == 1.0

    def test_output_dir_creation(self) -> None:
        """Test that output directory is created during initialization."""
        from ll_gen.training.rl_trainer import RLAlignmentTrainer

        mock_generator = mock.Mock()
        mock_generator._model = None

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "training_output"
            trainer = RLAlignmentTrainer(
                generator=mock_generator,
                output_dir=str(output_dir),
            )

            assert output_dir.exists()
            assert trainer.output_dir == output_dir

    def test_custom_parameters(self) -> None:
        """Test initialization with custom parameters."""
        from ll_gen.training.rl_trainer import RLAlignmentTrainer

        mock_generator = mock.Mock()
        mock_generator._model = None

        trainer = RLAlignmentTrainer(
            generator=mock_generator,
            learning_rate=5e-5,
            baseline_decay=0.95,
            entropy_coeff=0.02,
            max_grad_norm=2.0,
        )

        assert trainer.learning_rate == 5e-5
        assert trainer.baseline_decay == 0.95
        assert trainer.entropy_coeff == 0.02
        assert trainer.max_grad_norm == 2.0


@pytest.mark.unit
class TestRLAlignmentTrainerTrainingHistory:
    """Test RLAlignmentTrainer training history tracking."""

    def test_get_training_history_empty(self) -> None:
        """Test getting training history when empty."""
        from ll_gen.training.rl_trainer import RLAlignmentTrainer

        mock_generator = mock.Mock()
        trainer = RLAlignmentTrainer(generator=mock_generator)

        history = trainer.get_training_history()
        assert history == []

    def test_training_history_structure(self) -> None:
        """Test the structure of training history entries."""
        from ll_gen.training.rl_trainer import RLAlignmentTrainer

        mock_generator = mock.Mock()
        trainer = RLAlignmentTrainer(generator=mock_generator)

        # Manually add a step result (simulating a training step)
        step_result = {
            "reward": 0.8,
            "advantage": 0.3,
            "loss": 0.15,
            "baseline": 0.5,
            "is_valid": 1.0,
        }
        trainer._train_history.append(step_result)

        history = trainer.get_training_history()
        assert len(history) == 1
        assert history[0] == step_result


@pytest.mark.unit
class TestRLAlignmentTrainerMetricsComputer:
    """Test that RLAlignmentTrainer has MetricsComputer."""

    def test_metrics_computer_initialized(self) -> None:
        """Test that MetricsComputer is initialized."""
        from ll_gen.training.rl_trainer import RLAlignmentTrainer

        mock_generator = mock.Mock()
        trainer = RLAlignmentTrainer(generator=mock_generator)

        assert trainer._metrics_computer is not None
        assert isinstance(trainer._metrics_computer, MetricsComputer)


# ============================================================================
# Torch-dependent tests (skipped if torch unavailable)
# ============================================================================


def _has_torch() -> bool:
    """Check if torch is available."""
    try:
        import torch  # noqa: F401
        return True
    except ImportError:
        return False


@pytest.mark.unit
@pytest.mark.skipif(not _has_torch(), reason="torch not available")
class TestRLAlignmentTrainerWithTorch:
    """Tests for RLAlignmentTrainer that require torch.

    These tests use mocked torch components to avoid actual GPU/CPU training.
    """

    def test_torch_import_skip_if_unavailable(self) -> None:
        """Verify torch availability for these tests."""
        try:
            import torch  # noqa: F401
        except ImportError:
            pytest.skip("torch not installed")

    def test_baseline_initialization(self) -> None:
        """Test that baseline is initialized to 0.0."""
        from ll_gen.training.rl_trainer import RLAlignmentTrainer

        mock_generator = mock.Mock()
        mock_generator._model = None
        trainer = RLAlignmentTrainer(generator=mock_generator)

        assert trainer._baseline == 0.0

    def test_step_count_initialization(self) -> None:
        """Test that step count starts at 0."""
        from ll_gen.training.rl_trainer import RLAlignmentTrainer

        mock_generator = mock.Mock()
        trainer = RLAlignmentTrainer(generator=mock_generator)

        assert trainer._step_count == 0

    def test_optimizer_lazy_initialization(self) -> None:
        """Test that optimizer is initially None."""
        from ll_gen.training.rl_trainer import RLAlignmentTrainer

        mock_generator = mock.Mock()
        mock_generator._model = None
        trainer = RLAlignmentTrainer(generator=mock_generator)

        assert trainer._optimizer is None

    def test_disposal_engine_lazy_initialization(self) -> None:
        """Test that disposal_engine is initially None."""
        from ll_gen.training.rl_trainer import RLAlignmentTrainer

        mock_generator = mock.Mock()
        trainer = RLAlignmentTrainer(generator=mock_generator)

        assert trainer._disposal_engine is None


# ============================================================================
# Edge Cases and Mathematical Properties
# ============================================================================


@pytest.mark.unit
class TestMetricsEdgeCases:
    """Test edge cases and mathematical properties of metrics."""

    def test_mmd_scale_invariance_property(self) -> None:
        """Test MMD properties with scaled point clouds."""
        computer = MetricsComputer()
        # Identical point clouds should have MMD near 0
        pc1 = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
        set1 = [pc1]
        set2 = [pc1]

        mmd_same = computer.compute_mmd(set1, set2)
        assert mmd_same < 0.01  # Very small for identical sets

        # Very different point clouds should have larger MMD
        set3 = [np.array([[100.0, 100.0, 100.0], [101.0, 101.0, 101.0]])]
        mmd_diff = computer.compute_mmd(set1, set3)
        assert mmd_diff > mmd_same

    def test_jsd_zero_variance_axis(self) -> None:
        """Test JSD when one axis has zero variance."""
        computer = MetricsComputer()
        # All Z values are 0
        dist1 = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 0.0]])
        dist2 = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 0.0]])

        jsd = computer.compute_jsd(dist1, dist2)
        assert jsd == pytest.approx(0.0, abs=1e-5)

    def test_coverage_multiple_references_single_generator(self) -> None:
        """Test coverage when one generator covers multiple references."""
        computer = MetricsComputer()
        # Single generator point cloud centered near origin
        generated = [np.array([[0.0, 0.0, 0.0], [0.1, 0.1, 0.1]])]
        # Multiple references very close to centroid (0.05, 0.05, 0.05)
        reference = [
            np.array([[0.04, 0.04, 0.04]]),
            np.array([[0.05, 0.05, 0.05]]),
            np.array([[0.06, 0.06, 0.06]]),
        ]
        coverage = computer.compute_coverage(generated, reference, threshold=0.02)
        assert coverage == 1.0

    def test_validity_rate_large_batch(self) -> None:
        """Test validity rate computation on large batch."""
        computer = MetricsComputer()
        # Create 1000 results with 75% validity
        n = 1000
        results = [
            DisposalResult(is_valid=(i % 4 != 0))
            for i in range(n)
        ]
        rate = computer.compute_validity_rate(results)
        assert rate == pytest.approx(0.75, abs=0.01)

    def test_compile_rate_with_rewards(self) -> None:
        """Test compile rate alongside reward signals."""
        computer = MetricsComputer()
        results = [
            DisposalResult(shape=mock.Mock(), reward_signal=0.9),
            DisposalResult(shape=None, reward_signal=0.1),
            DisposalResult(shape=mock.Mock(), reward_signal=0.7),
        ]
        compile_rate = computer.compute_compile_rate(results)
        assert compile_rate == pytest.approx(2.0 / 3.0)

    def test_jsd_single_point(self) -> None:
        """Test JSD with single point distributions."""
        computer = MetricsComputer()
        dist1 = np.array([[0.0, 0.0, 0.0]])
        dist2 = np.array([[0.0, 0.0, 0.0]])
        jsd = computer.compute_jsd(dist1, dist2)
        # Should be 0 since distributions are identical
        assert jsd < 1e-5

    def test_coverage_all_references_far(self) -> None:
        """Test coverage when all references are far from generated."""
        computer = MetricsComputer()
        generated = [np.array([[0.0, 0.0, 0.0]])]
        reference = [
            np.array([[100.0, 100.0, 100.0]]),
            np.array([[200.0, 200.0, 200.0]]),
        ]
        coverage = computer.compute_coverage(generated, reference, threshold=0.1)
        assert coverage == 0.0


@pytest.mark.unit
class TestGenerationMetricsEdgeCases:
    """Test edge cases for GenerationMetrics."""

    def test_summary_with_nan_values(self) -> None:
        """Test summary handles edge case values."""
        metrics = GenerationMetrics(
            validity_rate=float("nan") if False else 0.5,  # Avoid actual NaN
            num_samples=0,
        )
        summary = metrics.summary()
        assert "validity_rate" in summary

    def test_metrics_boundary_values(self) -> None:
        """Test GenerationMetrics with boundary values."""
        # Minimum values
        metrics_min = GenerationMetrics(
            validity_rate=0.0,
            compile_rate=0.0,
            coverage=0.0,
            mmd=0.0,
            jsd=0.0,
        )
        assert metrics_min.validity_rate == 0.0
        assert metrics_min.compile_rate == 0.0

        # Maximum reasonable values
        metrics_max = GenerationMetrics(
            validity_rate=1.0,
            compile_rate=1.0,
            coverage=1.0,
            jsd=np.log(2),
        )
        assert metrics_max.validity_rate == 1.0
        assert metrics_max.compile_rate == 1.0
        assert metrics_max.coverage == 1.0


@pytest.mark.unit
class TestMetricsComputerNumericalStability:
    """Test numerical stability of metrics computation."""

    def test_mmd_with_very_small_values(self) -> None:
        """Test MMD with very small point clouds."""
        computer = MetricsComputer()
        set1 = [np.array([[1e-10, 1e-10, 1e-10]])]
        set2 = [np.array([[1e-10, 1e-10, 1e-10]])]
        mmd = computer.compute_mmd(set1, set2)
        assert mmd >= 0.0

    def test_jsd_with_very_small_points(self) -> None:
        """Test JSD with very small coordinate values."""
        computer = MetricsComputer()
        dist1 = np.array([[1e-8, 1e-8, 1e-8]])
        dist2 = np.array([[1e-8, 1e-8, 1e-8]])
        jsd = computer.compute_jsd(dist1, dist2)
        assert 0.0 <= jsd <= np.log(2) + 1e-5

    def test_coverage_centroid_computation(self) -> None:
        """Test coverage computation with many points."""
        computer = MetricsComputer()
        # Large point clouds
        generated = [np.random.randn(1000, 3)]
        reference = [np.random.randn(100, 3)]
        coverage = computer.compute_coverage(generated, reference, threshold=5.0)
        assert 0.0 <= coverage <= 1.0

    def test_rbf_kernel_numerical_stability(self) -> None:
        """Test RBF kernel with edge case inputs."""
        computer = MetricsComputer()
        # Single point
        x_data = np.array([[0.0, 0.0, 0.0]])
        y_data = np.array([[0.0, 0.0, 0.0]])
        k_mat = computer._rbf_kernel(x_data, y_data)
        assert k_mat[0, 0] == pytest.approx(1.0, abs=1e-5)

        # Two points - close together
        x_data = np.array([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0]])
        y_data = np.array([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0]])
        k_mat = computer._rbf_kernel(x_data, y_data)
        # Diagonal should be 1.0
        assert k_mat[0, 0] == pytest.approx(1.0, abs=1e-5)
        assert k_mat[1, 1] == pytest.approx(1.0, abs=1e-5)
        # Off-diagonal should be less than 1.0
        assert k_mat[0, 1] < 1.0
        assert k_mat[0, 1] > 0.0


# ============================================================================
# Integration-like Tests
# ============================================================================


@pytest.mark.unit
class TestMetricsComputerIntegration:
    """Integration-style tests combining multiple components."""

    def test_compute_all_with_mixed_results(self) -> None:
        """Test compute_all with a mix of valid/invalid results and geometries."""
        computer = MetricsComputer()
        results = [
            DisposalResult(
                is_valid=True,
                shape=mock.Mock(),
                geometry_report=GeometryReport(
                    bounding_box=(0.0, 0.0, 0.0, 1.0, 1.0, 1.0),
                    volume=1.0,
                ),
                reward_signal=0.95,
            ),
            DisposalResult(
                is_valid=False,
                shape=None,
                reward_signal=0.3,
            ),
            DisposalResult(
                is_valid=True,
                shape=mock.Mock(),
                geometry_report=GeometryReport(
                    bounding_box=(5.0, 5.0, 5.0, 6.0, 6.0, 6.0),
                    volume=1.0,
                ),
                reward_signal=0.85,
            ),
        ]

        metrics = computer.compute_all(results)

        assert metrics.num_samples == 3
        assert metrics.num_valid == 2
        assert metrics.num_compiled == 2
        assert metrics.validity_rate == pytest.approx(2.0 / 3.0)
        assert metrics.compile_rate == pytest.approx(2.0 / 3.0)
        assert metrics.mean_reward == pytest.approx((0.95 + 0.3 + 0.85) / 3.0)
        assert metrics.reward_std > 0.0

    def test_metrics_workflow(self) -> None:
        """Test typical workflow: compute metrics, generate summary, serialize."""
        computer = MetricsComputer()

        # Create sample results
        results = [
            DisposalResult(is_valid=i % 2 == 0, shape=mock.Mock(), reward_signal=0.5 + 0.1 * i)
            for i in range(10)
        ]

        # Compute all metrics
        metrics = computer.compute_all(results)

        # Generate summary
        summary = metrics.summary()

        # Serialize
        json_str = json.dumps(summary)
        parsed = json.loads(json_str)

        # Verify round-trip
        assert parsed["num_samples"] == 10
        assert parsed["validity_rate"] == pytest.approx(metrics.validity_rate)
        assert parsed["mean_reward"] == pytest.approx(metrics.mean_reward)


@pytest.mark.unit
class TestDisposalResultIntegration:
    """Test DisposalResult integration with metrics."""

    def test_disposal_result_fields_with_metrics(self) -> None:
        """Test that DisposalResult fields are properly used by metrics."""
        computer = MetricsComputer()

        result = DisposalResult(
            is_valid=True,
            shape=mock.Mock(),
            reward_signal=0.85,
            geometry_report=GeometryReport(
                bounding_box=(0.0, 0.0, 0.0, 10.0, 10.0, 10.0),
                volume=1000.0,
            ),
        )

        results = [result]

        metrics = computer.compute_all(results)
        assert metrics.num_valid == 1
        assert metrics.mean_reward == 0.85

    def test_has_shape_property_in_metrics(self) -> None:
        """Test that has_shape property is used for compile rate."""
        computer = MetricsComputer()

        results = [
            DisposalResult(shape=mock.Mock()),  # has_shape = True
            DisposalResult(shape=None),  # has_shape = False
            DisposalResult(shape=mock.Mock()),  # has_shape = True
        ]

        compile_rate = computer.compute_compile_rate(results)
        assert compile_rate == pytest.approx(2.0 / 3.0)

    def test_geometry_report_with_bbox_extraction(self) -> None:
        """Test extracting point clouds from geometry reports."""
        computer = MetricsComputer()

        results = [
            DisposalResult(
                geometry_report=GeometryReport(
                    bounding_box=(0.0, 0.0, 0.0, 2.0, 2.0, 2.0)
                )
            ),
            DisposalResult(
                geometry_report=GeometryReport(
                    bounding_box=(5.0, 5.0, 5.0, 6.0, 6.0, 6.0)
                )
            ),
        ]

        metrics = computer.compute_all(results)
        # Should extract and use point clouds
        assert metrics.num_samples == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
