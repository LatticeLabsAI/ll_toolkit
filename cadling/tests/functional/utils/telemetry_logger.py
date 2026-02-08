"""Enhanced telemetry logger for functional tests.

Provides structured logging with metrics tracking, timing, and detailed output
for debugging functional test runs.
"""

import logging
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, Any, Optional, List, Generator
from datetime import datetime
import json


class TelemetryLogger:
    """Enhanced logger for functional tests with metrics tracking."""

    def __init__(
        self,
        test_name: str,
        output_dir: Path,
        console_level: int = logging.INFO,
        file_level: int = logging.DEBUG
    ):
        """Initialize telemetry logger.

        Args:
            test_name: Name of the test being run
            output_dir: Directory for log output
            console_level: Logging level for console output (default: INFO)
            file_level: Logging level for file output (default: DEBUG)
        """
        self.test_name = test_name
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Create logs subdirectory
        self.log_dir = self.output_dir / "logs"
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Setup logging
        self.logger = logging.getLogger(f"functional.{test_name}")
        self.logger.setLevel(logging.DEBUG)

        # File handler - write to logs/ subdirectory
        log_file = self.log_dir / f"{test_name}.log"
        fh = logging.FileHandler(log_file)
        fh.setLevel(file_level)

        # Console handler
        ch = logging.StreamHandler()
        ch.setLevel(console_level)

        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)

        self.logger.addHandler(fh)
        self.logger.addHandler(ch)

        # Metrics tracking
        self.metrics: Dict[str, Any] = {
            'test_name': test_name,
            'start_time': datetime.now().isoformat(),
            'phases': [],
            'timings': {},
            'errors': [],
            'warnings': []
        }

        # Current phase tracking
        self.current_phase: Optional[str] = None
        self.phase_start_time: Optional[float] = None

    def start_phase(self, phase_name: str):
        """Start tracking a new phase.

        Args:
            phase_name: Name of the phase
        """
        if self.current_phase:
            self.end_phase()

        self.current_phase = phase_name
        self.phase_start_time = time.time()
        self.logger.info(f"Starting phase: {phase_name}")

    def end_phase(self):
        """End the current phase and record timing."""
        if not self.current_phase or not self.phase_start_time:
            return

        duration = time.time() - self.phase_start_time
        self.metrics['timings'][self.current_phase] = duration
        self.metrics['phases'].append({
            'name': self.current_phase,
            'duration': duration
        })

        self.logger.info(
            f"Completed phase: {self.current_phase} ({duration:.2f}s)"
        )

        self.current_phase = None
        self.phase_start_time = None

    def log_metric(self, name: str, value: Any):
        """Log a metric value.

        Args:
            name: Metric name
            value: Metric value
        """
        self.metrics[name] = value
        self.logger.debug(f"Metric: {name} = {value}")

    def log_error(self, message: str, exception: Optional[Exception] = None):
        """Log an error.

        Args:
            message: Error message
            exception: Optional exception object
        """
        error_info = {
            'message': message,
            'timestamp': datetime.now().isoformat()
        }

        if exception:
            error_info['exception'] = str(exception)
            error_info['type'] = type(exception).__name__

        self.metrics['errors'].append(error_info)
        self.logger.error(message, exc_info=exception is not None)

    def log_warning(self, message: str):
        """Log a warning.

        Args:
            message: Warning message
        """
        warning_info = {
            'message': message,
            'timestamp': datetime.now().isoformat()
        }
        self.metrics['warnings'].append(warning_info)
        self.logger.warning(message)

    def log_info(self, message: str):
        """Log an info message.

        Args:
            message: Info message
        """
        self.logger.info(message)

    def log_debug(self, message: str):
        """Log a debug message.

        Args:
            message: Debug message
        """
        self.logger.debug(message)

    def finalize(self):
        """Finalize logging and save metrics."""
        if self.current_phase:
            self.end_phase()

        self.metrics['end_time'] = datetime.now().isoformat()

        # Calculate total duration
        start = datetime.fromisoformat(self.metrics['start_time'])
        end = datetime.fromisoformat(self.metrics['end_time'])
        self.metrics['total_duration'] = (end - start).total_seconds()

        # Save metrics to JSON in logs directory
        metrics_file = self.log_dir / f"{self.test_name}_metrics.json"
        with open(metrics_file, 'w') as f:
            json.dump(self.metrics, f, indent=2)

        self.logger.info(f"Test completed in {self.metrics['total_duration']:.2f}s")
        self.logger.info(f"Metrics saved to: {metrics_file}")

    def get_metrics(self) -> Dict[str, Any]:
        """Get current metrics.

        Returns:
            Dictionary of metrics
        """
        return dict(self.metrics)

    # Convenience method aliases for standard logging interface
    def info(self, message: str) -> None:
        """Alias for log_info.

        Args:
            message: Info message
        """
        self.log_info(message)

    def warning(self, message: str) -> None:
        """Alias for log_warning.

        Args:
            message: Warning message
        """
        self.log_warning(message)

    def debug(self, message: str) -> None:
        """Alias for log_debug.

        Args:
            message: Debug message
        """
        self.log_debug(message)

    def error(self, message: str, exc_info: bool = False) -> None:
        """Log error with optional exception info.

        Args:
            message: Error message
            exc_info: Whether to include exception info
        """
        if exc_info:
            self.logger.error(message, exc_info=True)
            # Also track in metrics
            error_info = {
                'message': message,
                'timestamp': datetime.now().isoformat(),
                'exc_info': True
            }
            self.metrics['errors'].append(error_info)
        else:
            self.log_error(message)

    @contextmanager
    def stage(self, stage_name: str) -> Generator[None, None, None]:
        """Context manager for stage tracking.

        Args:
            stage_name: Name of the stage

        Yields:
            None
        """
        self.start_phase(stage_name)
        try:
            yield
        finally:
            self.end_phase()

    def log_file_info(self, file_path: Path) -> None:
        """Log information about a file.

        Args:
            file_path: Path to the file
        """
        file_path = Path(file_path)
        if file_path.exists():
            size = file_path.stat().st_size
            self.logger.info(f"File: {file_path.name} ({size:,} bytes)")
        else:
            self.logger.warning(f"File not found: {file_path}")

    def log_feature_statistics(self, features, name: str) -> None:
        """Log statistics for numpy array features.

        Args:
            features: Numpy array of features
            name: Name for the feature set
        """
        import numpy as np

        if not hasattr(features, 'size') or features.size == 0:
            self.logger.info(f"{name}: empty array")
            return

        self.logger.debug(
            f"{name}: shape={features.shape}, "
            f"min={features.min():.4f}, max={features.max():.4f}, "
            f"mean={features.mean():.4f}, std={features.std():.4f}"
        )

    def save_telemetry(self) -> None:
        """Alias for finalize - saves telemetry data."""
        self.finalize()
