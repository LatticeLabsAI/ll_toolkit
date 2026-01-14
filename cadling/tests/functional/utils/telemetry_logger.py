"""Enhanced telemetry logger for functional tests.

Provides structured logging with metrics tracking, timing, and detailed output
for debugging functional test runs.
"""

import logging
import time
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime
import json


class TelemetryLogger:
    """Enhanced logger for functional tests with metrics tracking."""

    def __init__(self, test_name: str, output_dir: Path):
        """Initialize telemetry logger.

        Args:
            test_name: Name of the test being run
            output_dir: Directory for log output
        """
        self.test_name = test_name
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Setup logging
        self.logger = logging.getLogger(f"functional.{test_name}")
        self.logger.setLevel(logging.DEBUG)

        # File handler
        log_file = self.output_dir / f"{test_name}.log"
        fh = logging.FileHandler(log_file)
        fh.setLevel(logging.DEBUG)

        # Console handler
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)

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

        # Save metrics to JSON
        metrics_file = self.output_dir / f"{self.test_name}_metrics.json"
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
