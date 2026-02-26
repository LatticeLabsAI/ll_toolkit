"""Global settings and configuration for CADling.

This module provides configuration settings for the CADling toolkit,
including paths, device settings, logging configuration, and defaults.

Classes:
    CADlingSettings: Global settings singleton
    LoggingConfig: Logging configuration
    DeviceConfig: Device and acceleration configuration
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, List
from enum import Enum

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings


class LogLevel(str, Enum):
    """Logging levels."""

    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class DeviceType(str, Enum):
    """Device types for computation."""

    CPU = "cpu"
    CUDA = "cuda"
    MPS = "mps"  # Apple Metal
    AUTO = "auto"  # Auto-detect


class LoggingConfig(BaseModel):
    """Logging configuration.

    Attributes:
        level: Logging level
        format: Log message format
        date_format: Date format for timestamps
        file_path: Optional path to log file
        console_output: Whether to output to console
    """

    level: LogLevel = LogLevel.INFO
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    date_format: str = "%Y-%m-%d %H:%M:%S"
    file_path: Optional[Path] = None
    console_output: bool = True


class DeviceConfig(BaseModel):
    """Device and acceleration configuration.

    Attributes:
        device: Device type (cpu, cuda, mps, auto)
        gpu_id: GPU device ID (for multi-GPU systems)
        num_threads: Number of threads for CPU operations
        enable_mixed_precision: Enable mixed precision (FP16/BF16)
    """

    device: DeviceType = DeviceType.AUTO
    gpu_id: int = 0
    num_threads: Optional[int] = None
    enable_mixed_precision: bool = False


class PathsConfig(BaseModel):
    """Paths configuration.

    Attributes:
        cache_dir: Directory for caching intermediate results
        artifacts_dir: Directory for model artifacts
        temp_dir: Temporary directory for processing
    """

    cache_dir: Path = Field(default=Path.home() / ".cache" / "cadling")
    artifacts_dir: Path = Field(default=Path.home() / ".cadling" / "artifacts")
    temp_dir: Path = Field(default=Path("/tmp") / "cadling")

    def ensure_dirs(self):
        """Create directories if they don't exist."""
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        self.temp_dir.mkdir(parents=True, exist_ok=True)


class ProcessingConfig(BaseModel):
    """Processing configuration.

    Attributes:
        max_workers: Maximum number of parallel workers
        batch_size: Default batch size for processing
        enable_caching: Enable result caching
        cache_ttl_seconds: Cache time-to-live in seconds
    """

    max_workers: int = 4
    batch_size: int = 32
    enable_caching: bool = True
    cache_ttl_seconds: int = 3600


class CADlingSettings(BaseSettings):
    """Global CADling settings.

    This class provides a centralized configuration for the entire
    CADling toolkit. Settings can be loaded from environment variables
    or configuration files.

    Attributes:
        logging: Logging configuration
        device: Device and acceleration configuration
        paths: Paths configuration
        processing: Processing configuration
        allowed_formats: Allowed CAD formats (empty = all)
        enable_telemetry: Enable anonymous usage telemetry
    """

    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    device: DeviceConfig = Field(default_factory=DeviceConfig)
    paths: PathsConfig = Field(default_factory=PathsConfig)
    processing: ProcessingConfig = Field(default_factory=ProcessingConfig)

    # Format restrictions
    allowed_formats: List[str] = Field(default_factory=list)

    # Feature flags
    enable_telemetry: bool = False
    enable_experimental_features: bool = False

    model_config = {
        "env_prefix": "CADLING_",
        "env_nested_delimiter": "__",
        "case_sensitive": False,
    }

    def configure_logging(self):
        """Configure Python logging based on settings."""
        log_level = getattr(logging, self.logging.level.value)

        handlers = []

        # Console handler
        if self.logging.console_output:
            console_handler = logging.StreamHandler()
            console_handler.setLevel(log_level)
            formatter = logging.Formatter(
                self.logging.format,
                datefmt=self.logging.date_format,
            )
            console_handler.setFormatter(formatter)
            handlers.append(console_handler)

        # File handler
        if self.logging.file_path:
            file_handler = logging.FileHandler(self.logging.file_path)
            file_handler.setLevel(log_level)
            formatter = logging.Formatter(
                self.logging.format,
                datefmt=self.logging.date_format,
            )
            file_handler.setFormatter(formatter)
            handlers.append(file_handler)

        # Configure cadling logger (not root, to avoid overriding host app config)
        cadling_logger = logging.getLogger("cadling")
        cadling_logger.setLevel(log_level)
        for handler in handlers:
            cadling_logger.addHandler(handler)

    def get_device(self) -> str:
        """Get device string for PyTorch/frameworks.

        Returns:
            Device string (e.g., "cuda:0", "cpu", "mps")
        """
        if self.device.device == DeviceType.AUTO:
            # Auto-detect
            try:
                import torch

                if torch.cuda.is_available():
                    return f"cuda:{self.device.gpu_id}"
                elif torch.backends.mps.is_available():
                    return "mps"
                else:
                    return "cpu"
            except ImportError:
                return "cpu"

        elif self.device.device == DeviceType.CUDA:
            return f"cuda:{self.device.gpu_id}"
        elif self.device.device == DeviceType.MPS:
            return "mps"
        else:
            return "cpu"

    def initialize(self):
        """Initialize settings (create directories, configure logging, etc.)."""
        self.paths.ensure_dirs()
        self.configure_logging()


# Global settings instance
_settings: Optional[CADlingSettings] = None


def get_settings() -> CADlingSettings:
    """Get global settings instance.

    Returns:
        CADlingSettings singleton
    """
    global _settings
    if _settings is None:
        _settings = CADlingSettings()
        _settings.initialize()
    return _settings


def reset_settings():
    """Reset settings to defaults (useful for testing)."""
    global _settings
    _settings = None
