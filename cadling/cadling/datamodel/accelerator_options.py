"""Hardware accelerator configuration options.

This module provides configuration options for hardware acceleration,
including GPU, TPU, and other accelerators for neural network inference
and computational geometry operations.

Classes:
    AcceleratorType: Types of hardware accelerators
    AcceleratorOptions: Configuration for hardware acceleration
"""

from __future__ import annotations

from enum import Enum
from typing import Optional, List

from pydantic import BaseModel, Field


class AcceleratorType(str, Enum):
    """Types of hardware accelerators."""

    CPU = "cpu"
    CUDA = "cuda"  # NVIDIA GPUs
    ROCM = "rocm"  # AMD GPUs
    MPS = "mps"  # Apple Metal Performance Shaders
    TPU = "tpu"  # Google TPUs
    AUTO = "auto"  # Auto-detect best available


class PrecisionType(str, Enum):
    """Precision types for computation."""

    FP32 = "fp32"  # 32-bit floating point
    FP16 = "fp16"  # 16-bit floating point (half precision)
    BF16 = "bf16"  # Brain float 16
    INT8 = "int8"  # 8-bit integer (quantized)
    MIXED = "mixed"  # Mixed precision (FP16/FP32)


class AcceleratorOptions(BaseModel):
    """Configuration options for hardware acceleration.

    Attributes:
        accelerator: Type of accelerator to use
        device_ids: List of device IDs (for multi-GPU)
        precision: Precision type for computation
        enable_cudnn_benchmark: Enable cuDNN autotuner (CUDA only)
        enable_tf32: Enable TF32 on Ampere GPUs (CUDA only)
        memory_limit_mb: Memory limit in megabytes
        num_threads: Number of CPU threads
        allow_growth: Allow GPU memory to grow as needed
        enable_xla: Enable XLA compilation (TensorFlow/JAX)
    """

    accelerator: AcceleratorType = AcceleratorType.AUTO
    device_ids: List[int] = Field(default_factory=lambda: [0])
    precision: PrecisionType = PrecisionType.FP32

    # CUDA-specific options
    enable_cudnn_benchmark: bool = True
    enable_tf32: bool = True

    # Memory management
    memory_limit_mb: Optional[int] = None
    allow_growth: bool = True

    # CPU options
    num_threads: Optional[int] = None

    # Compiler optimizations
    enable_xla: bool = False

    def get_torch_device(self) -> str:
        """Get PyTorch device string.

        Returns:
            Device string (e.g., "cuda:0", "cpu", "mps")
        """
        if self.accelerator == AcceleratorType.AUTO:
            try:
                import torch

                if torch.cuda.is_available():
                    return f"cuda:{self.device_ids[0]}"
                elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                    return "mps"
                else:
                    return "cpu"
            except ImportError:
                return "cpu"

        elif self.accelerator == AcceleratorType.CUDA:
            return f"cuda:{self.device_ids[0]}"
        elif self.accelerator == AcceleratorType.MPS:
            return "mps"
        else:
            return "cpu"

    def configure_torch(self):
        """Configure PyTorch with these accelerator options."""
        try:
            import torch

            # Set number of threads
            if self.num_threads is not None:
                torch.set_num_threads(self.num_threads)

            # CUDA-specific settings
            if self.accelerator in [AcceleratorType.CUDA, AcceleratorType.AUTO]:
                if torch.cuda.is_available():
                    # Enable cuDNN benchmark
                    torch.backends.cudnn.benchmark = self.enable_cudnn_benchmark

                    # Enable TF32
                    if hasattr(torch.backends.cuda, "matmul"):
                        torch.backends.cuda.matmul.allow_tf32 = self.enable_tf32
                        torch.backends.cudnn.allow_tf32 = self.enable_tf32

                    # Set memory limit
                    if self.memory_limit_mb is not None:
                        for device_id in self.device_ids:
                            torch.cuda.set_per_process_memory_fraction(
                                self.memory_limit_mb / (torch.cuda.get_device_properties(device_id).total_memory / 1024 / 1024),
                                device=device_id,
                            )

        except ImportError:
            pass  # PyTorch not available

    def get_precision_dtype(self):
        """Get PyTorch dtype for this precision setting.

        Returns:
            torch.dtype
        """
        try:
            import torch

            if self.precision == PrecisionType.FP32:
                return torch.float32
            elif self.precision == PrecisionType.FP16:
                return torch.float16
            elif self.precision == PrecisionType.BF16:
                return torch.bfloat16
            elif self.precision == PrecisionType.INT8:
                return torch.int8
            else:  # MIXED
                return torch.float16

        except ImportError:
            return None


class DistributedOptions(BaseModel):
    """Options for distributed training/inference.

    Attributes:
        enable_distributed: Enable distributed processing
        backend: Distributed backend ("nccl", "gloo", "mpi")
        world_size: Number of processes
        rank: Process rank
        master_addr: Master node address
        master_port: Master node port
    """

    enable_distributed: bool = False
    backend: str = "nccl"
    world_size: int = 1
    rank: int = 0
    master_addr: str = "localhost"
    master_port: int = 29500
