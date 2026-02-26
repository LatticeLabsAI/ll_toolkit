"""Configuration dataclasses for ll_gen.

Follows the same @dataclass pattern used in ll_stepnet.config and
geotoken.config — plain dataclasses with sensible defaults, no
runtime imports from heavy libraries.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class GenerationRoute(str, Enum):
    """Which generation path to use."""

    CODE_CADQUERY = "code_cadquery"
    CODE_OPENSCAD = "code_openscad"
    CODE_PYTHONOCC = "code_pythonocc"
    NEURAL_VAE = "neural_vae"
    NEURAL_DIFFUSION = "neural_diffusion"
    NEURAL_VQVAE = "neural_vqvae"


class CodeLanguage(str, Enum):
    """Supported code generation languages."""

    CADQUERY = "cadquery"
    OPENSCAD = "openscad"
    PYTHONOCC = "pythonocc"


class ErrorCategory(str, Enum):
    """Neural-interpretable error categories.

    OpenCASCADE's BRepCheck_Analyzer reports 37 distinct error codes.
    These 6 categories collapse them into signals that an LLM or RL
    reward function can act on.
    """

    INVALID_PARAMS = "invalid_params"
    TOPOLOGY_ERROR = "topology_error"
    BOOLEAN_FAILURE = "boolean_failure"
    SELF_INTERSECTION = "self_intersection"
    DEGENERATE_SHAPE = "degenerate_shape"
    TOLERANCE_VIOLATION = "tolerance_violation"


class StepSchema(str, Enum):
    """STEP export application protocol."""

    AP203 = "AP203"
    AP214 = "AP214"
    AP242 = "AP242"


class ErrorSeverity(str, Enum):
    """Severity of a validation finding."""

    CRITICAL = "critical"
    WARNING = "warning"
    INFO = "info"


# ---------------------------------------------------------------------------
# Configuration dataclasses
# ---------------------------------------------------------------------------

@dataclass
class RoutingConfig:
    """Configuration for the generation router."""

    # Keywords that trigger code generation (Path A)
    mechanical_keywords: List[str] = field(default_factory=lambda: [
        "extrude", "cut", "hole", "fillet", "chamfer", "thread", "bore",
        "counterbore", "countersink", "slot", "pocket", "boss", "rib",
        "shell", "loft", "sweep", "revolve", "mirror", "pattern",
        "mounting", "bracket", "plate", "bolt", "nut", "washer",
        "flange", "housing", "enclosure", "gear", "shaft", "bearing",
        "hinge", "clamp", "spacer", "standoff", "bushing", "collar",
        "pin", "key", "keyway", "groove", "channel", "rail", "guide",
    ])

    # Keywords that trigger OpenSCAD specifically
    openscad_keywords: List[str] = field(default_factory=lambda: [
        "union", "difference", "intersection", "hull", "minkowski",
        "openscad", "scad",
    ])

    # Keywords that trigger neural generation (Path B)
    freeform_keywords: List[str] = field(default_factory=lambda: [
        "smooth", "flowing", "sculpted", "organic", "aerodynamic",
        "freeform", "curved", "biomorphic", "blob", "amorphous",
        "ergonomic", "contoured", "streamlined", "natural",
    ])

    # Keywords for latent space exploration
    exploration_keywords: List[str] = field(default_factory=lambda: [
        "interpolate", "morph", "vary", "explore", "blend",
        "transition", "mix", "combine", "latent", "sample",
    ])

    # Keywords for VQ-VAE codebook generation
    codebook_keywords: List[str] = field(default_factory=lambda: [
        "quantize", "discrete", "codebook", "disentangle",
    ])

    # Confidence threshold below which we fall back to CODE_CADQUERY
    confidence_threshold: float = 0.3

    # Default route when no keywords match
    default_route: GenerationRoute = GenerationRoute.CODE_CADQUERY


@dataclass
class CodegenConfig:
    """Configuration for code generation (Path A)."""

    # LLM backend settings
    model_name: str = "claude-sonnet-4-20250514"
    api_provider: str = "anthropic"
    max_tokens: int = 4096
    temperature: float = 0.2

    # Execution settings
    execution_timeout: int = 30  # seconds
    max_retries: int = 3

    # Which code backend to prefer
    default_backend: CodeLanguage = CodeLanguage.CADQUERY

    # Sandbox restrictions
    allowed_modules: List[str] = field(default_factory=lambda: [
        "cadquery", "math", "numpy",
    ])

    # Whether to include few-shot examples in the system prompt
    include_examples: bool = True
    max_example_tokens: int = 2000


@dataclass
class DisposalConfig:
    """Configuration for the deterministic disposal engine."""

    # Validation tolerances
    tolerance: float = 1e-7
    angular_tolerance: float = 1e-5

    # Repair settings
    enable_auto_repair: bool = True
    max_repair_passes: int = 3

    # ShapeFix configuration
    shapefix_precision: float = 1e-7
    shapefix_max_tolerance: float = 1e-3
    shapefix_min_tolerance: float = 1e-7

    # Fuzzy boolean tolerance escalation for BOPAlgo failures
    fuzzy_tolerance_steps: List[float] = field(default_factory=lambda: [
        1e-7, 1e-6, 1e-5, 1e-4, 1e-3,
    ])

    # Manifold checking
    check_manifoldness: bool = True
    check_euler: bool = True
    check_watertightness: bool = True
    check_self_intersection: bool = True

    # Whether to always compute GeometryReport
    always_introspect: bool = True


@dataclass
class ExportConfig:
    """Configuration for STEP/STL export."""

    step_schema: StepSchema = StepSchema.AP214
    stl_linear_deflection: float = 0.1
    stl_angular_deflection: float = 0.5
    stl_ascii: bool = False

    # Multi-view rendering for visual verification
    render_views: List[str] = field(default_factory=lambda: [
        "front", "top", "right", "isometric",
    ])
    render_resolution: int = 512


@dataclass
class FeedbackConfig:
    """Configuration for feedback and reward signals."""

    # Reward components and their weights
    # 5 tiers (shape_exists, manifold, watertight, euler_valid,
    # no_self_intersection) sum to 0.8, leaving 0.2 for bonus
    # dimensions match.  Maximum possible reward = 1.0.
    validity_reward: float = 0.8
    shape_constructed_reward: float = 0.16
    repairable_reward: float = 0.0
    per_tier_reward: float = 0.16
    semantic_match_reward: float = 0.2
    critical_error_penalty: float = -0.1

    # Dimensional match tolerance for semantic verification
    dimension_tolerance_pct: float = 0.10  # 10%


@dataclass
class DatasetConfig:
    """Configuration for research dataset loading."""

    # Base paths (can be HuggingFace IDs or local paths)
    deepcad_path: str = "latticelabs/deepcad"
    abc_path: str = "latticelabs/abc"
    text2cad_path: str = "latticelabs/text2cad"
    sketchgraphs_path: str = "latticelabs/sketchgraphs"

    # Loading settings
    streaming: bool = True
    shuffle: bool = True
    shuffle_buffer_size: int = 10000
    max_samples: Optional[int] = None

    # Tokenization
    max_commands: int = 60
    quantization_bits: int = 8
    normalization_range: float = 2.0

    # Splits
    train_split: str = "train"
    val_split: str = "validation"
    test_split: str = "test"


@dataclass
class ConditioningConfig:
    """Configuration for the conditioning layer."""

    text_model: str = "bert-base-uncased"
    image_model: str = "dino_vits16"
    conditioning_dim: int = 768
    freeze_encoders: bool = True
    fusion_method: str = "concat"  # concat, average, text_only, image_only
    image_size: int = 224


@dataclass
class GeneratorConfig:
    """Configuration for neural generators."""

    vae_checkpoint: Optional[str] = None
    diffusion_checkpoint: Optional[str] = None
    vqvae_checkpoint: Optional[str] = None
    default_temperature: float = 0.8
    diffusion_inference_steps: int = 50
    diffusion_eta: float = 0.0
    vqvae_codebook_dim: int = 512
    latent_dim: int = 256
    max_seq_len: int = 60


@dataclass
class TrainingConfig:
    """Configuration for RL alignment training."""

    learning_rate: float = 1e-5
    batch_size: int = 4
    num_epochs: int = 10
    eval_interval: int = 5
    baseline_decay: float = 0.99
    entropy_coeff: float = 0.01
    max_grad_norm: float = 1.0
    checkpoint_dir: str = "checkpoints"


@dataclass
class LLGenConfig:
    """Top-level configuration for the ll_gen package."""

    routing: RoutingConfig = field(default_factory=RoutingConfig)
    codegen: CodegenConfig = field(default_factory=CodegenConfig)
    disposal: DisposalConfig = field(default_factory=DisposalConfig)
    export: ExportConfig = field(default_factory=ExportConfig)
    feedback: FeedbackConfig = field(default_factory=FeedbackConfig)
    datasets: DatasetConfig = field(default_factory=DatasetConfig)
    conditioning: ConditioningConfig = field(default_factory=ConditioningConfig)
    generators: GeneratorConfig = field(default_factory=GeneratorConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)

    # Global settings
    max_retries: int = 3
    output_dir: str = "output"
    log_level: str = "INFO"
    device: str = "cpu"


def get_ll_gen_config(**overrides) -> LLGenConfig:
    """Create an LLGenConfig with optional overrides.

    Supports nested overrides via dotted keys:
        get_ll_gen_config(**{"codegen.temperature": 0.5})

    Args:
        **overrides: Key-value pairs to override defaults.
            Top-level keys are set directly on LLGenConfig.
            Dotted keys like "codegen.temperature" are set on
            the corresponding sub-config.

    Returns:
        Configured LLGenConfig instance.
    """
    config = LLGenConfig()
    nested: Dict[str, Dict[str, object]] = {}

    for key, value in overrides.items():
        if "." in key:
            section, attr = key.split(".", 1)
            nested.setdefault(section, {})[attr] = value
        else:
            if hasattr(config, key):
                setattr(config, key, value)

    for section, attrs in nested.items():
        sub = getattr(config, section, None)
        if sub is not None:
            for attr, value in attrs.items():
                if hasattr(sub, attr):
                    setattr(sub, attr, value)

    return config
