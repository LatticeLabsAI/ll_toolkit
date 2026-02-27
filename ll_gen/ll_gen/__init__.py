"""ll_gen — Generation orchestration for the LatticeLabs CAD toolkit.

Implements the "Neural Propose, Deterministic Dispose" architecture:
neural generators produce typed proposals, and a deterministic engine
(OpenCASCADE via pythonocc-core) validates, repairs, and exports them.

Quick start::

    from ll_gen import GenerationOrchestrator

    orchestrator = GenerationOrchestrator()
    result = orchestrator.generate(
        "A mounting bracket with 4 bolt holes, 80mm wide, 3mm thick"
    )
    if result.is_valid:
        print(f"STEP file: {result.step_path}")

Proposal types::

    from ll_gen.proposals import (
        CodeProposal,
        CommandSequenceProposal,
        LatentProposal,
    )

Disposal engine (standalone)::

    from ll_gen.disposal import DisposalEngine

    engine = DisposalEngine()
    result = engine.dispose(my_proposal)

Dataset loaders::

    from ll_gen.datasets import load_deepcad, load_text2cad
"""
from __future__ import annotations

__version__ = "0.1.0"

# ---------------------------------------------------------------------------
# Core configuration
# ---------------------------------------------------------------------------
from ll_gen.config import (
    CodegenConfig,
    CodeLanguage,
    ConditioningConfig,
    DatasetConfig,
    DisposalConfig,
    ErrorCategory,
    ErrorSeverity,
    ExportConfig,
    FeedbackConfig,
    GeneratorConfig,
    GenerationRoute,
    LLGenConfig,
    RoutingConfig,
    StepSchema,
    TrainingConfig,
    get_ll_gen_config,
)

# ---------------------------------------------------------------------------
# Proposal protocol
# ---------------------------------------------------------------------------
from ll_gen.proposals import (
    BaseProposal,
    CodeProposal,
    CommandSequenceProposal,
    DisposalResult,
    GeometryReport,
    LatentProposal,
    RepairAction,
    ValidationFinding,
)

# ---------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------
from ll_gen.routing import GenerationRouter, RoutingDecision

# ---------------------------------------------------------------------------
# Disposal engine
# ---------------------------------------------------------------------------
from ll_gen.disposal import DisposalEngine

# ---------------------------------------------------------------------------
# Code generation proposers
# ---------------------------------------------------------------------------
from ll_gen.codegen import CadQueryProposer, OpenSCADProposer

# ---------------------------------------------------------------------------
# Pipeline orchestration
# ---------------------------------------------------------------------------
from ll_gen.pipeline import GenerationOrchestrator, VisualVerifier

# ---------------------------------------------------------------------------
# Feedback
# ---------------------------------------------------------------------------
from ll_gen.feedback import (
    build_code_feedback,
    build_neural_feedback,
    build_training_feedback,
    compute_reward,
)

# ---------------------------------------------------------------------------
# Conditioning
# ---------------------------------------------------------------------------
from ll_gen.conditioning import (
    ConditioningEmbeddings,
    ConstraintPrediction,
    ConstraintPredictor,
    ConstraintType,
    ImageConditioningEncoder,
    MultiModalConditioner,
    TextConditioningEncoder,
)

# ---------------------------------------------------------------------------
# Neural generators
# ---------------------------------------------------------------------------
from ll_gen.generators import (
    BaseNeuralGenerator,
    LatentSampler,
    NeuralDiffusionGenerator,
    NeuralVAEGenerator,
    NeuralVQVAEGenerator,
)

# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
from ll_gen.training import (
    GenerationMetrics,
    MetricsComputer,
    RLAlignmentTrainer,
)

# ---------------------------------------------------------------------------
# Embeddings
# ---------------------------------------------------------------------------
from ll_gen.embeddings import HybridShapeEncoder

__all__ = [
    # Config
    "LLGenConfig",
    "CodegenConfig",
    "CodeLanguage",
    "ConditioningConfig",
    "DisposalConfig",
    "ExportConfig",
    "FeedbackConfig",
    "GeneratorConfig",
    "GenerationRoute",
    "RoutingConfig",
    "StepSchema",
    "TrainingConfig",
    "ErrorCategory",
    "ErrorSeverity",
    "DatasetConfig",
    "get_ll_gen_config",
    # Proposals
    "BaseProposal",
    "CodeProposal",
    "CommandSequenceProposal",
    "LatentProposal",
    "DisposalResult",
    "GeometryReport",
    "RepairAction",
    "ValidationFinding",
    # Routing
    "GenerationRouter",
    "RoutingDecision",
    # Disposal
    "DisposalEngine",
    # Codegen
    "CadQueryProposer",
    "OpenSCADProposer",
    # Pipeline
    "GenerationOrchestrator",
    "VisualVerifier",
    # Feedback
    "build_code_feedback",
    "build_neural_feedback",
    "build_training_feedback",
    "compute_reward",
    # Conditioning
    "ConditioningEmbeddings",
    "ConstraintPrediction",
    "ConstraintPredictor",
    "ConstraintType",
    "ImageConditioningEncoder",
    "MultiModalConditioner",
    "TextConditioningEncoder",
    # Neural generators
    "BaseNeuralGenerator",
    "LatentSampler",
    "NeuralDiffusionGenerator",
    "NeuralVAEGenerator",
    "NeuralVQVAEGenerator",
    # Training
    "GenerationMetrics",
    "MetricsComputer",
    "RLAlignmentTrainer",
    # Embeddings
    "HybridShapeEncoder",
]
