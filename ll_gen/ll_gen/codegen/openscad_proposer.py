"""OpenSCAD code generator wrapper for ll_gen.

This module wraps cadling's OpenSCADGenerator to produce typed CodeProposal
objects that can be executed and validated by the ll_gen pipeline.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional

from ll_gen.config import CodegenConfig, CodeLanguage
from ll_gen.proposals.code_proposal import CodeProposal

logger = logging.getLogger(__name__)

# Lazy import cadling
_CADLING_AVAILABLE = False
try:
    from cadling.generation.codegen.openscad_generator import OpenSCADGenerator

    _CADLING_AVAILABLE = True
except ImportError:
    logger.debug("cadling not available; OpenSCADProposer will raise on use")


class OpenSCADProposer:
    """Wraps cadling's OpenSCADGenerator to produce typed CodeProposal objects.

    This class manages the lifecycle of cadling's OpenSCADGenerator, handling
    both initial generation from prompts and repair generation for code that
    has failed validation or execution.

    Attributes:
        config: The CodegenConfig for model selection and API provider.
        generator: The underlying cadling OpenSCADGenerator instance.

    Raises:
        ImportError: If cadling is not installed when propose() is called.
    """

    def __init__(self, config: Optional[CodegenConfig] = None) -> None:
        """Initialize the OpenSCADProposer.

        Args:
            config: Optional CodegenConfig specifying model_name and
                api_provider. If None, uses defaults from CodegenConfig.

        Side-effects:
            If cadling is available, creates an OpenSCADGenerator instance.
        """
        self.config = config or CodegenConfig()
        self.generator = None

        if _CADLING_AVAILABLE:
            self.generator = OpenSCADGenerator(
                model_name=self.config.model_name,
                api_provider=self.config.api_provider,
            )

    def propose(
        self,
        prompt: str,
        image_path: Optional[Path] = None,
        error_context: Optional[Dict] = None,
        attempt: int = 1,
    ) -> CodeProposal:
        """Generate an OpenSCAD code proposal from a prompt.

        For the first attempt (attempt=1), generates code from scratch.
        For retry attempts (attempt>1), uses the repair endpoint with the
        previous code and error message.

        Args:
            prompt: The natural language prompt describing the part.
            image_path: Optional path to a reference image (JPEG/PNG).
            error_context: Dictionary with keys:
                - "old_code" (str): The previous code that failed
                - "error_message" (str): The error from execution
                For retry attempts.
            attempt: Attempt number (1=initial, 2+=retry). Used to
                select generation vs repair mode.

        Returns:
            A CodeProposal wrapping the generated code with:
            - language set to OPENSCAD
            - imports_required extracted from the code
            - syntax_valid pre-checked via heuristic validation

        Raises:
            ImportError: If cadling is not installed.
            ValueError: If error_context is missing required keys on retry.
        """
        if not _CADLING_AVAILABLE:
            raise ImportError(
                "cadling is not installed. Install with: pip install cadling"
            )

        if self.generator is None:
            raise RuntimeError("OpenSCADGenerator failed to initialize")

        generated_code = ""

        if attempt == 1:
            # Initial generation from prompt
            generated_code = self.generator.generate(
                prompt=prompt,
                image_path=image_path,
            )
        else:
            # Repair mode: use error context
            if not error_context:
                raise ValueError(
                    "error_context required for retry attempts (attempt > 1)"
                )

            old_code = error_context.get("old_code", "")
            error_message = error_context.get("error_message", "")

            if not old_code or not error_message:
                raise ValueError(
                    "error_context must contain 'old_code' and 'error_message'"
                )

            generated_code = self.generator.repair(
                old_code=old_code,
                error_msg=error_message,
            )

        # Wrap in CodeProposal and validate
        proposal = CodeProposal(
            code=generated_code,
            language=CodeLanguage.OPENSCAD,
        )
        proposal.validate_syntax()

        return proposal

    def propose_batch(
        self,
        prompt: str,
        num_candidates: int = 3,
        image_path: Optional[Path] = None,
    ) -> List[CodeProposal]:
        """Generate multiple candidate OpenSCAD code proposals.

        This method calls the generator multiple times to produce diverse
        candidates. Useful for downstream filtering or ranking.

        Args:
            prompt: The natural language prompt describing the part.
            num_candidates: Number of distinct candidates to generate.
            image_path: Optional path to a reference image.

        Returns:
            A list of CodeProposal objects, each with:
            - language set to OPENSCAD
            - syntax_valid pre-checked via heuristic validation
            - code_hash set for deduplication

        Raises:
            ImportError: If cadling is not installed.
        """
        if not _CADLING_AVAILABLE:
            raise ImportError(
                "cadling is not installed. Install with: pip install cadling"
            )

        if self.generator is None:
            raise RuntimeError("OpenSCADGenerator failed to initialize")

        candidates = []
        for _ in range(num_candidates):
            generated_code = self.generator.generate(
                prompt=prompt,
                image_path=image_path,
            )

            proposal = CodeProposal(
                code=generated_code,
                language=CodeLanguage.OPENSCAD,
            )
            proposal.validate_syntax()
            candidates.append(proposal)

        return candidates
