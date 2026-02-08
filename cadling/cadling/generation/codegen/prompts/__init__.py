"""Prompt templates for code generation.

This package contains system prompts, user prompt templates, and repair
prompts used by the CadQuery and OpenSCAD code generators.

Files:
    cadquery_system.txt: CadQuery API reference and constraints for LLM
    openscad_system.txt: OpenSCAD syntax reference for LLM
    text_to_code.txt: Template for text-to-code generation
    image_to_code.txt: Template for image-to-code generation
    repair.txt: Template for error-to-fix generation
"""

from __future__ import annotations

import logging
from pathlib import Path

_log = logging.getLogger(__name__)

_PROMPTS_DIR = Path(__file__).parent


def load_prompt(name: str) -> str:
    """Load a prompt template from disk.

    Args:
        name: Filename of the prompt template (e.g., 'cadquery_system.txt')

    Returns:
        Prompt template text

    Raises:
        FileNotFoundError: If prompt file does not exist
    """
    prompt_path = _PROMPTS_DIR / name
    if not prompt_path.exists():
        raise FileNotFoundError(f"Prompt template not found: {prompt_path}")
    text = prompt_path.read_text(encoding="utf-8")
    _log.debug("Loaded prompt template: %s (%d chars)", name, len(text))
    return text


def load_cadquery_system_prompt() -> str:
    """Load the CadQuery system prompt.

    Returns:
        CadQuery system prompt text
    """
    return load_prompt("cadquery_system.txt")


def load_openscad_system_prompt() -> str:
    """Load the OpenSCAD system prompt.

    Returns:
        OpenSCAD system prompt text
    """
    return load_prompt("openscad_system.txt")


def load_text_to_code_prompt() -> str:
    """Load the text-to-code user prompt template.

    Returns:
        Text-to-code prompt template
    """
    return load_prompt("text_to_code.txt")


def load_image_to_code_prompt() -> str:
    """Load the image-to-code user prompt template.

    This template is designed for image-primary generation where the
    reference image is the main input.  It contains structured analysis
    steps (form identification, feature decomposition, dimension estimation,
    modeling sequence planning) plus multi-view and image-type handling
    guidance.

    Placeholders:
        {IMAGE_DATA}: Replaced with the base64-encoded image and metadata.
        {DESCRIPTION}: Replaced with supplementary text (may be empty).

    Returns:
        Image-to-code prompt template text.
    """
    return load_prompt("image_to_code.txt")


def load_repair_prompt() -> str:
    """Load the repair/error-fix prompt template.

    Returns:
        Repair prompt template
    """
    return load_prompt("repair.txt")


__all__ = [
    "load_prompt",
    "load_cadquery_system_prompt",
    "load_openscad_system_prompt",
    "load_text_to_code_prompt",
    "load_image_to_code_prompt",
    "load_repair_prompt",
]
