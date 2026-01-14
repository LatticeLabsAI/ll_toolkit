"""Synthetic data generation for CAD documents.

This package provides tools for generating synthetic training data
from CAD documents, including Q&A pairs for language model training.

Classes:
    CADQAGenerator: Generate Q&A pairs from CAD documents
    QAPair: Single question-answer pair with metadata
"""

from cadling.sdg.qa_generator import CADQAGenerator, QAPair

__all__ = [
    "CADQAGenerator",
    "QAPair",
]
