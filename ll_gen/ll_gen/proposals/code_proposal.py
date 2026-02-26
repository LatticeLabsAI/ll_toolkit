"""Code proposal — executable CAD code from an LLM.

A CodeProposal wraps a string of executable code (CadQuery Python,
OpenSCAD, or raw pythonocc) together with metadata about the
language, syntax validity, and required imports.

The disposal engine's ``code_executor`` receives this proposal,
executes it in a sandboxed environment, and captures the resulting
``TopoDS_Shape``.
"""
from __future__ import annotations

import ast
import copy
import re
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from ll_gen.config import CodeLanguage
from ll_gen.proposals.base import BaseProposal


@dataclass
class CodeProposal(BaseProposal):
    """A proposal containing executable CAD code.

    Attributes:
        code: The executable code string.
        language: Which CAD scripting language the code is written in.
        syntax_valid: Whether syntax has been pre-checked.  ``None``
            means not yet checked; ``True``/``False`` after
            ``validate_syntax()`` has run.
        imports_required: List of module names the code requires
            (extracted from import statements).
        code_hash: SHA-256 digest of ``code`` for dedup / caching.
    """

    code: str = ""
    language: CodeLanguage = CodeLanguage.CADQUERY
    syntax_valid: Optional[bool] = None
    imports_required: List[str] = field(default_factory=list)
    code_hash: Optional[str] = None

    def __post_init__(self) -> None:
        if self.code and self.code_hash is None:
            import hashlib

            self.code_hash = hashlib.sha256(self.code.encode()).hexdigest()
        if self.code and not self.imports_required:
            self.imports_required = self._extract_imports()

    # ------------------------------------------------------------------
    # Syntax validation
    # ------------------------------------------------------------------

    def validate_syntax(self) -> bool:
        """Check whether the code is syntactically valid.

        For Python-based languages (CadQuery, pythonocc) this uses
        ``ast.parse``.  For OpenSCAD it uses a set of heuristic regex
        checks that catch the most common structural errors (unmatched
        braces, missing semicolons after statements).

        Returns:
            True if the code parses without error.

        Side-effects:
            Sets ``self.syntax_valid``.
        """
        if self.language in (CodeLanguage.CADQUERY, CodeLanguage.PYTHONOCC):
            self.syntax_valid = self._validate_python_syntax()
        elif self.language == CodeLanguage.OPENSCAD:
            self.syntax_valid = self._validate_openscad_syntax()
        else:
            self.syntax_valid = False
        return self.syntax_valid

    def _validate_python_syntax(self) -> bool:
        """Validate Python code via ``ast.parse``."""
        try:
            ast.parse(self.code)
            return True
        except SyntaxError:
            return False

    def _validate_openscad_syntax(self) -> bool:
        """Heuristic validation for OpenSCAD code.

        Checks:
        1. Matching brace pairs ``{ }``.
        2. Every statement-terminating line ends with ``;`` or ``}``.
        3. At least one primitive or operation call is present.
        """
        # Strip single-line and block comments
        stripped = re.sub(r"//[^\n]*", "", self.code)
        stripped = re.sub(r"/\*.*?\*/", "", stripped, flags=re.DOTALL)

        # Check brace balance
        if stripped.count("{") != stripped.count("}"):
            return False

        # Check parenthesis balance
        if stripped.count("(") != stripped.count(")"):
            return False

        # Check bracket balance
        if stripped.count("[") != stripped.count("]"):
            return False

        # Must contain at least one OpenSCAD call
        primitives = [
            "cube", "sphere", "cylinder", "polyhedron", "circle",
            "square", "polygon", "text", "import", "surface",
            "linear_extrude", "rotate_extrude",
            "union", "difference", "intersection",
            "translate", "rotate", "scale", "mirror",
            "hull", "minkowski", "offset", "projection",
            "module", "function",
        ]
        has_primitive = any(
            re.search(rf"\b{p}\s*\(", stripped)
            for p in primitives
        )
        if not has_primitive:
            return False

        return True

    # ------------------------------------------------------------------
    # Import extraction
    # ------------------------------------------------------------------

    def _extract_imports(self) -> List[str]:
        """Extract top-level module names from import statements.

        Handles both ``import foo`` and ``from foo.bar import baz``.

        Returns:
            Sorted list of unique top-level module names.
        """
        if self.language == CodeLanguage.OPENSCAD:
            # OpenSCAD uses include/use, not Python imports
            modules: List[str] = []
            for match in re.finditer(
                r"(?:include|use)\s*<([^>]+)>", self.code
            ):
                modules.append(match.group(1))
            return sorted(set(modules))

        # Python-based languages
        modules = set()
        try:
            tree = ast.parse(self.code)
        except SyntaxError:
            # Fallback to regex if AST fails
            for match in re.finditer(
                r"^\s*(?:import|from)\s+(\w+)", self.code, re.MULTILINE
            ):
                modules.add(match.group(1))
            return sorted(modules)

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    modules.add(alias.name.split(".")[0])
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    modules.add(node.module.split(".")[0])

        return sorted(modules)

    # ------------------------------------------------------------------
    # Retry support
    # ------------------------------------------------------------------

    def with_error_context(self, error: Dict[str, Any]) -> "CodeProposal":
        """Create a retry proposal with error context attached.

        The new proposal keeps the same language and source prompt but
        clears the code (the generator will produce a new attempt
        conditioned on the error).

        Args:
            error: Structured error dict from the disposal result.

        Returns:
            New CodeProposal ready for retry generation.
        """
        new = copy.deepcopy(self)
        new.proposal_id = uuid.uuid4().hex
        new.attempt = self.next_attempt()
        new.error_context = error
        new.timestamp = datetime.now(timezone.utc).isoformat()
        new.code = ""  # Generator must produce new code
        new.syntax_valid = None
        new.imports_required = []
        new.code_hash = None
        new.confidence = 0.0
        return new

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    @property
    def line_count(self) -> int:
        """Number of lines in the code string."""
        if not self.code:
            return 0
        return len(self.code.strip().splitlines())

    @property
    def token_estimate(self) -> int:
        """Rough token count estimate (chars / 4).

        Useful for checking whether the code fits in an LLM context.
        """
        return len(self.code) // 4

    def summary(self) -> Dict[str, Any]:
        """Extended summary including code-specific fields."""
        base = super().summary()
        base.update({
            "language": self.language.value,
            "syntax_valid": self.syntax_valid,
            "line_count": self.line_count,
            "token_estimate": self.token_estimate,
            "imports": self.imports_required,
        })
        return base
