#!/usr/bin/env python3
"""Generate API-reference MDX for each LatticeLabs package from Python docstrings.

This is the docs site's API-generation step (SPEC-2 M5 / FR-11..FR-14). It uses
griffe to **statically** parse each package (it never imports the packages, so no
torch / pythonocc / OpenMP is required), then renders one Markdown reference page
per package into ``site/src/content/docs/<pkg>/reference.md``.

Design notes:
  - Only the **public** API is documented: names a package exports at its top
    level (via ``__all__`` where present), filtered to symbols actually defined
    inside the package (so re-exports are kept but imported third-party names
    like ``numpy``/``torch`` are dropped).
  - ``ll_ocadr`` has an empty top-level ``__init__``; its public surface lives in
    explicit modules, so it is configured below.
  - The script is **fail-loud**: if a configured package fails to load or yields
    zero documented symbols, it exits non-zero so the build fails (FR-13).

Run: ``python3 scripts/gen_api.py`` (or ``npm run gen:api`` from ``site/``).
"""

from __future__ import annotations

import sys
from pathlib import Path

try:
    import griffe
except ImportError:  # pragma: no cover - clear message when the dep is missing
    sys.stderr.write(
        "error: griffe is not installed. Install it with `pip install griffe` "
        "(it is a pure-Python static analyzer; it does not import the packages).\n"
    )
    raise SystemExit(2)

REPO = Path(__file__).resolve().parents[2]
DOCS = REPO / "site" / "src" / "content" / "docs"
GITHUB_BLOB = "https://github.com/LatticeLabsAI/ll_toolkit/blob/main"

# (import_name, search_path relative to repo, docs dir name, explicit modules).
# An empty `modules` list means "document the package's top-level public surface".
PACKAGES = [
    ("cadling", "cadling", "cadling", []),
    ("stepnet", "ll_stepnet", "ll_stepnet", []),
    ("geotoken", "geotoken", "geotoken", []),
    (
        "ll_ocadr",
        ".",
        "ll_ocadr",
        [
            "run_ll_ocadr_hf",
            "vllm.latticelabs_ocadr",
            "vllm.lattice_encoder.geometry_net",
            "vllm.lattice_encoder.shape_net",
        ],
    ),
    ("ll_gen", "ll_gen", "ll_gen", []),
    ("ll_clouds", "ll_clouds", "ll_clouds", []),
]

# Human-readable package descriptions for the page frontmatter.
DESCRIPTIONS = {
    "cadling": "Public API reference for cadling, generated from docstrings.",
    "ll_stepnet": "Public API reference for ll_stepnet (the stepnet package), generated from docstrings.",
    "geotoken": "Public API reference for geotoken, generated from docstrings.",
    "ll_ocadr": "Public API reference for ll_ocadr, generated from docstrings.",
    "ll_gen": "Public API reference for ll_gen, generated from docstrings.",
    "ll_clouds": "Public API reference for ll_clouds, generated from docstrings.",
}


def esc(text: str) -> str:
    """Escape characters that markdown/MDX would otherwise treat as tags."""
    return text.replace("<", "&lt;").replace(">", "&gt;")


def resolve(member):
    """Return the concrete (non-alias) griffe object, or None if unresolvable."""
    try:
        if getattr(member, "is_alias", False):
            return member.final_target
        return member
    except Exception:
        return None


def in_package(obj, pkg_dir: Path) -> bool:
    """True if `obj` is defined inside the package source directory."""
    fp = getattr(obj, "filepath", None)
    if fp is None:
        return False
    try:
        return str(Path(fp)).startswith(str(pkg_dir))
    except Exception:
        return False


def source_link(obj) -> str:
    """Build a GitHub source link `file.py:line` for an object, or ''."""
    fp = getattr(obj, "filepath", None)
    lineno = getattr(obj, "lineno", None)
    if fp is None:
        return ""
    try:
        rel = Path(fp).resolve().relative_to(REPO)
    except Exception:
        return ""
    anchor = f"#L{lineno}" if lineno else ""
    label = f"{rel}{':' + str(lineno) if lineno else ''}"
    return f"[`{label}`]({GITHUB_BLOB}/{rel}{anchor})"


def format_signature(name: str, obj) -> str:
    """Build a readable call signature from a function/method's parameters."""
    parts = []
    for p in getattr(obj, "parameters", []) or []:
        if p.name in ("self", "cls"):
            continue
        kind = str(getattr(p, "kind", ""))
        prefix = ""
        if "var_positional" in kind:
            prefix = "*"
        elif "var_keyword" in kind:
            prefix = "**"
        token = f"{prefix}{p.name}"
        if prefix == "" and p.annotation is not None:
            token += f": {p.annotation}"
        if prefix == "" and p.default is not None:
            token += f" = {p.default}"
        parts.append(token)
    sig = f"{name}({', '.join(parts)})"
    returns = getattr(obj, "returns", None)
    if returns is not None:
        sig += f" -> {returns}"
    return sig


def render_docstring(obj) -> str:
    ds = getattr(obj, "docstring", None)
    if ds is None or not ds.value:
        return ""
    return esc(ds.value.strip())


def render_function(name: str, obj, level: str = "##") -> list[str]:
    out = [f"{level} `{name}`", ""]
    out.append("```python")
    out.append(format_signature(name, obj))
    out.append("```")
    link = source_link(obj)
    if link:
        out.append("")
        out.append(f"Source: {link}")
    doc = render_docstring(obj)
    if doc:
        out += ["", doc]
    out.append("")
    return out


def render_class(name: str, obj, pkg_dir: Path) -> list[str]:
    bases = getattr(obj, "bases", None) or []
    base_str = f"({', '.join(str(b) for b in bases)})" if bases else ""
    out = [f"## `class {name}{base_str}`", ""]
    link = source_link(obj)
    if link:
        out.append(f"Source: {link}")
        out.append("")
    doc = render_docstring(obj)
    if doc:
        out += [doc, ""]

    # Public methods (+ the constructor), defined on this class.
    methods = []
    for mname, member in getattr(obj, "members", {}).items():
        target = resolve(member)
        if target is None or "FUNCTION" not in str(target.kind):
            continue
        if mname != "__init__" and mname.startswith("_"):
            continue
        methods.append((mname, target))
    if methods:
        out += ["**Methods**", ""]
        for mname, target in methods:
            out += render_function(mname, target, level="###")
    return out


def collect_public(module, pkg_dir: Path):
    """Return (classes, functions) public symbols defined in the package."""
    classes, functions = [], []
    for name, member in module.members.items():
        if not getattr(member, "is_public", False):
            continue
        target = resolve(member)
        if target is None:
            continue
        if not in_package(target, pkg_dir):
            continue  # drop imported third-party names (numpy, torch, ...)
        kind = str(target.kind)
        if "CLASS" in kind:
            classes.append((name, target))
        elif "FUNCTION" in kind:
            functions.append((name, target))
    classes.sort(key=lambda x: x[0])
    functions.sort(key=lambda x: x[0])
    return classes, functions


def get_submodule(root, dotted: str):
    """Walk a dotted module path through griffe `.members`."""
    obj = root
    for part in dotted.split("."):
        obj = obj.members[part]
    return obj


def generate_one(import_name, search_rel, docs_name, modules) -> int:
    """Generate the reference page for one package. Returns symbol count."""
    search = REPO if search_rel == "." else REPO / search_rel
    root = griffe.load(import_name, search_paths=[str(search)])
    pkg_dir = Path(root.filepath).parent if root.filepath else (REPO / search_rel)

    targets = []
    if modules:
        for dotted in modules:
            try:
                targets.append((dotted, get_submodule(root, dotted)))
            except KeyError:
                sys.stderr.write(
                    f"error: {import_name}: module '{dotted}' not found\n"
                )
                raise SystemExit(1)
    else:
        targets.append((import_name, root))

    body: list[str] = []
    total = 0
    for dotted, module in targets:
        classes, functions = collect_public(module, pkg_dir)
        if not (classes or functions):
            continue
        if modules:
            body += [f"## Module `{import_name}.{dotted}`", ""]
        for name, obj in classes:
            body += render_class(name, obj, pkg_dir)
            total += 1
        for name, obj in functions:
            body += render_function(name, obj)
            total += 1

    if total == 0:
        sys.stderr.write(
            f"error: {import_name}: produced zero documented symbols\n"
        )
        raise SystemExit(1)

    out_dir = DOCS / docs_name
    out_dir.mkdir(parents=True, exist_ok=True)
    frontmatter = [
        "---",
        "title: API Reference",
        f"description: {DESCRIPTIONS[docs_name]}",
        "editUrl: false",
        "sidebar:",
        "  label: API Reference",
        "  order: 9",
        "---",
        "",
        "<!-- This page is GENERATED from Python docstrings by site/scripts/gen_api.py.",
        "     Do not edit by hand — change the source docstring instead. -->",
        "",
        f"Generated from the `{import_name}` package source. "
        f"Each symbol links to its definition on GitHub.",
        "",
    ]
    out_file = out_dir / "reference.md"
    out_file.write_text("\n".join(frontmatter + body).rstrip() + "\n")
    return total


def main() -> None:
    grand_total = 0
    for import_name, search_rel, docs_name, modules in PACKAGES:
        try:
            count = generate_one(import_name, search_rel, docs_name, modules)
        except SystemExit:
            raise
        except Exception as exc:  # fail loud on any unexpected error
            sys.stderr.write(
                f"error: failed to generate reference for {import_name}: "
                f"{type(exc).__name__}: {exc}\n"
            )
            raise SystemExit(1)
        print(f"  {docs_name}: {count} symbols → {docs_name}/reference.md")
        grand_total += count
    print(f"gen_api: generated {grand_total} symbols across {len(PACKAGES)} packages")


if __name__ == "__main__":
    main()
