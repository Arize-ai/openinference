# /// script
# requires-python = ">=3.11"
# dependencies = ["griffe==2.4.1"]
# ///
"""
Extract the public API of the Python packages in `python/` for the site.

Run from `site/` with `pnpm python-api` (which runs `uv run scripts/python-api.py`).
It writes `src/generated/python-api.json`, which the pages under
`src/pages/python/` render as the Python API reference at `/python/`.

The packages are read with griffe (https://mkdocstrings.github.io/griffe/),
which parses the source statically: nothing is imported, so the libraries the
instrumentors patch (openai, langchain, ...) don't need to be installed.

What counts as public, per module:

- with `__all__`, the names it lists;
- without it, the names that don't start with `_` and are defined in the
  module, not imported into it.

A package is documented from its top-level module down through its public
submodules. An object re-exported by a module higher up is documented there
only, once.
"""

from __future__ import annotations

import json
import logging
import re
import sys
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import griffe
import tomllib

SITE = Path(__file__).resolve().parent.parent
REPO = SITE.parent
PYTHON = REPO / "python"
OUTPUT = SITE / "src" / "generated" / "python-api.json"

GITHUB = "https://github.com/Arize-ai/openinference"
BRANCH = "main"
PYPI = "https://pypi.org/project"

INSTRUMENTATION_PREFIX = "openinference-instrumentation-"

# Public names that are module plumbing, not API.
SKIPPED_NAMES = {"logger", "__version__"}
# Submodules that hold packaging details, shown in the page header instead.
SKIPPED_MODULES = {"version", "package"}
# Dunder methods worth documenting; the rest are protocol plumbing.
DOCUMENTED_DUNDERS = {"__call__"}

DOCSTRING_STYLE: Literal["google"] = "google"


# --- Packages -----------------------------------------------------------------


@dataclass
class Package:
    name: str
    directory: Path
    """The package's directory, e.g. `python/instrumentation/openinference-instrumentation-x`."""
    module: str
    """The import path of its top-level module, e.g. `openinference.instrumentation.openai`."""
    group: str
    """`core` or `instrumentation`."""
    project: dict[str, Any]
    """The `[project]` table of its `pyproject.toml`."""
    version_file: Path | None


def module_of(source: Path) -> str:
    """The deepest package under `src/`: `src/openinference/instrumentation/openai`."""
    path = source
    while True:
        children = [
            child
            for child in sorted(path.iterdir())
            if child.is_dir() and not child.name.startswith((".", "_"))
        ]
        if (path / "__init__.py").exists() or len(children) != 1:
            break
        path = children[0]
    return ".".join(path.relative_to(source).parts)


def discover() -> list[Package]:
    directories = [
        PYTHON / "openinference-instrumentation",
        PYTHON / "openinference-semantic-conventions",
        *sorted((PYTHON / "instrumentation").iterdir()),
    ]
    packages = []
    for directory in directories:
        pyproject = directory / "pyproject.toml"
        if not pyproject.exists() or not (directory / "src").is_dir():
            continue
        config = tomllib.loads(pyproject.read_text())
        project = config["project"]
        version_path = config.get("tool", {}).get("hatch", {}).get("version", {}).get("path")
        packages.append(
            Package(
                name=project["name"],
                directory=directory,
                module=module_of(directory / "src"),
                group="instrumentation"
                if project["name"].startswith(INSTRUMENTATION_PREFIX)
                else "core",
                project=project,
                version_file=directory / version_path if version_path else None,
            )
        )
    return packages


def version_of(package: Package) -> str | None:
    if package.version_file is None or not package.version_file.exists():
        return package.project.get("version")
    match = re.search(r"""__version__\s*=\s*["']([^"']+)["']""", package.version_file.read_text())
    return match.group(1) if match else None


def instrumentor_of(module: griffe.Module) -> dict[str, Any] | None:
    """
    The package's OpenTelemetry instrumentor: the class in its top-level module
    that subclasses `BaseInstrumentor`, which gives it `instrument()` and
    `uninstrument()`. Packages that ship a span processor instead have none.
    """
    for name, member in module.members.items():
        if name.startswith("_") or not isinstance(member, griffe.Class):
            continue
        if not any(str(base).endswith("BaseInstrumentor") for base in member.bases):
            continue
        instrument = member.members.get("_instrument")
        source = "\n".join(instrument.lines) if isinstance(instrument, griffe.Function) else ""
        return {
            "name": name,
            "path": member.path,
            # Whether `instrument()` reads a `config=TraceConfig(...)` keyword.
            "acceptsConfig": "config" in source,
        }
    return None


# --- Rendering helpers --------------------------------------------------------


def repo_path(path: Path) -> str:
    return path.resolve().relative_to(REPO).as_posix()


def source_url(obj: griffe.Object | griffe.Alias) -> str | None:
    if not isinstance(obj.filepath, Path):
        return None
    url = f"{GITHUB}/blob/{BRANCH}/{repo_path(obj.filepath)}"
    if obj.lineno and obj.kind is not griffe.Kind.MODULE:
        url += f"#L{obj.lineno}"
        if obj.endlineno and obj.endlineno != obj.lineno:
            url += f"-L{obj.endlineno}"
    return url


Part = str | dict[str, str]


def expression(value: str | griffe.Expr | None) -> list[Part] | None:
    """
    An annotation or default as text, with each name that resolves to an
    object kept as `{"text", "path"}`, so the site can link the ones it
    documents.
    """
    if value is None:
        return None
    if isinstance(value, str):
        return [value]
    parts: list[Part] = []
    for element in value.iterate(flat=True):
        if isinstance(element, str):
            if parts and isinstance(parts[-1], str):
                parts[-1] += element
            else:
                parts.append(element)
        elif isinstance(element, griffe.ExprName):
            try:
                path = element.canonical_path
            except Exception:
                path = element.name
            parts.append({"text": element.name, "path": path})
    return parts


def parameters(function: griffe.Function, *, drop_self: bool) -> list[dict[str, Any]]:
    result = []
    for index, parameter in enumerate(function.parameters):
        if drop_self and index == 0 and parameter.name in {"self", "cls"}:
            continue
        result.append(
            {
                "name": parameter.name,
                "kind": parameter.kind.value if parameter.kind else "positional or keyword",
                "annotation": expression(parameter.annotation),
                "default": expression(parameter.default),
            }
        )
    return result


def docstring(obj: griffe.Object | griffe.Alias) -> list[dict[str, Any]]:
    """The docstring as sections: text, parameters, returns, raises, examples..."""
    if not obj.docstring or not obj.docstring.value.strip():
        return []
    sections = []
    for section in obj.docstring.parse(DOCSTRING_STYLE):
        kind = section.kind.value
        value = section.value
        if kind == "text":
            sections.append({"kind": "text", "value": value})
        elif kind == "admonition":
            sections.append(
                {
                    "kind": "admonition",
                    "title": section.title or value.annotation or "Note",
                    "value": value.description,
                }
            )
        elif kind == "examples":
            sections.append(
                {
                    "kind": "examples",
                    "items": [
                        {"kind": item_kind.value, "value": text} for item_kind, text in value
                    ],
                }
            )
        elif kind == "deprecated":
            sections.append(
                {"kind": "deprecated", "version": value.version, "value": value.description}
            )
        elif isinstance(value, list):
            sections.append(
                {
                    "kind": kind,
                    "title": section.title,
                    "items": [
                        {
                            "name": getattr(item, "name", None),
                            "annotation": expression(getattr(item, "annotation", None)),
                            "default": expression(getattr(item, "default", None)),
                            "description": getattr(item, "description", ""),
                        }
                        for item in value
                    ],
                }
            )
    return sections


# --- Walking the API ----------------------------------------------------------


class Extractor:
    def __init__(self, loader: griffe.GriffeLoader) -> None:
        self.loader = loader
        # Every object documented so far, by path, so a re-export isn't
        # documented twice and annotations can link to it.
        self.documented: set[str] = set()
        # Path (including alias paths) -> (package, anchor).
        self.index: dict[str, tuple[str, str]] = {}
        self.package: Package | None = None

    def is_inside(self, obj: griffe.Object) -> bool:
        """Whether the object's source lives in the package being documented."""
        assert self.package is not None
        return isinstance(obj.filepath, Path) and obj.filepath.resolve().is_relative_to(
            (self.package.directory / "src").resolve()
        )

    def public_members(
        self, module: griffe.Module
    ) -> Iterator[tuple[str, griffe.Object | griffe.Alias]]:
        exports = module.exports
        names = (
            [name if isinstance(name, str) else name.name for name in exports]
            if exports is not None
            else [name for name in module.members if not name.startswith("_")]
        )
        for name in names:
            if name in SKIPPED_NAMES or name not in module.members:
                continue
            member = module.members[name]
            if exports is None and (member.is_alias or member.is_attribute):
                # Without `__all__`, module-level names (a logger, a TypeVar,
                # constants copied from the semantic conventions) are usually
                # incidental: document the classes and functions only.
                continue
            if isinstance(member, griffe.Alias):
                try:
                    if member.final_target.is_module:
                        continue
                except (griffe.AliasResolutionError, griffe.CyclicAliasError):
                    pass
            elif member.is_module:
                continue
            yield name, member

    def submodules(self, module: griffe.Module) -> Iterator[griffe.Module]:
        for name, member in module.members.items():
            if name.startswith("_") or name in SKIPPED_MODULES or member.is_alias:
                continue
            if isinstance(member, griffe.Module) and self.is_inside(member):
                yield member

    def module(self, module: griffe.Module) -> list[dict[str, Any]]:
        """This module's entry, then its public submodules', depth first."""
        members = []
        for name, member in self.public_members(module):
            entry = self.member(name, member, parent=module.path)
            if entry is not None:
                members.append(entry)
        result = []
        if members or module.path == (self.package and self.package.module):
            result.append(
                {
                    "path": module.path,
                    "docstring": docstring(module),
                    "source": source_url(module),
                    "members": members,
                }
            )
        for submodule in self.submodules(module):
            result.extend(self.module(submodule))
        return result

    def anchor(self, path: str, target: str) -> None:
        assert self.package is not None
        self.index.setdefault(path, (self.package.name, target))

    def member(
        self, name: str, member: griffe.Object | griffe.Alias, *, parent: str
    ) -> dict[str, Any] | None:
        path = f"{parent}.{name}"
        if isinstance(member, griffe.Alias):
            try:
                target = member.final_target
            except (griffe.AliasResolutionError, griffe.CyclicAliasError):
                # Re-exported from a package we don't read (e.g. OpenTelemetry).
                self.anchor(path, path)
                return {
                    "kind": "reexport",
                    "name": name,
                    "path": path,
                    "target": member.target_path,
                }
            if target.path in self.documented:
                # Link to where it's documented, which may be another
                # package's page (core, read first), not this one.
                self.index.setdefault(path, self.index[target.path])
                return None
            obj: griffe.Object = target
        else:
            obj = member
            if obj.path in self.documented:
                return None

        self.documented.add(obj.path)
        self.anchor(path, path)
        self.anchor(obj.path, path)
        entry = self.object(obj, name=name, path=path)
        if member.is_alias and obj.path != path:
            entry["canonical"] = obj.path
        return entry

    def object(self, obj: griffe.Object, *, name: str, path: str) -> dict[str, Any]:
        base: dict[str, Any] = {
            "name": name,
            "path": path,
            "docstring": docstring(obj),
            "source": source_url(obj),
        }
        if isinstance(obj, griffe.Class):
            return {**base, **self.klass(obj, path=path)}
        if isinstance(obj, griffe.Function):
            return {**base, **self.function(obj, drop_self=False)}
        if isinstance(obj, griffe.Attribute):
            return {**base, **attribute(obj)}
        if isinstance(obj, griffe.TypeAlias):
            return {**base, "kind": "type alias", "value": expression(obj.value)}
        return {**base, "kind": obj.kind.value}

    def function(self, function: griffe.Function, *, drop_self: bool) -> dict[str, Any]:
        labels = sorted(function.labels)
        kind = "property" if "property" in labels else "function"
        if kind == "property":
            return {
                "kind": "property",
                "labels": [label for label in labels if label != "property"],
                "annotation": expression(function.returns),
            }
        return {
            "kind": "method" if drop_self else "function",
            "labels": labels,
            "parameters": parameters(
                function, drop_self=drop_self and "staticmethod" not in labels
            ),
            "returns": expression(function.returns),
        }

    def klass(self, cls: griffe.Class, *, path: str) -> dict[str, Any]:
        bases = [expression(base) for base in cls.bases]
        # Show what a class inherits from private bases (they aren't
        # documented); a public base is linked instead.
        members: dict[str, griffe.Object | griffe.Alias] = {}
        for base in reversed(cls.mro()):
            if base.name.startswith("_") and self.is_inside(base):
                members.update(base.members)
        members.update(cls.members)

        init = members.get("__init__")
        signature = None
        if isinstance(init, griffe.Function):
            signature = parameters(init, drop_self=True)
        elif "dataclass" in cls.labels:
            signature = [
                {
                    "name": attr.name,
                    "kind": "positional or keyword",
                    "annotation": expression(attr.annotation),
                    "default": expression(attr.value),
                }
                for attr in cls.attributes.values()
                if not attr.name.startswith("_") and "class-attribute" not in attr.labels
            ] or None

        is_enum = any(
            isinstance(part, dict) and part["path"].startswith("enum.")
            for base in bases
            if base
            for part in base
        )

        children = []
        for name, child in members.items():
            if name.startswith("_") and name not in DOCUMENTED_DUNDERS:
                continue
            if child.is_alias:
                continue
            child_path = f"{path}.{name}"
            self.anchor(child_path, child_path)
            self.anchor(child.path, child_path)
            entry: dict[str, Any] = {
                "name": name,
                "path": child_path,
                "docstring": docstring(child),
                "source": source_url(child),
            }
            if isinstance(child, griffe.Function):
                entry.update(self.function(child, drop_self=True))
            elif isinstance(child, griffe.Attribute):
                entry.update(attribute(child))
            elif isinstance(child, griffe.Class):
                entry.update({"kind": "class", "bases": [expression(b) for b in child.bases]})
            else:
                continue
            children.append(entry)

        return {
            "kind": "enum" if is_enum else "class",
            "labels": sorted(cls.labels),
            "bases": bases,
            "parameters": signature,
            "members": children,
        }


def attribute(attr: griffe.Attribute) -> dict[str, Any]:
    value = expression(attr.value)
    # Long literal values (a dict of defaults, a regex) say little in a
    # signature line; keep them short.
    if value is not None:
        text = "".join(part if isinstance(part, str) else part["text"] for part in value)
        if len(text) > 120:
            value = [text[:117] + "..."]
    return {
        "kind": "attribute",
        "labels": sorted(attr.labels),
        "annotation": expression(attr.annotation),
        "value": value,
    }


# --- Main ---------------------------------------------------------------------


def main() -> int:
    # griffe logs a warning per docstring it can't fully parse (a parameter
    # documented but missing from the signature, and so on). They aren't
    # this script's to fix; keep the output to a summary.
    logging.getLogger("griffe").setLevel(logging.ERROR)

    packages = discover()
    loader = griffe.GriffeLoader(
        search_paths=[str(package.directory / "src") for package in packages],
        docstring_parser=DOCSTRING_STYLE,
    )
    for top in sorted({package.module.split(".")[0] for package in packages}):
        loader.load(top)
    loader.resolve_aliases(implicit=False, external=False)

    extractor = Extractor(loader)
    # Core first, so instrumentors' re-exports of core objects link to core.
    packages.sort(key=lambda package: (package.group != "core", package.name))
    output = []
    for package in packages:
        extractor.package = package
        module = loader.modules_collection[package.module]
        assert isinstance(module, griffe.Module)
        readme = package.directory / "README.md"
        instruments = package.project.get("optional-dependencies", {}).get("instruments", [])
        output.append(
            {
                "name": package.name,
                "group": package.group,
                "module": package.module,
                "description": package.project.get("description", ""),
                "version": version_of(package),
                "requiresPython": package.project.get("requires-python"),
                "instruments": instruments,
                "instrumentor": instrumentor_of(module),
                "pypi": f"{PYPI}/{package.name}/",
                "source": f"{GITHUB}/tree/{BRANCH}/{repo_path(package.directory)}",
                "readme": f"{GITHUB}/blob/{BRANCH}/{repo_path(readme)}"
                if readme.exists()
                else None,
                "modules": extractor.module(module),
            }
        )

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(
        json.dumps(
            {
                "packages": output,
                "index": {
                    path: {"package": name, "anchor": anchor}
                    for path, (name, anchor) in sorted(extractor.index.items())
                },
            },
            indent=1,
        )
        + "\n"
    )
    members = sum(len(module["members"]) for package in output for module in package["modules"])
    print(
        f"Wrote {len(output)} packages, {members} public objects to {OUTPUT.relative_to(SITE)}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
