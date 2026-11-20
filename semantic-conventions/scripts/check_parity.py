#!/usr/bin/env python3
"""Verify that the machine-readable model has not drifted from the rest of the repo.

The model under `semantic-conventions/model` is the source of truth for
OpenInference attribute keys and well-known values. This check keeps the rest
of the repository from diverging from it. It compares the model against:

- `spec/attributes/semantic-conventions.md`: every attribute key in the tables
  and every row of the well-known value tables. Those tables are generated from
  the model (`make generate-spec`), so their contents cannot drift; what this
  catches is a model attribute that no table on the page selects.
- `python/openinference-semantic-conventions`: every attribute key constant and
  every enum value. That module is generated from the model too
  (`make generate-code`), so this guards the template: a class that stops
  selecting an attribute, or an enum that is not wired up.

Comparison is on wire-format values (attribute keys and enum values), never on
constant names. Differences that exist today are listed in `KNOWN_DRIFT` with
the side that is missing the value; the check fails on any difference that is
not listed, and on any listed difference that no longer exists.

Run from anywhere:

    uv run --no-project --with pyyaml python3 semantic-conventions/scripts/check_parity.py
"""

from __future__ import annotations

import ast
import re
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = REPO_ROOT / "semantic-conventions" / "model"
SPEC_ATTRIBUTES = REPO_ROOT / "spec" / "attributes" / "semantic-conventions.md"
PYTHON_TRACE = (
    REPO_ROOT / "python/openinference-semantic-conventions/src/openinference/semconv/trace/__init__.py"
)
PYTHON_RESOURCE = (
    REPO_ROOT / "python/openinference-semantic-conventions/src/openinference/semconv/resource/__init__.py"
)

# Spec sections that document OpenTelemetry's own attributes.
SPEC_UPSTREAM_SECTIONS = {"Exception"}

# Enum attribute in the model -> (heading of its value table in the spec, Python enum class).
# `None` means that side has no enumerated list to compare.
ENUMS: dict[str, tuple[str | None, str | None]] = {
    "openinference.span.kind": (None, "OpenInferenceSpanKindValues"),
    "input.mime_type": (None, "OpenInferenceMimeTypeValues"),
    "output.mime_type": (None, "OpenInferenceMimeTypeValues"),
    "annotation.annotator_kind": (None, "OpenInferenceAnnotatorKindValues"),
    "evaluation.annotator_kind": (None, "OpenInferenceAnnotatorKindValues"),
    "llm.system": ("LLM System Values", "OpenInferenceLLMSystemValues"),
    "llm.provider": ("LLM Provider Values", "OpenInferenceLLMProviderValues"),
    "decision.system": (None, "OpenInferenceDecisionSystemValues"),
    "decision.provider": (None, "OpenInferenceDecisionProviderValues"),
    "message_content.type": (None, None),
}

# Differences that exist today. Each entry is (what, value, side missing it).
# Fix the drift and delete the entry; do not add entries to make CI pass
# without a tracking issue.
KNOWN_DRIFT: set[tuple[str, str, str]] = {
    # Key prefixes exposed as Python constants. Not attributes in their own
    # right: the spec and model list the concrete `*_details.<kind>` keys.
    ("attribute", "llm.token_count.prompt_details", "model"),
    ("attribute", "llm.cost.prompt_details", "model"),
    ("attribute", "llm.cost.completion_details", "model"),
}


def split_row(line: str) -> list[str]:
    """Split a markdown table row on pipes that are not inside backticks."""
    cells: list[str] = []
    current = ""
    in_code = False
    for char in line.strip():
        if char == "`":
            in_code = not in_code
        if char == "|" and not in_code:
            cells.append(current.strip())
            current = ""
        else:
            current += char
    cells.append(current.strip())
    return cells[1:-1]


def load_model() -> tuple[set[str], set[str], dict[str, set[str]]]:
    """Return (keys defined in the model, upstream keys it references, enum values by key)."""
    defined: set[str] = set()
    referenced: set[str] = set()
    enums: dict[str, set[str]] = {}
    for path in sorted(MODEL_DIR.rglob("*.yaml")):
        if path.name == "manifest.yaml":
            continue
        document = yaml.safe_load(path.read_text()) or {}
        for attribute in document.get("attributes", []):
            defined.add(attribute["key"])
            if isinstance(attribute.get("type"), dict):
                enums[attribute["key"]] = {str(m["value"]) for m in attribute["type"]["members"]}
        signals = document.get("attribute_groups", []) + document.get("spans", [])
        for signal in signals:
            referenced.update(a["ref"] for a in signal.get("attributes", []) if "ref" in a)
    return defined, referenced - defined, enums


def load_spec() -> tuple[set[str], dict[str, set[str]]]:
    """Return (attribute keys in the spec tables, well-known values by table heading)."""
    keys: set[str] = set()
    values: dict[str, set[str]] = {}
    section = heading = ""
    for line in SPEC_ATTRIBUTES.read_text().splitlines():
        if line.startswith("## "):
            section = heading = line[3:].strip()
        elif line.startswith("### "):
            heading = line[4:].strip()
        if section in SPEC_UPSTREAM_SECTIONS or not line.startswith("| `"):
            continue
        cells = split_row(line)
        names = re.findall(r"`([^`]+)`", cells[0])
        if len(cells) == 5:  # Attribute | Stability | Type | Description | Examples
            keys.update(names)
        elif len(cells) == 3 and heading.endswith("Values"):  # Value | Description | Stability
            values.setdefault(heading, set()).update(names)
    return keys, values


def load_python() -> tuple[set[str], dict[str, set[str]]]:
    """Return (attribute key constants, enum values by class name) without importing the package."""
    keys: set[str] = set()
    enums: dict[str, dict[str, str]] = {}
    for path in (PYTHON_TRACE, PYTHON_RESOURCE):
        for node in ast.parse(path.read_text()).body:
            if not isinstance(node, ast.ClassDef):
                continue
            is_enum = any(isinstance(b, ast.Name) and b.id == "Enum" for b in node.bases)
            for statement in node.body:
                if not isinstance(statement, ast.Assign) or not isinstance(statement.targets[0], ast.Name):
                    continue
                name, value = statement.targets[0].id, statement.value
                if isinstance(value, ast.Constant) and isinstance(value.value, str):
                    resolved = value.value
                elif (  # e.g. `TYPESAFE = OpenInferenceLLMSystemValues.TYPESAFE.value`
                    isinstance(value, ast.Attribute)
                    and value.attr == "value"
                    and isinstance(value.value, ast.Attribute)
                    and isinstance(value.value.value, ast.Name)
                ):
                    resolved = enums[value.value.value.id][value.value.attr]
                else:
                    continue
                if is_enum:
                    enums.setdefault(node.name, {})[name] = resolved
                else:
                    keys.add(resolved)
    return keys, {name: set(members.values()) for name, members in enums.items()}


def main() -> int:
    for path in (MODEL_DIR, SPEC_ATTRIBUTES, PYTHON_TRACE, PYTHON_RESOURCE):
        if not path.exists():
            sys.exit(f"check_parity: missing {path}")

    model_defined, model_upstream, model_enums = load_model()
    model_keys = model_defined | model_upstream
    spec_keys, spec_values = load_spec()
    python_keys, python_enums = load_python()

    # (what, value, side missing it)
    drift: set[tuple[str, str, str]] = set()
    for key in model_keys - spec_keys:
        drift.add(("attribute", key, "spec"))
    for key in spec_keys - model_keys:
        drift.add(("attribute", key, "model"))
    for key in model_keys - python_keys:
        drift.add(("attribute", key, "python"))
    for key in python_keys - model_keys:
        drift.add(("attribute", key, "model"))

    for key in model_enums.keys() - ENUMS.keys():
        sys.exit(f"check_parity: add the enum attribute '{key}' to ENUMS in {Path(__file__).name}")
    for key, (spec_heading, python_class) in ENUMS.items():
        if key not in model_enums:
            sys.exit(f"check_parity: '{key}' is listed in ENUMS but is not an enum in the model")
        sides: dict[str, set[str]] = {"model": model_enums[key]}
        if spec_heading is not None:
            if spec_heading not in spec_values:
                sys.exit(f"check_parity: no '{spec_heading}' value table in {SPEC_ATTRIBUTES}")
            sides["spec"] = spec_values[spec_heading]
        if python_class is not None:
            if python_class not in python_enums:
                sys.exit(f"check_parity: no enum class '{python_class}' in {PYTHON_TRACE}")
            sides["python"] = python_enums[python_class]
        for side, values in sides.items():
            for value in model_enums[key] - values:
                drift.add((key, value, side))
            for value in values - model_enums[key]:
                drift.add((key, value, "model"))

    unexpected = sorted(drift - KNOWN_DRIFT)
    resolved = sorted(KNOWN_DRIFT - drift)
    print(
        f"model: {len(model_defined)} attributes defined, {len(model_upstream)} referenced from OpenTelemetry; "
        f"spec: {len(spec_keys)}; python: {len(python_keys)}; enums compared: {len(ENUMS)}"
    )
    if unexpected:
        print(f"FAIL: {len(unexpected)} difference(s) between the model, the spec, and the Python package:")
        for what, value, missing in unexpected:
            subject = f"attribute `{value}`" if what == "attribute" else f"`{what}` value `{value}`"
            print(f"  - {subject} is missing from the {missing}")
    if resolved:
        print(f"FAIL: {len(resolved)} KNOWN_DRIFT entries no longer apply; delete them from {Path(__file__).name}:")
        for entry in resolved:
            print(f"  - {entry}")
    if unexpected or resolved:
        return 1
    print(f"OK: no unexpected drift ({len(KNOWN_DRIFT)} known difference(s) tracked in KNOWN_DRIFT).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
