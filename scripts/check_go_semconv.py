#!/usr/bin/env python3
"""Verify that the Go semconv package covers every attribute key declared
in the canonical Python semconv source.

Fails fast (exit 1) if any string value declared as
`ATTR_CONSTANT = "value"` in
`python/openinference-semantic-conventions/src/openinference/semconv/trace/__init__.py`
does not appear as the right-hand side of some `Name = "value"` line in
the Go semconv package files.

We compare on the *values* (the wire-format keys), not the constant
names, because Go and Python use different naming conventions
(CamelCase vs ALL_CAPS) for the same constants. The wire-format value
is the thing the collector cares about and the thing that must not
drift across SDKs.

Run from the repo root:

    python3 scripts/check_go_semconv.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
PYTHON_SOURCE = (
    REPO_ROOT
    / "python"
    / "openinference-semantic-conventions"
    / "src"
    / "openinference"
    / "semconv"
    / "trace"
    / "__init__.py"
)
GO_SOURCES = [
    REPO_ROOT / "go" / "openinference-semantic-conventions" / "attributes.go",
    REPO_ROOT / "go" / "openinference-semantic-conventions" / "enums.go",
]

# Python value patterns we extract:
#   FOO = "bar.baz"
#   FOO: SomeType = "bar.baz"
# We deliberately accept only top-level / class-level string assignments
# (no f-strings, no concatenations) — anything more dynamic isn't a
# semconv "constant" worth pinning.
PY_ASSIGN = re.compile(r'^\s+([A-Z][A-Z0-9_]*)\s*(?::[^=]+)?=\s*"([^"]+)"')

# Go pattern is simpler: `Name = "value"` or `Name SomeType = "value"`
# inside a const block. We grep all such lines across the listed files.
GO_ASSIGN = re.compile(r'\s+([A-Z][A-Za-z0-9_]*)\s*(?:[A-Za-z][A-Za-z0-9_]*\s*)?=\s*"([^"]+)"')


def extract_values(path: Path, pattern: re.Pattern[str]) -> set[str]:
    if not path.is_file():
        sys.exit(f"check_go_semconv: missing source file {path}")
    out: set[str] = set()
    for line in path.read_text().splitlines():
        m = pattern.search(line)
        if m:
            out.add(m.group(2))
    return out


def main() -> int:
    py_values = extract_values(PYTHON_SOURCE, PY_ASSIGN)
    go_values: set[str] = set()
    for go_src in GO_SOURCES:
        go_values |= extract_values(go_src, GO_ASSIGN)

    missing = py_values - go_values
    if missing:
        print(
            "Go semconv is missing the following wire-format keys "
            "that the Python source declares:",
            file=sys.stderr,
        )
        for v in sorted(missing):
            print(f"  - {v}", file=sys.stderr)
        print(
            "\nAdd the constants in go/openinference-semantic-conventions/attributes.go or .../enums.go, "
            "then re-run this script.",
            file=sys.stderr,
        )
        return 1

    # Inverse direction is informational: Go may declare values that
    # don't yet exist in Python (e.g., during a parity gap). We report
    # them but do NOT fail.
    extras = go_values - py_values
    if extras:
        print(
            f"info: {len(extras)} Go value(s) not present in Python "
            "(probably intentional — Go-side additions land first):"
        )
        for v in sorted(extras):
            print(f"  + {v}")

    print(
        f"OK — {len(py_values)} Python wire-format keys all present "
        f"in Go semconv ({len(go_values)} total Go constants)."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
