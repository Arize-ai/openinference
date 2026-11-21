#!/usr/bin/env python3
"""Judge the diagnostics of `weaver registry check --diagnostic-format json`.

The registry is checked against the shared OpenTelemetry policy package, which
has no way to exempt a single attribute from a single rule. A handful of
OpenInference attribute keys predate the registry and cannot be renamed, so the
accepted findings are listed, each with a reason, in `policies/exceptions.json`.

This script fails when:

- Weaver reports an error that is not a policy finding (e.g. a model that does
  not parse or resolve),
- a policy violation is not listed in the exceptions file, or
- an exception no longer matches any finding, so the list cannot go stale.

Usage:

    check_policies.py <diagnostics.json> <exceptions.json> [--weaver-exit-code N]

Standard library only, so it runs anywhere `python3` does.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


def load_diagnostics(path: Path) -> list[dict[str, Any]]:
    text = path.read_text().strip()
    if not text:
        return []
    diagnostics = json.loads(text)
    if not isinstance(diagnostics, list):
        sys.exit(f"check_policies: expected a JSON list in {path}")
    return diagnostics


def is_policy_finding(diagnostic: dict[str, Any]) -> bool:
    error = diagnostic.get("error")
    return isinstance(error, dict) and error.get("type") == "policy_violation"


def is_warning(diagnostic: dict[str, Any]) -> bool:
    return diagnostic.get("diagnostic", {}).get("severity") == "Warning"


def matches(exception: dict[str, Any], finding: dict[str, Any]) -> bool:
    if exception["id"] != finding.get("id"):
        return False
    context = finding.get("context") or {}
    return all(context.get(key) == value for key, value in exception.get("context", {}).items())


def describe(finding: dict[str, Any]) -> str:
    return f"{finding.get('id')}: {finding.get('message')}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("diagnostics", type=Path)
    parser.add_argument("exceptions", type=Path)
    parser.add_argument("--weaver-exit-code", type=int, default=0)
    args = parser.parse_args()

    diagnostics = load_diagnostics(args.diagnostics)
    if args.weaver_exit_code != 0 and not diagnostics:
        # Weaver never got as far as checking the registry (not installed, the
        # container could not start, ...). Its own error is on stderr above.
        print(f"FAIL: weaver did not run (exit code {args.weaver_exit_code}). See its error output above.")
        return 1
    exceptions: list[dict[str, Any]] = json.loads(args.exceptions.read_text())
    for exception in exceptions:
        if not exception.get("id") or not exception.get("reason"):
            sys.exit(f"check_policies: every exception needs an 'id' and a 'reason': {exception}")

    findings = [d["error"]["violation"] for d in diagnostics if is_policy_finding(d)]
    errors = [d for d in diagnostics if not is_policy_finding(d) and not is_warning(d)]
    warnings = sum(1 for d in diagnostics if is_warning(d))

    violations = [f for f in findings if f.get("level") == "violation"]
    advisories = [f for f in findings if f.get("level") != "violation"]
    unexpected = [f for f in violations if not any(matches(e, f) for e in exceptions)]
    accepted = [f for f in violations if any(matches(e, f) for e in exceptions)]
    # After a registry error the policies may not have run at all, so a missing
    # finding says nothing about whether an exception is still needed.
    stale = [] if errors else [e for e in exceptions if not any(matches(e, f) for f in violations)]

    failed = False
    if errors:
        failed = True
        print(f"FAIL: {len(errors)} registry error(s):")
        for error in errors:
            print(f"  - {error.get('diagnostic', {}).get('message', error)}")
    if unexpected:
        failed = True
        print(f"FAIL: {len(unexpected)} policy violation(s):")
        for finding in unexpected:
            print(f"  - {describe(finding)}")
    if stale:
        failed = True
        print(f"FAIL: {len(stale)} exception(s) in {args.exceptions} no longer match a finding; remove them:")
        for exception in stale:
            print(f"  - {exception['id']} {json.dumps(exception.get('context', {}))}")
    if args.weaver_exit_code != 0 and not (errors or violations):
        failed = True
        print(f"FAIL: weaver exited with code {args.weaver_exit_code} but reported no error or violation.")

    for finding in advisories:
        print(f"note ({finding.get('level')}): {describe(finding)}")
    if accepted:
        print(f"{len(accepted)} accepted exception(s), see {args.exceptions}:")
        for finding in accepted:
            print(f"  - {finding.get('id')} {json.dumps(finding.get('context', {}))}")
    print(f"{len(findings)} policy finding(s), {warnings} warning(s).")
    if failed:
        return 1
    print("OK: registry is valid and passes all policies.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
