#!/usr/bin/env python3
"""Assert that every OpenInference policy rule still reports its violation.

`make check` proves that the model breaks no rule. It cannot tell a rule that
is satisfied from a rule that no longer fires, e.g. because a Weaver upgrade
changed the shape of the policy input. `tests/policies/violations` is a registry
that breaks each rule once, and `expected.json` beside it lists the findings.

This script fails when:

- an expected finding is not reported,
- a finding is reported that is not expected, or
- a rule in the policy directory has no expected finding, so that a new rule
  cannot be added without a fixture.

Usage:

    check_policy_fixtures.py <diagnostics.json> <expected.json> <policy-dir>

Standard library only.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any

RULE_ID = re.compile(r'"id":\s*"([a-z0-9_]+)"')


def normalize(finding: dict[str, Any]) -> str:
    kept = {"id": finding.get("id"), "context": finding.get("context") or {}}
    if finding.get("signal_name"):
        kept["signal_name"] = finding["signal_name"]
    return json.dumps(kept, sort_keys=True)


def main() -> int:
    if len(sys.argv) != 4:
        sys.exit(__doc__)
    diagnostics_path, expected_path, policy_dir = (Path(arg) for arg in sys.argv[1:])

    text = diagnostics_path.read_text().strip()
    diagnostics = json.loads(text) if text else []
    reported = {
        normalize(d["error"]["violation"])
        for d in diagnostics
        if isinstance(d.get("error"), dict) and d["error"].get("type") == "policy_violation"
    }
    expected_findings = json.loads(expected_path.read_text())
    expected = {normalize(finding) for finding in expected_findings}
    rules = {rule for policy in sorted(policy_dir.glob("*.rego")) for rule in RULE_ID.findall(policy.read_text())}
    if not rules:
        print(f"FAIL: found no policy rules in {policy_dir}")
        return 1

    failed = False
    for title, findings in (
        ("expected finding(s) not reported; the rule no longer fires", expected - reported),
        ("finding(s) reported but not expected", reported - expected),
    ):
        if findings:
            failed = True
            print(f"FAIL: {len(findings)} {title}:")
            for finding in sorted(findings):
                print(f"  - {finding}")
    uncovered = rules - {finding["id"] for finding in expected_findings}
    if uncovered:
        failed = True
        print(f"FAIL: {len(uncovered)} rule(s) in {policy_dir} have no fixture in {expected_path}:")
        for rule in sorted(uncovered):
            print(f"  - {rule}")
    if failed:
        return 1
    print(f"OK: all {len(rules)} OpenInference policy rule(s) report their violation.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
