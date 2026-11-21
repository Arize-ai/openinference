#!/usr/bin/env python3
"""Assert what `weaver registry live-check` says about sample attribute keys.

OpenInference flattens lists of objects into one attribute per leaf, so the keys
on the wire (`llm.input_messages.0.message.role`) are not the keys in the model
(`llm.input_messages`). The model declares list attributes as key prefixes so
that Weaver matches the flattened keys. This test pins that behavior: every key
in `tests/live-check/known.json` must resolve to a registry attribute, and every
key in `tests/live-check/unknown.json` must be reported as missing.

Matching a prefix is all Weaver does for those attributes: it does not look at
what follows the prefix. `tests/live-check/unvalidated.json` documents that gap
with keys that have a valid prefix and a leaf that is misspelled, has the wrong
type, or is not an item of that list. They are accepted today. When leaves are
validated (https://github.com/Arize-ai/openinference/issues/4419) this check
fails, and those keys move to `unknown.json`.

Usage:

    check_live.py known|unknown|unvalidated <live-check-output.json>

Standard library only.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Iterator

MISSING = "missing_attribute"

# The only advice Weaver gives for a key it matched to a key prefix.
PREFIX_MATCH_ADVICE = {"template_attribute", "not_stable"}


def documents(text: str) -> Iterator[Any]:
    """Weaver writes one or more JSON documents back to back."""
    decoder = json.JSONDecoder()
    position = 0
    while position < len(text):
        if text[position].isspace():
            position += 1
            continue
        document, position = decoder.raw_decode(text, position)
        yield document


def samples(node: Any) -> Iterator[tuple[str, set[str]]]:
    """Yield (attribute key, advice types) for every checked attribute."""
    if isinstance(node, dict):
        if "name" in node and "live_check_result" in node:
            advice = (node["live_check_result"] or {}).get("all_advice", [])
            yield node["name"], {a.get("advice_type") or a.get("id") for a in advice}
        for value in node.values():
            yield from samples(value)
    elif isinstance(node, list):
        for value in node:
            yield from samples(value)


def check_unvalidated(checked: list[tuple[str, set[str]]]) -> int:
    caught = [(key, advice - PREFIX_MATCH_ADVICE) for key, advice in checked if advice - PREFIX_MATCH_ADVICE]
    if caught:
        print(f"FAIL: {len(caught)} of {len(checked)} key(s) with an invalid leaf are no longer accepted:")
        for key, advice in caught:
            print(f"  - {key}: {', '.join(sorted(advice))}")
        print("Leaves are being validated. Move these keys to tests/live-check/unknown.json.")
        return 1
    print(f"OK: known gap, all {len(checked)} key(s) with a valid prefix and an invalid leaf are still accepted (#4419).")
    return 0


def main() -> int:
    if len(sys.argv) != 3 or sys.argv[1] not in ("known", "unknown", "unvalidated"):
        sys.exit(__doc__)
    expect_known = sys.argv[1] == "known"
    checked = [s for document in documents(Path(sys.argv[2]).read_text()) for s in samples(document)]
    if not checked:
        print(f"FAIL: no checked attributes found in {sys.argv[2]}")
        return 1
    if sys.argv[1] == "unvalidated":
        return check_unvalidated(checked)
    wrong = [key for key, advice in checked if (MISSING in advice) == expect_known]
    if wrong:
        verdict = "missing from the registry" if expect_known else "unexpectedly found in the registry"
        print(f"FAIL: {len(wrong)} of {len(checked)} sample key(s) {verdict}:")
        for key in wrong:
            print(f"  - {key}")
        return 1
    verdict = "resolve to a registry attribute" if expect_known else "are reported as missing"
    print(f"OK: all {len(checked)} {sys.argv[1]} sample key(s) {verdict}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
