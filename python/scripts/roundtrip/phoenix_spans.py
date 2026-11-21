"""Read a Phoenix project's spans back over the REST API and print them as a tree.

    python python/scripts/roundtrip/phoenix_spans.py --project NAME [--phoenix URL] [--attributes]

A round-trip script imports it instead, so it checks what Phoenix stored, not what it exported:

    sys.path.insert(0, "<repo>/python/scripts/roundtrip")
    from phoenix_spans import children, wait_for_spans
    spans = wait_for_spans("http://localhost:6006", project, expected=2)

Standard library only. Each span is Phoenix's JSON: `name`, `context.trace_id`,
`context.span_id`, `parent_id`, `span_kind`, `status_code`, `events`, and `attributes` with
nested keys flattened (`llm.input_messages.0.message.role`, `metadata.<key>`).
"""

import argparse
import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any

Span = dict[str, Any]


def fetch_spans(phoenix: str, project: str) -> list[Span]:
    """Every span in the project, or [] when the project doesn't exist yet."""
    spans: list[Span] = []
    cursor = None
    while True:
        query = urllib.parse.urlencode({"limit": "1000", **({"cursor": cursor} if cursor else {})})
        url = f"{phoenix.rstrip('/')}/v1/projects/{urllib.parse.quote(project)}/spans?{query}"
        try:
            with urllib.request.urlopen(url, timeout=10) as response:
                body = json.load(response)
        except urllib.error.HTTPError as error:
            if error.code == 404:
                return []
            raise
        spans.extend(body["data"])
        cursor = body.get("next_cursor")
        if not cursor:
            return spans


def wait_for_spans(phoenix: str, project: str, expected: int, timeout: float = 60) -> list[Span]:
    """Poll until the project holds at least `expected` spans, then read once more and return.

    Returns whatever arrived by the deadline, so a short list is the caller's failure to report.
    """
    deadline = time.monotonic() + timeout
    spans: list[Span] = []
    while time.monotonic() < deadline:
        spans = fetch_spans(phoenix, project)
        if len(spans) >= expected:
            time.sleep(1)  # let a straggler from the same flush land
            return fetch_spans(phoenix, project)
        time.sleep(0.5)
    return spans


def children(spans: list[Span], parent: Span) -> list[Span]:
    """Spans whose parent is `parent`, in start order."""
    span_id = parent["context"]["span_id"]
    return sorted(
        (s for s in spans if s.get("parent_id") == span_id), key=lambda s: s["start_time"]
    )


def roots(spans: list[Span]) -> list[Span]:
    """Spans with no parent stored in `spans`, in start order."""
    ids = {s["context"]["span_id"] for s in spans}
    return sorted(
        (s for s in spans if s.get("parent_id") not in ids), key=lambda s: s["start_time"]
    )


def render(spans: list[Span], attributes: bool = False) -> str:
    lines: list[str] = []

    def walk(span: Span, depth: int) -> None:
        pad = "  " * depth
        lines.append(f"{pad}{span['name']} [{span['span_kind']} {span['status_code']}]")
        if attributes:
            for key, value in sorted((span.get("attributes") or {}).items()):
                lines.append(f"{pad}    {key} = {json.dumps(value, default=str)[:200]}")
            for event in span.get("events") or []:
                lines.append(f"{pad}    event {event['name']}")
        for child in children(spans, span):
            walk(child, depth + 1)

    for root in roots(spans):
        walk(root, 0)
    return "\n".join(lines)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--project", required=True)
    parser.add_argument(
        "--phoenix", default=os.environ.get("PHOENIX_HOST", "http://localhost:6006")
    )
    parser.add_argument("--attributes", action="store_true", help="print attributes and events")
    parser.add_argument("--json", help="also write the raw spans to this path")
    args = parser.parse_args()
    found = fetch_spans(args.phoenix, args.project)
    if args.json:
        with open(args.json, "w") as f:
            json.dump(found, f, indent=2)
    print(f"{len(found)} spans in {args.project}")
    print(render(found, args.attributes))
