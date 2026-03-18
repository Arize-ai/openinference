from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from opentelemetry.sdk.trace import ReadableSpan

from openinference.semconv.trace import SpanAttributes


def get_spans_by_kind(spans: Sequence[ReadableSpan], kind: str) -> list[ReadableSpan]:
    return sorted(
        [
            span
            for span in spans
            if span.attributes
            and span.attributes.get(SpanAttributes.OPENINFERENCE_SPAN_KIND) == kind
        ],
        key=lambda span: (span.start_time, span.name),
    )


def pop_prefixed(attributes: dict[str, Any], prefix: str) -> dict[str, Any]:
    matched = {
        key: attributes.pop(key) for key in sorted(list(attributes)) if key.startswith(prefix)
    }
    return matched


INPUT_VALUE = SpanAttributes.INPUT_VALUE
INPUT_MIME_TYPE = SpanAttributes.INPUT_MIME_TYPE
OUTPUT_VALUE = SpanAttributes.OUTPUT_VALUE
OUTPUT_MIME_TYPE = SpanAttributes.OUTPUT_MIME_TYPE
TOOL_NAME = SpanAttributes.TOOL_NAME
TOOL_DESCRIPTION = SpanAttributes.TOOL_DESCRIPTION
TOOL_PARAMETERS = SpanAttributes.TOOL_PARAMETERS
GRAPH_NODE_ID = SpanAttributes.GRAPH_NODE_ID
SESSION_ID = SpanAttributes.SESSION_ID
USER_ID = SpanAttributes.USER_ID
METADATA = SpanAttributes.METADATA
TAG_TAGS = SpanAttributes.TAG_TAGS
LLM_PROMPT_TEMPLATE = SpanAttributes.LLM_PROMPT_TEMPLATE
LLM_PROMPT_TEMPLATE_VERSION = SpanAttributes.LLM_PROMPT_TEMPLATE_VERSION
LLM_PROMPT_TEMPLATE_VARIABLES = SpanAttributes.LLM_PROMPT_TEMPLATE_VARIABLES
OPENINFERENCE_SPAN_KIND = SpanAttributes.OPENINFERENCE_SPAN_KIND
JSON = "application/json"
TEXT = "text/plain"
