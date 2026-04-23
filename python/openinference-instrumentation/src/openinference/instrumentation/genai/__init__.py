"""
OpenInference -> OpenTelemetry GenAI semantic convention bridge.

This sub-package exposes the GenAI attribute names and a set of pure,
composable mappers that convert OpenInference span attributes into their
GenAI equivalents.

The conversion is opt-in: enable it via ``TraceConfig(enable_genai_semconv=True)``
or the ``OPENINFERENCE_ENABLE_GENAI_SEMCONV`` environment variable. When
enabled, the OITracer emits GenAI attributes alongside the existing
OpenInference attributes so that both sets can be consumed in parallel.
"""

from .attributes import GenAIAttributes
from .mappers import (
    MAPPERS,
    convert_oi_to_genai,
    map_agent,
    map_conversation,
    map_invocation_parameters,
    map_messages,
    map_model_name,
    map_provider,
    map_retrieval,
    map_span_kind,
    map_token_counts,
    map_tool_call,
    map_tools,
)
from .values import (
    GenAIMessagePartTypeValues,
    GenAIOperationNameValues,
    GenAIOutputTypeValues,
    GenAIProviderNameValues,
    GenAIToolTypeValues,
)

__all__ = [
    "GenAIAttributes",
    "GenAIMessagePartTypeValues",
    "GenAIOperationNameValues",
    "GenAIOutputTypeValues",
    "GenAIProviderNameValues",
    "GenAIToolTypeValues",
    "MAPPERS",
    "convert_oi_to_genai",
    "map_agent",
    "map_conversation",
    "map_invocation_parameters",
    "map_messages",
    "map_model_name",
    "map_provider",
    "map_retrieval",
    "map_span_kind",
    "map_token_counts",
    "map_tool_call",
    "map_tools",
]
