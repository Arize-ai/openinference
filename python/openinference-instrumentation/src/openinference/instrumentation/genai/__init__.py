"""
OpenInference -> OpenTelemetry GenAI semantic convention bridge.

This sub-package exposes a set of pure, composable mappers that convert
OpenInference span attributes into their OpenTelemetry GenAI equivalents.

Attribute names and most enumerated values come directly from the upstream
``opentelemetry.semconv._incubating.attributes.gen_ai_attributes`` module.
Enum values that are defined by the GenAI spec but not yet present in the
Python incubating module (``GenAiMessagePartTypeValues``,
``GenAiToolTypeValues``) live in :mod:`.values` until upstream catches up.

The conversion is opt-in: enable it via ``TraceConfig(enable_genai_semconv=True)``
or the ``OPENINFERENCE_ENABLE_GENAI_SEMCONV`` environment variable. When
enabled, the OITracer emits GenAI attributes alongside the existing
OpenInference attributes so that both sets can be consumed in parallel.
"""

from opentelemetry.semconv._incubating.attributes.gen_ai_attributes import (
    GenAiOperationNameValues,
    GenAiOutputTypeValues,
    GenAiProviderNameValues,
)

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
    GenAiMessagePartTypeValues,
    GenAiToolTypeValues,
)

__all__ = [
    "GenAiMessagePartTypeValues",
    "GenAiOperationNameValues",
    "GenAiOutputTypeValues",
    "GenAiProviderNameValues",
    "GenAiToolTypeValues",
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
