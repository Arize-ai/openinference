"""
Shared GenAI semantic convention constants for OpenInference instrumentations.

This module provides constants from OpenTelemetry's GenAI semantic conventions,
along with additional constants commonly used across instrumentations for mapping
GenAI attributes to OpenInference format.
"""

from opentelemetry.semconv._incubating.attributes.gen_ai_attributes import (
    GEN_AI_OPERATION_NAME,
    GEN_AI_REQUEST_FREQUENCY_PENALTY,
    GEN_AI_REQUEST_MAX_TOKENS,
    GEN_AI_REQUEST_MODEL,
    GEN_AI_REQUEST_PRESENCE_PENALTY,
    GEN_AI_REQUEST_SEED,
    GEN_AI_REQUEST_STOP_SEQUENCES,
    GEN_AI_REQUEST_TEMPERATURE,
    GEN_AI_REQUEST_TOP_K,
    GEN_AI_REQUEST_TOP_P,
    GEN_AI_SYSTEM,
    GEN_AI_TOOL_CALL_ID,
    GEN_AI_TOOL_DESCRIPTION,
    GEN_AI_TOOL_NAME,
    GEN_AI_USAGE_INPUT_TOKENS,
    GEN_AI_USAGE_OUTPUT_TOKENS,
)

# Re-export official OTEL constants
__all__ = [
    "GEN_AI_OPERATION_NAME",
    "GEN_AI_REQUEST_FREQUENCY_PENALTY",
    "GEN_AI_REQUEST_MAX_TOKENS",
    "GEN_AI_REQUEST_MODEL",
    "GEN_AI_REQUEST_PRESENCE_PENALTY",
    "GEN_AI_REQUEST_SEED",
    "GEN_AI_REQUEST_STOP_SEQUENCES",
    "GEN_AI_REQUEST_TEMPERATURE",
    "GEN_AI_REQUEST_TOP_K",
    "GEN_AI_REQUEST_TOP_P",
    "GEN_AI_SYSTEM",
    "GEN_AI_TOOL_CALL_ID",
    "GEN_AI_TOOL_DESCRIPTION",
    "GEN_AI_TOOL_NAME",
    "GEN_AI_USAGE_INPUT_TOKENS",
    "GEN_AI_USAGE_OUTPUT_TOKENS",
    # Additional string constants not yet in OTEL semconv
    "GenAIAttributes",
    "GenAIEventNames",
]


# String constants that aren't yet in the official OTEL package
# These are commonly used across instrumentations
class GenAIAttributes:
    """GenAI attribute names that aren't yet in official OTEL semconv."""

    # Usage attributes (some may be deprecated in favor of input/output_tokens)
    USAGE_PROMPT_TOKENS = "gen_ai.usage.prompt_tokens"  # Deprecated
    USAGE_COMPLETION_TOKENS = "gen_ai.usage.completion_tokens"  # Deprecated
    USAGE_TOTAL_TOKENS = "gen_ai.usage.total_tokens"

    # Agent attributes
    AGENT_NAME = "gen_ai.agent.name"
    AGENT_TOOLS = "gen_ai.agent.tools"

    # Tool attributes
    TOOL_CALL_ID = "gen_ai.tool.call.id"

    # Legacy/deprecated attributes
    PROMPT = "gen_ai.prompt"  # Deprecated
    COMPLETION = "gen_ai.completion"  # Deprecated


class GenAIEventNames:
    """GenAI event names for OTEL events."""

    SYSTEM_MESSAGE = "gen_ai.system.message"
    USER_MESSAGE = "gen_ai.user.message"
    ASSISTANT_MESSAGE = "gen_ai.assistant.message"
    TOOL_MESSAGE = "gen_ai.tool.message"
    CHOICE = "gen_ai.choice"
