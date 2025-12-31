"""
Strands-specific GenAI semantic convention constants.

This module provides constants from OpenTelemetry's GenAI semantic conventions,
along with additional constants commonly used for mapping GenAI attributes
to OpenInference format.

These constants are kept local to the strands instrumentation package to avoid
coupling issues with the volatile gen_ai conventions as different 3rd party tools
update their OTEL usage at different rates.
"""

import json
from typing import Any

from opentelemetry.semconv._incubating.attributes.gen_ai_attributes import (
    GEN_AI_REQUEST_MODEL,
    GEN_AI_TOOL_CALL_ID,
    GEN_AI_TOOL_NAME,
    GEN_AI_USAGE_INPUT_TOKENS,
    GEN_AI_USAGE_OUTPUT_TOKENS,
)

# Re-export official OTEL constants
__all__ = [
    "GEN_AI_REQUEST_MODEL",
    "GEN_AI_TOOL_CALL_ID",
    "GEN_AI_TOOL_NAME",
    "GEN_AI_USAGE_INPUT_TOKENS",
    "GEN_AI_USAGE_OUTPUT_TOKENS",
    # Additional constants
    "GEN_AI_REQUEST_MAX_TOKENS",
    "GEN_AI_REQUEST_TEMPERATURE",
    "GEN_AI_REQUEST_TOP_P",
    "GenAIAttributes",
    "GenAIEventNames",
    "safe_json_dumps",
]

# GenAI request parameter attribute keys (semantic convention names)
# These are not yet in the official OTEL semconv package
GEN_AI_REQUEST_MAX_TOKENS = "gen_ai.request.max_tokens"
GEN_AI_REQUEST_TEMPERATURE = "gen_ai.request.temperature"
GEN_AI_REQUEST_TOP_P = "gen_ai.request.top_p"


class GenAIAttributes:
    """GenAI attribute names used by Strands agents."""

    # Usage attributes
    USAGE_PROMPT_TOKENS = "gen_ai.usage.prompt_tokens"
    USAGE_COMPLETION_TOKENS = "gen_ai.usage.completion_tokens"
    USAGE_TOTAL_TOKENS = "gen_ai.usage.total_tokens"

    # Agent attributes
    AGENT_NAME = "gen_ai.agent.name"
    AGENT_TOOLS = "gen_ai.agent.tools"

    # Tool attributes
    TOOL_CALL_ID = "gen_ai.tool.call.id"

    # Legacy/deprecated attributes
    PROMPT = "gen_ai.prompt"
    COMPLETION = "gen_ai.completion"


class GenAIEventNames:
    """GenAI event names for OTEL events."""

    SYSTEM_MESSAGE = "gen_ai.system.message"
    USER_MESSAGE = "gen_ai.user.message"
    ASSISTANT_MESSAGE = "gen_ai.assistant.message"
    TOOL_MESSAGE = "gen_ai.tool.message"
    CHOICE = "gen_ai.choice"


def safe_json_dumps(obj: Any, **kwargs: Any) -> str:
    """
    A convenience wrapper around `json.dumps` that ensures that any object can
    be safely encoded without a `TypeError` and that non-ASCII Unicode
    characters are not escaped.
    """
    return json.dumps(obj, default=str, ensure_ascii=False, **kwargs)
