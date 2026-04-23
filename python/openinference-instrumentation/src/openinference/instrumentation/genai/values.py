"""
Enumerated values used by the OpenInference -> GenAI bridge that are defined
by the OpenTelemetry GenAI semantic conventions spec but not yet available in
``opentelemetry.semconv._incubating.attributes.gen_ai_attributes``.

Values that are already exposed by the incubating module (e.g.
``GenAiOperationNameValues``, ``GenAiProviderNameValues``,
``GenAiOutputTypeValues``) are imported directly from there and are not
re-defined in this module.
"""

from enum import Enum


class GenAiMessagePartTypeValues(Enum):
    """Values for the ``type`` field of message parts inside
    ``gen_ai.input.messages`` / ``gen_ai.output.messages``.
    """

    TEXT = "text"
    TOOL_CALL = "tool_call"
    TOOL_CALL_RESPONSE = "tool_call_response"
    REASONING = "reasoning"
    SERVER_TOOL_CALL = "server_tool_call"
    SERVER_TOOL_CALL_RESPONSE = "server_tool_call_response"
    BLOB = "blob"
    URI = "uri"
    FILE = "file"


class GenAiToolTypeValues(Enum):
    """Values for ``gen_ai.tool.type``."""

    FUNCTION = "function"
    EXTENSION = "extension"
    DATASTORE = "datastore"
